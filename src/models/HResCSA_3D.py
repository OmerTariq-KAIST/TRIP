"""
Enhanced HResCSA model with support for both 2D and 3D trajectory prediction
"""

import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from torchstat import stat


def conv_dw(in_planes, out_planes, kernel_size, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv1d(in_planes, in_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size // 2) * dilation, dilation=dilation, groups=in_planes, bias=False),
        nn.Conv1d(in_planes, out_planes, kernel_size=1, bias=False)
    )

class TemporalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super(TemporalSelfAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        # Transpose to match the shape (batch_size, sequence_length, channels) -> (batch_size, channels, sequence_length)
        x = x.transpose(1, 2)
        out, _ = self.attn(x, x, x)
        return out.transpose(1, 2)  # Return to (batch_size, sequence_length, channels)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, dilation=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = conv_dw(in_planes, out_planes, kernel_size, stride, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_dw(out_planes, out_planes, kernel_size, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_planes)

        # Using temporal attention
        self.ta = TemporalSelfAttention(out_planes)
        self.ca = ChannelAttention(out_planes)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply attention mechanisms
        out = self.ca(out) * out  # Channel attention
        out = self.ta(out) + out  # Temporal self-attention

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, dilation=1, downsample=None):
        super(Bottleneck1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, in_planes, kernel_size=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.conv2 = conv_dw(out_planes, out_planes, kernel_size, stride, dilation)
        self.bn2 = nn.BatchNorm1d(out_planes)
        self.conv3 = nn.Conv1d(out_planes, out_planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.ta = TemporalSelfAttention(out_planes * 4)
        self.ca = ChannelAttention(out_planes * 4)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Apply attention mechanisms
        out = self.ca(out) * out  # Channel attention
        out = self.ta(out) + out  # Temporal self-attention

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class MLPOutputModule(nn.Module):
    """
    Enhanced MLP output module supporting both 2D and 3D output
    """
    def __init__(self, in_planes, num_outputs, **kwargs):
        super(MLPOutputModule, self).__init__()
        mlp_dim = kwargs.get('mlp_dim', 512)
        num_layers = kwargs.get('num_layers', 2)
        dropout = kwargs.get('dropout', 0.3)

        # Global average pooling to reduce temporal dimension
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Transition layer for dimensionality reduction if specified
        trans_planes = kwargs.get('trans_planes', None)
        if trans_planes is not None:
            self.transition = nn.Sequential(
                nn.Conv1d(in_planes, trans_planes, kernel_size=1, bias=False),
                nn.BatchNorm1d(trans_planes))
            in_planes = trans_planes
        else:
            self.transition = None

        # Adaptive MLP layers based on output dimension
        mlp_layers = []
        for i in range(num_layers - 1):
            mlp_layers.append(nn.Linear(in_planes if i == 0 else mlp_dim, mlp_dim))
            mlp_layers.append(nn.ReLU(True))
            mlp_layers.append(nn.Dropout(dropout))
        
        # Final output layer
        mlp_layers.append(nn.Linear(mlp_dim, num_outputs))
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Store output dimension for reference
        self.num_outputs = num_outputs

    def get_dropout(self):
        return [m for m in self.mlp if isinstance(m, nn.Dropout)]

    def forward(self, x):
        if self.transition is not None:
            x = self.transition(x)
        x = self.global_avg_pool(x)  # Perform global average pooling
        x = x.view(x.size(0), -1)    # Flatten to (batch_size, channels)
        return self.mlp(x)            # Pass through the MLP layers

class GlobAvgOutputModule(nn.Module):
    """
    Global average output module supporting configurable output dimensions
    """
    def __init__(self, in_planes, num_outputs):
        super(GlobAvgOutputModule, self).__init__()
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_planes, num_outputs)
        self.num_outputs = num_outputs

    def get_dropout(self):
        return []

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class HResCSA(nn.Module):
    """
    Enhanced HResCSA supporting both 2D and 3D trajectory prediction
    """
    def __init__(self, num_inputs, num_outputs, block_type, group_sizes, base_plane=64, 
                 output_block=None, zero_init_residual=False, **kwargs):
        super(HResCSA, self).__init__()
        self.base_plane = base_plane
        self.inplanes = self.base_plane
        self.num_outputs = num_outputs

        # Input module
        self.input_block = nn.Sequential(
            nn.Conv1d(num_inputs, self.inplanes, kernel_size=5, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # Residual groups
        self.planes = [self.base_plane * (2 ** i) for i in range(len(group_sizes))]
        kernel_size = kwargs.get('kernel_size', 3)
        strides = [1] + [2] * (len(group_sizes) - 1)
        dilations = [1] * len(group_sizes)
        groups = [self._make_residual_group1d(block_type, self.planes[i], kernel_size, group_sizes[i], strides[i], dilations[i])
                  for i in range(len(group_sizes))]
        self.residual_groups = nn.Sequential(*groups)

        # Output module with configurable dimensions
        if output_block is None:
            self.output_block = GlobAvgOutputModule(self.planes[-1] * block_type.expansion, num_outputs)
        else:
            self.output_block = output_block(self.planes[-1] * block_type.expansion, num_outputs, **kwargs)

        self._initialize(zero_init_residual)
        
        print(f"HResCSA initialized with {num_outputs}D output (num_outputs={num_outputs})")

    def _make_residual_group1d(self, block_type, planes, kernel_size, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_type.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block_type.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block_type.expansion))
        layers = []
        layers.append(block_type(self.inplanes, planes, kernel_size=kernel_size, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block_type.expansion
        for _ in range(1, blocks):
            layers.append(block_type(self.inplanes, planes, kernel_size=kernel_size, dilation=dilation))

        return nn.Sequential(*layers)

    def _initialize(self, zero_init_residual):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck1D):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock1D):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x = self.input_block(x)
        x = self.residual_groups(x)
        x = self.output_block(x)
        return x

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def reset_output_block(self, block_type, num_outputs, **kwargs):
        """
        Reset output block for different output dimensions
        """
        self.output_block = MLPOutputModule(self.planes[-1] * block_type.expansion, num_outputs, **kwargs)
        self.num_outputs = num_outputs
        
        # Initialize the new output block
        self.output_block.apply(self._initialize)
        print(f'Output block reset for {num_outputs}D prediction')

# Factory function for easy model creation
def create_hrescsa(num_inputs=6, num_outputs=2, **kwargs):
    """
    Factory function to create HResCSA model
    
    Args:
        num_inputs: Number of input channels (default: 6 for IMU)
        num_outputs: Number of output dimensions (2 for 2D, 3 for 3D)
        **kwargs: Additional arguments for model configuration
    """
    mlp_config = kwargs.get('mlp_config', {
        'mlp_dim': 512,
        'num_layers': 2,
        'dropout': 0.3,
        'trans_planes': 128
    })
    
    model = HResCSA(
        num_inputs=num_inputs, 
        num_outputs=num_outputs, 
        block_type=BasicBlock1D, 
        group_sizes=[2, 2, 2, 2], 
        base_plane=64, 
        output_block=MLPOutputModule, 
        kernel_size=3, 
        **mlp_config
    )
    
    return model

if __name__ == '__main__':
    # Test both 2D and 3D models
    print("Testing 2D model:")
    net_2d = create_hrescsa(num_inputs=6, num_outputs=2)
    x_image = Variable(torch.randn(1, 6, 200))
    y_2d = net_2d(x_image)
    print(f"2D output shape: {y_2d.shape}")
    print(f"2D model parameters: {net_2d.get_num_params()}")
    
    print("\nTesting 3D model:")
    net_3d = create_hrescsa(num_inputs=6, num_outputs=3)
    y_3d = net_3d(x_image)
    print(f"3D output shape: {y_3d.shape}")
    print(f"3D model parameters: {net_3d.get_num_params()}")
    
    # Test FLOPs if available
    try:
        from pthflops import count_ops
        print(f"\n2D Model FLOPs: {count_ops(net_2d, x_image)}")
        print(f"3D Model FLOPs: {count_ops(net_3d, x_image)}")
    except ImportError:
        print("\npthflops not available for FLOP counting")