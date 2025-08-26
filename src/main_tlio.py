"""
👨‍💻 Copyright (C) $2024 Omer Tariq KAIST. - All Rights Reserved

Project: TRIP

"""

import os
import time
from os import path as osp
import numpy as np
import torch
import json
import math

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import interp1d
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from sklearn import metrics
from preprocess.data_processor import *
from metric_tlio import compute_ate_rte, compute_absolute_trajectory_error, compute_drift
# from models.MobileNetV2 import *
# from models.MobileNet import *
# from models.MnasNet import *
# from models.IMUNet import *
# from models.EfficientnetB0 import *
#from models.ResNet1D_dws import *
from models.HResCSA import *
from utils import *
import argparse
args = argparse.Namespace()
from torch.autograd import Variable

_fc_config = {'fc_dim': 512, 'in_dim': 7, 'dropout': 0.5, 'trans_planes': 128}

# Define MLP-specific configuration
_mlp_config = {
    'mlp_dim': 512,         # Hidden dimension for MLP layers
    'num_layers': 2,        # Number of MLP layers
    'dropout': 0.3,         # Dropout probability for MLP layers
    'trans_planes': 128     # Optional: dimensionality reduction in transition layer
}

def get_model(arch):
    global _output_channel, _input_channel
    
    # Set defaults if not initialized
    if '_output_channel' not in globals() or _output_channel is None:
        n_class = 2
        print(f"Warning: _output_channel not set, using default n_class = {n_class}")
    else:
        n_class = _output_channel
        print(f"Using n_class = {n_class} based on dataset target dimension")
    
    # For TLIO, we're using averaged features, so input is still 6
    num_inputs = 6
    
    arch = args.arch
    if arch == 'HResCSA':
        network = HResCSA(num_inputs=num_inputs, num_outputs=n_class, block_type=BasicBlock1D, 
                         group_sizes=[2, 2, 2, 2], base_plane=64, output_block=MLPOutputModule, 
                         kernel_size=3, **_mlp_config)
    else:
        raise ValueError('Invalid architecture: ', args.arch)
    return network


def run_test(network, data_loader, device, eval_mode=True):
    """
    Updated run_test function to handle both TRIP and TLIO data formats.
    """
    targets_all = []
    preds_all = []
    if eval_mode:
        network.eval()
    
    for bid, batch in enumerate(data_loader):
        with torch.no_grad():
            # Handle different batch formats
            if isinstance(batch, dict):
                # TLIO format
                feat = batch["feats"]["imu0"]  # [B, 6, T]
                targ = batch["targ_dt_World"]  # [B, T, 3] or [B, 1, 3]
                
                # Handle target shape variations
                if len(targ.shape) == 3:
                    if targ.shape[1] == 1:
                        targ = targ.squeeze(1)  # [B, 3]
                    else:
                        targ = targ[:, -1, :]  # [B, 3] - take last timestep
                elif len(targ.shape) == 2:
                    pass  # Already [B, 3]
                else:
                    targ = targ.view(-1, 3)  # Reshape to [B, 3]
                    
                # Ensure feature format is correct for the network
                if len(feat.shape) == 3:  # [B, 6, T]
                    if feat.shape[2] != network.window_size if hasattr(network, 'window_size') else 200:
                        # Pad or truncate if needed
                        target_size = getattr(network, 'window_size', 200)
                        if feat.shape[2] < target_size:
                            padding = torch.zeros(feat.shape[0], feat.shape[1], 
                                                target_size - feat.shape[2], 
                                                device=feat.device)
                            feat = torch.cat([feat, padding], dim=2)
                        else:
                            feat = feat[:, :, :target_size]
                elif len(feat.shape) == 2:  # [B, 6]
                    # Expand to sequence format
                    target_size = getattr(network, 'window_size', 200)
                    feat = feat.unsqueeze(2).repeat(1, 1, target_size)
                    
            else:
                # Original TRIP format: (feat, targ, _, _)
                feat, targ, _, _ = batch
            
            feat, targ = feat.to(device), targ.to(device)
            
            # Get prediction
            if hasattr(network, 'forward') and callable(getattr(network, 'forward')):
                pred = network(feat).cpu().detach().numpy()
            else:
                # Handle different network output formats
                output = network(feat)
                if isinstance(output, tuple):
                    pred = output[0].cpu().detach().numpy()  # Take first output (mean)
                else:
                    pred = output.cpu().detach().numpy()
            
            targets_all.append(targ.cpu().detach().numpy())
            preds_all.append(pred)

    targets_all = np.concatenate(targets_all, axis=0)
    preds_all = np.concatenate(preds_all, axis=0)
    return targets_all, preds_all

def add_summary(writer, loss, step, mode):
    names = '{0}_loss/loss_x,{0}_loss/loss_y,{0}_loss/loss_z,{0}_loss/loss_sin,{0}_loss/loss_cos'.format(
        mode).split(',')

    for i in range(loss.shape[0]):
        writer.add_scalar(names[i], loss[i], step)
    writer.add_scalar('{}_loss/avg'.format(mode), np.mean(loss), step)


# Moving average function for test
def simple_moving_average(data, window_size):
    """
    Calculate the simple moving average of a given data array using a specified window size.
    
    Args:
        data: The input data array.
        window_size: The size of the moving window.
    
    Returns:
        The array of simple moving averages.
    """
    cumsum = np.cumsum(data)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size


# Add this import at the top of main.py after other imports
from torch.utils.data import Dataset

# Replace the get_dataset_from_list function in main.py with this enhanced version
def get_dataset_from_list_with_tlio(root_dir, list_path, args, mode, **kwargs):
    """
    Enhanced get_dataset_from_list function that supports TLIO datasets
    """
    if args.dataset == 'tlio':
        # For TLIO, use the root_dir as the dataset path
        data_list = ['']  # TLIO sequence will handle the data loading
        
    elif args.dataset == 'oxiod':
        if mode == 'train':
            root_dir = root_dir + '/train'
            data_list = os.listdir(root_dir)
        else:
            root_dir = root_dir + '/validation'
            data_list = os.listdir(root_dir)
    else:
        # Handle other datasets
        if list_path and osp.exists(list_path):
            with open(list_path) as f:
                data_list = [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
        else:
            print(f"Warning: List file not found: {list_path}")
            data_list = ['']  # Fallback

    return get_dataset(root_dir, data_list, args, mode=mode, **kwargs)


def get_dataset(root_dir, data_list, args, **kwargs):
    """
    get_dataset function that supports TLIO datasets
    """
    mode = kwargs.get('mode', 'train')

    random_shift, shuffle, transforms, grv_only = 0, False, None, False
    if mode == 'train':
        random_shift = args.step_size // 2
        shuffle = True
        transforms = RandomHoriRotate(math.pi * 2)
    elif mode == 'val':
        shuffle = True
    elif mode == 'test':
        shuffle = False
        grv_only = True

    # Select sequence type based on dataset
    if args.dataset == 'ronin':
        seq_type = GlobSpeedSequence
    elif args.dataset == 'ridi':
        seq_type = RIDIGlobSpeedSequence
    elif args.dataset == 'oxiod':
        seq_type = OXIODSequence
    elif args.dataset == 'tlio':  # Add TLIO support
        from preprocess.tlio_integration import TLIOSequence  # Import from the integration file
        seq_type = TLIOSequence
    elif args.dataset in ['imunet', 'kiod', 'inaiod']:
        seq_type = ProposedSequence
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    dataset = StridedSequenceDataset(
        seq_type, root_dir, data_list, args.cache_path, args.step_size, args.window_size,
        random_shift=random_shift, transform=transforms,
        shuffle=shuffle, grv_only=grv_only, max_ori_error=args.max_ori_error)

    global _input_channel, _output_channel
    _input_channel, _output_channel = dataset.feature_dim, dataset.target_dim
    return dataset

def get_dataset_from_list(root_dir, list_path, args, mode, **kwargs):
    if args.dataset == 'oxiod':
        if (mode == 'train'):
            root_dir = root_dir + '/train'
            data_list = os.listdir(root_dir)
        else:
            root_dir = root_dir + '/validation'
            data_list = os.listdir(root_dir)
    else:
        with open(list_path) as f:
            data_list = [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']

    return get_dataset(root_dir, data_list, args, **kwargs)


# Fix for the tensor shape issue in main.py

# Replace the training loop section in the train() function with this:

def train(args, **kwargs):
    import os
    import os.path as osp
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter

    start_t = time.time()
    print(args.root_dir)

    # Handle TLIO vs other datasets differently
    if args.dataset == 'tlio':
        print("Using TLIO's native dataloader...")
        
        # Import and use TLIO's native dataloader
        try:
            from dataloader.tlio_data import TlioData
            
            # Create TLIO data object with proper configuration
            tlio_data = TlioData(
                data_path=args.root_dir,
                batch_size=args.batch_size,
                num_workers=4,
                persistent_workers=True,
                decimator=args.step_size,
                dataset_style="mmap",  # Use memory mapping for large datasets
                data_window_config={
                    "window_size": args.window_size,
                    "step_period_us": 5000,  # 5ms = 200Hz
                    "data_in_t0_yaw_normalized_frame": False,
                    "input_sensors": ["imu0"],
                    "data_style": "resampled",
                },
                augmentation_options={
                    "do_bias_shift": True,
                    "bias_shift_options": {
                        "accel_bias_range": 0.2,
                        "gyro_bias_range": 0.05,
                        "accel_noise_std": 0.02,
                        "gyro_noise_std": 0.01,
                    },
                    "perturb_gravity": True,
                    "perturb_gravity_theta_range": 5.0,
                },
            )
            
            # Prepare TLIO data
            tlio_data.prepare_data()
            train_loader = tlio_data.train_dataloader()
            val_loader = tlio_data.val_dataloader() if args.val_list else None
            
            # Set dimensions for TLIO
            global _input_channel, _output_channel
            _input_channel, _output_channel = 6, 3
            
            # Create dummy dataset objects for compatibility with existing logging
            class TLIODatasetInfo:
                def __init__(self):
                    self.feature_dim = 6
                    self.target_dim = 3
                def __len__(self):
                    return len(train_loader.dataset) if hasattr(train_loader, 'dataset') else 10000
            
            train_dataset = TLIODatasetInfo()
            val_dataset = TLIODatasetInfo() if val_loader else None
            
            print(f"TLIO dataset loaded using native dataloader")
            
        except ImportError as e:
            print(f"Error importing TLIO: {e}")
            raise ImportError("TLIO dependencies not available. Please install TLIO.")
            
    else:
        # Use TRIP's original dataloader for non-TLIO datasets
        print(f"Using TRIP's original dataloader for {args.dataset}...")
        train_dataset = get_dataset_from_list(args.root_dir, args.train_list, args, mode='train')
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        
        val_dataset = None
        val_loader = None
        if args.val_list is not None and args.val_list != '':
            val_dataset = get_dataset_from_list(args.root_dir, args.val_list, args, mode='val')
            val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    end_t = time.time()
    print(f'Dataset loaded. Feature size: {train_dataset.feature_dim}, target size: {train_dataset.target_dim}. Time usage: {end_t - start_t:.3f}s')
    print(f'Number of train samples: {len(train_dataset)}')
    
    if val_dataset:
        print(f'Number of val samples: {len(val_dataset)}')

    device = torch.device('cuda:0' if torch.cuda.is_available() and not args.cpu else 'cpu')

    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)
        os.makedirs(osp.join(args.out_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(osp.join(args.out_dir, 'logs'), exist_ok=True)

    # Create model AFTER dataset is loaded so _output_channel is set correctly
    network = get_model(args).to(device)
    print(f"Model created with output dimension: {_output_channel}")
    print(f"Model parameters: {network.get_num_params()}")
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, eps=1e-12)

    start_epoch = 0
    if args.continue_from is not None and osp.exists(args.continue_from):
        checkpoints = torch.load(args.continue_from, weights_only=False) 
        start_epoch = checkpoints.get('epoch', 0)
        network.load_state_dict(checkpoints['model_state_dict'])
        optimizer.load_state_dict(checkpoints['optimizer_state_dict'])

    summary_writer = SummaryWriter(osp.join(args.out_dir, 'logs')) if args.out_dir else None

    step = 0
    best_val_metric = np.inf
    train_losses_all, val_losses_all, diff_losses_all = [], [], []
    best_avg_diff_loss = np.inf

    for epoch in range(start_epoch, args.epochs):
        start_t = time.time()
        network.train()
        train_outs, train_targets = [], []
        
        print(f"Starting epoch {epoch+1}/{args.epochs}")
        print(f"Total batches in epoch: {len(train_loader)}")
        
        for batch_id, batch in enumerate(train_loader):
            # Add progress monitoring every 100 batches
            if batch_id % 100 == 0:
                batch_time = time.time() - start_t
                avg_batch_time = batch_time / (batch_id + 1) if batch_id > 0 else 0
                eta_minutes = (len(train_loader) - batch_id) * avg_batch_time / 60
                print(f"  Batch {batch_id}/{len(train_loader)} (Epoch {epoch}) - ETA: {eta_minutes:.1f}min")
            
            # Handle different batch formats
            if args.dataset == 'tlio':
                # TLIO native format: batch is a dict
                feat = batch["feats"]["imu0"]  # [B, 6, T]
                targ = batch["targ_dt_World"]  # [B, 1, 3] or similar
                
                # Handle different target shapes from TLIO
                if len(targ.shape) == 3:
                    if targ.shape[1] == 1:
                        targ = targ.squeeze(1)  # [B, 3]
                    else:
                        targ = targ[:, -1, :]  # [B, 3] - take last timestep
                elif len(targ.shape) == 2:
                    pass  # Already [B, 3]
                else:
                    targ = targ.view(-1, 3)  # Reshape to [B, 3]
                
                # CRITICAL FIX: Handle TLIO features for 1D CNN
                if len(feat.shape) == 3:  # [B, 6, T]
                    # For HResCSA (1D CNN), we need to ensure proper input format
                    # Option 1: Use the full sequence if T matches window_size
                    if feat.shape[2] == args.window_size:
                        # Keep as [B, 6, T] - this is correct for 1D CNN
                        pass
                    else:
                        # Option 2: Pad or truncate to match window_size
                        if feat.shape[2] < args.window_size:
                            # Pad with zeros
                            padding = torch.zeros(feat.shape[0], feat.shape[1], 
                                                args.window_size - feat.shape[2], 
                                                device=feat.device)
                            feat = torch.cat([feat, padding], dim=2)
                        else:
                            # Truncate to window_size
                            feat = feat[:, :, :args.window_size]
                elif len(feat.shape) == 2:  # [B, 6]
                    # If we only have averaged features, expand to create a sequence
                    # Repeat the features across the time dimension
                    feat = feat.unsqueeze(2).repeat(1, 1, args.window_size)  # [B, 6, window_size]
                    
            else:
                # TRIP format: (feat, targ, seq_id, frame_id)
                feat, targ, _, _ = batch
            
            feat, targ = feat.to(device), targ.to(device)
            
            # Debug: print shapes for first batch
            if epoch == 0 and batch_id == 0:
                print(f"Debug - First batch shapes after processing:")
                print(f"  Input features: {feat.shape}")
                print(f"  Target: {targ.shape}")
                print(f"  Expected input format for HResCSA: [batch_size, 6, sequence_length]")
            
            optimizer.zero_grad()
            pred = network(feat)
            
            # Debug: print prediction shape for first batch
            if epoch == 0 and batch_id == 0:
                print(f"  Model prediction: {pred.shape}")
                print(f"  Shapes match: {pred.shape == targ.shape}")
            
            loss = criterion(pred, targ)
            diff = torch.abs(pred - targ)
            mean_diff = torch.mean(diff)

            total_loss = loss + mean_diff
            total_loss.backward()
            optimizer.step()
            
            train_outs.append(pred.cpu().detach().numpy())
            train_targets.append(targ.cpu().detach().numpy())
            step += 1
            
            # Show timing estimate after first batch
            if batch_id == 0:
                first_batch_time = time.time() - start_t
                estimated_epoch_time_hours = (first_batch_time * len(train_loader)) / 3600
                print(f"  First batch took {first_batch_time:.2f}s")
                print(f"  Estimated epoch time: {estimated_epoch_time_hours:.2f} hours")

        train_outs = np.concatenate(train_outs, axis=0)
        train_targets = np.concatenate(train_targets, axis=0)
        train_losses = np.mean((train_outs - train_targets) ** 2, axis=0)
        mean_diff_losses = np.mean(np.abs(train_outs - train_targets), axis=0)
        train_losses_all.append(np.mean(train_losses))
        diff_losses_all.append(np.mean(mean_diff_losses))

        print(f"DEBUG: Checking checkpoint saving...")
        print(f"DEBUG: args.out_dir = '{args.out_dir}'")
        print(f"DEBUG: Current train loss = {np.mean(train_losses):.6f}")

        # Initialize best_train_loss if not exists
        if 'best_train_loss' not in locals():
            best_train_loss = np.inf
            print(f"DEBUG: Initializing best_train_loss = {best_train_loss}")

        # Check if this is the best model so far
        current_train_loss = np.mean(train_losses)
        if current_train_loss < best_train_loss:
            best_train_loss = current_train_loss
            print(f"DEBUG: NEW BEST! Saving checkpoint...")
            
            if args.out_dir and args.out_dir != '':
                try:
                    # Make sure checkpoints directory exists
                    checkpoint_dir = osp.join(args.out_dir, 'checkpoints')
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    
                    model_path = osp.join(checkpoint_dir, 'checkpoint_best.pt')
                    torch.save({
                        'model_state_dict': network.state_dict(), 
                        'epoch': epoch, 
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_train_loss': best_train_loss,
                        'train_loss': current_train_loss,
                        'diff_loss': np.mean(mean_diff_losses),
                    }, model_path)
                    print(f'✅ BEST MODEL SAVED: {model_path}')
                except Exception as e:
                    print(f'❌ ERROR saving best model: {e}')
                    import traceback
                    traceback.print_exc()
            else:
                print(f'❌ ERROR: args.out_dir is empty or None!')
        else:
            print(f"DEBUG: No improvement. Current: {current_train_loss:.6f}, Best: {best_train_loss:.6f}")

        # Periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"DEBUG: Epoch {epoch+1} - saving periodic checkpoint...")
            
            if args.out_dir and args.out_dir != '':
                try:
                    # Make sure checkpoints directory exists
                    checkpoint_dir = osp.join(args.out_dir, 'checkpoints')
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    
                    periodic_path = osp.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
                    torch.save({
                        'model_state_dict': network.state_dict(), 
                        'epoch': epoch, 
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': current_train_loss,
                        'diff_loss': np.mean(mean_diff_losses),
                    }, periodic_path)
                    print(f'✅ PERIODIC CHECKPOINT SAVED: {periodic_path}')
                except Exception as e:
                    print(f'❌ ERROR saving periodic checkpoint: {e}')
                    import traceback
                    traceback.print_exc()
            else:
                print(f'❌ ERROR: args.out_dir is empty or None!')

        # Add this to also save immediately since we're at epoch 37+
        if epoch >= 37:  # Force save since we're already deep in training
            print(f"DEBUG: Force saving current model at epoch {epoch}...")
            if args.out_dir and args.out_dir != '':
                try:
                    checkpoint_dir = osp.join(args.out_dir, 'checkpoints')
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    
                    force_path = osp.join(checkpoint_dir, f'checkpoint_epoch_{epoch}_force.pt')
                    torch.save({
                        'model_state_dict': network.state_dict(), 
                        'epoch': epoch, 
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': current_train_loss,
                        'diff_loss': np.mean(mean_diff_losses),
                    }, force_path)
                    print(f'✅ FORCE CHECKPOINT SAVED: {force_path}')
                except Exception as e:
                    print(f'❌ ERROR force saving: {e}')
                    import traceback
                    traceback.print_exc()





        if summary_writer:
            summary_writer.add_scalar('Loss/train', np.mean(train_losses), epoch)
            summary_writer.add_scalar('Diff/train', np.mean(mean_diff_losses), epoch)
        
        # Validation phase
        if val_loader:
            network.eval()
            val_outs, val_targets = [], []
            
            with torch.no_grad():
                for _, batch in enumerate(val_loader):
                    # Handle different batch formats for validation
                    if args.dataset == 'tlio':
                        feat = batch["feats"]["imu0"]
                        targ = batch["targ_dt_World"]
                        
                        # Handle target shapes
                        if len(targ.shape) == 3:
                            if targ.shape[1] == 1:
                                targ = targ.squeeze(1)
                            else:
                                targ = targ[:, -1, :]
                        elif len(targ.shape) == 2:
                            pass
                        else:
                            targ = targ.view(-1, 3)
                            
                        # Handle feature shapes for validation (same as training)
                        if len(feat.shape) == 3:  # [B, 6, T]
                            if feat.shape[2] == args.window_size:
                                pass
                            else:
                                if feat.shape[2] < args.window_size:
                                    padding = torch.zeros(feat.shape[0], feat.shape[1], 
                                                        args.window_size - feat.shape[2], 
                                                        device=feat.device)
                                    feat = torch.cat([feat, padding], dim=2)
                                else:
                                    feat = feat[:, :, :args.window_size]
                        elif len(feat.shape) == 2:  # [B, 6]
                            feat = feat.unsqueeze(2).repeat(1, 1, args.window_size)
                    else:
                        feat, targ, _, _ = batch
                    
                    feat, targ = feat.to(device), targ.to(device)
                    pred = network(feat)
                    val_outs.append(pred.cpu().detach().numpy())
                    val_targets.append(targ.cpu().detach().numpy())

            val_outs = np.concatenate(val_outs, axis=0)
            val_targets = np.concatenate(val_targets, axis=0)
            val_losses = np.mean((val_outs - val_targets) ** 2, axis=0)
            diff_losses = np.mean(np.abs(val_outs - val_targets), axis=0)
            val_losses_all.append(np.mean(val_losses))
            diff_losses_all.append(np.mean(diff_losses))
            print(f'Epoch {epoch}, Validation Loss: {np.mean(val_losses):.6f}, Differential Loss: {np.mean(diff_losses):.6f}')
            
            scheduler.step(np.mean(val_losses))

            if np.mean(diff_losses) < best_avg_diff_loss:
                best_avg_diff_loss = np.mean(diff_losses)
                model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_best.pt')
                torch.save({'model_state_dict': network.state_dict(), 'epoch': epoch, 'optimizer_state_dict': optimizer.state_dict()}, model_path)
                print(f'New best model saved based on differential loss to {model_path}')

    print('Training complete')
    final_model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_latest.pt')
    torch.save({'model_state_dict': network.state_dict(), 'epoch': epoch, 'optimizer_state_dict': optimizer.state_dict()}, final_model_path)
    print(f'Checkpoint saved to {final_model_path}')

    return train_losses_all, val_losses_all, diff_losses_all


# def recon_traj_with_preds(dataset, preds, seq_id=0, **kwargs):
#     """
#     Reconstruct trajectory with predicted global velocities.
#     Updated to support both 2D and 3D trajectories.
#     """
#     # Determine if we're using TLIO (3D) or original datasets (2D)
#     use_3d = hasattr(dataset, 'get_ts_last_imu_us') and hasattr(dataset, 'get_gt_traj_center_window_times')
#     output_dim = preds.shape[1]  # Should be 2 for 2D datasets, 3 for TLIO
    
#     if use_3d:
#         # TLIO dataset - use native methods
#         ts = dataset.get_ts_last_imu_us(seq_id) * 1e-6  # Convert to seconds
        
#         # Get ground truth trajectory
#         if hasattr(dataset, 'get_gt_traj_center_window_times'):
#             try:
#                 gt_data = dataset.get_gt_traj_center_window_times(seq_id)
#                 if isinstance(gt_data, tuple) and len(gt_data) == 2:
#                     # Returns (rotation, position)
#                     gt_rot, gt_pos = gt_data
#                     pos_gt = gt_pos  # Already in correct format
#                 else:
#                     # Returns SE3 matrices
#                     gt_traj_SE3 = gt_data
#                     pos_gt = gt_traj_SE3[:, :3, 3]  # Extract positions
#             except:
#                 # Fallback: create dummy ground truth
#                 print("Warning: Could not extract ground truth trajectory, using dummy data")
#                 pos_gt = np.zeros((len(ts), 3))
#         else:
#             pos_gt = np.zeros((len(ts), 3))
            
#         # Calculate time differences
#         if len(ts) > 1:
#             dts = np.mean(ts[1:] - ts[:-1])
#         else:
#             dts = 0.005  # Default 200Hz
            
#         # Reconstruct trajectory from velocity predictions
#         pos = np.zeros([preds.shape[0] + 1, output_dim])
#         if len(pos_gt) > 0:
#             pos[0] = pos_gt[0, :output_dim]  # Initialize with first GT position
        
#         # Integrate velocities
#         pos[1:] = np.cumsum(preds * dts, axis=0) + pos[0]
        
#         # Interpolate if needed to match ground truth timestamps
#         if len(pos) != len(pos_gt):
#             from scipy.interpolate import interp1d
#             ts_pred = np.linspace(ts[0], ts[-1], len(pos))
#             if len(ts) > 1:
#                 pos_interp = interp1d(ts_pred, pos, axis=0, fill_value="extrapolate")(ts)
#                 pos = pos_interp
            
#     else:
#         # Original TRIP datasets (2D)
#         if hasattr(dataset, 'ts') and hasattr(dataset, 'gt_pos'):
#             ts = dataset.ts[seq_id]
#             gt_pos = dataset.gt_pos[seq_id]
#         else:
#             # Fallback for datasets without direct access
#             ts = np.arange(len(preds)) * 0.05  # Assume 20Hz
#             gt_pos = np.zeros((len(preds), 2))
            
#         ind = np.array([i[1] for i in dataset.index_map if i[0] == seq_id], dtype=int)
#         dts = np.mean(ts[ind[1:]] - ts[ind[:-1]]) if len(ind) > 1 else 0.05
        
#         pos = np.zeros([preds.shape[0] + 2, output_dim])
#         pos[0] = gt_pos[0, :output_dim]
#         pos[1:-1] = np.cumsum(preds[:, :output_dim] * dts, axis=0) + pos[0]
#         pos[-1] = pos[-2]
        
#         # Interpolate to match timestamps
#         ts_ext = np.concatenate([[ts[0] - 1e-06], ts[ind], [ts[-1] + 1e-06]], axis=0)
#         from scipy.interpolate import interp1d
#         pos = interp1d(ts_ext, pos, axis=0)(ts)
    
#     return pos

def recon_traj_with_preds(dataset, preds, seq_id=0, **kwargs):
    """
    Reconstruct trajectory with predicted global velocities.
    Updated to support both 2D and 3D trajectories.
    """
    # Determine if we're using TLIO (3D) or original datasets (2D)
    use_3d = hasattr(dataset, 'get_ts_last_imu_us') and hasattr(dataset, 'get_gt_traj_center_window_times')
    output_dim = preds.shape[1]  # Should be 2 for 2D datasets, 3 for TLIO
    
    if use_3d:
        # TLIO dataset - use native methods
        ts = dataset.get_ts_last_imu_us(seq_id) * 1e-6  # Convert to seconds
        
        # Get ground truth trajectory
        if hasattr(dataset, 'get_gt_traj_center_window_times'):
            try:
                gt_data = dataset.get_gt_traj_center_window_times(seq_id)
                if isinstance(gt_data, tuple) and len(gt_data) == 2:
                    # Returns (rotation, position)
                    gt_rot, gt_pos = gt_data
                    pos_gt = gt_pos  # Already in correct format
                else:
                    # Returns SE3 matrices
                    gt_traj_SE3 = gt_data
                    pos_gt = gt_traj_SE3[:, :3, 3]  # Extract positions
            except:
                # Fallback: create dummy ground truth
                print("Warning: Could not extract ground truth trajectory, using dummy data")
                pos_gt = np.zeros((len(ts), 3))
        else:
            pos_gt = np.zeros((len(ts), 3))
            
        # Calculate time differences
        if len(ts) > 1:
            dts = np.mean(ts[1:] - ts[:-1])
        else:
            dts = 0.005  # Default 200Hz
            
        # Reconstruct trajectory from velocity predictions
        pos = np.zeros([preds.shape[0] + 1, output_dim])
        if len(pos_gt) > 0:
            pos[0] = pos_gt[0, :output_dim]  # Initialize with first GT position
        
        # Integrate velocities
        pos[1:] = np.cumsum(preds * dts, axis=0) + pos[0]
        
        # Interpolate if needed to match ground truth timestamps
        if len(pos) != len(pos_gt):
            from scipy.interpolate import interp1d
            ts_pred = np.linspace(ts[0], ts[-1], len(pos))
            if len(ts) > 1:
                pos_interp = interp1d(ts_pred, pos, axis=0, fill_value="extrapolate")(ts)
                pos = pos_interp
            
    else:
        # Original TRIP datasets (2D)
        if hasattr(dataset, 'ts') and hasattr(dataset, 'gt_pos'):
            ts = dataset.ts[seq_id]
            gt_pos = dataset.gt_pos[seq_id]
        else:
            # Fallback for datasets without direct access
            ts = np.arange(len(preds)) * 0.05  # Assume 20Hz
            gt_pos = np.zeros((len(preds), 2))
            
        ind = np.array([i[1] for i in dataset.index_map if i[0] == seq_id], dtype=int)
        dts = np.mean(ts[ind[1:]] - ts[ind[:-1]]) if len(ind) > 1 else 0.05
        
        pos = np.zeros([preds.shape[0] + 2, output_dim])
        pos[0] = gt_pos[0, :output_dim]
        pos[1:-1] = np.cumsum(preds[:, :output_dim] * dts, axis=0) + pos[0]
        pos[-1] = pos[-2]
        
        # Interpolate to match timestamps
        ts_ext = np.concatenate([[ts[0] - 1e-06], ts[ind], [ts[-1] + 1e-06]], axis=0)
        from scipy.interpolate import interp1d
        pos = interp1d(ts_ext, pos, axis=0)(ts)
    
    return pos


def test_sequence(args):
    """
    Updated test sequence function to support both TRIP and TLIO datasets.
    """
    # IMPORTANT: Set global dimensions for TLIO BEFORE creating the model
    if args.dataset == 'tlio':
        global _input_channel, _output_channel
        _input_channel = 6  # IMU data: 3 accel + 3 gyro
        _output_channel = 3  # 3D velocity: vx, vy, vz
        print(f"TLIO test mode: Set _input_channel={_input_channel}, _output_channel={_output_channel}")
    
    # Handle dataset loading based on type
    if args.dataset == 'tlio':
        print("Testing TLIO dataset...")

        # Use TLIO's native dataloader
        from dataloader.tlio_data import TlioData
        from dataloader.memmapped_sequences_dataset import MemMappedSequencesDataset
        from dataloader.constants import DatasetGenerationParams
        
        # Setup data window config for testing
        data_window_config = DatasetGenerationParams(
            window_size=args.window_size,
            step_period_us=5000,  # 5ms = 200Hz
            prediction_times_us=[0],
            starting_point_time_us=0,
            generate_data_period_us=5000,
            decimator=args.step_size,
            express_in_t0_yaw_normalized_frame=False,
            input_sensors=["imu0"],
            data_style="resampled",
        )
        
        # Get test list for TLIO
        if args.test_path is not None:
            test_data_list = [args.test_path.split('/')[-1]]
            root_dir = args.root_dir
        else:
            # Try to load from test list file, or scan directory
            test_list_path = os.path.join(args.root_dir, 'test_list.txt')
            if os.path.exists(test_list_path):
                print(f"Loading test list from: {test_list_path}")
                with open(test_list_path) as f:
                    test_data_list = [s.strip() for s in f.readlines() if len(s.strip()) > 0 and not s.strip().startswith('#')]
            else:
                # Scan directory for sequence folders
                print(f"No test_list.txt found, scanning directory: {args.root_dir}")
                test_data_list = []
                if os.path.exists(args.root_dir):
                    for item in os.listdir(args.root_dir):
                        item_path = os.path.join(args.root_dir, item)
                        if os.path.isdir(item_path):
                            # Check if it looks like a TLIO sequence (has imu0_resampled.npy)
                            if os.path.exists(os.path.join(item_path, 'imu0_resampled.npy')):
                                test_data_list.append(item)
                
                if not test_data_list:
                    raise ValueError(f"No TLIO sequences found in {args.root_dir}. Please ensure the directory contains valid TLIO data or provide a test_list.txt file.")
                
                print(f"Found {len(test_data_list)} TLIO sequences: {test_data_list[:5]}{'...' if len(test_data_list) > 5 else ''}")
            
            root_dir = args.root_dir
                
    else:
        # Original TRIP dataset handling
        if args.dataset != 'px4' and args.dataset != 'oxiod':
            if args.test_path is not None:
                if args.test_path[-1] == '/':
                    args.test_path = args.test_path[:-1]
                root_dir = osp.split(args.test_path)[0]
                test_data_list = [osp.split(args.test_path)[1]]
            elif args.test_list is not None and args.test_list != '':
                root_dir = args.root_dir
                with open(args.test_list) as f:
                    test_data_list = [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
            else:
                raise ValueError('Either test_path or test_list must be specified.')
        else:
            root_dir = args.root_dir + '/validation'
            test_data_list = os.listdir(root_dir)
            if args.dataset == 'px4':
                args.step_size = 1

    if args.out_dir is not None and not osp.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    device = torch.device('cpu')
    checkpoint = torch.load(args.model_path, map_location=lambda storage, location: storage, weights_only=False)

    # Create model
    network = get_model(args)
    network.load_state_dict(checkpoint['model_state_dict'])
    network.eval().to(device)
    print('Model {} loaded to device {}.'.format(args.model_path, device))

    total_test_samples = 0

    preds_seq, targets_seq, losses_seq, ate_all, rte_all, d_drift_all = [], [], [], [], [], []
    traj_lens = []

    pred_per_min = 200 * 60
    start_time = time.time()

    for data in test_data_list:
        sequence_start_time = time.time()
        print(f"Processing sequence: {data}")
        
        try:
            if args.dataset == 'tlio':
                # Use TLIO dataset loading
                seq_dataset = MemMappedSequencesDataset(
                    args.root_dir,
                    "test",
                    data_window_config,
                    sequence_subset=[data],
                    store_in_ram=True,
                    verbose=True
                )
            else:
                # Use original TRIP dataset loading
                seq_dataset = get_dataset(root_dir, [data], args, mode='test')
            
            seq_loader = DataLoader(seq_dataset, batch_size=1024, shuffle=False)

            num_samples = len(seq_dataset)
            total_test_samples += num_samples
            print(f"Sequence {data}: {num_samples} samples")
            
        except Exception as e:
            print(f"Error loading dataset {data}: {e}")
            continue

        # Run inference
        try:
            targets, preds = run_test(network, seq_loader, device, True)
            losses = np.mean((targets - preds) ** 2, axis=0)
            preds_seq.append(preds)
            targets_seq.append(targets)
            losses_seq.append(losses)

            sequence_end_time = time.time()
            print(f"Inference time for sequence {data}: {sequence_end_time - sequence_start_time:.2f} seconds")

        except Exception as e:
            print(f"Error during inference for {data}: {e}")
            continue

        # Trajectory reconstruction and metrics
        try:
            if args.dataset != 'px4':
                pos_pred = recon_traj_with_preds(seq_dataset, preds)
                
                # Get ground truth positions
                if args.dataset == 'tlio':
                    # For TLIO, get 3D ground truth
                    try:
                        gt_data = seq_dataset.get_gt_traj_center_window_times(0)
                        if isinstance(gt_data, tuple) and len(gt_data) == 2:
                            gt_rot, pos_gt = gt_data
                        else:
                            # SE3 matrices
                            pos_gt = gt_data[:, :3, 3]
                    except:
                        print(f"Warning: Could not get ground truth for {data}")
                        pos_gt = np.zeros((len(pos_pred), 3))
                else:
                    # Original 2D datasets
                    pos_gt = seq_dataset.gt_pos[0][:, :2]
                    pos_pred = pos_pred[:, :2]  # Ensure 2D for original datasets

                # Calculate metrics
                traj_lens.append(np.sum(np.linalg.norm(pos_gt[1:] - pos_gt[:-1], axis=1)))
                ate, rte = compute_ate_rte(pos_pred, pos_gt, pred_per_min)
                d_drift = compute_drift(pos_pred, pos_gt)
                ate_all.append(ate)
                rte_all.append(rte)
                d_drift_all.append(d_drift)

                pos_cum_error = np.linalg.norm(pos_pred - pos_gt, axis=1)
                print('Sequence {}, loss {} / {}, ate {:.6f}, rte {:.6f}, d_drift {:.2f}%'.format(
                    data, losses, np.mean(losses), ate, rte, d_drift))

                # Plotting
                kp = preds.shape[1]
                if kp == 2:
                    targ_names = ['vx', 'vy']
                elif kp == 3:
                    targ_names = ['vx', 'vy', 'vz']

                plt.figure('{}'.format(data), figsize=(16, 9))
                
                if args.dataset == 'tlio' and kp == 3:
                    # 3D plotting for TLIO
                    ax = plt.subplot2grid((kp, 2), (0, 0), rowspan=kp - 1, projection='3d')
                    ax.plot(pos_pred[:, 0], pos_pred[:, 1], pos_pred[:, 2], label='ResTCA')
                    ax.plot(pos_gt[:, 0], pos_gt[:, 1], pos_gt[:, 2], label='Ground truth')
                    ax.set_title(data)

                        # Set axis labels with increased font size
                    ax.set_xlabel('X (m)', fontsize=14)
                    ax.set_ylabel('Y (m)', fontsize=14)
                    ax.set_zlabel('Z (m)', fontsize=14)
                    
                    # Increase tick label font size
                    ax.tick_params(axis='x', labelsize=14)
                    ax.tick_params(axis='y', labelsize=14)
                    ax.tick_params(axis='z', labelsize=14)
                    
                    # Remove grey background color in grid
                    ax.xaxis.pane.fill = False
                    ax.yaxis.pane.fill = False
                    ax.zaxis.pane.fill = False
                    
                    # Make pane edges white/transparent
                    ax.xaxis.pane.set_edgecolor('white')
                    ax.yaxis.pane.set_edgecolor('white') 
                    ax.zaxis.pane.set_edgecolor('white')
                    
                    # Optional: Make grid lines lighter
                    ax.grid(True, alpha=0.3)
                    
                    # Adjust legend to center of the figure (positioned to avoid data overlap)
                    ax.legend(['ResTCA', 'Ground truth'], 
                            loc='upper center', 
                            bbox_to_anchor=(0.5, 0.3),
                            frameon=True,
                            fancybox=False,
                            shadow=False,
                            ncol=2,
                            fontsize=12)
                else:
                    # 2D plotting for original datasets
                    plt.subplot2grid((kp, 2), (0, 0), rowspan=kp - 1)
                    plt.plot(pos_pred[:, 0], pos_pred[:, 1], label='Predicted')
                    plt.plot(pos_gt[:, 0], pos_gt[:, 1], label='Ground truth')
                    plt.title(data)
                    plt.axis('equal')
                    plt.legend(['Predicted', 'Ground truth'])
                
                plt.subplot2grid((kp, 2), (kp - 1, 0))
                plt.plot(pos_cum_error)
                plt.legend(['ATE:{:.3f}, RTE:{:.3f}'.format(ate_all[-1], rte_all[-1])])
                
                # Plot velocity predictions
                if hasattr(seq_dataset, 'index_map'):
                    ind = np.array([i[1] for i in seq_dataset.index_map if i[0] == 0], dtype=int)
                else:
                    ind = np.arange(len(preds))
                    
                for i in range(kp):
                    plt.subplot2grid((kp, 2), (i, 1))
                    plt.plot(ind, preds[:, i])
                    plt.plot(ind, targets[:, i])
                    plt.legend(['Predicted', 'Ground truth'])
                    plt.title('{}, error: {:.6f}'.format(targ_names[i], losses[i]))
                plt.tight_layout()

                if args.show_plot:
                    plt.show()

                if args.out_dir is not None and osp.isdir(args.out_dir):
                    # Save results
                    if args.dataset == 'tlio':
                        np.save(osp.join(args.out_dir, data + '_tlio.npy'),
                                np.concatenate([pos_pred, pos_gt], axis=1))
                    else:
                        np.save(osp.join(args.out_dir, data + '_gsn.npy'),
                                np.concatenate([pos_pred[:, :2], pos_gt[:, :2]], axis=1))
                    plt.savefig(osp.join(args.out_dir, data + '_gsn.png'))

                plt.close('all')
                
        except Exception as e:
            print(f"Error processing trajectory for {data}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Final statistics
    losses_seq = np.stack(losses_seq, axis=0)
    losses_avg = np.mean(losses_seq, axis=1)

    end_time = time.time()
    print(f"Total inference time for the test sequence: {end_time - start_time:.2f} seconds")
    print(f"Total test samples across all sequences: {total_test_samples}")

    if args.dataset != 'px4' and len(ate_all) > 0:
        if args.out_dir is not None and osp.isdir(args.out_dir):
            with open(osp.join(args.out_dir, 'losses.csv'), 'w') as f:
                if losses_seq.shape[1] == 2:
                    f.write('seq,vx,vy,avg,ate,rte,d_drift\n')
                else:
                    f.write('seq,vx,vy,vz,avg,ate,rte,d_drift\n')
                for i in range(losses_seq.shape[0]):
                    f.write('{},'.format(test_data_list[i]))
                    for j in range(losses_seq.shape[1]):
                        f.write('{:.6f},'.format(losses_seq[i][j]))
                    f.write('{:.6f},{:.6f},{:.6f},{:.2f}\n'.format(
                        losses_avg[i], ate_all[i], rte_all[i], d_drift_all[i]))

        print('----------\nOverall loss: {}/{}, avg ATE:{}, avg RTE:{}, avg D_drift:{}%'.format(
            np.average(losses_seq, axis=0), np.average(losses_avg), 
            np.mean(ate_all), np.mean(rte_all), np.mean(d_drift_all)))

    return losses_avg


def write_config(args):
    if args.out_dir:
        with open(osp.join(args.out_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f)

# Add this to your main.py file in the main section where dataset cases are handled

# In the main section, find this part and add the TLIO case:

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_list', default='', type=str)
    parser.add_argument('--val_list', type=str, default='')
    parser.add_argument('--test_list', type=str, default='')
    parser.add_argument('--test_path', type=str, default=None)
    parser.add_argument('--root_dir', type=str, default='', help='Path to data directory')
    parser.add_argument('--cache_path', type=str, default=None, help='Path to cache folder to store processed data')
    parser.add_argument('--dataset', type=str, default='oxiod',
                        choices=['ronin', 'ridi', 'imunet', 'oxiod', 'kiod', 'inaiod', 'tlio'])
    parser.add_argument('--max_ori_error', type=float, default=20.0)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--window_size', type=int, default=200)
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--lr', type=float, default=1e-04)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--arch', type=str, default='HResCSA',
             choices=['MobileNet', 'MobileNetV2','MnasNet', 'EfficientNet', 'IMUNet', 'ResNet1D', 'HResCSA'])
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--run_ekf', action='store_true')
    parser.add_argument('--fast_test', action='store_true')
    parser.add_argument('--show_plot', action='store_true')
    parser.add_argument('--test_status', type=str, default='seen', choices=['seen', 'unseen'])
    parser.add_argument('--continue_from', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default='')
    parser.add_argument('--model_path', type=str, default='')

    parser.add_argument('--feature_sigma', type=float, default=0.00001)
    parser.add_argument('--target_sigma', type=float, default=0.00001)

    args = parser.parse_args()

    np.set_printoptions(formatter={'all': lambda x: '{:.6f}'.format(x)})
    dataset = args.dataset
    import os

    # Get the current working directory
    current_dir = os.getcwd()
    
    from pathlib import Path
    path = Path(current_dir)
    current_dir = str(path.parent.absolute())

    # Print the current working directory
    print("Current working directory: {0}".format(current_dir))

    # Train
    if args.mode == 'train':
        if dataset == 'ronin':
            args.train_list = current_dir +'/dataset/ronin/list_train.txt'
            args.val_list = current_dir + '/dataset/ronin/list_val.txt'
            args.root_dir = current_dir + '/dataset/ronin/train_dataset_1'
            args.out_dir = current_dir + '/src/output/Train_out/' + args.arch + '/ronin'
            
        elif dataset == 'ridi':
            args.train_list = current_dir + '/dataset/ridi/data_publish_v2/list_train_publish_v2.txt'
            args.val_list = current_dir + '/dataset/ridi/data_publish_v2/list_test_publish_v2.txt'
            args.root_dir = current_dir + '/dataset/ridi/data_publish_v2'
            args.out_dir = current_dir + '/src/output/Train_out/' + args.arch + '/ridi'

        # IMUNet dataset
        elif dataset == 'imunet':
            args.train_list = current_dir +'/dataset/imunet/list_train.txt'
            args.val_list = current_dir +'/dataset/imunet/list_test.txt'
            args.root_dir = current_dir + '/dataset/imunet'
            args.out_dir = current_dir +'/src/output/Train_out/' + args.arch + '/imunet'

            # added the functionality of KAIST-N1 dataset
        elif dataset == 'kiod':
            args.train_list = current_dir +'/dataset/KIOD/list_train.txt'
            args.val_list = current_dir +'/dataset/KIOD/list_test.txt'
            args.root_dir = current_dir + '/dataset/KIOD'
            args.out_dir = current_dir +'/src/output/Train_out/' + args.arch + '/KIOD'
        
        # INA-IOD dataset
        elif dataset == 'inaiod':
            args.train_list = current_dir +'/dataset/INAIOD/list_train.txt'
            args.val_list = current_dir +'/dataset/INAIOD/list_test.txt'
            args.root_dir = current_dir + '/dataset/INAIOD'
            args.out_dir = current_dir +'/src/output/Train_out/' + args.arch + '/INAIOD'

        elif dataset == 'oxiod':
            args.train_list = ''
            args.val_list = ''
            args.root_dir = current_dir + '/dataset/oxiod'
            args.out_dir = current_dir +'/src/output/Train_out/' + args.arch + '/oxiod'
            
# Add this debug section right after the TLIO case in the main function

        elif dataset == 'tlio':
            args.train_list = ''  # TLIO doesn't use list files
            args.val_list = ''
            # Keep the root_dir as provided by command line argument
            # args.root_dir is already set by --root_dir argument
            args.out_dir = current_dir + '/src/output/Train_out/' + args.arch + '/tlio'
            
            # ADD THESE DEBUG PRINTS
            print(f"DEBUG TLIO PATHS:")
            print(f"  current_dir: {current_dir}")
            print(f"  args.arch: {args.arch}")
            print(f"  args.out_dir: {args.out_dir}")
            print(f"  out_dir exists: {os.path.exists(args.out_dir)}")
            
            # Try to create the directory explicitly
            try:
                os.makedirs(args.out_dir, exist_ok=True)
                print(f"  ✅ Successfully created/verified: {args.out_dir}")
            except Exception as e:
                print(f"  ❌ Failed to create directory: {e}")
            
            # Check if we can write to it
            test_file = os.path.join(args.out_dir, 'test_write.txt')
            try:
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                print(f"  ✅ Write permissions OK")
            except Exception as e:
                print(f"  ❌ Cannot write to directory: {e}")

        # Also add this debug print right before calling train(args)
        print(f"\nFINAL DEBUG before training:")
        print(f"args.out_dir = '{args.out_dir}'")
        print(f"args.dataset = '{args.dataset}'")
        print(f"args.mode = '{args.mode}'")

        train(args)

    # Test    
    elif args.mode == 'test':
        if args.test_status == 'unseen':
            if dataset != 'ronin':
                raise ValueError('Undefined mode')
        if dataset == 'ronin':
            args.model_path = current_dir + '/src/output/Train_out/' + args.arch + \
                            '/ronin/checkpoints/checkpoint_best.pt'
            
            if args.test_status == 'seen':
                args.root_dir =  current_dir + '/dataset/ronin/seen_subjects_test_set'
                args.test_list = current_dir + '/dataset/ronin/list_test_seen.txt'
                args.out_dir = current_dir + '/src/output/Test_out/ronin/seen/'  + args.arch
            else:
                args.root_dir = current_dir + '/dataset/ronin/unseen_subjects_test_set'
                args.test_list = current_dir + '/dataset/ronin/list_test_unseen.txt'
                args.out_dir = current_dir + '/src/output/Test_out/ronin/unseen/'  + args.arch

        elif dataset == 'ridi':
            args.model_path = current_dir + '/src/output/Train_out/' + args.arch + \
                              '/ridi/checkpoints/checkpoint_best.pt'
            args.test_list = current_dir + '/dataset/ridi/data_publish_v2/list_test_publish_v2.txt'
            args.root_dir = current_dir + '/dataset/ridi/data_publish_v2'
            args.out_dir = current_dir + '/src/output/Test_out/ridi/' + args.arch

        #IMUNet dataset
        elif dataset == 'imunet':
            args.model_path = current_dir + '/src/output/Train_out/' + args.arch + '/imunet/checkpoints' \
                                                                                    '/checkpoint_best.pt'           
            args.root_dir = current_dir + '/dataset/imunet' 
    
            args.test_list = current_dir + '/dataset/imunet/list_test.txt'      
            args.out_dir = current_dir + '/output/Test_out/imunet/seen/' + args.arch

            # K-IOD dataset
        elif dataset == 'kiod':
            args.model_path = current_dir + '/src/output/Train_out/' + args.arch + '/KIOD/checkpoints' \
                                                                                    '/checkpoint_best.pt'
            args.test_list = current_dir + '/dataset/KIOD/list_test.txt'
            args.root_dir = current_dir + '/dataset/KIOD'
            args.out_dir = current_dir + '/src/output/Test_out/KIOD' + args.arch
            
        # INA-IOD dataset
        elif dataset == 'inaiod':
            args.model_path = current_dir + '/src/output/Train_out/' + args.arch + '/INAIOD/checkpoints' \
                                                                                    '/checkpoint_best.pt'
            args.test_list = current_dir + '/dataset/INAIOD/list_test.txt'
            args.root_dir = current_dir + '/dataset/INAIOD'
            args.out_dir = current_dir + '/src/output/Test_out/INAIOD' + args.arch

        elif dataset == 'oxiod':
            args.model_path =  current_dir + '/src/output/Train_out/' + args.arch + '/oxiod/checkpoints/checkpoint_latest.pt'
            args.test_list = current_dir +  '/dataset/oxiod/'
            args.root_dir = current_dir + '/dataset/oxiod'
            args.out_dir = current_dir + '/src/output/Test_out/oxiod/' + args.arch
            
        # ADD THIS TLIO TEST CASE:
        elif dataset == 'tlio':
            args.model_path = current_dir + '/src/output/Train_out/' + args.arch + '/tlio/checkpoints/checkpoint_best.pt'
            args.test_list = ''
            # Keep the root_dir as provided by command line argument
            args.out_dir = current_dir + '/src/output/Test_out/tlio/' + args.arch
            
        test_sequence(args)
    else:
        raise ValueError('Undefined mode')