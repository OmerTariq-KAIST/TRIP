"""
TLIO Integration for TRIP
This module provides the TLIOSequence class that integrates TLIO dataset with TRIP's existing structure
Save this file as: preprocess/tlio_integration.py
"""

import numpy as np
import torch
import os
from os import path as osp
import json
from preprocess.data_processor import CompiledSequence

# Handle TLIO imports with proper fallback
try:
    from dataloader.tlio_data import TlioData
    from dataloader.memmapped_sequences_dataset import MemMappedSequencesDataset
    from dataloader.constants import DatasetGenerationParams
    TLIO_AVAILABLE = True
    print("✓ TLIO modules imported successfully")
except ImportError as e:
    print(f"Warning: TLIO imports failed: {e}")
    print("TLIO functionality will not be available")
    TLIO_AVAILABLE = False


class TLIOSequence(CompiledSequence):
    """
    TLIO Dataset sequence - integrates with TRIP's existing data pipeline
    Features: raw angular rate and acceleration (includes gravity)
    """
    feature_dim = 6
    target_dim = 3  # 3D displacement (dx, dy, dz)
    aux_dim = 8     # Compatible with TRIP's aux format

    def __init__(self, data_path=None, **kwargs):
        super().__init__(**kwargs)
        self.ts, self.features, self.targets, self.orientations, self.gt_pos = None, None, None, None, None
        self.w = kwargs.get('interval', 1)
        self.info = {}
        
        # TLIO specific parameters
        self.window_size = kwargs.get('window_size', 200)
        self.step_size = kwargs.get('step_size', 10)
        
        if data_path is not None:
            self.load(data_path)

    def load(self, data_path):
        """Load TLIO dataset using TLIO's native data loader"""
        self.info['path'] = osp.split(data_path)[-1] if data_path else 'tlio_dataset'
        self.info['ori_source'] = 'tlio_dataset'
        
        print(f"Loading TLIO dataset from: {data_path}")
        
        if not TLIO_AVAILABLE:
            raise ImportError("TLIO modules are not available. Please install TLIO dependencies.")
            
        # Check if TLIO dataset structure exists
        if not self._check_tlio_structure(data_path):
            raise FileNotFoundError(f"TLIO dataset structure not found at {data_path}")
            
        try:
            # Create TLIO dataset using the proper TLIO dataloader
            genparams = DatasetGenerationParams(
                window_size=self.window_size,
                step_period_us=5000,  # 5ms = 200Hz
                generate_data_period_us=5000,
                prediction_times_us=[0],
                starting_point_time_us=0,
                decimator=self.step_size,
                express_in_t0_yaw_normalized_frame=False,
                input_sensors=["imu0"],
                data_style="resampled",
            )
            
            # Create dataset for training split
            dataset = MemMappedSequencesDataset(
                data_path=data_path,
                split='train',
                genparams=genparams,
                store_in_ram=False,
                verbose=True,
            )
            
            print(f"TLIO dataset loaded: {len(dataset)} samples")
            
            # Convert TLIO data to TRIP continuous format
            self._convert_tlio_to_trip_format(dataset)
            
        except Exception as e:
            print(f"Error loading TLIO dataset: {e}")
            raise

    def _check_tlio_structure(self, data_path):
        """Check if the data path has TLIO structure"""
        if not data_path or not osp.exists(data_path):
            print(f"Data path does not exist: {data_path}")
            return False
        
        # Look for train_list.txt which is required for TLIO
        train_list_path = osp.join(data_path, 'train_list.txt')
        if not osp.exists(train_list_path):
            print(f"TLIO train_list.txt not found at: {train_list_path}")
            return False
        
        # Check if we have at least one sequence directory
        try:
            with open(train_list_path, 'r') as f:
                sequences = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]
            
            if not sequences:
                print("No sequences found in train_list.txt")
                return False
            
            # Check first sequence has required files
            first_seq_path = osp.join(data_path, sequences[0])
            if not osp.exists(first_seq_path):
                print(f"Sequence directory not found: {first_seq_path}")
                return False
            
            # Look for imu0_resampled.npy and description
            imu_file = osp.join(first_seq_path, 'imu0_resampled.npy')
            desc_file = osp.join(first_seq_path, 'imu0_resampled_description.json')
            
            if not osp.exists(imu_file):
                print(f"IMU file not found: {imu_file}")
                return False
            if not osp.exists(desc_file):
                print(f"Description file not found: {desc_file}")
                return False
            
            print("✓ TLIO dataset structure validation passed")
            return True
            
        except Exception as e:
            print(f"Error checking TLIO structure: {e}")
            return False

    def _convert_tlio_to_trip_format(self, dataset):
        """Convert TLIO dataset to TRIP's continuous sequence format"""
        print("Converting TLIO data to TRIP format...")
        
        # Process samples from TLIO dataset
        all_features = []
        all_targets = []
        all_timestamps = []
        
        num_samples = min(len(dataset), 10000)  # Limit for memory efficiency
        
        for i in range(num_samples):
            try:
                sample = dataset[i]
                
                # TLIO sample structure:
                # sample is a dict with keys: 'feats', 'targ_dt_World', 'ts_us', etc.
                
                # Extract IMU features - TLIO format: feats['imu0'] is [C, T] 
                feat_imu = sample["feats"]["imu0"]  # Shape: [6, window_size]
                if isinstance(feat_imu, torch.Tensor):
                    feat_imu = feat_imu.numpy()
                
                # Take middle timestep from window for continuous data
                mid_idx = feat_imu.shape[1] // 2
                feat_single = feat_imu[:, mid_idx]  # Shape: [6]
                
                # Extract 3D displacement target
                targ_dt = sample["targ_dt_World"]  # Shape varies
                if isinstance(targ_dt, torch.Tensor):
                    targ_dt = targ_dt.numpy()
                
                # Handle different target shapes
                if len(targ_dt.shape) == 3:
                    # Shape: [batch, time, 3] - take last timestep
                    targ_single = targ_dt[0, -1, :]  # [3]
                elif len(targ_dt.shape) == 2:
                    # Shape: [time, 3] - take last timestep
                    targ_single = targ_dt[-1, :]     # [3]
                else:
                    # Shape: [3] or flattened
                    targ_single = targ_dt.flatten()[:3]
                
                # Extract timestamp
                ts_sample = sample.get("ts_us", i * 5000)  # Default to 5ms intervals
                if isinstance(ts_sample, torch.Tensor):
                    ts_sample = ts_sample.numpy()
                
                # Convert microseconds to seconds
                if isinstance(ts_sample, (list, np.ndarray)) and len(ts_sample) > 0:
                    ts_single = float(ts_sample[0]) / 1e6
                else:
                    ts_single = float(ts_sample) / 1e6
                
                all_features.append(feat_single)
                all_targets.append(targ_single)
                all_timestamps.append(ts_single)
                
                if i % 1000 == 0:
                    print(f"Processed {i}/{num_samples} TLIO samples...")
                    
            except Exception as e:
                print(f"Error processing TLIO sample {i}: {e}")
                continue
        
        if not all_features:
            raise ValueError("No valid TLIO samples found")
        
        # Convert to numpy arrays - continuous sequence data
        self.features = np.array(all_features)  # Shape: [N, 6]
        self.targets = np.array(all_targets)    # Shape: [N, 3]
        self.ts = np.array(all_timestamps)      # Shape: [N]
        
        # Sort by timestamp to ensure proper order
        sort_indices = np.argsort(self.ts)
        self.features = self.features[sort_indices]
        self.targets = self.targets[sort_indices]
        self.ts = self.ts[sort_indices]
        
        # Create orientations and positions for compatibility
        n_samples = len(self.features)
        self.orientations = np.tile([1, 0, 0, 0], (n_samples, 1))  # Identity quaternions
        
        # Create ground truth positions by integrating displacements
        self.gt_pos = np.zeros((n_samples, 3))
        if n_samples > 0:
            self.gt_pos[0] = [0, 0, 0]  # Start at origin
            self.gt_pos[1:] = np.cumsum(self.targets[:-1], axis=0)
        
        print(f"✓ Converted TLIO data - Features: {self.features.shape}, Targets: {self.targets.shape}")
        print(f"  Time range: {self.ts[0]:.3f}s to {self.ts[-1]:.3f}s")
        print(f"  Feature ranges: {[f'{self.features[:, i].min():.3f} to {self.features[:, i].max():.3f}' for i in range(6)]}")

    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_aux(self):
        # Return in TRIP's expected aux format: [timestamp, quaternion(4), position(3)]
        if len(self.orientations.shape) == 1:
            orientations = self.orientations.reshape(-1, 4)
        else:
            orientations = self.orientations
            
        return np.concatenate([
            self.ts.reshape(-1, 1), 
            orientations, 
            self.gt_pos
        ], axis=1)

    def get_meta(self):
        return f'{self.info["path"]}: TLIO dataset, features: {self.feature_dim}, targets: {self.target_dim}, samples: {len(self.features)}'


# Test function for the TLIO integration
def test_tlio_sequence(data_path=None):
    """Test the TLIOSequence class"""
    print("Testing TLIOSequence...")
    
    try:
        # Create sequence
        seq = TLIOSequence(data_path=data_path, window_size=200, step_size=10)
        
        print(f"✓ Sequence created successfully")
        print(f"  Features shape: {seq.get_feature().shape}")
        print(f"  Targets shape: {seq.get_target().shape}")
        print(f"  Aux shape: {seq.get_aux().shape}")
        print(f"  Meta: {seq.get_meta()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test with a dummy path - modify as needed
    test_path = "/path/to/tlio/dataset"
    test_tlio_sequence(test_path)