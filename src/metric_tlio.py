import numpy as np
def compute_ate_rte(pos_pred, pos_gt, pred_per_min):
    """
    Compute Absolute Trajectory Error (ATE) and Relative Trajectory Error (RTE).
    Updated to support both 2D and 3D trajectories.
    
    FIXED VERSION - RTE and ATE should be different values
    """
    # Ensure both trajectories have the same length
    min_len = min(len(pos_pred), len(pos_gt))
    pos_pred = pos_pred[:min_len]
    pos_gt = pos_gt[:min_len]
    
    # Absolute Trajectory Error (ATE)
    # ATE measures the RMSE of absolute position differences
    ate = np.sqrt(np.mean(np.linalg.norm(pos_pred - pos_gt, axis=1) ** 2))
    
    # Relative Trajectory Error (RTE) 
    # RTE measures the RMSE of relative motion errors over fixed time intervals
    
    # Use much shorter intervals for RTE (1-second intervals instead of 1-minute)
    interval_length = max(10, pred_per_min // 60)  # 1-second intervals (200 samples at 200Hz)
    
    # If trajectory is too short for any intervals, use smaller intervals
    if len(pos_pred) < interval_length:
        interval_length = max(2, len(pos_pred) // 4)  # At least 4 intervals
    
    segment_errors = []
    
    # Calculate RTE over shorter, overlapping segments
    step = max(1, interval_length // 4)  # Overlapping intervals for better statistics
    
    for i in range(0, len(pos_pred) - interval_length + 1, step):
        # Get segment start and end indices
        start_idx = i
        end_idx = i + interval_length
        
        # Calculate relative displacement over this interval
        pred_rel = pos_pred[end_idx] - pos_pred[start_idx]
        gt_rel = pos_gt[end_idx] - pos_gt[start_idx]
        
        # Error in relative displacement
        rel_error = np.linalg.norm(pred_rel - gt_rel)
        segment_errors.append(rel_error)
    
    # Calculate RTE as RMSE of all relative errors
    if len(segment_errors) > 0:
        rte = np.sqrt(np.mean(np.array(segment_errors) ** 2))
    else:
        # Emergency fallback: use half-trajectory relative error
        if len(pos_pred) >= 2:
            mid_idx = len(pos_pred) // 2
            pred_rel = pos_pred[-1] - pos_pred[mid_idx]
            gt_rel = pos_gt[-1] - pos_gt[mid_idx]
            rte = np.linalg.norm(pred_rel - gt_rel)
        else:
            rte = ate
    
    # Debug info (remove after testing)
    print(f"DEBUG ATE/RTE: trajectory_length={len(pos_pred)}, interval_length={interval_length}, num_segments={len(segment_errors)}")
    print(f"DEBUG ATE/RTE: ATE={ate:.6f}, RTE={rte:.6f}, RTE>ATE: {rte > ate}")
    
    return ate, rte


def compute_drift(pos_pred, pos_gt):
    """
    Compute drift percentage: final position error / total trajectory length.
    Updated to support both 2D and 3D trajectories.
    """
    # Ensure both trajectories have the same length
    min_len = min(len(pos_pred), len(pos_gt))
    pos_pred = pos_pred[:min_len]
    pos_gt = pos_gt[:min_len]
    
    if len(pos_gt) < 2:
        return 0.0
    
    # Calculate total trajectory length
    gt_diffs = pos_gt[1:] - pos_gt[:-1]
    total_distance = np.sum(np.linalg.norm(gt_diffs, axis=1))
    
    if total_distance == 0:
        return 0.0
    
    # Calculate final position error
    final_error = np.linalg.norm(pos_pred[-1] - pos_gt[-1])
    
    # Drift as percentage
    drift_percentage = (final_error / total_distance) * 100
    
    return drift_percentage


def compute_absolute_trajectory_error(pos_pred, pos_gt):
    """
    Compute mean and RMS of absolute trajectory error.
    Updated to support both 2D and 3D trajectories.
    """
    # Ensure both trajectories have the same length
    min_len = min(len(pos_pred), len(pos_gt))
    pos_pred = pos_pred[:min_len]
    pos_gt = pos_gt[:min_len]
    
    # Point-wise errors
    errors = np.linalg.norm(pos_pred - pos_gt, axis=1)
    
    # Statistics
    mean_error = np.mean(errors)
    rms_error = np.sqrt(np.mean(errors ** 2))
    max_error = np.max(errors)
    std_error = np.std(errors)
    
    return {
        'mean': mean_error,
        'rms': rms_error,
        'max': max_error,
        'std': std_error,
        'errors': errors
    }


def align_trajectories_3d(pos_pred, pos_gt, method='start'):
    """
    Align predicted trajectory with ground truth trajectory.
    Supports different alignment methods for 3D trajectories.
    
    Args:
        pos_pred: Predicted trajectory [N, 3] or [N, 2]
        pos_gt: Ground truth trajectory [N, 3] or [N, 2]
        method: 'start' - align starting points
                'se3' - full SE(3) alignment (translation + rotation)
                'similarity' - similarity transform (translation + rotation + scale)
    
    Returns:
        pos_pred_aligned: Aligned predicted trajectory
        transform: Transformation applied
    """
    # Ensure both trajectories have the same length and dimensions
    min_len = min(len(pos_pred), len(pos_gt))
    pos_pred = pos_pred[:min_len]
    pos_gt = pos_gt[:min_len]
    
    if method == 'start':
        # Simple alignment: translate to align starting points
        offset = pos_gt[0] - pos_pred[0]
        pos_pred_aligned = pos_pred + offset
        transform = {'translation': offset, 'rotation': None, 'scale': 1.0}
        
    elif method == 'se3' and pos_pred.shape[1] >= 3:
        # SE(3) alignment using SVD (Umeyama algorithm)
        try:
            from scipy.spatial.transform import Rotation
            
            # Center the trajectories
            centroid_pred = np.mean(pos_pred, axis=0)
            centroid_gt = np.mean(pos_gt, axis=0)
            
            centered_pred = pos_pred - centroid_pred
            centered_gt = pos_gt - centroid_gt
            
            # Cross-covariance matrix
            H = centered_pred.T @ centered_gt
            
            # SVD
            U, S, Vt = np.linalg.svd(H)
            
            # Rotation matrix
            R = Vt.T @ U.T
            
            # Ensure proper rotation (det(R) = 1)
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            # Apply transformation
            pos_pred_aligned = (R @ centered_pred.T).T + centroid_gt
            
            transform = {
                'translation': centroid_gt - R @ centroid_pred,
                'rotation': R,
                'scale': 1.0
            }
            
        except Exception as e:
            print(f"SE(3) alignment failed: {e}, falling back to start alignment")
            return align_trajectories_3d(pos_pred, pos_gt, method='start')
            
    elif method == 'similarity':
        # Similarity transform (translation + rotation + uniform scale)
        try:
            # Center the trajectories
            centroid_pred = np.mean(pos_pred, axis=0)
            centroid_gt = np.mean(pos_gt, axis=0)
            
            centered_pred = pos_pred - centroid_pred
            centered_gt = pos_gt - centroid_gt
            
            # Scale estimation
            scale = np.sqrt(np.sum(centered_gt ** 2) / np.sum(centered_pred ** 2))
            
            # Apply scale
            scaled_pred = centered_pred * scale
            
            if pos_pred.shape[1] >= 3:
                # Cross-covariance matrix
                H = scaled_pred.T @ centered_gt
                
                # SVD for rotation
                U, S, Vt = np.linalg.svd(H)
                R = Vt.T @ U.T
                
                # Ensure proper rotation
                if np.linalg.det(R) < 0:
                    Vt[-1, :] *= -1
                    R = Vt.T @ U.T
                
                # Apply transformation
                pos_pred_aligned = (R @ scaled_pred.T).T + centroid_gt
                
                transform = {
                    'translation': centroid_gt - R @ (scale * centroid_pred),
                    'rotation': R,
                    'scale': scale
                }
            else:
                # 2D case
                pos_pred_aligned = scaled_pred + centroid_gt
                transform = {
                    'translation': centroid_gt - scale * centroid_pred,
                    'rotation': None,
                    'scale': scale
                }
                
        except Exception as e:
            print(f"Similarity alignment failed: {e}, falling back to start alignment")
            return align_trajectories_3d(pos_pred, pos_gt, method='start')
    else:
        # Default to start alignment
        return align_trajectories_3d(pos_pred, pos_gt, method='start')
    
    return pos_pred_aligned, transform


def plot_trajectory_comparison(pos_pred, pos_gt, title="Trajectory Comparison", save_path=None):
    """
    Plot trajectory comparison supporting both 2D and 3D trajectories.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Determine if 2D or 3D
    is_3d = pos_pred.shape[1] >= 3 and pos_gt.shape[1] >= 3
    
    fig = plt.figure(figsize=(15, 5))
    
    if is_3d:
        # 3D trajectory plot
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot(pos_pred[:, 0], pos_pred[:, 1], pos_pred[:, 2], 'b-', label='Predicted', linewidth=2)
        ax1.plot(pos_gt[:, 0], pos_gt[:, 1], pos_gt[:, 2], 'r-', label='Ground Truth', linewidth=2)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.legend()
        ax1.set_title('3D Trajectory')
        
        # Top view (X-Y)
        ax2 = fig.add_subplot(132)
        ax2.plot(pos_pred[:, 0], pos_pred[:, 1], 'b-', label='Predicted', linewidth=2)
        ax2.plot(pos_gt[:, 0], pos_gt[:, 1], 'r-', label='Ground Truth', linewidth=2)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.legend()
        ax2.set_title('Top View (X-Y)')
        ax2.axis('equal')
        ax2.grid(True)
        
    else:
        # 2D trajectory plot
        ax1 = fig.add_subplot(131)
        ax1.plot(pos_pred[:, 0], pos_pred[:, 1], 'b-', label='Predicted', linewidth=2)
        ax1.plot(pos_gt[:, 0], pos_gt[:, 1], 'r-', label='Ground Truth', linewidth=2)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.legend()
        ax1.set_title('2D Trajectory')
        ax1.axis('equal')
        ax1.grid(True)
        
        # Duplicate for consistency
        ax2 = ax1
    
    # Error plot
    ax3 = fig.add_subplot(133)
    errors = np.linalg.norm(pos_pred - pos_gt, axis=1)
    time_stamps = np.arange(len(errors))
    ax3.plot(time_stamps, errors, 'g-', linewidth=2)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Position Error (m)')
    ax3.set_title(f'Position Error\nMean: {np.mean(errors):.3f}m, RMS: {np.sqrt(np.mean(errors**2)):.3f}m')
    ax3.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    return fig


def evaluate_trajectory_metrics(pos_pred, pos_gt, dt=None, align_method='start'):
    """
    Comprehensive trajectory evaluation with multiple metrics.
    
    Args:
        pos_pred: Predicted trajectory [N, 2/3]
        pos_gt: Ground truth trajectory [N, 2/3]  
        dt: Time step (optional, for velocity-based metrics)
        align_method: Alignment method for trajectories
    
    Returns:
        Dictionary of metrics
    """
    # Align trajectories
    pos_pred_aligned, transform = align_trajectories_3d(pos_pred, pos_gt, method=align_method)
    
    # Basic trajectory metrics
    ate_metrics = compute_absolute_trajectory_error(pos_pred_aligned, pos_gt)
    
    # Calculate ATE and RTE
    ate, rte = compute_ate_rte(pos_pred_aligned, pos_gt, pred_per_min=200*60)
    
    # Calculate drift
    drift = compute_drift(pos_pred_aligned, pos_gt)
    
    # Trajectory length comparison
    pred_length = np.sum(np.linalg.norm(np.diff(pos_pred_aligned, axis=0), axis=1))
    gt_length = np.sum(np.linalg.norm(np.diff(pos_gt, axis=0), axis=1))
    length_error = abs(pred_length - gt_length) / gt_length * 100 if gt_length > 0 else 0
    
    # Velocity metrics (if dt provided)
    vel_metrics = {}
    if dt is not None and dt > 0:
        # Calculate velocities
        vel_pred = np.diff(pos_pred_aligned, axis=0) / dt
        vel_gt = np.diff(pos_gt, axis=0) / dt
        
        vel_errors = np.linalg.norm(vel_pred - vel_gt, axis=1)
        vel_metrics = {
            'velocity_rmse': np.sqrt(np.mean(vel_errors ** 2)),
            'velocity_mean_error': np.mean(vel_errors),
            'velocity_max_error': np.max(vel_errors)
        }
    
    # Compile all metrics
    metrics = {
        'ate': ate,
        'rte': rte,
        'drift_percentage': drift,
        'rmse': ate_metrics['rms'],
        'mean_error': ate_metrics['mean'],
        'max_error': ate_metrics['max'],
        'std_error': ate_metrics['std'],
        'trajectory_length_error_percent': length_error,
        'predicted_length': pred_length,
        'ground_truth_length': gt_length,
        'alignment_method': align_method,
        'transform': transform,
        **vel_metrics
    }
    
    return metrics