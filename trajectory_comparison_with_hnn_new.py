#!/usr/bin/env python3

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import argparse
import pickle
import json
import glob
from collections import defaultdict

# 添加项目路径
PROJECT_PATH = '/home/jiayinliu/Desktop/ControlledHamiltonianGAN_0'
sys.path.append(PROJECT_PATH)
sys.path.append(os.path.join(PROJECT_PATH, 'src'))

# 添加hamiltonian-nn路径
HAMILTONIAN_NN_PATH = os.path.join(PROJECT_PATH, 'hamiltonian-nn')
sys.path.append(HAMILTONIAN_NN_PATH)

# 导入HGAN模块
try:
    from hgan.hgn.environments.environment_factory import EnvFactory
    from hgan.hgn_datasets import constant_physics_hgn
    HGAN_AVAILABLE = True
except ImportError as e:
    print(f"⚠ HGAN modules not available: {e}")
    HGAN_AVAILABLE = False

# 导入HNN模块
try:
    from nn_models import MLP
    from hnn import HNN
    from utils import integrate_model
    HNN_AVAILABLE = True
except ImportError as e:
    print(f"⚠ HNN modules not available: {e}")
    HNN_AVAILABLE = False

def create_results_directory(system_name, mask=None):
    """创建结果目录结构"""
    # 确定系统名称
    if system_name == 'NObjectGravity' and mask is not None:
        num_objects = np.sum(mask > 0)
        system_folder = f"{num_objects}_body"
    else:
        system_mapping = {
            'Spring': 'mass_spring',
            'Pendulum': 'pendulum',
            'ChaoticPendulum': 'double_pendulum'
        }
        system_folder = system_mapping.get(system_name, system_name.lower())
    
    results_dir = os.path.join(PROJECT_PATH, 'results', system_folder)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def load_custom_hnn_model(model_path, input_dim=8, hidden_dim=200):
    """Load custom trained HNN model from .pth file"""
    try:
        # Import required modules
        from nn_models import MLP
        from hnn import HNN
        
        # Create model architecture (matching retrain script)
        nn_model = MLP(input_dim, hidden_dim, 2, 'tanh')
        model = HNN(input_dim, differentiable_model=nn_model,
                   field_type='solenoidal', baseline=False)
        
        # Load state dict from .pth file
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        return model
        
    except Exception as e:
        print(f"Failed to load custom HNN model: {e}")
        return None


def generate_hnn_trajectory(hnn_model, q_initial, p_initial, n_steps=30, dt=0.05, model_path=None):
    """使用HNN模型生成轨迹"""
    if hnn_model is None:
        return None
        
    total_time = dt * (n_steps - 1)
    t_span = (0, total_time)
    t_eval = np.linspace(t_span[0], t_span[1], n_steps)
            
    if q_initial.ndim == 0:
        q_initial = np.array([q_initial])
    if p_initial.ndim == 0:
        p_initial = np.array([p_initial])
    hnn_initial_condition = np.concatenate([q_initial, p_initial])
        
    sol = integrate_model(hnn_model, t_span, hnn_initial_condition, 
                        t_eval=t_eval, rtol=1e-8, atol=1e-10)
    trajectory = sol.y.T
    return trajectory


def process_single_file(data_file, hnn_model, system, system_args):
    """处理单个数据文件"""
    # try:
    data = np.load(data_file)
    fake_traj = data['fake']
    mask = data['mask']
    system_name = data['system_name'].item()
    
    # Extract q and p from fake trajectory
    q_fake = system.extract_q(fake_traj, mask, frame_idx=0)
    q_fake_next = system.extract_q(fake_traj, mask, frame_idx=1)
    
    # Calculate momentum
    delta = 0.05
    mass = np.atleast_1d(system_args["mass"])
    p_fake = mass[:, None] * (q_fake_next - q_fake) / delta
    
    # Generate HNN trajectory
    hnn_traj = generate_hnn_trajectory(
        hnn_model, q_fake.flatten(), p_fake.flatten(), 
        n_steps=fake_traj.shape[0], dt=0.05
    )
    
    if hnn_traj is None:
        return None
    
    system._rollout = hnn_traj.transpose()
    hnn_coords_full = system._convert_to_t2n_format()
    
    if system_name == 'NObjectGravity':
        real_traj = system.calculate_fixed_rollout(q_fake,p_fake,fake_traj.shape[0],delta)
    else:
        real_traj = system.calculate_fixed_rollout(q_fake.flatten(),p_fake.flatten(),fake_traj.shape[0],delta)

    # Get all valid particles for comparison
    valid_idx = np.where(mask > 0)[0]
    if len(valid_idx) == 0:
        return None
    
    # Calculate MSE for each particle
    particle_mses_hgan = []
    particle_mses_hnn = []
    
    for i, particle_idx in enumerate(valid_idx):
        # Extract coordinates for this particle
        real_coords = real_traj[:, :, particle_idx]  # (T, 2)
        fake_coords = fake_traj[:, :, particle_idx]  # (T, 2)
        
        # Extract HNN coordinates for this particle
        if hnn_coords_full.ndim == 3:
            hnn_coords = hnn_coords_full[:, :, particle_idx]  # (T, 2)
        else:
            # If HNN only has one particle, use it for all
            hnn_coords = hnn_coords_full[:, :]  # (T, 2)
        
        # Calculate MSE for this particle
        hgan_mse = np.mean((fake_coords - real_coords) ** 2)
        hnn_mse = np.mean((hnn_coords - real_coords) ** 2)
        
        particle_mses_hgan.append(hgan_mse)
        particle_mses_hnn.append(hnn_mse)
    
    # Return results for this file
    return {
        'filename': os.path.basename(data_file),
        'system_name': system_name,
        'n_particles': len(valid_idx),
        'mean_hgan_mse': np.mean(particle_mses_hgan),
        'mean_hnn_mse': np.mean(particle_mses_hnn),
        'particle_mses_hgan': particle_mses_hgan,
        'particle_mses_hnn': particle_mses_hnn,
        'trajectories': {
            'real': real_traj,
            'fake': fake_traj,
            'hnn': hnn_coords_full
        },
        'mask': mask,
        'valid_idx': valid_idx
    }
    
    # except Exception as e:
    #     print(f"Error processing {data_file}: {e}")
    #     return None


def plot_best_trajectory(best_result, results_dir):
    """绘制最佳结果的轨迹对比图"""
    real_traj = best_result['trajectories']['real']
    fake_traj = best_result['trajectories']['fake']
    hnn_coords_full = best_result['trajectories']['hnn']
    valid_idx = best_result['valid_idx']
    n_valid_particles = len(valid_idx)
    
    # Create visualization
    fig, axes = plt.subplots(2, n_valid_particles, figsize=(5*n_valid_particles, 10))
    if n_valid_particles == 1:
        axes = axes.reshape(2, 1)
    
    for i, particle_idx in enumerate(valid_idx):
        # Extract coordinates for this particle
        real_coords = real_traj[:, :, particle_idx]  # (T, 2)
        fake_coords = fake_traj[:, :, particle_idx]  # (T, 2)
        
        # Extract HNN coordinates for this particle
        if hnn_coords_full.ndim == 3:
            hnn_coords = hnn_coords_full[:, :, particle_idx]  # (T, 2)
        else:
            # If HNN only has one particle, use it for all
            hnn_coords = hnn_coords_full[:, :]  # (T, 2)
        
        # Plot position trajectories
        ax_pos = axes[0, i]
        ax_pos.plot(real_coords[:, 0], real_coords[:, 1], 'k-', linewidth=2, label='Real', alpha=0.8)
        ax_pos.plot(fake_coords[:, 0], fake_coords[:, 1], 'b--', linewidth=2, label='HGAN', alpha=0.7)
        ax_pos.plot(hnn_coords[:, 0], hnn_coords[:, 1], 'r:', linewidth=2, label='HNN', alpha=0.7)
        ax_pos.set_title(f'Particle {particle_idx} - Position Trajectory\n(Best: {best_result["filename"]})')
        ax_pos.set_xlabel('X Position')
        ax_pos.set_ylabel('Y Position')
        ax_pos.legend()
        ax_pos.grid(True, alpha=0.3)
        ax_pos.axis('equal')
        
        # Plot time series
        ax_time = axes[1, i]
        time_steps = np.arange(len(real_coords))
        ax_time.plot(time_steps, real_coords[:, 0], 'k-', linewidth=2, label='Real X', alpha=0.8)
        ax_time.plot(time_steps, fake_coords[:, 0], 'b--', linewidth=2, label='HGAN X', alpha=0.7)
        ax_time.plot(time_steps, hnn_coords[:, 0], 'r:', linewidth=2, label='HNN X', alpha=0.7)
        ax_time.plot(time_steps, real_coords[:, 1], 'k-', linewidth=1, label='Real Y', alpha=0.6)
        ax_time.plot(time_steps, fake_coords[:, 1], 'b--', linewidth=1, label='HGAN Y', alpha=0.5)
        ax_time.plot(time_steps, hnn_coords[:, 1], 'r:', linewidth=1, label='HNN Y', alpha=0.5)
        ax_time.set_title(f'Particle {particle_idx} - Time Series')
        ax_time.set_xlabel('Time Step')
        ax_time.set_ylabel('Coordinate Value')
        ax_time.legend()  
        ax_time.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot to results directory
    plot_filename = os.path.join(results_dir, f'best_trajectory_comparison_{best_result["system_name"]}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_filename

def main():
    if not HGAN_AVAILABLE:
        print("Error: HGAN modules not available!")
        return
    
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='Compare HGAN and HNN trajectories for a folder of data files')
    parser.add_argument('--data_folder', type=str, default='/home/jiayinliu/Desktop/ControlledHamiltonianGAN_0/data/1P',
                        help='Path to folder containing .npz data files')
    parser.add_argument('--results_folder', type=str, default='/home/jiayinliu/Desktop/ControlledHamiltonianGAN_0/results',
                        help='Path to save results')
    args = parser.parse_args()
    
    data_folder = args.data_folder
    if not os.path.exists(data_folder):
        print(f"Data folder not found: {data_folder}")
        return
    
    # Find all .npz files in the folder
    npz_files = glob.glob(os.path.join(data_folder, '*_data.npz'))
    if not npz_files:
        print(f"No *_data.npz files found in {data_folder}")
        return
    
    print(f"Found {len(npz_files)} data files to process")
    
    # Load first file to determine system type and setup
    first_data = np.load(npz_files[0])
    system_name = first_data['system_name'].item()
    mask = first_data['mask']
    
    # Determine dataset system name based on system_name and mask
    if system_name == 'NObjectGravity':
        num_objects = np.sum(mask > 0)
        if num_objects == 2:
            dataset_system_name = 'two_body'
        elif num_objects == 3:
            dataset_system_name = 'three_body'
        else:
            dataset_system_name = 'two_body'  # default
    else:
        # Map system names to dataset names
        system_mapping = {
            'Pendulum': 'pendulum',
            'Spring': 'mass_spring',
            'ChaoticPendulum': 'double_pendulum'
        }
        dataset_system_name = system_mapping.get(system_name, 'two_body')
    
    print(f"Detected system: {system_name} ({dataset_system_name})")
    
    # Load the appropriate HNN model for this system type
    model_path = os.path.join(PROJECT_PATH, f'hnn_model/custom_hnn_{dataset_system_name}_model.pth')
    if not os.path.exists(model_path):
        print(f"HNN model not found: {model_path}")
        return
    
    # Determine input dimension based on system type
    if dataset_system_name in ['mass_spring', 'pendulum']:
        input_dim = 2 # 2 position + 2 momentum for single particle systems
    elif dataset_system_name == 'double_pendulum':
        input_dim = 4  # 4 position + 4 momentum for double pendulum
    elif dataset_system_name == 'two_body':
        input_dim = 8  # 4 position + 4 momentum for two body
    elif dataset_system_name == 'three_body':
        input_dim = 12  # 6 position + 6 momentum for three body
    else:
        input_dim = 8  # default
    
    print(f"Loading HNN model: {os.path.basename(model_path)} (input_dim={input_dim})")
    hnn_model = load_custom_hnn_model(model_path, input_dim=input_dim)
    if hnn_model is None:
        print("Failed to load HNN model!")
        return
    print("✓ HNN model loaded successfully")
    
    # Get system args and create environment once
    print(f"Setting up environment for {system_name} system...")
    system_args = constant_physics_hgn[dataset_system_name]
    system_args = {
        k: ([item() if callable(item) else item for item in v] if isinstance(v, list) else 
            (v() if callable(v) else v))
        for k, v in system_args.items()
    }
    
    system = EnvFactory.get_environment(system_name, **system_args)
    print(f"✓ Environment created successfully")
    
    # Process all files
    results = []
    
    print(f"\nProcessing {len(npz_files)} files...")
    for i, data_file in enumerate(npz_files):
        print(f"Processing {i+1}/{len(npz_files)}: {os.path.basename(data_file)}")
        
        result = process_single_file(data_file, hnn_model, system, system_args)
        if result is not None:
            results.append(result)
    
    if not results:
        print("No files were successfully processed!")
        return
    
    print(f"\nSuccessfully processed {len(results)} files")
    
    # Calculate overall averages
    overall_avg_hgan = np.mean([r['mean_hgan_mse'] for r in results])
    overall_avg_hnn = np.mean([r['mean_hnn_mse'] for r in results])
    
    print(f"\n=== OVERALL AVERAGE RESULTS ===")
    print(f"System: {system_name} ({dataset_system_name})")
    print(f"Files processed: {len(results)}")
    print(f"Overall Average HGAN MSE: {overall_avg_hgan:.6f}")
    print(f"Overall Average HNN MSE:  {overall_avg_hnn:.6f}")
    print(f"Overall MSE Ratio (HGAN/HNN): {overall_avg_hgan/overall_avg_hnn:.3f}")
    print(f"{'✓ HGAN better overall' if overall_avg_hgan < overall_avg_hnn else '✗ HNN better overall'}")
    
    # Find best performing case (lowest HGAN MSE)
    best_result = min(results, key=lambda x: x['mean_hgan_mse'])
    print(f"\n=== BEST PERFORMING CASE ===")
    print(f"File: {best_result['filename']}")
    print(f"System: {best_result['system_name']} ({best_result['n_particles']} particles)")
    print(f"HGAN MSE: {best_result['mean_hgan_mse']:.6f}")
    print(f"HNN MSE: {best_result['mean_hnn_mse']:.6f}")
    print(f"Improvement: {(1 - best_result['mean_hgan_mse']/best_result['mean_hnn_mse'])*100:.1f}%")
    
    # Create results directory for the system type
    system_name = best_result['system_name']
    if system_name == 'NObjectGravity':
        n_particles = best_result['n_particles']
        results_dir = os.path.join(PROJECT_PATH, 'results', f'{n_particles}_body')
    else:
        system_mapping = {
            'Spring': 'mass_spring',
            'Pendulum': 'pendulum',
            'ChaoticPendulum': 'double_pendulum'
        }
        system_folder = system_mapping.get(system_name, system_name.lower())
        results_dir = os.path.join(PROJECT_PATH, 'results', system_folder)
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot best trajectory
    plot_filename = plot_best_trajectory(best_result, results_dir)
    print(f"Best trajectory plot saved to: {plot_filename}")
    
    # Save comprehensive results to JSON
    comprehensive_results = {
        'data_folder': data_folder,
        'system_name': system_name,
        'dataset_system_name': dataset_system_name,
        'processed_files': len(results),
        'total_files_found': len(npz_files),
        'overall_averages': {
            'hgan_mse': overall_avg_hgan,
            'hnn_mse': overall_avg_hnn,
            'improvement_ratio': overall_avg_hgan / overall_avg_hnn
        },
        'best_case': {
            'filename': best_result['filename'],
            'system_name': best_result['system_name'],
            'n_particles': best_result['n_particles'],
            'hgan_mse': best_result['mean_hgan_mse'],
            'hnn_mse': best_result['mean_hnn_mse'],
            'improvement_ratio': best_result['mean_hgan_mse'] / best_result['mean_hnn_mse']
        },
        'individual_results': [
            {
                'filename': r['filename'],
                'system_name': r['system_name'], 
                'n_particles': r['n_particles'],
                'hgan_mse': r['mean_hgan_mse'],
                'hnn_mse': r['mean_hnn_mse'],
                'improvement_ratio': r['mean_hgan_mse'] / r['mean_hnn_mse']
            }
            for r in results
        ]
    }
    
    results_json_path = os.path.join(results_dir, 'folder_comparison_results.json')
    with open(results_json_path, 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    print(f"Comprehensive results saved to: {results_json_path}")
    print("\n=== ANALYSIS COMPLETE ===")
    print(f"Processed {len(results)}/{len(npz_files)} files successfully")
    print(f"Results saved in: {results_dir}")
    print(f"Best case visualization: {os.path.basename(plot_filename)}")

if __name__ == "__main__":
    main()
