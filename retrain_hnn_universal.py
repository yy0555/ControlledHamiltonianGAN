#!/usr/bin/env python3

"""
通用HNN重训练脚本 - 使用EnvFactory和HGN常量物理参数
支持: mass_spring, pendulum, double_pendulum, two_body, three_body
"""

import os
import sys
import numpy as np
import torch
import argparse
import time
import logging

# 添加路径
PROJECT_PATH = '/home/jiayinliu/Desktop/ControlledHamiltonianGAN_0'
sys.path.append(os.path.join(PROJECT_PATH, 'hamiltonian-nn'))
sys.path.append(os.path.join(PROJECT_PATH, 'src'))
sys.path.append(os.path.join(PROJECT_PATH, 'src/hgan/hgn/environments'))

from nn_models import MLP
from hnn import HNN
from utils import L2_loss
import scipy.integrate

# HGAN环境导入
from environment_factory import EnvFactory
from hgan.hgn_datasets import constant_physics_hgn

class UniversalHNNRetrainer:
    """通用HNN重训练器 - 使用EnvFactory和HGN参数"""
    
    def __init__(self, system_name, dt=0.05):
        """
        Args:
            system_name: HGN系统名称 ('mass_spring', 'pendulum', 'double_pendulum', 'two_body', 'three_body')
            dt: 时间步长
        """
        self.hgn_system_name = system_name
        self.dt = dt
        
        # 系统名称映射：HGN -> Environment类名
        self.system_mapping = {
            "mass_spring": "Spring",
            "pendulum": "Pendulum", 
            "double_pendulum": "ChaoticPendulum",
            "two_body": "NObjectGravity",
            "three_body": "NObjectGravity",
        }
        
        # 获取环境类名
        self.env_class_name = self.system_mapping[system_name]
        
        # 获取HGN常量物理参数
        self.system_params = self._get_hgn_constant_params(system_name)
        
        # 使用EnvFactory创建环境实例
        self.env = EnvFactory.get_environment(self.env_class_name, **self.system_params)
        self.system_dim = self._get_system_dimension()
        
        print(f"初始化 {system_name} 系统重训练器")
        print(f"  HGN系统名: {system_name}")
        print(f"  环境类名: {self.env_class_name}")
        print(f"  系统维度: {self.system_dim}")
        print(f"  物理参数: {self.system_params}")
        print(f"  时间步长: {dt}")
    
    def _get_hgn_constant_params(self, system_name):
        """获取HGN常量物理参数"""
        params_config = constant_physics_hgn[system_name]
        params = {}
        
        for key, value in params_config.items():
            if isinstance(value, list):
                # 处理质量列表
                params[key] = [v() for v in value]
            else:
                params[key] = value()
        
        return params
    
    def _get_system_dimension(self):
        """获取系统维度 - 基于环境类名"""
        if self.env_class_name == 'Spring':
            return 2  # [q, p]
        elif self.env_class_name == 'Pendulum':
            return 2  # [q, p]
        elif self.env_class_name == 'ChaoticPendulum':
            return 4  # [q1, q2, p1, p2]
        elif self.env_class_name == 'NObjectGravity':
            # 从环境实例获取物体数量
            n_objects = getattr(self.env, 'n_objects', 2)
            return 4 * n_objects  # [x1,y1,x2,y2,...,px1,py1,px2,py2,...]
        else:
            return 2
    
    def _dynamics_wrapper(self, t, coords):
        """动力学函数包装器 - 使用环境的动力学"""
        return self.env._dynamics(t, coords)
    
    def get_trajectory(self, t_span=[0, 1.45], y0=None, radius=None, noise_std=0.1):
        """生成单个轨迹 - 使用环境的初始条件采样和动力学"""
        n_points = int((t_span[1] - t_span[0]) / self.dt) + 1
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        
        # 生成初始条件
        if y0 is None:
            # 使用环境的采样方法获取初始条件
            if radius is None:
                radius_bounds = self.env.get_default_radius_bounds()
                radius = np.random.uniform(radius_bounds[0], radius_bounds[1])
            
            # 不同环境的采样参数格式不同
            if self.env_class_name == 'ChaoticPendulum':
                # ChaoticPendulum需要单个radius值
                self.env._sample_init_conditions(radius)
            else:
                # 其他环境需要radius_bound tuple
                self.env._sample_init_conditions((radius, radius))
            
            # 将环境的q,p转换为状态向量格式
            if self.env_class_name in ['Spring', 'Pendulum']:
                y0 = np.concatenate([self.env.q, self.env.p])
            elif self.env_class_name == 'ChaoticPendulum':
                y0 = np.concatenate([self.env.q, self.env.p])
            elif self.env_class_name == 'NObjectGravity':
                # gravity系统: [q, p] -> [x1,y1,x2,..., px1,py1,px2,...]
                y0 = np.concatenate([self.env.q.flatten(), self.env.p.flatten()])
            else:
                y0 = np.random.rand(self.system_dim) * 2 - 1
        # 积分
        sol = scipy.integrate.solve_ivp(
            self._dynamics_wrapper, t_span, y0, t_eval=t_eval
        )
        
        coords = sol.y
        # 计算导数
        dydt = []
        for i in range(len(t_eval)):
            state = coords[:, i]
            dydt.append(self._dynamics_wrapper(t_eval[i], state))
        dydt = np.array(dydt).T
        # 添加噪声
        coords += np.random.randn(*coords.shape) * noise_std
        
        return coords, dydt, t_eval
    
    def get_dataset(self, seed=0, samples=160, test_split=0.8):
        """生成训练数据集"""
        data = {'meta': {
            'system': self.hgn_system_name,
            'params': self.system_params,
            'dt': self.dt,
            'samples': samples,
            'seed': seed  # 记录随机种子
        }}
        
        np.random.seed(seed)  # 固定随机种子以确保可重复性
        xs, dxs = [], []
        
        for s in range(samples):            
            coords, dydt, t = self.get_trajectory()
            xs.append(coords.T)
            dxs.append(dydt.T)
        
        data['x'] = np.concatenate(xs)
        data['dx'] = np.concatenate(dxs)
        
        split_ix = int(len(data['x']) * test_split)
        split_data = {}
        for k in ['x', 'dx']:
            split_data[k] = data[k][:split_ix]
            split_data['test_' + k] = data[k][split_ix:]
        
        return split_data
    
    def train_hnn(self, total_steps=1000, learning_rate=1e-3, hidden_dim=200):
        """训练HNN模型"""
        
        # 简单logger配置 - 只输出消息内容
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        logger = logging.getLogger(__name__)
        
        # GPU设备检测和设置
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  使用设备: {device}")
        if torch.cuda.is_available():
            print(f"  GPU名称: {torch.cuda.get_device_name(0)}")
            print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
                
        output_dim = self.system_dim if False else 2  # baseline=False，所以输出维度固定为2
        nn_model = MLP(self.system_dim, hidden_dim, output_dim, 'tanh')
        hnn_model = HNN(self.system_dim, nn_model, field_type='solenoidal', baseline=False)
        
        # 模型移动到GPU
        hnn_model = hnn_model.to(device)
        
        # 确保HNN模型的permutation tensor也在正确的设备上
        if hasattr(hnn_model, 'M') and hnn_model.M is not None:
            hnn_model.M = hnn_model.M.to(device)
        
        optim = torch.optim.Adam(hnn_model.parameters(), learning_rate, weight_decay=1e-4)
        
        stats = {'train_loss': [], 'test_loss': []}
        
        # 时间统计
        start_time = time.time()
        step_start_time = start_time
        
        for step in range(total_steps + 1):
            data = self.get_dataset()
 
            # 数据移动到GPU
            x = torch.tensor(data['x'], requires_grad=True, dtype=torch.float32).to(device)
            test_x = torch.tensor(data['test_x'], requires_grad=True, dtype=torch.float32).to(device)
            dxdt = torch.tensor(data['dx'], dtype=torch.float32).to(device)
            test_dxdt = torch.tensor(data['test_dx'], dtype=torch.float32).to(device)

            dxdt_hat = hnn_model.time_derivative(x)
            loss = L2_loss(dxdt, dxdt_hat)
            loss.backward()
            optim.step()
            optim.zero_grad()
            
            test_dxdt_hat = hnn_model.time_derivative(test_x)
            test_loss = L2_loss(test_dxdt, test_dxdt_hat)
            
            stats['train_loss'].append(loss.item())
            stats['test_loss'].append(test_loss.item())
            
            if step % 100 == 0:
                current_time = time.time()
                step_elapsed = current_time - step_start_time
                total_elapsed = current_time - start_time
                remaining_steps = total_steps - step
                estimated_remaining_time = (step_elapsed / 100) * remaining_steps if step > 0 else 0
                
                logger.info(f"Step {step:4d} | train_loss: {loss.item():.6e} | test_loss: {test_loss.item():.6e} | time: {step_elapsed:.1f}s | total: {total_elapsed:.1f}s | eta: {estimated_remaining_time:.1f}s")
                
                step_start_time = current_time
        
        # 训练完成时间统计
        total_training_time = time.time() - start_time
        logger.info(f"训练完成 | 总用时: {total_training_time:.1f}s ({total_training_time/60:.1f}min)")
        
        return hnn_model, stats



def main():
    parser = argparse.ArgumentParser(description='通用HNN重训练脚本 - 使用HGN常量物理参数')
    parser.add_argument('--system', type=str, required=True,
                       choices=['mass_spring', 'pendulum', 'double_pendulum', 'two_body', 'three_body'],
                       help='HGN物理系统类型')
    parser.add_argument('--dt', type=float, default=0.05, help='时间步长')
    parser.add_argument('--total_steps', type=int, default=50000, help='训练步数')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='学习率')
    parser.add_argument('--hidden_dim', type=int, default=200, help='隐藏层维度')
    
    args = parser.parse_args()
    
    print("="*60)
    print(f"通用HNN重训练 - {args.system} 系统")
    print("="*60)
    
    # 创建重训练器并训练
    retrainer = UniversalHNNRetrainer(args.system, args.dt)
    model, stats = retrainer.train_hnn(
        total_steps=args.total_steps,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim
    )
    
    # 保存模型 - 使用清晰的命名
    output_filename = f'custom_hnn_{args.system}_model.pth'
    output_path = os.path.join(PROJECT_PATH, output_filename)
    torch.save(model.state_dict(), output_path)
    print(f"\\n✓ 模型保存到: {output_path}")
    
    print(f"\\n现在可以使用这个重新训练的{args.system}模型进行公平对比！")

if __name__ == "__main__":
    main()
