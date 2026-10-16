#!/usr/bin/env python3
"""Training utilities: environment setup and saving of final results."""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from collections import defaultdict
import time

logger = logging.getLogger(__name__)

class EarlyStopping:
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best_weights: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.should_stop = False
        self.best_weights = None
        
        self.is_better = self._get_is_better_func()
    
    def _get_is_better_func(self):
        if self.mode == 'min':
            return lambda current, best: current < best - self.min_delta
        else:
            return lambda current, best: current > best + self.min_delta
    
    def step(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return True
        
        if self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False
    
    def reset(self):
        self.best_score = None
        self.counter = 0
        self.should_stop = False
        self.best_weights = None

class MetricsTracker:
    
    def __init__(self):
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        self.epoch_times = []
        self.best_epoch = 0
        self.best_val_loss = float('inf')
    
    def update(self, train_metrics: Dict, val_metrics: Dict):
        for key, value in train_metrics.items():
            self.train_metrics[key].append(value)
        
        for key, value in val_metrics.items():
            self.val_metrics[key].append(value)
        
        if val_metrics.get('total_loss', float('inf')) < self.best_val_loss:
            self.best_val_loss = val_metrics['total_loss']
            self.best_epoch = len(self.val_metrics['total_loss']) - 1
    
    def get_summary(self) -> Dict:
        summary = {
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'total_epochs': len(self.val_metrics.get('total_loss', [])),
            'final_train_loss': self.train_metrics.get('total_loss', [0])[-1],
            'final_val_loss': self.val_metrics.get('total_loss', [0])[-1],
            'final_train_error': self.train_metrics.get('mean_error_meters', [0])[-1],
            'final_val_error': self.val_metrics.get('mean_error_meters', [0])[-1]
        }
        return summary
    
    def plot_metrics(self, save_path: Optional[str] = None):
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Métricas de Treinamento', fontsize=16)
        
        epochs = range(len(self.train_metrics['total_loss']))
        
        axes[0, 0].plot(epochs, self.train_metrics['total_loss'], 'b-', label='Treino')
        axes[0, 0].plot(epochs, self.val_metrics['total_loss'], 'r-', label='Validação')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Época')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(epochs, self.train_metrics['mean_error_meters'], 'g-', label='Treino')
        axes[0, 1].plot(epochs, self.val_metrics['mean_error_meters'], 'orange', label='Validação')
        axes[0, 1].set_title('Erro Médio')
        axes[0, 1].set_xlabel('Época')
        axes[0, 1].set_ylabel('Erro (m)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(epochs, self.val_metrics['precision_200m'], 'purple', label='<200m')
        axes[1, 0].plot(epochs, self.val_metrics['precision_500m'], 'brown', label='<500m')
        axes[1, 0].set_title('Precisão')
        axes[1, 0].set_xlabel('Época')
        axes[1, 0].set_ylabel('Precisão (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].axis('off')
        summary = self.get_summary()
        summary_text = f"\n        Final Summary:\n        \n        Best Epoch: {summary['best_epoch']}\n        Best Val Loss: {summary['best_val_loss']:.4f}\n        \n        Final Error: {summary['final_val_error']:.1f}m\n        Epochs: {summary['total_epochs']}\n        "
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved to: {save_path}")
        
        plt.show()

class PerformanceAnalyzer:
    
    def __init__(self):
        self.predictions = None
        self.targets = None
        self.errors = None
        self.analysis_results = {}
    
    def analyze(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict:
        self.predictions = predictions.cpu().numpy()
        self.targets = targets.cpu().numpy()
        
        self.errors = np.linalg.norm(self.predictions - self.targets, axis=1)
        
        self.analysis_results = {
            'mean_error': np.mean(self.errors),
            'median_error': np.median(self.errors),
            'std_error': np.std(self.errors),
            'min_error': np.min(self.errors),
            'max_error': np.max(self.errors),
            'q25_error': np.percentile(self.errors, 25),
            'q75_error': np.percentile(self.errors, 75),
            'q90_error': np.percentile(self.errors, 90),
            'q95_error': np.percentile(self.errors, 95),
            'q99_error': np.percentile(self.errors, 99),
            
            'precision_10m': np.mean(self.errors <= 0.01) * 100,
            'precision_50m': np.mean(self.errors <= 0.05) * 100,
            'precision_100m': np.mean(self.errors <= 0.1) * 100,
            'precision_200m': np.mean(self.errors <= 0.2) * 100,
            'precision_500m': np.mean(self.errors <= 0.5) * 100,
            'precision_1000m': np.mean(self.errors <= 1.0) * 100,
            
            'correlation_x': np.corrcoef(self.predictions[:, 0], self.targets[:, 0])[0, 1],
            'correlation_y': np.corrcoef(self.predictions[:, 1], self.targets[:, 1])[0, 1],
            
            'bias_x': np.mean(self.predictions[:, 0] - self.targets[:, 0]),
            'bias_y': np.mean(self.predictions[:, 1] - self.targets[:, 1]),
            
            'num_samples': len(self.errors)
        }
        
        return self.analysis_results
    
    def create_visualizations(self, save_dir: str = "/content/outputs"):
        
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        plt.figure(figsize=(12, 8))
        plt.hist(self.errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(np.mean(self.errors), color='red', linestyle='--', 
                   label=f'Média: {np.mean(self.errors):.3f}')
        plt.axvline(np.median(self.errors), color='green', linestyle='--', 
                   label=f'Mediana: {np.median(self.errors):.3f}')
        plt.xlabel('Erro (normalizado)')
        plt.ylabel('Frequência')
        plt.title('Distribuição de Erros')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        axes[0].scatter(self.targets[:, 0], self.predictions[:, 0], alpha=0.6, s=1)
        axes[0].plot([self.targets[:, 0].min(), self.targets[:, 0].max()], 
                    [self.targets[:, 0].min(), self.targets[:, 0].max()], 'r--')
        axes[0].set_xlabel('X Verdadeiro')
        axes[0].set_ylabel('X Predito')
        axes[0].set_title(f'Coordenada X (R²={self.analysis_results["correlation_x"]:.3f})')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].scatter(self.targets[:, 1], self.predictions[:, 1], alpha=0.6, s=1)
        axes[1].plot([self.targets[:, 1].min(), self.targets[:, 1].max()], 
                    [self.targets[:, 1].min(), self.targets[:, 1].max()], 'r--')
        axes[1].set_xlabel('Y Verdadeiro')
        axes[1].set_ylabel('Y Predito')
        axes[1].set_title(f'Coordenada Y (R²={self.analysis_results["correlation_y"]:.3f})')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'predictions_vs_targets.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        error_percentiles = [
            self.errors[self.errors <= np.percentile(self.errors, 25)],
            self.errors[(self.errors > np.percentile(self.errors, 25)) & 
                       (self.errors <= np.percentile(self.errors, 50))],
            self.errors[(self.errors > np.percentile(self.errors, 50)) & 
                       (self.errors <= np.percentile(self.errors, 75))],
            self.errors[self.errors > np.percentile(self.errors, 75)]
        ]
        
        plt.figure(figsize=(10, 6))
        plt.boxplot(error_percentiles, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        plt.ylabel('Erro (normalizado)')
        plt.title('Distribution of Errors by Quartile')
        plt.grid(True, alpha=0.3)
        plt.savefig(save_dir / 'error_quartiles.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Visualizations saved to: {save_dir}")

class GPUMonitor:
    
    def __init__(self):
        self.memory_usage = []
        self.timestamps = []
        self.is_available = torch.cuda.is_available()
        
        if self.is_available:
            self.device_name = torch.cuda.get_device_name(0)
            self.total_memory = torch.cuda.get_device_properties(0).total_memory
        else:
            self.device_name = "CPU"
            self.total_memory = 0
    
    def log_usage(self):
        if self.is_available:
            memory_used = torch.cuda.memory_allocated()
            self.memory_usage.append(memory_used)
            self.timestamps.append(time.time())
    
    def get_summary(self) -> Dict:
        if not self.is_available:
            return {"device": "CPU", "memory_usage": None}
        
        memory_mb = [m / 1024**2 for m in self.memory_usage]
        
        return {
            "device": self.device_name,
            "total_memory_gb": self.total_memory / 1024**3,
            "peak_memory_mb": max(memory_mb) if memory_mb else 0,
            "avg_memory_mb": np.mean(memory_mb) if memory_mb else 0,
            "current_memory_mb": memory_mb[-1] if memory_mb else 0
        }
    
    def plot_usage(self, save_path: Optional[str] = None):
        if not self.is_available or not self.memory_usage:
            print('⚠️ GPU data are not available')
            return
        
        memory_mb = [m / 1024**2 for m in self.memory_usage]
        times = [(t - self.timestamps[0]) / 60 for t in self.timestamps]
        
        plt.figure(figsize=(12, 6))
        plt.plot(times, memory_mb, 'b-', linewidth=2)
        plt.xlabel('Tempo (minutos)')
        plt.ylabel('Memória GPU (MB)')
        plt.title(f"GPU Memory Use - {self.device_name}")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
    **kwargs
):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat(),
        **kwargs
    }
    
    torch.save(checkpoint, filepath)
    print(f"💾 Checkpoint salvo: {filepath}")

def load_checkpoint(filepath: str, model: nn.Module, optimizer: torch.optim.Optimizer) -> Dict:
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"✅ Checkpoint carregado: {filepath}")
    print(f"   Época: {checkpoint['epoch']}")
    print(f"   Loss: {checkpoint['loss']:.4f}")
    
    return checkpoint

def calculate_model_size(model: nn.Module) -> Dict:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    model_size_bytes = total_params * 4
    model_size_mb = model_size_bytes / (1024 * 1024)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params,
        'model_size_mb': model_size_mb,
        'model_size_bytes': model_size_bytes
    }

def estimate_training_time(
    num_epochs: int,
    num_batches: int,
    batch_size: int,
    model_size_mb: float,
    device: str = 'cuda'
) -> Dict:
    
    if device == 'cuda':
        if 'A100' in torch.cuda.get_device_name(0):
            time_per_batch = 0.1
        elif 'V100' in torch.cuda.get_device_name(0):
            time_per_batch = 0.15
        elif 'T4' in torch.cuda.get_device_name(0):
            time_per_batch = 0.3
        else:
            time_per_batch = 0.5
    else:
        time_per_batch = 2.0
    
    time_per_batch *= (1 + model_size_mb / 100)
    
    time_per_epoch = num_batches * time_per_batch
    total_time = num_epochs * time_per_epoch
    
    return {
        'time_per_batch_sec': time_per_batch,
        'time_per_epoch_min': time_per_epoch / 60,
        'total_time_min': total_time / 60,
        'total_time_hours': total_time / 3600,
        'estimated_completion': datetime.now().timestamp() + total_time
    }

def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def create_summary_report(
    config: Dict,
    training_results: Dict,
    model_info: Dict,
    save_path: str
) -> str:
    
    report = f"\n# ARPIA GNN v4 TRAINING REPORT\n## Google Colab Pipeline\n\n### Configuration\n- **GPU**: {config.get('gpu_name', 'N/A')}\n- **Batch Size**: {config.get('batch_size', 'N/A')}\n- **Hidden Dim**: {config.get('hidden_dim', 'N/A')}\n- **Epochs**: {training_results.get('total_epochs', 'N/A')}\n- **Phases**: {training_results.get('phases_completed', 'N/A')}\n\n### Results\n- **Final Error**: {training_results.get('final_error', 0):.1f}m\n- **Best Error**: {training_results.get('best_error', 0):.1f}m\n- **Accuracy <200m**: {training_results.get('final_precision_200m', 0):.1f}%\n- **Accuracy <500m**: {training_results.get('final_precision_500m', 0):.1f}%\n- **Improvement**: {training_results.get('improvement', 0):.1f}m\n\n### Model\n- **Parameters**: {model_info.get('total_params', 0):,}\n- **Size**: {model_info.get('model_size_mb', 0):.1f} MB\n- **Node Types**: {len(model_info.get('node_types', []))}\n- **Edge Types**: {len(model_info.get('edge_types', []))}\n\n### Time\n- **Duration**: {training_results.get('training_time_minutes', 0):.1f} minutes\n- **Timestamp**: {training_results.get('timestamp', 'N/A')}\n\n### Evaluation\n"
    
    final_error = training_results.get('final_error', float('inf'))
    if final_error < 150:
        evaluation = "🌟 EXCELENTE!"
    elif final_error < 250:
        evaluation = "⭐ MUITO BOM!"
    elif final_error < 400:
        evaluation = "✅ BOM!"
    else:
        evaluation = "⚠️ REGULAR"
    
    report += f"**{evaluation}**\n"
    
    with open(save_path, 'w') as f:
        f.write(report)
    
    print(f"📄 Report saved to: {save_path}")
    return report

if __name__ == "__main__":
    
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    
    for epoch in range(20):
        loss = 1.0 - epoch * 0.05 + np.random.normal(0, 0.01)
        improved = early_stopping.step(loss)
        
        if early_stopping.should_stop:
            print(f"Early stopping at epoch {epoch}")
            break
    
    tracker = MetricsTracker()
    
    gpu_monitor = GPUMonitor()
    gpu_monitor.log_usage()
    
    print('✅ Utilities tested successfully!')
    

def setup_training_environment(device='cuda'):
    print('🔧 Configuring the training environment...')
    
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        
        gpu_info = {
            'name': torch.cuda.get_device_name(0),
            'memory_total': torch.cuda.get_device_properties(0).total_memory / 1024**3,
            'memory_allocated': torch.cuda.memory_allocated() / 1024**3
        }
        
        print(f"✅ GPU: {gpu_info['name']}")
        print(f"💾 VRAM: {gpu_info['memory_total']:.1f} GB")
    else:
        gpu_info = {'name': 'CPU', 'memory_total': 0, 'memory_allocated': 0}
        print("✅ Usando CPU")
    
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('seaborn')
    
    return {
        'device': device,
        'gpu_info': gpu_info,
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available()
    }


def save_final_results(model, history, evaluation_results, save_dir):
    from pathlib import Path
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    print(f"💾 Saving the final results to {save_path}...")
    
    model_path = save_path / 'final_model.pt'
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to: {model_path}")
    
    history_path = save_path / 'training_history.json'
    with open(history_path, 'w') as f:
        json_history = {}
        for key, values in history.items():
            if isinstance(values, list):
                json_history[key] = [
                    v.tolist() if hasattr(v, 'tolist') else v
                    for v in values
                ]
            else:
                json_history[key] = values
        json.dump(json_history, f, indent=2)
    print(f"✅ History saved to: {history_path}")
    
    eval_path = save_path / 'evaluation_results.json'
    with open(eval_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    print(f"✅ Evaluation saved to: {eval_path}")
    
    if evaluation_results and history:
        best_results = evaluation_results.get('val', evaluation_results.get('test', {}))
        summary = {
            'final_metrics': best_results,
            'total_epochs': len(history.get('epoch', [])),
            'best_loss': min(history.get('loss', [float('inf')])),
            'model_size': calculate_model_size(model),
            'timestamp': datetime.now().isoformat()
        }
        
        summary_path = save_path / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✅ Resumo salvo: {summary_path}")
    
    print(f"🎉 All results saved to {save_path}")
    
    return str(save_path) 
