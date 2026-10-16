"""Training configuration (model, phases, loss weights) for Google Colab."""
INTERNAL_DATA_ROOT = __import__("os").environ.get("ARPIA_ROOT", ".")
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@dataclass
class DataConfig:
    batch_size: int = 2048
    num_neighbors: List[int] = None
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 16
    
    def __post_init__(self):
        if self.num_neighbors is None:
            self.num_neighbors = [128, 96, 64]

@dataclass
class ModelConfig:
    hidden_channels: int = 2048
    out_channels: int = 1024
    num_layers: int = 16
    heads: int = 64
    dropout: float = 0.1
    
    use_rf_attention: bool = True
    rf_attention_heads: int = 8
    use_node_type_embedding: bool = True
    node_type_embedding_dim: int = 32
    use_skip_connections: bool = True
    use_layer_norm: bool = True
    
    use_multi_scale: bool = True
    local_gnn_layers: int = 6
    regional_gnn_layers: int = 5
    global_context_layers: int = 5
    
    use_advanced_features: bool = True
    composite_features: bool = True
    spatial_features: bool = True
    temporal_features: bool = False

@dataclass
class LossConfig:
    position_weight: float = 10.0
    physics_weight: float = 2.0
    connection_weight: float = 3.0
    non_connection_weight: float = 5.0
    triangular_weight: float = 1.0
    precision_weight: float = 2.0
    
    spatial_dispersion_weight: float = 0.8
    smoothness_weight: float = 0.1
    feature_reconstruction_weight: float = 0.001
    movement_penalty_weight: float = 0.1
    rf_physics_weight: float = 1.5
    boundary_weight: float = 0.5
    topology_weight: float = 0.3
    local_precision_weight: float = 1.0
    adaptive_threshold_weight: float = 0.8
    
    use_weight_annealing: bool = True
    use_haversine: bool = True
    adaptive_thresholds: bool = True
    use_numba_acceleration: bool = True
    log_verbose: bool = True

@dataclass
class PhaseConfig:
    name: str
    epochs: int
    lr: float
    movement_scale: float
    loss_weights: Dict[str, float]
    special_config: Dict[str, Union[int, float, bool]] = None
    
    def __post_init__(self):
        if self.special_config is None:
            self.special_config = {}

@dataclass
class OptimizerConfig:
    type: str = 'adam'
    weight_decay: float = 1e-5
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = False

@dataclass
class SchedulerConfig:
    type: str = 'plateau'
    factor: float = 0.5
    patience: int = 10
    threshold: float = 1e-4
    min_lr: float = 1e-6
    
    T_max: int = 100
    eta_min: float = 1e-6

@dataclass
class EDAConfig:
    enabled: bool = True
    frequency: int = 20
    save_plots: bool = True
    save_data: bool = True
    detailed_analysis: bool = True
    
    plot_dpi: int = 300
    plot_format: str = 'png'
    plot_style: str = 'seaborn-v0_8'
    
    error_percentiles: List[int] = None
    accuracy_thresholds: List[int] = None
    
    def __post_init__(self):
        if self.error_percentiles is None:
            self.error_percentiles = [25, 50, 75, 90, 95, 99]
        if self.accuracy_thresholds is None:
            self.accuracy_thresholds = [50, 100, 200, 500, 1000]

@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    patience: int = 50
    min_delta: float = 1.0
    metric: str = 'mean_error'
    mode: str = 'min'

@dataclass
class CheckpointConfig:
    enabled: bool = True
    save_frequency: int = 10
    save_best_only: bool = True
    save_dir: str = 'checkpoints'
    filename_template: str = 'arpia_gnn_v4_epoch_{epoch}.pt'

@dataclass
class ValidationConfig:
    enabled: bool = True
    frequency: int = 1
    physics_validation: bool = True
    geometric_validation: bool = True
    
    max_error_threshold: float = 5000.0
    min_accuracy_threshold: float = 0.1
    
    max_connection_distance: float = 1000.0
    min_non_connection_distance: float = 200.0

class ColabConfig:
    
    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.loss = LossConfig()
        self.optimizer = OptimizerConfig()
        self.scheduler = SchedulerConfig()
        self.eda = EDAConfig()
        self.early_stopping = EarlyStoppingConfig()
        self.checkpoint = CheckpointConfig()
        self.validation = ValidationConfig()
        
        self.mixed_precision = True
        self.compile_model = True
        self.use_tensorboard = True
        self.save_memory = True
        
        self.total_epochs = 300
        self.phases = self._create_advanced_phases()
        
        self.device = 'cuda'
        self.num_threads = 2
        self.random_seed = 42
        
        self.output_dir = f"{INTERNAL_DATA_ROOT}/RESULTADOS"
        self.experiment_name = 'arpia_gnn_v4_colab'
        
        self.log_level = 'INFO'
        self.log_to_file = True
        self.log_to_tensorboard = True
        
        self.physics = self._create_physics_config()
        
        self.colab_optimization = self._create_colab_optimization()
        
        self.logging = self._create_logging_config()
    
    def _create_advanced_phases(self):
        return {
            'phase_1': PhaseConfig(
                name='foundation_learning',
                epochs=100,
                lr=1e-3,
                movement_scale=1.0,
                loss_weights={
                    'position_weight': 10.0,
                    'physics_weight': 1.0,
                    'connection_weight': 2.0,
                    'non_connection_weight': 3.0,
                    'triangular_weight': 0.5,
                    'precision_weight': 1.0,
                    'spatial_dispersion_weight': 0.5,
                    'smoothness_weight': 0.1,
                    'rf_physics_weight': 1.0
                },
                special_config={
                    'use_weight_annealing': True,
                    'gradient_clipping': 1.0,
                    'warmup_epochs': 10
                }
            ),
            'phase_2': PhaseConfig(
                name='dynamics_stabilization',
                epochs=100,
                lr=5e-4,
                movement_scale=0.8,
                loss_weights={
                    'position_weight': 8.0,
                    'physics_weight': 2.0,
                    'connection_weight': 3.0,
                    'non_connection_weight': 4.0,
                    'triangular_weight': 1.0,
                    'precision_weight': 2.0,
                    'spatial_dispersion_weight': 0.8,
                    'smoothness_weight': 0.2,
                    'rf_physics_weight': 1.5,
                    'movement_penalty_weight': 0.1
                },
                special_config={
                    'use_weight_annealing': True,
                    'gradient_clipping': 0.5,
                    'adaptive_thresholds': True
                }
            ),
            'phase_3': PhaseConfig(
                name='precision_focusing',
                epochs=100,
                lr=1e-4,
                movement_scale=0.5,
                loss_weights={
                    'position_weight': 15.0,
                    'physics_weight': 2.0,
                    'connection_weight': 3.0,
                    'non_connection_weight': 5.0,
                    'triangular_weight': 1.0,
                    'precision_weight': 3.0,
                    'spatial_dispersion_weight': 1.0,
                    'smoothness_weight': 0.3,
                    'rf_physics_weight': 2.0,
                    'movement_penalty_weight': 0.2,
                    'local_precision_weight': 2.0,
                    'adaptive_threshold_weight': 1.0
                },
                special_config={
                    'use_weight_annealing': False,
                    'gradient_clipping': 0.3,
                    'adaptive_thresholds': True,
                    'precision_mode': True
                }
            )
        }
    
    def _create_colab_optimization(self):
        return {
            'memory_optimization': {
                'gradient_checkpointing': True,
                'mixed_precision': True,
                'optimize_graph': True,
                'memory_efficient_attention': True
            },
            'performance_optimization': {
                'compile_model': True,
                'use_channels_last': True,
                'pin_memory': True,
                'non_blocking_transfer': True
            },
            'storage_optimization': {
                'compress_checkpoints': True,
                'save_optimizer_state': False,
                'cleanup_old_checkpoints': True
            }
        }
    
    def _create_physics_config(self):
        return {
            'rf_parameters': {
                'frequency_ghz': 1.8,
                'max_range_km': 35.0,
                'min_signal_strength': -120.0,
                'path_loss_exponent': 3.5,
                'shadowing_std': 8.0
            },
            'triangulation': {
                'min_stations': 3,
                'max_stations': 6,
                'gdop_threshold': 5.0,
                'convergence_threshold': 1e-6
            },
            'validation': {
                'max_velocity_ms': 100.0,
                'max_acceleration_ms2': 10.0,
                'consistency_threshold': 0.95
            }
        }
    
    def _create_logging_config(self):
        return {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file_logging': True,
            'console_logging': True,
            'tensorboard_logging': True,
            'wandb_logging': False,
            'log_frequency': 10
        }
    
    def get_total_epochs(self):
        return sum(phase.epochs for phase in self.phases.values())
    
    def get_phase_info(self, phase_number):
        phase_key = f'phase_{phase_number}'
        return self.phases.get(phase_key, None)
    
    def validate_config(self):
        errors = []
        
        if self.data.batch_size <= 0:
            errors.append("Batch size deve ser positivo")
        
        if self.model.hidden_channels <= 0:
            errors.append("Hidden channels deve ser positivo")
        
        if self.total_epochs <= 0:
            errors.append('Total epochs must be positive')
        
        total_phase_epochs = sum(phase.epochs for phase in self.phases.values())
        if total_phase_epochs != self.total_epochs:
            errors.append(f"Total epochs of the phases ({total_phase_epochs}) does not match the configured total ({self.total_epochs})")
        
        for phase_name, phase in self.phases.items():
            if phase.lr <= 0:
                errors.append(f"Learning rate of {phase_name} must be positive")
        
        if self.physics['rf_parameters']['frequency_ghz'] <= 0:
            errors.append('RF frequency must be positive')
        
        if errors:
            raise ValueError('Configuration errors were found:\n' + "\n".join(errors))
        
        return True
    
    def to_dict(self):
        def convert_dataclass(obj):
            if hasattr(obj, '__dict__'):
                result = {}
                for key, value in obj.__dict__.items():
                    if hasattr(value, '__dict__'):
                        result[key] = convert_dataclass(value)
                    elif isinstance(value, dict):
                        result[key] = {k: convert_dataclass(v) if hasattr(v, '__dict__') else v 
                                     for k, v in value.items()}
                    else:
                        result[key] = value
                return result
            return obj
        
        return convert_dataclass(self)
    
    def save_config(self, filepath):
        config_dict = self.to_dict()
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    @classmethod
    def load_from_json(cls, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        config = cls()
        
        dataclass_sections = {
            'data': config.data,
            'model': config.model,
            'loss': config.loss,
            'optimizer': config.optimizer,
            'scheduler': config.scheduler,
            'eda': config.eda,
            'early_stopping': config.early_stopping,
            'checkpoint': config.checkpoint,
            'validation': config.validation,
        }
        for section_name, section_obj in dataclass_sections.items():
            if section_name in data:
                for key, value in data[section_name].items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
        
        simple_attrs = [
            'mixed_precision', 'compile_model', 'use_tensorboard', 'save_memory',
            'total_epochs', 'device', 'num_threads', 'random_seed',
            'output_dir', 'experiment_name', 'log_level', 'log_to_file',
            'log_to_tensorboard',
        ]
        for attr in simple_attrs:
            if attr in data:
                setattr(config, attr, data[attr])
        
        for dict_attr in ['physics', 'colab_optimization', 'logging']:
            if dict_attr in data:
                setattr(config, dict_attr, data[dict_attr])
        
        if 'phases' in data:
            phases = {}
            for phase_key, phase_data in data['phases'].items():
                phases[phase_key] = PhaseConfig(
                    name=phase_data['name'],
                    epochs=phase_data['epochs'],
                    lr=phase_data['lr'],
                    movement_scale=phase_data['movement_scale'],
                    loss_weights=phase_data['loss_weights'],
                    special_config=phase_data.get('special_config', {})
                )
            config.phases = phases
        
        print(f"✅ Configuration loaded from: {filepath}")
        print(f"   Experimento: {config.experiment_name}")
        print(f"   Épocas: {config.get_total_epochs()}")
        print(f"   Loss: physics={config.loss.physics_weight}, conn={config.loss.connection_weight}, non_conn={config.loss.non_connection_weight}")
        
        config.validate_config()
        return config
    
    def print_summary(self):
        print("=" * 60)
        print('ARPIA GNN v4 CONFIGURATION - GOOGLE COLAB')
        print("=" * 60)
        print(f"Total number of epochs: {self.get_total_epochs()}")
        print(f"Batch size: {self.data.batch_size}")
        print(f"Hidden channels: {self.model.hidden_channels}")
        print(f"Number of layers used: {self.model.num_layers}")
        print(f"Attention heads: {self.model.heads}")
        print(f"Mixed precision: {self.mixed_precision}")
        print(f"Compile model: {self.compile_model}")
        print()
        
        print("FASES DE TREINAMENTO:")
        print("-" * 40)
        for phase_name, phase in self.phases.items():
            print(f"{phase.name}: {phase.epochs} epochs, LR={phase.lr}")
        print()
        
        print("CONFIGURAÇÕES DE LOSS:")
        print("-" * 40)
        phase_1 = self.phases['phase_1']
        for weight_name, weight_value in phase_1.loss_weights.items():
            print(f"{weight_name}: {weight_value}")
        print()
        
        print("CONFIGURAÇÕES DE FÍSICA:")
        print("-" * 40)
        rf_params = self.physics['rf_parameters']
        print(f"Frequência: {rf_params['frequency_ghz']} GHz")
        print(f"Maximum range in km: {rf_params['max_range_km']} km")
        print(f"Minimum signal strength: {rf_params['min_signal_strength']} dBm")
        print("=" * 60)

def create_default_config():
    config = ColabConfig()
    config.validate_config()
    return config

def create_custom_config(
    batch_size=1024,
    hidden_channels=1024,
    num_layers=12,
    heads=32,
    total_epochs=300,
    use_mixed_precision=True,
    eda_frequency=20,
    **kwargs
):
    config = ColabConfig()
    
    config.data.batch_size = batch_size
    config.model.hidden_channels = hidden_channels
    config.model.num_layers = num_layers
    config.model.heads = heads
    config.total_epochs = total_epochs
    config.mixed_precision = use_mixed_precision
    config.eda.frequency = eda_frequency
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    config.validate_config()
    return config

default_config = create_default_config()

if __name__ == "__main__":
    config = create_default_config()
    config.print_summary()
    
    config.save_config('/tmp/config_exemplo.json')
    print('Configuration saved to /tmp/config_exemplo.json') 
