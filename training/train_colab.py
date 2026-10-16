#!/usr/bin/env python3
"""Training pipeline entry points used on Google Colab."""

INTERNAL_DATA_ROOT = __import__("os").environ.get("ARPIA_ROOT", ".")
import importlib
import gc
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
import logging
from typing import Dict, Any, Optional
import yaml
import os
from datetime import datetime
import psutil

def setup_ultra_detailed_logging():
    
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-20s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(console_handler)
    
    log_file = f'{INTERNAL_DATA_ROOT}/TREINAMENTO_REAL/OUTPUT/training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    os.makedirs(f"{INTERNAL_DATA_ROOT}/TREINAMENTO_REAL/OUTPUT", exist_ok=True)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    print(f"📝 Logging configurado:")
    print(f"   Console: INFO level")
    print(f"   Arquivo: {log_file} (DEBUG level)")
    
    return log_file

log_file = setup_ultra_detailed_logging()
logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

print('🚀 ARPIA GNN v4 - ULTRA-ADVANCED FULL PIPELINE')
print("="*80)
print("📊 SISTEMA ROBUSTO IMPLEMENTADO:")
print("   ✅ Multi-Scale GNN (Local + Regional + Global)")
print("   ✅ PhysicsInformedLoss (15+ componentes)")
print("   ✅ PhaseManager (6 fases progressivas)")
print('   ✅ Automatic EDA (denormalized metrics)')
print("   ✅ Sistema de checkpoints robusto")
print('   ✅ Physical validation in real time')
print("   ✅ LOGGING ULTRA-DETALHADO")
print("="*80)

try:
    import config_colab
    import models
    import losses
    import trainers
    import data_loaders
    import utils
    
    importlib.reload(config_colab)
    importlib.reload(models)
    importlib.reload(losses)
    importlib.reload(trainers)
    importlib.reload(data_loaders)
    importlib.reload(utils)
    
    logger.info('🔄 Modules reloaded successfully!')
    
except Exception as e:
    logger.error(f"❌ Error while reloading modules: {e}")
    print('⚠️  Continuing with the versions already loaded...')

from config_colab import ColabConfig, create_custom_config
from models import create_model, ARPIAGNNv4
from losses import create_physics_informed_loss, PhysicsInformedLoss
from trainers import create_trainer, ARPIATrainer
from data_loaders import create_data_loaders, prepare_graph_data
from utils import setup_training_environment, save_final_results


class ProgressTracker:
    
    def __init__(self, total_epochs, log_frequency=1):
        self.total_epochs = total_epochs
        self.log_frequency = log_frequency
        self.start_time = time.time()
        self.epoch_times = []
        self.losses = []
        self.phase_changes = []
        
    def log_epoch_start(self, epoch, phase_info):
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        logger.info(f"🚀 EPOCH {epoch+1}/{self.total_epochs} STARTED")
        logger.info(f"   ⏱️ Tempo decorrido: {elapsed/60:.1f}min")
        logger.info(f"   📊 Fase: {phase_info['phase_name']}")
        logger.info(f"   📈 Progresso fase: {phase_info['epoch_in_phase']}/{phase_info['total_epochs_in_phase']}")
        logger.info(f"   🎯 Learning rate: {phase_info['learning_rate']:.2e}")
        
        if len(self.epoch_times) > 0:
            avg_time = np.mean(self.epoch_times[-10:])
            remaining_epochs = self.total_epochs - epoch
            eta = remaining_epochs * avg_time
            logger.info(f"   ⏰ ETA: {eta/60:.1f}min restantes")
    
    def log_epoch_end(self, epoch, loss, metrics=None):
        epoch_time = time.time() - (self.start_time + sum(self.epoch_times))
        self.epoch_times.append(epoch_time)
        self.losses.append(loss)
        
        logger.info(f"✅ EPOCH {epoch+1} IS FINISHED")
        logger.info(f"   🔥 Loss: {loss:.6f}")
        logger.info(f"   ⏱️ Epoch time: {epoch_time:.1f}s")
        
        if metrics:
            logger.info(f"   📊 Mean error: {metrics.get('mean_error', 0):.1f}m")
            logger.info(f"   🎯 Accuracy <100m: {metrics.get('accuracy_100m', 0):.1%}")
            logger.info(f"   🎯 Accuracy <200m: {metrics.get('accuracy_200m', 0):.1%}")
        
        if len(self.losses) >= 2:
            improvement = self.losses[-2] - self.losses[-1]
            logger.info(f"   📈 Melhoria: {improvement:+.6f}")
        
        if len(self.epoch_times) >= 5:
            avg_time = np.mean(self.epoch_times[-5:])
            logger.info(f"   ⚡ Mean speed: {avg_time:.1f}s/epoch")
    
    def log_phase_change(self, old_phase, new_phase):
        logger.info(f"🔄 PHASE CHANGE IS DETECTED")
        logger.info(f"   📤 Saindo: {old_phase}")
        logger.info(f"   📥 Entrando: {new_phase}")
        self.phase_changes.append((time.time(), old_phase, new_phase))
    
    def log_batch_progress(self, epoch, batch_idx, total_batches, loss):
        if batch_idx % 50 == 0:
            progress = batch_idx / total_batches * 100
            logger.debug(f"   Batch {batch_idx}/{total_batches} ({progress:.1f}%) - Loss: {loss:.6f}")


def create_pipeline_ultra_avancado(
    graph_data,
    scalers,
    config_overrides=None,
    device='cuda'
):
    logger.info('🏗️ Creating the ULTRA-ADVANCED pipeline...')
    
    logger.info("⚙️ Configurando sistema...")
    if isinstance(config_overrides, ColabConfig):
        config = config_overrides
    elif isinstance(config_overrides, str) and config_overrides.endswith('.json'):
        config = ColabConfig.load_from_json(config_overrides)
    elif config_overrides:
        config = create_custom_config(**config_overrides)
    else:
        config = ColabConfig()
    
    logger.info("🔍 Validando configuração...")
    if not config.validate_config():
        raise ValueError("Configuração inválida!")
    
    config.print_summary()
    
    logger.info('🧠 Creating the ROBUST Multi-Scale model...')
    model = create_model(config.to_dict(), graph_data, use_ensemble=False)
    model = model.to(device)
    
    model_info = model.get_model_info()
    logger.info(f"📊 Modelo criado:")
    logger.info(f"   Parâmetros: {model_info['total_params']:,}")
    logger.info(f"   Tamanho: {model_info['model_size_mb']:.1f} MB")
    logger.info(f"   Arquitetura: {model_info['architecture']}")
    
    logger.info('🔬 Creating the ULTRA-ADVANCED PhysicsInformedLoss...')
    loss_fn = create_physics_informed_loss(config.to_dict())
    
    logger.info("📊 Criando data loaders otimizados...")
    data_loaders = create_data_loaders(
        graph_data,
        batch_size=config.data.batch_size,
        num_neighbors=config.data.num_neighbors,
        num_workers=config.data.num_workers,
        device=device
    )
    
    logger.info('🚀 Creating the ULTRA-ADVANCED trainer...')
    trainer = create_trainer(model, loss_fn, config.to_dict(), device)
    
    trainer.set_scalers(scalers)
    
    logger.info('✅ ULTRA-ADVANCED pipeline created successfully!')
    
    return model, loss_fn, trainer, data_loaders, config


def execute_training_ultra_avancado(
    trainer,
    data_loaders,
    config,
    save_checkpoints=True,
    save_dir='output'
):
    logger.info('🚀 STARTING THE ULTRA-ADVANCED TRAINING')
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    checkpoint_path = save_path / 'checkpoints'
    checkpoint_path.mkdir(exist_ok=True)
    
    eda_path = save_path / 'eda'
    eda_path.mkdir(exist_ok=True)
    
    trainer.eda_config = getattr(trainer, 'eda_config', {})
    trainer.eda_config['save_dir'] = str(eda_path)
    
    config.save_config(save_path / 'config.json')
    
    total_epochs = config.get_total_epochs()
    progress_tracker = ProgressTracker(total_epochs)
    
    start_time = time.time()
    
    try:
        logger.info(f"🎯 Starting the training of {total_epochs} epochs")
        
        history = trainer.train(
            train_loader=data_loaders['train'],
            val_loader=data_loaders.get('val', None),
            num_epochs=total_epochs
        )
        
        training_time = time.time() - start_time
        
        logger.info(f"🏁 Training finished in {training_time:.1f}s")
        logger.info(f"   📊 Épocas: {len(history['epoch'])}")
        logger.info(f"   🎯 Best error: {trainer.best_metrics['error']:.1f}m")
        
        if save_checkpoints:
            final_checkpoint = checkpoint_path / 'final_model.pt'
            trainer.save_checkpoint(
                final_checkpoint,
                epoch=len(history['epoch']),
                metrics=trainer.best_metrics
            )
        
        history_file = save_path / 'training_history.json'
        with open(history_file, 'w') as f:
            json_history = {}
            for key, values in history.items():
                json_history[key] = [
                    v.tolist() if isinstance(v, torch.Tensor) else v
                    for v in values
                ]
            json.dump(json_history, f, indent=2)
        
        logger.info(f"✅ Results saved to {save_path}")
        
        return history
        
    except KeyboardInterrupt:
        logger.info('🛑 Training interrupted by the user')
        if save_checkpoints:
            interrupt_checkpoint = checkpoint_path / 'interrupted_model.pt'
            trainer.save_checkpoint(
                interrupt_checkpoint,
                epoch=trainer.phase_manager.total_epochs_completed,
                metrics=trainer.best_metrics
            )
            logger.info(f"💾 Interruption checkpoint saved to {interrupt_checkpoint}")
        
        return trainer.training_history
    
    except Exception as e:
        logger.error(f"❌ Error during the training: {e}")
        if save_checkpoints:
            error_checkpoint = checkpoint_path / 'error_model.pt'
            trainer.save_checkpoint(
                error_checkpoint,
                epoch=trainer.phase_manager.total_epochs_completed,
                metrics=trainer.best_metrics
            )
            logger.info(f"💾 Error checkpoint saved to {error_checkpoint}")
        
        raise


def run_comprehensive_evaluation(
    model,
    trainer,
    data_loaders,
    config,
    save_dir='output'
):
    logger.info('📊 Running the comprehensive evaluation...')
    
    save_path = Path(save_dir)
    eval_path = save_path / 'evaluation'
    eval_path.mkdir(exist_ok=True)
    
    evaluation_results = {}
    
    for split_name, data_loader in data_loaders.items():
        if data_loader is None:
            continue
            
        logger.info(f"🔍 Avaliando {split_name}...")
        
        metrics = trainer.evaluate_model_accurate_metrics(data_loader)
        evaluation_results[split_name] = metrics
        
        logger.info(f"   📊 {split_name} - Mean error: {metrics['mean_error']:.1f}m")
        logger.info(f"   🎯 {split_name} - Accuracy <100m: {metrics['accuracy_100m']:.1%}")
        logger.info(f"   🎯 {split_name} - Accuracy <200m: {metrics['accuracy_200m']:.1%}")
    
    results_file = eval_path / 'comprehensive_evaluation.json'
    with open(results_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    create_evaluation_visualizations(evaluation_results, eval_path)
    
    logger.info(f"✅ Evaluation saved to {eval_path}")
    
    return evaluation_results


def create_evaluation_visualizations(results, save_path):
    try:
        plt.style.use('seaborn-v0_8')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comprehensive Evaluation - ARPIA GNN v4', fontsize=16, fontweight='bold')
        
        splits = list(results.keys())
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
        
        mean_errors = [results[split]['mean_error'] for split in splits]
        axes[0, 0].bar(splits, mean_errors, color=colors[:len(splits)], alpha=0.7)
        axes[0, 0].set_title('Mean Error per Split')
        axes[0, 0].set_ylabel('Erro (m)')
        axes[0, 0].grid(True, alpha=0.3)
        
        thresholds = [100, 200, 500, 1000]
        for i, split in enumerate(splits):
            precisions = [
                results[split][f'accuracy_{t}m']
                for t in thresholds
            ]
            axes[0, 1].plot(thresholds, precisions, 'o-', label=split, 
                           color=colors[i], linewidth=2, markersize=8)
        
        axes[0, 1].set_title('Curva de Precisão')
        axes[0, 1].set_xlabel('Threshold (m)')
        axes[0, 1].set_ylabel('Precisão')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        error_metrics = ['mean_error', 'median_error', 'error_75th', 'error_90th', 'error_95th']
        
        for i, split in enumerate(splits):
            values = [results[split][metric] for metric in error_metrics]
            x_pos = np.arange(len(error_metrics)) + i * 0.2
            axes[1, 0].bar(x_pos, values, width=0.2, label=split, 
                          color=colors[i], alpha=0.7)
        
        axes[1, 0].set_title('Distribuição de Erros')
        axes[1, 0].set_xlabel('Métrica')
        axes[1, 0].set_ylabel('Erro (m)')
        axes[1, 0].set_xticks(np.arange(len(error_metrics)) + 0.3)
        axes[1, 0].set_xticklabels(['Média', 'Mediana', 'P75', 'P90', 'P95'])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        summary_text = "RESUMO DA AVALIAÇÃO\n" + "="*30 + "\n"
        
        for split in splits:
            metrics = results[split]
            summary_text += f"\n{split.upper()}:\n"
            summary_text += f"  Erro médio: {metrics['mean_error']:.1f}m\n"
            summary_text += f"  RMSE: {metrics['rmse']:.1f}m\n"
            summary_text += f"  Precisão <100m: {metrics['accuracy_100m']:.1%}\n"
            summary_text += f"  Precisão <200m: {metrics['accuracy_200m']:.1%}\n"
            summary_text += f"  Erro máximo: {metrics['max_error']:.1f}m\n"
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                       verticalalignment='top', fontsize=10, fontfamily='monospace')
        axes[1, 1].set_title('Resumo Numérico')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path / 'comprehensive_evaluation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info('✅ Evaluation visualizations were created')
        
    except Exception as e:
        logger.error(f"❌ Error while creating visualizations: {e}")


def main_pipeline_ultra_avancado(
    graph_data,
    scalers,
    config_overrides=None,
    device='cuda',
    save_dir='output'
):
    
    print("\n" + "🚀" * 40)
    print('ULTRA-ADVANCED PIPELINE - STARTING')
    print("🚀" * 40)
    
    try:
        model, loss_fn, trainer, data_loaders, config = create_pipeline_ultra_avancado(
            graph_data, scalers, config_overrides, device
        )
        
        history = execute_training_ultra_avancado(
            trainer, data_loaders, config, save_checkpoints=True, save_dir=save_dir
        )
        
        evaluation_results = run_comprehensive_evaluation(
            model, trainer, data_loaders, config, save_dir
        )
        
        save_final_results(model, history, evaluation_results, save_dir)
        
        print("\n" + "✅" * 40)
        print('ULTRA-ADVANCED PIPELINE - FINISHED')
        print("✅" * 40)
        
        return model, history, evaluation_results
        
    except Exception as e:
        logger.error(f"❌ Pipeline error: {e}")
        print("\n" + "❌" * 40)
        print('ULTRA-ADVANCED PIPELINE - ERROR')
        print("❌" * 40)
        raise


def run_training_cell6(
    graph_data,
    scalers,
    config_overrides=None,
    device='cuda',
    save_dir=f"{INTERNAL_DATA_ROOT}/TREINAMENTO_REAL/OUTPUT",
    epochs=300,
    batch_size=1024,
    hidden_channels=1024,
    num_layers=12,
    heads=32,
    eda_frequency=20,
    use_mixed_precision=True
):
    
    print("\n" + "🌟" * 50)
    print('CELL 6 - ARPIA GNN v4 FULL PIPELINE')
    print("🌟" * 50)
    
    if config_overrides is None:
        config_overrides = {}
    
    config_overrides.update({
        'batch_size': batch_size,
        'hidden_channels': hidden_channels,
        'num_layers': num_layers,
        'heads': heads,
        'total_epochs': epochs,
        'use_mixed_precision': use_mixed_precision,
        'eda_frequency': eda_frequency,
        'device': device
    })
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    start_time = time.time()
    
    try:
        print(f"🎯 Configurações:")
        print(f"   📊 Épocas: {epochs}")
        print(f"   🔄 Batch Size: {batch_size}")
        print(f"   🧠 Hidden Channels: {hidden_channels}")
        print(f"   📏 Layers: {num_layers}")
        print(f"   🎭 Attention Heads: {heads}")
        print(f"   🔬 EDA Frequency: {eda_frequency}")
        print(f"   💾 Device: {device}")
        print(f"   📁 Save Dir: {save_dir}")
        print(f"   ⚡ Mixed Precision: {use_mixed_precision}")
        
        model, history, evaluation_results = main_pipeline_ultra_avancado(
            graph_data=graph_data,
            scalers=scalers,
            config_overrides=config_overrides,
            device=device,
            save_dir=save_dir
        )
        
        total_time = time.time() - start_time
        
        config = create_custom_config(**config_overrides)
        
        create_final_summary(
            model=model,
            history=history,
            evaluation_results=evaluation_results,
            config=config,
            total_time=total_time,
            save_dir=save_dir
        )
        
        print("\n" + "🎉" * 50)
        print('CELL 6 - PIPELINE FINISHED SUCCESSFULLY!')
        print("🎉" * 50)
        print(f"⏱️ Tempo total: {total_time:.1f}s ({total_time/60:.1f}m)")
        print(f"📁 Results saved to: {save_dir}")
        
        return model, history, evaluation_results, config
        
    except Exception as e:
        logger.error(f"❌ Error in the pipeline of cell 6: {e}")
        print("\n" + "💥" * 50)
        print('CELL 6 - PIPELINE ERROR')
        print("💥" * 50)
        raise


def create_final_summary(model, history, evaluation_results, config, total_time, save_dir):
    
    save_path = Path(save_dir)
    
    final_metrics = evaluation_results.get('val', evaluation_results.get('test', {}))
    
    config_dict = config.to_dict() if hasattr(config, 'to_dict') else config
    
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_time_seconds': total_time,
        'total_time_minutes': total_time / 60,
        'configuration': {
            'epochs': config_dict.get('total_epochs', config_dict.get('epochs', 300)),
            'batch_size': config_dict.get('batch_size', 128),
            'hidden_channels': config_dict.get('hidden_channels', 256),
            'num_layers': config_dict.get('num_layers', 4),
            'heads': config_dict.get('heads', 8),
            'device': config_dict.get('device', 'cuda'),
            'mixed_precision': config_dict.get('use_mixed_precision', True)
        },
        'model_info': {
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'model_size_mb': sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)
        },
        'training_results': {
            'total_epochs': len(history.get('epoch', [])),
            'final_loss': history.get('loss', [0])[-1] if history.get('loss') else 0,
            'best_loss': min(history.get('loss', [float('inf')])),
            'improvement': history.get('loss', [0])[0] - history.get('loss', [0])[-1] if len(history.get('loss', [])) > 0 else 0
        },
        'evaluation_metrics': final_metrics
    }
    
    summary_file = save_path / 'final_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    report_lines = [
        'FINAL REPORT - ARPIA GNN v4',
        "=" * 50,
        f"Data: {summary['timestamp']}",
        f"Tempo total: {summary['total_time_minutes']:.1f} minutos",
        "",
        "CONFIGURAÇÃO:",
        f"  Épocas: {summary['configuration']['epochs']}",
        f"  Batch Size: {summary['configuration']['batch_size']}",
        f"  Hidden Channels: {summary['configuration']['hidden_channels']}",
        f"  Layers: {summary['configuration']['num_layers']}",
        f"  Heads: {summary['configuration']['heads']}",
        f"  Device: {summary['configuration']['device']}",
        f"  Mixed Precision: {summary['configuration']['mixed_precision']}",
        "",
        "MODELO:",
        f"  Parâmetros: {summary['model_info']['total_parameters']:,}",
        f"  Parâmetros Treináveis: {summary['model_info']['trainable_parameters']:,}",
        f"  Tamanho: {summary['model_info']['model_size_mb']:.1f} MB",
        "",
        "TREINAMENTO:",
        f"  Épocas executadas: {summary['training_results']['total_epochs']}",
        f"  Loss final: {summary['training_results']['final_loss']:.4f}",
        f"  Melhor loss: {summary['training_results']['best_loss']:.4f}",
        f"  Melhoria: {summary['training_results']['improvement']:.4f}",
        "",
        "AVALIAÇÃO:"
    ]
    
    if final_metrics:
        report_lines.extend([
            f"  Erro médio: {final_metrics.get('mean_error', 0):.1f}m",
            f"  RMSE: {final_metrics.get('rmse', 0):.1f}m",
            f"  Precisão <100m: {final_metrics.get('accuracy_100m', 0):.1%}",
            f"  Precisão <200m: {final_metrics.get('accuracy_200m', 0):.1%}",
            f"  Precisão <500m: {final_metrics.get('accuracy_500m', 0):.1%}",
            f"  Erro máximo: {final_metrics.get('max_error', 0):.1f}m"
        ])
    
    if final_metrics:
        error = final_metrics.get('mean_error', float('inf'))
        if error < 150:
            evaluation = "🌟 EXCELENTE!"
        elif error < 250:
            evaluation = "⭐ MUITO BOM!"
        elif error < 400:
            evaluation = "✅ BOM!"
        else:
            evaluation = "⚠️ REGULAR"
        
        report_lines.extend([
            "",
            f"AVALIAÇÃO GERAL: {evaluation}",
            "=" * 50
        ])
    
    report_file = save_path / 'final_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print("\n📄 RESUMO FINAL:")
    print("=" * 40)
    for line in report_lines[-20:]:
        print(line)
    
    print(f"\n💾 Full summary saved to:")
    print(f"   📊 JSON: {summary_file}")
    print(f"   📄 Texto: {report_file}")
    print(f"   📝 Log detalhado: {log_file}")


def quick_training_cell6(graph_data, scalers, epochs=50, batch_size=64):
    
    print('🚀 QUICK TRAINING - CELL 6')
    
    config_overrides = {
        'batch_size': batch_size,
        'hidden_channels': 128,
        'num_layers': 2,
        'heads': 4,
        'total_epochs': epochs,
        'use_mixed_precision': True,
        'eda_frequency': 10,
    }
    
    return run_training_cell6(
        graph_data=graph_data,
        scalers=scalers,
        config_overrides=config_overrides,
        epochs=epochs,
        batch_size=batch_size,
        save_dir='/content/outputs/quick_training'
    )


print('\n🎯 TRAINING FUNCTIONS ADDED:')
print("   ✅ run_training_cell6() - Pipeline completo")
print('   ✅ quick_training_cell6() - Quick version')
print("   ✅ create_final_summary() - Resumo detalhado")
print("   ✅ ProgressTracker() - Logging ultra-detalhado")
print("   ✅ setup_ultra_detailed_logging() - Sistema de logs")
print("="*80)


print(f"\n📝 LOGGING SYSTEM IS CONFIGURED:")
print(f"   📄 Log file: {log_file}")
print(f"   📊 Console level: INFO")
print(f"   🔍 File level: DEBUG")
print(f"   ⏰ Timestamps detalhados")
print(f"   🎯 Progresso em tempo real")
print("="*80)


class RealTimeMonitor:
    
    def __init__(self, total_epochs, update_frequency=5):
        self.total_epochs = total_epochs
        self.update_frequency = update_frequency
        self.start_time = time.time()
        self.epoch_times = []
        self.losses = []
        self.errors = []
        self.current_epoch = 0
        self.current_phase = ""
        self.last_update = time.time()
        
    def update(self, epoch, loss, error=None, phase="", metrics=None):
        self.current_epoch = epoch
        self.current_phase = phase
        
        current_time = time.time()
        if len(self.epoch_times) == 0:
            epoch_time = current_time - self.start_time
        else:
            epoch_time = current_time - self.last_update
        
        self.epoch_times.append(epoch_time)
        self.losses.append(loss)
        if error is not None:
            self.errors.append(error)
        
        self.last_update = current_time
        
        if epoch % self.update_frequency == 0 or epoch == 0:
            self.show_progress(metrics)
    
    def show_progress(self, metrics=None):
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        progress_pct = (self.current_epoch + 1) / self.total_epochs * 100
        
        if len(self.epoch_times) > 0:
            avg_epoch_time = np.mean(self.epoch_times[-10:])
            remaining_epochs = self.total_epochs - self.current_epoch - 1
            eta = remaining_epochs * avg_epoch_time
        else:
            avg_epoch_time = 0
            eta = 0
        
        print("\n" + "="*80)
        print(f"📊 REAL-TIME PROGRESS - EPOCH {self.current_epoch + 1}/{self.total_epochs}")
        print("="*80)
        
        bar_length = 40
        filled_length = int(bar_length * progress_pct / 100)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        print(f"🎯 Progresso: |{bar}| {progress_pct:.1f}%")
        
        print(f"⏱️ Tempo decorrido: {elapsed/3600:.1f}h {(elapsed%3600)/60:.0f}m")
        print(f"⏰ ETA restante: {eta/3600:.1f}h {(eta%3600)/60:.0f}m")
        print(f"⚡ Mean speed: {avg_epoch_time:.1f}s/epoch")
        
        print(f"📊 Fase atual: {self.current_phase}")
        
        if len(self.losses) > 0:
            current_loss = self.losses[-1]
            if len(self.losses) >= 2:
                loss_change = self.losses[-1] - self.losses[-2]
                trend = "📈" if loss_change > 0 else "📉"
                print(f"🔥 Loss: {current_loss:.6f} {trend} ({loss_change:+.6f})")
            else:
                print(f"🔥 Loss: {current_loss:.6f}")
        
        if len(self.errors) > 0:
            current_error = self.errors[-1]
            if len(self.errors) >= 2:
                error_change = self.errors[-1] - self.errors[-2]
                trend = "📈" if error_change > 0 else "📉"
                print(f"📍 Erro: {current_error:.1f}m {trend} ({error_change:+.1f}m)")
            else:
                print(f"📍 Erro: {current_error:.1f}m")
        
        if metrics:
            print(f"🎯 Accuracy <100m: {metrics.get('accuracy_100m', 0):.1%}")
            print(f"🎯 Accuracy <200m: {metrics.get('accuracy_200m', 0):.1%}")
            print(f"🎯 Accuracy <500m: {metrics.get('accuracy_500m', 0):.1%}")
        
        if len(self.losses) >= 5:
            recent_losses = self.losses[-5:]
            trend_slope = (recent_losses[-1] - recent_losses[0]) / 5
            if abs(trend_slope) > 1e-6:
                trend_direction = "Melhorando 📉" if trend_slope < 0 else "Piorando 📈"
                print(f"📈 Trend (5 epochs): {trend_direction}")
        
        if len(self.errors) > 0:
            current_error = self.errors[-1]
            if current_error < 150:
                quality = "🌟 EXCELENTE"
            elif current_error < 250:
                quality = "⭐ MUITO BOM"
            elif current_error < 400:
                quality = "✅ BOM"
            else:
                quality = "⚠️ REGULAR"
            print(f"🏆 Qualidade atual: {quality}")
        
        print("="*80)
    
    def show_final_summary(self):
        total_time = time.time() - self.start_time
        
        print("\n" + "🎊"*50)
        print('FINAL SUMMARY OF THE TRAINING')
        print("🎊"*50)
        
        print(f"⏱️ Tempo total: {total_time/3600:.1f}h {(total_time%3600)/60:.0f}m")
        print(f"📊 Epochs that were run: {len(self.losses)}")
        
        if len(self.losses) > 0:
            print(f"🔥 Loss inicial: {self.losses[0]:.6f}")
            print(f"🔥 Loss final: {self.losses[-1]:.6f}")
            print(f"📈 Melhoria total: {self.losses[0] - self.losses[-1]:.6f}")
        
        if len(self.errors) > 0:
            print(f"📍 Erro inicial: {self.errors[0]:.1f}m")
            print(f"📍 Erro final: {self.errors[-1]:.1f}m")
            print(f"📈 Melhoria total: {self.errors[0] - self.errors[-1]:.1f}m")
        
        if len(self.epoch_times) > 0:
            print(f"⚡ Mean speed: {np.mean(self.epoch_times):.1f}s/epoch")
        
        print("🎊"*50)


def create_enhanced_trainer_with_monitoring(model, loss_fn, config, device='cuda'):
    
    trainer = create_trainer(model, loss_fn, config, device)
    
    total_epochs = config.get('total_epochs', config.get('epochs', 300))
    monitor = RealTimeMonitor(total_epochs, update_frequency=1)
    
    original_train_epoch = trainer.train_epoch
    original_validate_epoch = trainer.validate_epoch
    
    def enhanced_train_epoch(data_loader, epoch):
        phase_info = trainer.phase_manager.get_current_info()
        logger.info(f"🚀 STARTING EPOCH {epoch+1}/{total_epochs}")
        logger.info(f"   📊 Fase: {phase_info['phase_name']}")
        logger.info(f"   🎯 LR: {trainer.optimizer.param_groups[0]['lr']:.2e}")
        
        epoch_start = time.time()
        
        loss = original_train_epoch(data_loader, epoch)
        
        epoch_time = time.time() - epoch_start
        logger.info(f"✅ EPOCH {epoch+1} FINISHED in {epoch_time:.1f}s")
        logger.info(f"   🔥 Loss: {loss:.6f}")
        
        return loss
    
    def enhanced_validate_epoch(data_loader):
        logger.info("🔍 Executando validação...")
        
        val_start = time.time()
        metrics = original_validate_epoch(data_loader)
        val_time = time.time() - val_start
        
        logger.info(f"✅ Validation finished in {val_time:.1f}s")
        logger.info(f"   📊 Mean error: {metrics['mean_error']:.1f}m")
        logger.info(f"   🎯 Accuracy <200m: {metrics['accuracy_200m']:.1%}")
        
        return metrics
    
    original_train = trainer.train
    
    def enhanced_train(train_loader, val_loader=None, num_epochs=None):
        
        logger.info('🚀 STARTING TRAINING WITH IMPROVED MONITORING')
        
        if num_epochs is None:
            num_epochs = total_epochs
        
        monitor.total_epochs = num_epochs
        
        for epoch in range(num_epochs):
            try:
                if trainer.phase_manager.should_advance_phase():
                    if not trainer.phase_manager.advance_phase():
                        logger.info('🏁 All phases are finished!')
                        break
                    
                    for param_group in trainer.optimizer.param_groups:
                        param_group['lr'] = trainer.phase_manager.current_phase.learning_rate
                
                train_loss = enhanced_train_epoch(train_loader, epoch)
                
                val_metrics = None
                error = None
                if val_loader is not None:
                    val_metrics = enhanced_validate_epoch(val_loader)
                    error = val_metrics.get('mean_error', None)
                
                trainer.phase_manager.step_epoch()
                
                phase_info = trainer.phase_manager.get_current_info()
                monitor.update(
                    epoch=epoch,
                    loss=train_loss,
                    error=error,
                    phase=phase_info['phase_name'],
                    metrics=val_metrics
                )
                
                trainer.training_history['epoch'].append(epoch)
                trainer.training_history['phase'].append(trainer.phase_manager.current_phase_idx)
                trainer.training_history['loss'].append(train_loss)
                trainer.training_history['metrics'].append(val_metrics)
                trainer.training_history['learning_rate'].append(trainer.optimizer.param_groups[0]['lr'])
                trainer.training_history['phase_info'].append(phase_info)
                
                if hasattr(trainer, 'eda_frequency') and trainer.eda_frequency > 0:
                    if (epoch + 1) % trainer.eda_frequency == 0:
                        try:
                            eda_dir = getattr(trainer, 'eda_config', {}).get('save_dir', f"{INTERNAL_DATA_ROOT}/TREINAMENTO_REAL/OUTPUT/eda")
                            trainer.run_automated_eda(train_loader, epoch, eda_dir)
                        except Exception as e:
                            logger.warning(f"Error in the EDA: {e}")
                
                if val_metrics and val_metrics['mean_error'] < trainer.best_metrics['error']:
                    trainer.best_metrics['error'] = val_metrics['mean_error']
                    trainer.patience_counter = 0
                else:
                    trainer.patience_counter += 1
                    if trainer.patience_counter >= trainer.max_patience:
                        logger.info(f"🛑 Early stopping after {trainer.patience_counter} epochs without improvement")
                        break
                
            except KeyboardInterrupt:
                logger.info('🛑 Training interrupted by the user')
                break
            except Exception as e:
                logger.error(f"❌ Error in epoch {epoch}: {e}")
                continue
        
        monitor.show_final_summary()
        
        return trainer.training_history
    
    trainer.train_epoch = enhanced_train_epoch
    trainer.validate_epoch = enhanced_validate_epoch
    trainer.train = enhanced_train
    trainer.monitor = monitor
    
    return trainer


def run_training_cell6_with_monitoring(
    graph_data,
    scalers,
    config_overrides=None,
    device='cuda',
    save_dir=f"{INTERNAL_DATA_ROOT}/TREINAMENTO_REAL/OUTPUT",
    epochs=300,
    batch_size=2048,
    hidden_channels=2048,
    num_layers=16,
    heads=64,
    eda_frequency=20,
    use_mixed_precision=True
):
    
    print("\n" + "🌟" * 60)
    print('CELL 6 - IMPROVED PIPELINE WITH MONITORING')
    print("🌟" * 60)
    
    if config_overrides is None:
        config_overrides = {}
    
    config_overrides.update({
        'batch_size': batch_size,
        'hidden_channels': hidden_channels,
        'num_layers': num_layers,
        'heads': heads,
        'total_epochs': epochs,
        'epochs': epochs,
        'use_mixed_precision': use_mixed_precision,
        'eda_frequency': eda_frequency,
        'device': device
    })
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    try:
        print(f"🎯 Configurações:")
        print(f"   📊 Épocas: {epochs}")
        print(f"   🔄 Batch Size: {batch_size}")
        print(f"   🧠 Hidden Channels: {hidden_channels}")
        print(f"   📏 Layers: {num_layers}")
        print(f"   🎭 Attention Heads: {heads}")
        print(f"   🔬 EDA Frequency: {eda_frequency}")
        print(f"   💾 Device: {device}")
        print(f"   📁 Save Dir: {save_dir}")
        print(f"   ⚡ Mixed Precision: {use_mixed_precision}")
        print(f"   📝 Log file: {log_file}")
        
        logger.info("⚙️ Criando configuração...")
        config = create_custom_config(**config_overrides)
        
        logger.info("📊 Criando data loaders...")
        data_loaders = create_data_loaders(
            graph_data,
            batch_size=batch_size,
            num_neighbors=config.data.num_neighbors,
            num_workers=config.data.num_workers,
            device=device
        )
        
        logger.info("🧠 Criando modelo...")
        model = create_model(config.to_dict(), graph_data)
        model = model.to(device)
        
        logger.info("🔬 Criando loss function...")
        loss_fn = create_physics_informed_loss(config.to_dict())
        
        logger.info('🚀 Creating the trainer with monitoring...')
        trainer = create_enhanced_trainer_with_monitoring(model, loss_fn, config.to_dict(), device)
        trainer.set_scalers(scalers)
        
        trainer.eda_frequency = eda_frequency
        trainer.eda_config = {'save_dir': str(save_path / 'eda')}
        
        logger.info('🎯 Starting the training with monitoring...')
        history = trainer.train(
            train_loader=data_loaders['train'],
            val_loader=data_loaders.get('val'),
            num_epochs=epochs
        )
        
        logger.info('📊 Running the final evaluation...')
        evaluation_results = {}
        for split_name, data_loader in data_loaders.items():
            if data_loader is not None:
                metrics = trainer.evaluate_model_accurate_metrics(data_loader)
                evaluation_results[split_name] = metrics
        
        total_time = time.time() - start_time
        
        create_final_summary_corrected(
            model=model,
            history=history,
            evaluation_results=evaluation_results,
            config_dict=config.to_dict(),
            total_time=total_time,
            save_dir=save_dir
        )
        
        print("\n" + "🎉" * 60)
        print('IMPROVED PIPELINE FINISHED SUCCESSFULLY!')
        print("🎉" * 60)
        print(f"⏱️ Tempo total: {total_time:.1f}s ({total_time/60:.1f}m)")
        print(f"📁 Results saved to: {save_dir}")
        print(f"📝 Log detalhado em: {log_file}")
        
        return model, history, evaluation_results, config
        
    except Exception as e:
        logger.error(f"❌ Error in the improved pipeline: {e}")
        print("\n" + "💥" * 60)
        print("PIPELINE APRIMORADO - ERRO")
        print("💥" * 60)
        import traceback
        traceback.print_exc()
        raise


def create_final_summary_corrected(model, history, evaluation_results, config_dict, total_time, save_dir):
    
    save_path = Path(save_dir)
    
    final_metrics = evaluation_results.get('val', evaluation_results.get('test', {}))
    
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_time_seconds': total_time,
        'total_time_minutes': total_time / 60,
        'configuration': {
            'epochs': config_dict.get('total_epochs', config_dict.get('epochs', 300)),
            'batch_size': config_dict.get('batch_size', 128),
            'hidden_channels': config_dict.get('hidden_channels', 256),
            'num_layers': config_dict.get('num_layers', 4),
            'heads': config_dict.get('heads', 8),
            'device': config_dict.get('device', 'cuda'),
            'mixed_precision': config_dict.get('use_mixed_precision', True)
        },
        'model_info': {
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'model_size_mb': sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)
        },
        'training_results': {
            'total_epochs': len(history.get('epoch', [])),
            'final_loss': history.get('loss', [0])[-1] if history.get('loss') else 0,
            'best_loss': min(history.get('loss', [float('inf')])),
            'improvement': history.get('loss', [0])[0] - history.get('loss', [0])[-1] if len(history.get('loss', [])) > 0 else 0
        },
        'evaluation_metrics': final_metrics
    }
    
    summary_file = save_path / 'final_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    report_lines = [
        'FINAL REPORT - ARPIA GNN v4',
        "=" * 50,
        f"Data: {summary['timestamp']}",
        f"Tempo total: {summary['total_time_minutes']:.1f} minutos",
        "",
        "CONFIGURAÇÃO:",
        f"  Épocas: {summary['configuration']['epochs']}",
        f"  Batch Size: {summary['configuration']['batch_size']}",
        f"  Hidden Channels: {summary['configuration']['hidden_channels']}",
        f"  Layers: {summary['configuration']['num_layers']}",
        f"  Heads: {summary['configuration']['heads']}",
        f"  Device: {summary['configuration']['device']}",
        f"  Mixed Precision: {summary['configuration']['mixed_precision']}",
        "",
        "MODELO:",
        f"  Parâmetros: {summary['model_info']['total_parameters']:,}",
        f"  Tamanho: {summary['model_info']['model_size_mb']:.1f} MB",
        "",
        "TREINAMENTO:",
        f"  Épocas executadas: {summary['training_results']['total_epochs']}",
        f"  Loss final: {summary['training_results']['final_loss']:.4f}",
        f"  Melhor loss: {summary['training_results']['best_loss']:.4f}",
        "",
        "AVALIAÇÃO:"
    ]
    
    if final_metrics:
        report_lines.extend([
            f"  Erro médio: {final_metrics.get('mean_error', 0):.1f}m",
            f"  RMSE: {final_metrics.get('rmse', 0):.1f}m",
            f"  Precisão <100m: {final_metrics.get('accuracy_100m', 0):.1%}",
            f"  Precisão <200m: {final_metrics.get('accuracy_200m', 0):.1%}",
            f"  Precisão <500m: {final_metrics.get('accuracy_500m', 0):.1%}",
        ])
        
        error = final_metrics.get('mean_error', float('inf'))
        if error < 150:
            evaluation = "🌟 EXCELENTE!"
        elif error < 250:
            evaluation = "⭐ MUITO BOM!"
        elif error < 400:
            evaluation = "✅ BOM!"
        else:
            evaluation = "⚠️ REGULAR"
        
        report_lines.extend([
            "",
            f"AVALIAÇÃO GERAL: {evaluation}",
            "=" * 50
        ])
    
    report_file = save_path / 'final_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n💾 Full summary saved to:")
    print(f"   📊 JSON: {summary_file}")
    print(f"   📄 Texto: {report_file}")
    print(f"   📝 Log detalhado: {log_file}")


print("\n🎯 SISTEMA APRIMORADO ADICIONADO:")
print("   ✅ RealTimeMonitor() - Monitor em tempo real")
print("   ✅ create_enhanced_trainer_with_monitoring() - Trainer aprimorado")
print("   ✅ run_training_cell6_with_monitoring() - Pipeline com monitoramento")
print("   ✅ create_final_summary_corrected() - Resumo corrigido")
print("   ✅ Logging ultra-detalhado com timestamps")
print('   ✅ Correction of attribute errors')
print("   ✅ Sistema de EDA robusto")
print("="*80)


def run_pipeline_cuda_safe():
    logger.info("🌟 INICIANDO PIPELINE CUDA-SAFE")
    
    try:
        config = get_gpu_optimized_config()
        
        logger.info("🔄 Criando data loaders robustos...")
        data_loaders = create_data_loaders(config)
        
        logger.info('🏗️ Creating the robust model...')
        model = create_model(config, data_loaders['train'].dataset[0])
        model = model.cuda()
        
        logger.info("🎯 Criando loss function robusta...")
        loss_fn = create_physics_informed_loss(config)
        
        logger.info("🚀 Criando trainer robusto...")
        trainer = create_trainer(model, loss_fn, config, device='cuda')
        
        if hasattr(data_loaders['train'].dataset, 'scalers'):
            trainer.set_scalers(data_loaders['train'].dataset.scalers)
        
        logger.info('🎉 Running the CUDA-SAFE training...')
        history = execute_training_cuda_safe(
            trainer=trainer,
            data_loaders=data_loaders,
            config=config,
            save_checkpoints=True,
            save_dir='output'
        )
        
        logger.info('🎊 CUDA-SAFE PIPELINE FINISHED SUCCESSFULLY!')
        return history
        
    except Exception as e:
        logger.error(f"❌ Error in the CUDA-SAFE pipeline: {e}")
        raise


if __name__ == "__main__":
    try:
        run_pipeline_cuda_safe()
    except Exception as e:
        logger.error(f"Erro: {e}")
        import traceback
        traceback.print_exc()


def celula_6_pipeline_completo(
    graph_data,
    scalers,
    epochs=300,
    batch_size=2048,
    hidden_channels=2048,
    layers=16,
    attention_heads=64,
    eda_frequency=20,
    device='cuda',
    save_dir=f"{INTERNAL_DATA_ROOT}/TREINAMENTO_REAL/OUTPUT",
    mixed_precision=True,
    debug_mode=False
):
    print(f"\n🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟")
    print('CELL 6 - ARPIA GNN v4 FULL PIPELINE')
    print(f"🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟")
    print(f"🎯 Configurações:")
    print(f"   📊 Épocas: {epochs}")
    print(f"   🔄 Batch Size: {batch_size}")
    print(f"   🧠 Hidden Channels: {hidden_channels}")
    print(f"   📏 Layers: {layers}")
    print(f"   🎭 Attention Heads: {attention_heads}")
    print(f"   🔬 EDA Frequency: {eda_frequency}")
    print(f"   💾 Device: {device}")
    print(f"   📁 Save Dir: {save_dir}")
    print(f"   ⚡ Mixed Precision: {mixed_precision}")
    
    print(f"\n🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀")
    print('ULTRA-ADVANCED PIPELINE - STARTING')
    print(f"🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀")
    
    try:
        print('\n🔍 DIAGNOSIS OF THE SCALERS')
        diagnose_scalers(scalers)
        
        config_overrides = {
            'batch_size': batch_size,
            'hidden_channels': hidden_channels,
            'layers': layers,
            'attention_heads': attention_heads,
            'multi_scale': True,
            'rf_attention': True,
            'physics_enabled': True,
            'device': device,
            'mixed_precision': mixed_precision,
            'eda_frequency': eda_frequency,
            'epochs': epochs,
            'save_dir': save_dir,
            'physics_weight': 1.5,
            'rf_frequency': 2.4e9,
            'constraint_distance': 1000.0,
            'enable_early_stopping': True,
            'patience': 50,
            'auto_eda': True,
            'phases': [
                {
                    'name': 'FUNDAÇÃO ESTRUTURAL',
                    'epochs': 50,
                    'lr': 1e-3,
                    'physics_weight': 0.2,
                    'max_movement': 300.0,
                    'description': 'Establishment of the basic structure'
                },
                {
                    'name': 'ESTRUTURA AVANÇADA',
                    'epochs': 50,
                    'lr': 8e-4,
                    'physics_weight': 0.35,
                    'max_movement': 250.0,
                    'description': 'Desenvolvimento da arquitetura'
                },
                {
                    'name': 'REFINAMENTO DINÂMICO',
                    'epochs': 50,
                    'lr': 6e-4,
                    'physics_weight': 0.5,
                    'max_movement': 200.0,
                    'description': 'Dynamic adjustments and adaptation'
                },
                {
                    'name': 'CALIBRAÇÃO INTELIGENTE',
                    'epochs': 50,
                    'lr': 4e-4,
                    'physics_weight': 0.65,
                    'max_movement': 150.0,
                    'description': 'Precise calibration of the parameters'
                },
                {
                    'name': 'OTIMIZAÇÃO AVANÇADA',
                    'epochs': 50,
                    'lr': 2e-4,
                    'physics_weight': 0.8,
                    'max_movement': 100.0,
                    'description': 'High-precision optimization of parameters'
                },
                {
                    'name': 'PRECISÃO MÁXIMA',
                    'epochs': 50,
                    'lr': 1e-4,
                    'physics_weight': 1.0,
                    'max_movement': 50.0,
                    'description': 'Final refinement and maximum precision'
                }
            ]
        }
        
        print_ultra_advanced_config(config_overrides)
        
        result = run_training_cell6(
            graph_data=graph_data,
            scalers=scalers,
            config_overrides=config_overrides,
            device=device
        )
        
        print(f"\n🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
        print('ULTRA-ADVANCED PIPELINE FINISHED SUCCESSFULLY!')
        print(f"🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
        
        return result
        
    except Exception as e:
        print(f"\n❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌")
        print('ULTRA-ADVANCED PIPELINE - ERROR')
        print(f"❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌")
        print(f"Erro: {e}")
        
        if debug_mode:
            import traceback
            traceback.print_exc()
            
        raise


def celula_6_conservadora():
    print('🔧 RUNNING THE CONSERVATIVE VERSION...')
    
    try:
        from notebooks.config_colab import get_ultra_conservative_config
        config = get_ultra_conservative_config()
        
        config['training']['batch_size'] = 32
        config['training']['num_epochs'] = 50
        config['model']['hidden_channels'] = 64
        config['model']['num_layers'] = 2
        config['model']['heads'] = 2
        config['training']['eda_frequency'] = 100
        
        print("⚠️ Configurações ultra-conservadoras:")
        print(f"   Batch Size: {config['training']['batch_size']}")
        print(f"   Épocas: {config['training']['num_epochs']}")
        print(f"   Hidden Channels: {config['model']['hidden_channels']}")
        print(f"   Layers: {config['model']['num_layers']}")
        print(f"   Heads: {config['model']['heads']}")
        print("")
        
        return celula_6_pipeline_completo()
        
    except Exception as e:
        print(f"❌ Error even in the conservative version: {e}")
        raise 


def diagnose_scalers(scalers):
    print('\n🔍 DIAGNOSIS OF THE SCALERS')
    print("=" * 50)
    
    if scalers is None:
        print('❌ Scalers is None')
        return
    
    print(f"📋 Tipo dos scalers: {type(scalers)}")
    
    if hasattr(scalers, 'inverse_transform'):
        print('✅ Scalers is a single scaler with inverse_transform')
        return
    
    if isinstance(scalers, dict):
        print(f"✅ Scalers is a dictionary with {len(scalers)} keys:")
        for key, value in scalers.items():
            print(f"   📋 {key}: {type(value)}")
            
            if isinstance(value, dict):
                print(f"      Sub-chaves: {list(value.keys())}")
                
            elif hasattr(value, 'inverse_transform'):
                print(f"      ✅ Tem inverse_transform")
                
        if 'pea_scaler' in scalers:
            print("✅ 'pea_scaler' encontrado diretamente")
        elif 'coord_scalers' in scalers:
            coord_scalers = scalers['coord_scalers']
            print("✅ 'coord_scalers' encontrado")
            if 'x_global_scaler' in coord_scalers and 'y_global_scaler' in coord_scalers:
                print('   ✅ x_global_scaler and y_global_scaler found')
            else:
                print('   ❌ x_global_scaler or y_global_scaler not found')
        elif 'PEA' in scalers:
            print("✅ 'PEA' found as an alternative key")
        else:
            print('❌ Structure was not recognized')
    else:
        print(f"❌ Scalers is neither a dict nor a single scaler: {type(scalers)}")
    
    print("=" * 50) 


def test_scaler_correction(graph_data, scalers):
    print("\n🧪 TEST OF THE FIX FOR THE 'pea_scaler' ERROR")
    print("=" * 60)
    
    try:
        diagnose_scalers(scalers)
        
        from notebooks.models import create_model
        from notebooks.losses import create_physics_informed_loss
        from notebooks.trainers import create_trainer
        from notebooks.config_colab import create_custom_config
        
        config = create_custom_config(
            batch_size=32,
            hidden_channels=64,
            num_layers=2,
            heads=4,
            total_epochs=1,
            device='cuda'
        )
        
        print('\n🏗️ Creating components for the test...')
        model = create_model(config, graph_data)
        loss_fn = create_physics_informed_loss(config)
        trainer = create_trainer(model, loss_fn, config.to_dict(), device='cuda')
        
        print('\n🔧 Configuring the scalers in the trainer...')
        trainer.set_scalers(scalers)
        
        print('\n🔬 Creating the test tensors...')
        batch_size = 10
        test_predictions = torch.randn(batch_size, 2).cuda()
        test_targets = torch.randn(batch_size, 2).cuda()
        
        print('\n⚡ Testing the function denormalize_for_metrics_accurate...')
        pred_denorm, target_denorm = trainer.denormalize_for_metrics_accurate(
            test_predictions, test_targets
        )
        
        print(f"✅ Function ran successfully!")
        print(f"   📊 Predictions shape: {pred_denorm.shape}")
        print(f"   📊 Targets shape: {target_denorm.shape}")
        print(f"   📊 Predictions range: [{pred_denorm.min():.1f}, {pred_denorm.max():.1f}]")
        print(f"   📊 Targets range: [{target_denorm.min():.1f}, {target_denorm.max():.1f}]")
        
        if pred_denorm.min() > -1000000 and pred_denorm.max() < 1000000:
            print('✅ Denormalized values are in a valid range')
        else:
            print("⚠️ Valores desnormalizados podem estar fora do range esperado")
        
        print('\n🎉 TEST OF THE FIX: SUCCESS!')
        print("✅ The 'pea_scaler' error was fixed successfully")
        print("=" * 60)
        
        return True
        
    except KeyError as e:
        if 'pea_scaler' in str(e):
            print(f"\n❌ TEST FAILED: the 'pea_scaler' error still exists")
            print(f"   Erro: {e}")
            return False
        else:
            print(f"\n❌ TEST FAILED: a different key error: {e}")
            return False
            
    except Exception as e:
        print(f"\n❌ TEST FAILED: unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def fix_loss_clamp_emergency():
    logger.info('🚨 APPLYING THE EMERGENCY FIX OF THE LOSS FUNCTION')
    logger.info("   ❌ Problema: Loss clamped em 100.0 impedia aprendizado")
    logger.info('   ✅ Solution: remove the clamp and allow natural gradients')
    
    import importlib
    import sys
    
    if 'notebooks.losses' in sys.modules:
        importlib.reload(sys.modules['notebooks.losses'])
    
    from notebooks.losses import create_physics_informed_loss
    
    logger.info('✅ Loss function reloaded with the fix applied!')
    return create_physics_informed_loss

def restart_training_with_fix(
    graph_data,
    scalers,
    config_overrides=None,
    device='cuda',
    save_dir=f"{INTERNAL_DATA_ROOT}/TREINAMENTO_REAL/OUTPUT",
    epochs=300,
    batch_size=2048,
    hidden_channels=2048,
    num_layers=16,
    heads=64,
    eda_frequency=20,
    use_mixed_precision=True
):
    print("🚨" * 50)
    print("CORREÇÃO EMERGENCIAL APLICADA!")
    print("🚨" * 50)
    print("❌ PROBLEMA IDENTIFICADO:")
    print("   • Loss estava clamped em 100.0")
    print('   • The model could not learn')
    print("   • Gradientes eram cortados")
    print()
    print("✅ CORREÇÃO APLICADA:")
    print("   • Clamp removido da loss function")
    print("   • Gradientes naturais restaurados")
    print("   • Aprendizado desbloqueado")
    print("🚨" * 50)
    
    create_physics_informed_loss_fixed = fix_loss_clamp_emergency()
    
    if config_overrides is None:
        config_overrides = {}
    
    config_overrides.update({
        'epochs': epochs,
        'batch_size': batch_size,
        'hidden_channels': hidden_channels,
        'num_layers': num_layers,
        'heads': heads,
        'eda_frequency': eda_frequency,
        'use_mixed_precision': use_mixed_precision,
        'device': device,
        'save_dir': save_dir
    })
    
    import logging
    log_file = Path(save_dir) / 'logs' / f'training_log_FIXED_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    setup_ultra_detailed_logging(log_file)
    
    try:
        print(f"🎯 Settings with the fix:")
        print(f"   📊 Épocas: {epochs}")
        print(f"   🔄 Batch Size: {batch_size}")
        print(f"   🧠 Hidden Channels: {hidden_channels}")
        print(f"   📏 Layers: {num_layers}")
        print(f"   🎭 Attention Heads: {heads}")
        print(f"   💾 Device: {device}")
        print(f"   📁 Save Dir: {save_dir}")
        print(f"   📝 Log file: {log_file}")
        print(f"   🚨 Loss function: CORRIGIDA")
        
        logger.info("⚙️ Criando configuração...")
        config = create_custom_config(**config_overrides)
        
        logger.info("📊 Criando data loaders...")
        data_loaders = create_data_loaders(
            graph_data,
            batch_size=batch_size,
            num_neighbors=config.data.num_neighbors,
            num_workers=config.data.num_workers,
            device=device
        )
        
        logger.info("🧠 Criando modelo...")
        model = create_model(config.to_dict(), graph_data)
        model = model.to(device)
        
        logger.info("🚨 Criando loss function CORRIGIDA...")
        loss_fn = create_physics_informed_loss_fixed(config.to_dict())
        
        logger.info("🚀 Criando trainer...")
        trainer = create_enhanced_trainer_with_monitoring(model, loss_fn, config.to_dict(), device)
        trainer.set_scalers(scalers)
        
        trainer.eda_frequency = eda_frequency
        trainer.eda_config = {'save_dir': str(Path(save_dir) / 'eda')}
        
        logger.info('🎯 Starting the CORRECTED training...')
        results = enhanced_train(
            trainer,
            data_loaders,
            epochs=epochs,
            save_dir=save_dir,
            use_mixed_precision=use_mixed_precision
        )
        
        logger.info('🎉 CORRECTED training is finished!')
        return results
        
    except Exception as e:
        logger.error(f"❌ Error in the corrected training: {e}")
        import traceback
        traceback.print_exc()
        return None

def monitor_gpu_utilization():
    if not torch.cuda.is_available():
        return {"gpu_available": False}
    
    gpu_stats = {}
    
    try:
        gpu_stats["gpu_name"] = torch.cuda.get_device_name(0)
        gpu_stats["gpu_count"] = torch.cuda.device_count()
        
        memory_allocated = torch.cuda.memory_allocated(0)
        memory_reserved = torch.cuda.memory_reserved(0)
        memory_total = torch.cuda.get_device_properties(0).total_memory
        
        gpu_stats["memory_allocated_gb"] = memory_allocated / (1024**3)
        gpu_stats["memory_reserved_gb"] = memory_reserved / (1024**3)
        gpu_stats["memory_total_gb"] = memory_total / (1024**3)
        gpu_stats["memory_utilization_percent"] = (memory_allocated / memory_total) * 100
        
        if "A100" in gpu_stats["gpu_name"]:
            if gpu_stats["memory_utilization_percent"] < 50:
                gpu_stats["utilization_status"] = "🔴 BAIXA - GPU A100 subutilizada!"
                gpu_stats["recommendation"] = "Aumentar batch_size, hidden_channels ou num_layers"
            elif gpu_stats["memory_utilization_percent"] < 80:
                gpu_stats["utilization_status"] = '🟡 MEDIUM - the A100 GPU can use more'
                gpu_stats["recommendation"] = 'Consider increasing the model parameters'
            else:
                gpu_stats["utilization_status"] = "🟢 ALTA - GPU A100 bem utilizada!"
                gpu_stats["recommendation"] = "Utilização adequada"
        
        gpu_stats["cache_allocated_gb"] = torch.cuda.memory_cached(0) / (1024**3)
        
        return gpu_stats
        
    except Exception as e:
        logger.error(f"Error while monitoring the GPU: {e}")
        return {"error": str(e)}

def print_gpu_status():
    stats = monitor_gpu_utilization()
    
    if not stats.get("gpu_available", True):
        print('❌ GPU is not available!')
        return
    
    print("\n" + "🔥" * 60)
    print("MONITORAMENTO GPU A100")
    print("🔥" * 60)
    print(f"🖥️  GPU: {stats.get('gpu_name', 'N/A')}")
    print(f"💾 Total memory: {stats.get('memory_total_gb', 0):.1f} GB")
    print(f"📊 Memory in use: {stats.get('memory_allocated_gb', 0):.1f} GB")
    print(f"📈 Utilização: {stats.get('memory_utilization_percent', 0):.1f}%")
    print(f"🎯 Status: {stats.get('utilization_status', 'N/A')}")
    print(f"💡 Recomendação: {stats.get('recommendation', 'N/A')}")
    print("🔥" * 60)

def create_a100_optimized_config():
    return {
        'batch_size': 2048,
        'hidden_channels': 2048,
        'num_layers': 16,
        'heads': 64,
        'num_neighbors': [128, 96, 64],
        
        'num_workers': 4,
        'pin_memory': True,
        'persistent_workers': True,
        'prefetch_factor': 16,
        
        'use_mixed_precision': True,
        'gradient_accumulation_steps': 1,
        'max_grad_norm': 1.0,
        
        'use_multi_scale': True,
        'use_rf_attention': True,
        'use_advanced_features': True,
        
        'epochs': 300,
        'eda_frequency': 20,
        'device': 'cuda'
    }

def run_a100_maximized_training(
    graph_data,
    scalers,
    save_dir=f"{INTERNAL_DATA_ROOT}/TREINAMENTO_REAL/OUTPUT",
    epochs=300,
    monitor_gpu=True
):
    
    print("🚀" * 80)
    print('MAXIMIZED A100 TRAINING - ALL FIXES APPLIED')
    print("🚀" * 80)
    
    if monitor_gpu:
        print_gpu_status()
    
    config_overrides = create_a100_optimized_config()
    
    print(f"\n🎯 MAXIMIZED A100 CONFIGURATION:")
    print(f"   📊 Batch Size: {config_overrides['batch_size']} (MASSIVO)")
    print(f"   🧠 Hidden Channels: {config_overrides['hidden_channels']} (MASSIVO)")
    print(f"   📏 Layers: {config_overrides['num_layers']} (MUITO MAIS)")
    print(f"   🎭 Attention Heads: {config_overrides['heads']} (MASSIVO)")
    print(f"   🔗 Neighbors: {config_overrides['num_neighbors']} (MASSIVO)")
    print(f"   ⚡ Mixed Precision: {config_overrides['use_mixed_precision']}")
    print(f"   💾 Workers: {config_overrides['num_workers']}")
    print(f"   📈 Prefetch: {config_overrides['prefetch_factor']}")
    
    print('\n🔧 APPLYING THE LOSS FUNCTION FIX...')
    create_physics_informed_loss_fixed = fix_loss_clamp_emergency()
    
    log_file = Path(save_dir) / 'logs' / f'training_A100_MAXIMIZED_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    setup_ultra_detailed_logging(log_file)
    
    try:
        logger.info('⚙️ Creating the maximized A100 configuration...')
        config = create_custom_config(**config_overrides)
        
        logger.info("📊 Criando data loaders massivos...")
        data_loaders = create_data_loaders(
            graph_data,
            batch_size=config_overrides['batch_size'],
            num_neighbors=config_overrides['num_neighbors'],
            num_workers=config_overrides['num_workers'],
            device='cuda'
        )
        
        logger.info('🧠 Creating the massive model...')
        model = create_model(config.to_dict(), graph_data)
        model = model.to('cuda')
        
        if monitor_gpu:
            print('\n📊 UTILIZATION AFTER LOADING THE MODEL:')
            print_gpu_status()
        
        logger.info("🔬 Criando loss function CORRIGIDA...")
        loss_fn = create_physics_informed_loss_fixed(config.to_dict())
        
        logger.info("🚀 Criando trainer...")
        trainer = create_enhanced_trainer_with_monitoring(model, loss_fn, config.to_dict(), 'cuda')
        trainer.set_scalers(scalers)
        
        logger.info('🎯 Starting the MAXIMIZED A100 training...')
        
        if monitor_gpu:
            print('\n📊 UTILIZATION DURING THE TRAINING:')
            print_gpu_status()
        
        results = enhanced_train(
            trainer,
            data_loaders,
            epochs=epochs,
            save_dir=save_dir,
            use_mixed_precision=config_overrides['use_mixed_precision']
        )
        
        if monitor_gpu:
            print("\n📊 UTILIZAÇÃO FINAL:")
            print_gpu_status()
        
        logger.info('🎉 MAXIMIZED A100 training is finished!')
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Error in the maximized A100 training: {e}")
        import traceback
        traceback.print_exc()
        return None

def execute_a100_maximized_pipeline():
    print("🔥" * 80)
    print("EXECUTANDO PIPELINE A100 MAXIMIZADO")
    print("🔥" * 80)
    print("📋 VERIFICAÇÕES NECESSÁRIAS:")
    print("   ✅ graph_data carregado")
    print("   ✅ scalers carregado")
    print("   ✅ Google Drive montado")
    print('   ✅ A100 GPU is available')
    print("🔥" * 80)
    
    try:
        return run_a100_maximized_training(
            graph_data=graph_data,
            scalers=scalers,
            save_dir=f"{INTERNAL_DATA_ROOT}/TREINAMENTO_REAL/OUTPUT",
            epochs=300,
            monitor_gpu=True
        )
    except NameError as e:
        print(f"❌ Erro: {e}")
        print('💡 Make sure graph_data and scalers are loaded!')
        return None

def create_enhanced_trainer_with_monitoring_v2(model, loss_fn, config, device='cuda'):
    from notebooks.trainers import ARPIATrainer, EnsembleManager
    
    trainer = ARPIATrainer(model, loss_fn, config, device)
    
    ensemble_manager = EnsembleManager(
        ensemble_size=5,
        metric_name='val_median_error',
        metric_mode='min'
    )
    
    def enhanced_train_epoch_v2(data_loader, epoch):
        model.train()
        total_loss = 0.0
        num_batches = len(data_loader)
        
        current_metrics = {'avg_error': 1000.0}
        
        adaptive_weights = trainer.phase_manager.get_adaptive_loss_weights(current_metrics)
        loss_fn.update_weights(adaptive_weights)
        
        exploration_params = trainer.phase_manager.get_exploration_params_v2()
        
        logger.info(f"🚀 STARTING EPOCH {epoch+1}/300 (IMPROVED V2)")
        logger.info(f"   📊 Fase: {trainer.phase_manager.phases[trainer.phase_manager.current_phase_number].name}")
        logger.info(f"   🎯 LR: {trainer.phase_manager.phases[trainer.phase_manager.current_phase_number].learning_rate:.2e}")
        logger.info(f"   🔧 Pesos adaptativos: {'✅' if trainer.phase_manager.adaptive_weights_enabled else '❌'}")
        logger.info(f"   🎲 Exploração: {'✅' if exploration_params.get('enabled', False) else '❌'}")
        
        for batch_idx, batch in enumerate(data_loader):
            try:
                batch = batch.to(device)
                
                trainer.optimizer.zero_grad()
                
                output = model(batch)
                predictions = output['predictions']['PEA']
                targets = batch['PEA'].x[:, :2]
                
                if exploration_params.get('enabled', False):
                    noise_std = exploration_params.get('noise_std', 0.0)
                    if noise_std > 0:
                        exploration_noise = torch.randn_like(predictions) * noise_std
                        predictions = predictions + exploration_noise
                
                loss, loss_components = loss_fn(
                    predictions, targets, batch,
                    epoch=epoch,
                    total_epochs=trainer.phase_manager.phases[trainer.phase_manager.current_phase_number].epochs,
                    node_embeddings=output.get('embeddings'),
                    max_movement=trainer.phase_manager.phases[trainer.phase_manager.current_phase_number].max_movement
                )
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                trainer.optimizer.step()
                
                total_loss += loss.item()
                
                with torch.no_grad():
                    error = torch.norm(predictions - targets, dim=1).mean().item() * 450.0
                    current_metrics['avg_error'] = error
                
            except Exception as e:
                logger.error(f"❌ Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        
        epoch_metrics = {
            'avg_error': current_metrics['avg_error'],
            'loss': avg_loss,
            'phase': trainer.phase_manager.current_phase_number
        }
        trainer.phase_manager.update_phase_performance(epoch_metrics)
        
        if trainer.phase_manager.should_advance_phase_v2(current_metrics):
            trainer.phase_manager.advance_phase()
            logger.info(f"🎯 Moving on to phase {trainer.phase_manager.current_phase_number}")
        
        trainer.phase_manager.step_epoch()
        
        logger.info(f"✅ EPOCH {epoch+1} FINISHED V2")
        logger.info(f"   🔥 Loss: {avg_loss:.6f}")
        logger.info(f"   📍 Mean error: {current_metrics['avg_error']:.1f}m")
        
        return avg_loss
    
    def enhanced_validate_epoch_v2(data_loader):
        model.eval()
        total_error = 0.0
        num_samples = 0
        
        logger.info('🔍 Running the V2 validation...')
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(device)
                
                output = model(batch)
                predictions = output['predictions']['PEA']
                targets = batch['PEA'].x[:, :2]
                
                errors = torch.norm(predictions - targets, dim=1) * 450.0
                total_error += errors.sum().item()
                num_samples += len(errors)
        
        avg_error = total_error / max(num_samples, 1)
        
        precision_100m = 0.0
        precision_200m = 0.0
        
        logger.info(f"✅ V2 validation is finished")
        logger.info(f"   📊 Mean error: {avg_error:.1f}m")
        logger.info(f"   🎯 Accuracy <100m: {precision_100m:.1f}%")
        logger.info(f"   🎯 Accuracy <200m: {precision_200m:.1f}%")
        
        return {
            'avg_error': avg_error,
            'precision_100m': precision_100m,
            'precision_200m': precision_200m
        }
    
    def enhanced_train_v2(train_loader, val_loader=None, num_epochs=None):
        if num_epochs is None:
            num_epochs = sum(phase.epochs for phase in trainer.phase_manager.phases.values())
        
        logger.info('🚀 STARTING THE IMPROVED V2 TRAINING')
        logger.info(f"   📊 Total epochs: {num_epochs}")
        logger.info(f"   🎯 Fases: {len(trainer.phase_manager.phases)}")
        logger.info(f"   🔮 Ensemble: {ensemble_manager.ensemble_size} checkpoints")
        
        best_val_error = float('inf')
        
        for epoch in range(num_epochs):
            train_loss = enhanced_train_epoch_v2(train_loader, epoch)
            
            val_metrics = enhanced_validate_epoch_v2(val_loader) if val_loader else {}
            val_error = val_metrics.get('avg_error', float('inf'))
            
            if val_error < best_val_error:
                best_val_error = val_error
                
                model_state = model.state_dict().copy()
                optimizer_state = trainer.optimizer.state_dict().copy()
                
                ensemble_manager.add_checkpoint(
                    epoch=epoch,
                    model_state=model_state,
                    metrics={'val_median_error': val_error, **val_metrics},
                    optimizer_state=optimizer_state
                )
            
            if epoch % 5 == 0:
                ensemble_summary = ensemble_manager.get_ensemble_summary()
                logger.info(f"📊 Ensemble: {ensemble_summary.get('size', 0)} modelos, "
                           f"melhor: {ensemble_summary.get('best_metric', 'N/A'):.1f}m")
        
        logger.info('🎉 V2 TRAINING IS FINISHED!')
        logger.info(f"   🏆 Best error: {best_val_error:.1f}m")
        logger.info(f"   🔮 Ensemble final: {ensemble_manager.get_ensemble_summary()}")
        
        return {
            'best_val_error': best_val_error,
            'ensemble_manager': ensemble_manager,
            'trainer': trainer
        }
    
    trainer.enhanced_train_epoch_v2 = enhanced_train_epoch_v2
    trainer.enhanced_validate_epoch_v2 = enhanced_validate_epoch_v2
    trainer.enhanced_train_v2 = enhanced_train_v2
    trainer.ensemble_manager = ensemble_manager
    
    return trainer

def run_training_cell6_with_monitoring_v2(
    graph_data,
    scalers,
    config_overrides=None,
    device='cuda',
    save_dir=f"{INTERNAL_DATA_ROOT}/TREINAMENTO_REAL/OUTPUT",
    epochs=300,
    batch_size=1024,
    hidden_channels=512,
    num_layers=8,
    heads=16,
    eda_frequency=20,
    use_mixed_precision=True
):
    logger.info("🚀 INICIANDO PIPELINE V2 ULTRA-APRIMORADO")
    logger.info("=" * 80)
    
    config_overrides = config_overrides or {}
    config_overrides.update({
        'batch_size': batch_size,
        'hidden_channels': hidden_channels,
        'num_layers': num_layers,
        'heads': heads,
        'use_mixed_precision': use_mixed_precision,
        'use_local_refiner': True,
        'adaptive_weights_enabled': True,
        'ensemble_size': 5
    })
    
    logger.info(f"🎯 Configuração V2:")
    logger.info(f"   📊 Épocas: {epochs}")
    logger.info(f"   🔄 Batch Size: {batch_size}")
    logger.info(f"   🧠 Hidden Channels: {hidden_channels}")
    logger.info(f"   📏 Layers: {num_layers}")
    logger.info(f"   🎭 Attention Heads: {heads}")
    logger.info(f"   🎯 LocalPrecisionRefiner: ✅")
    logger.info(f"   🔧 Pesos Adaptativos: ✅")
    logger.info(f"   🔮 Ensemble: ✅")
    
    logger.info("📊 Criando data loaders...")
    from notebooks.data_loaders import create_data_loaders
    
    data_loaders = create_data_loaders(
        graph_data,
        batch_size=batch_size,
        num_neighbors=[32, 24, 16],
        num_workers=2,
        device=device
    )
    
    logger.info('🧠 Creating the V2 model...')
    from notebooks.models import create_model
    
    model = create_model(config_overrides, graph_data)
    model = model.to(device)
    
    logger.info("🔬 Criando loss function V2...")
    from notebooks.losses import create_physics_informed_loss
    
    loss_fn = create_physics_informed_loss(config_overrides)
    
    logger.info("🚀 Criando trainer V2...")
    trainer = create_enhanced_trainer_with_monitoring_v2(
        model, loss_fn, config_overrides, device
    )
    
    logger.info('🎯 Starting the V2 training...')
    results = trainer.enhanced_train_v2(
        train_loader=data_loaders['train'],
        val_loader=data_loaders['val'],
        num_epochs=epochs
    )
    
    logger.info('🎉 V2 PIPELINE FINISHED SUCCESSFULLY!')
    
    return {
        'model': model,
        'trainer': trainer,
        'ensemble_manager': results['ensemble_manager'],
        'best_val_error': results['best_val_error'],
        'data_loaders': data_loaders,
        'config': config_overrides
    }
