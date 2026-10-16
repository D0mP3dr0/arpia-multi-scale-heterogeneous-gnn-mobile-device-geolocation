#!/usr/bin/env python3
"""Ablation study of the ARPIA components."""

INTERNAL_DATA_ROOT = __import__("os").environ.get("ARPIA_ROOT", ".")
import sys
import os
import io
import torch
import torch.nn as nn
import numpy as np
import json
import time
import logging
import gc
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

global IN_COLAB
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if not IN_COLAB and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from config_colab import ColabConfig, create_default_config, create_custom_config
from models import ARPIAGNNv4, create_model
from losses import PhysicsInformedLoss, create_physics_informed_loss
from trainers import ARPIATrainer, create_trainer
from data_loaders import ARPIADataLoader, create_data_loaders, prepare_graph_data
from train_colab import create_pipeline_ultra_avancado, execute_training_ultra_avancado, run_training_cell6_with_monitoring

print("🔬 ARPIA V3 ABLATION STUDY - PIPELINE CORRETO")
print("="*60)
print('✅ Using the pipeline that ALREADY WORKS')
print('✅ Without any generic solutions')
print("✅ Sem invenções")
print("="*60)


class ARPIAAblationStudy:
    
    def __init__(self):
        logger.info("🔬 Inicializando ARPIA Ablation Study...")
        
        self._configure_memory_optimizations()
        
        if IN_COLAB:
            self.base_dir = Path(f"{INTERNAL_DATA_ROOT}/TREINAMENTO_REAL")
        else:
            self.base_dir = Path(__file__).parent
            
        self.output_dir = self.base_dir / "ablation_results_arpia_v3"
        self.output_dir.mkdir(exist_ok=True)
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info(f"🚀 Usando GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device('cpu')
            logger.info(f"🔧 Device configurado: {self.device}")
        
        self.base_config = create_default_config()
        
        self.ablation_components = self._define_ablation_components()
        
        self.results = {}
        
        logger.info("✅ Ablation Study inicializado")
    
    def _configure_memory_optimizations(self):
        logger.info('🔧 Configuring memory optimizations...')
        
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        
        if torch.cuda.is_available():
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                logger.info("   ✅ Flash Attention habilitado")
            except:
                logger.info('   ⚠️ Flash Attention not available')
            
            torch.cuda.empty_cache()
            
            logger.info('   ✅ Memory optimizations are configured')
    
    def _configure_data_loader_optimizations(self, config_overrides):
        config_overrides['num_neighbors'] = [32, 24, 16]
        config_overrides['num_workers'] = 2
        config_overrides['pin_memory'] = False
        config_overrides['persistent_workers'] = False
        
        logger.info("   🔧 Data loader otimizado:")
        logger.info(f"      Neighbors: {config_overrides['num_neighbors']}")
        logger.info(f"      Workers: {config_overrides['num_workers']}")
        logger.info(f"      Pin memory: {config_overrides['pin_memory']}")
        
        return config_overrides
    
    def _cleanup_memory(self, stage_name=""):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            gc.collect()
            
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            
            logger.info(f"   🧹 Memory cleanup {stage_name}:")
            logger.info(f"      📊 GPU Alocada: {allocated:.2f}GB")
            logger.info(f"      📊 GPU Reservada: {reserved:.2f}GB")
    
    def _define_ablation_components(self):
        return {
            'baseline': {
                'name': 'Baseline Completo',
                'description': 'Full model without any modifications',
                'modifications': {}
            },
            'no_local_precision': {
                'name': 'Sem LocalPrecisionRefiner',
                'description': 'Removes the local precision refinement',
                'modifications': {
                    'model': {'use_local_precision_refiner': False}
                }
            },
            'no_physics_loss': {
                'name': 'Sem Physics-Informed Loss',
                'description': 'Removes the physical components of the loss',
                'modifications': {
                    'loss': {
                        'physics_weight': 0.0,
                        'rf_physics_weight': 0.0,
                        'geometric_weight': 0.0
                    }
                }
            },
            'no_attention': {
                'name': 'Sem Attention',
                'description': 'Removes the attention mechanism entirely',
                'modifications': {
                    'model': {'use_attention': False}
                }
            },
            'reduced_layers': {
                'name': 'Camadas Reduzidas',
                'description': 'Reduces the number of GNN layers',
                'modifications': {
                    'model': {'num_layers': 2}
                }
            }
        }
    
    def _load_data(self):
        logger.info("📊 Carregando dados...")
        
        graph_path = self.base_dir / "arpia_hetero_graph_v3.pt"
        metadata_path = self.base_dir / "graph_metadata_v3.json"
        scalers_path = self.base_dir / "scalers.pkl"
        
        if not graph_path.exists():
            raise FileNotFoundError(f"The file was not found: {graph_path}")
        
        logger.info('🔄 Loading the heterogeneous graph...')
        graph_data = torch.load(
            graph_path, 
            map_location=self.device,
            weights_only=False
        )
        
        graph_data = prepare_graph_data(graph_data, self.device)
        
        logger.info("🔄 Carregando metadata...")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info("🔄 Carregando scalers...")
        try:
            scalers = torch.load(scalers_path, map_location='cpu')
            logger.info('✅ Scalers loaded successfully!')
        except Exception as e:
            logger.warning(f"⚠️ Error while loading scalers: {e}")
            logger.info("🔄 Tentando carregar com pickle...")
            try:
                import pickle
                with open(scalers_path, 'rb') as f:
                    scalers = pickle.load(f)
                logger.info("✅ Scalers carregados com pickle!")
            except Exception as e2:
                logger.warning(f"⚠️ Error while loading scalers with pickle: {e2}")
                logger.info('🔄 Using the default scalers...')
                scalers = None
        
        logger.info('✅ Data loaded successfully')
        return graph_data, metadata, scalers
    
    def _create_config_with_modifications(self, component_name, modifications):
        config_dict = self.base_config.to_dict()
        
        for category, changes in modifications.items():
            if category in config_dict:
                config_dict[category].update(changes)
            else:
                config_dict[category] = changes
        
        config_dict['device'] = str(self.device)
        
        return config_dict
    
    def _train_model(self, config_dict, component_name):
        logger.info(f"\n🔄 Treinando: {component_name}")
        
        try:
            self._cleanup_memory(f"antes de {component_name}")
            
            original_batch_size = config_dict['data']['batch_size']
            optimized_batch_size = min(256, original_batch_size // 8)
            
            original_hidden_channels = config_dict['model']['hidden_channels']
            optimized_hidden_channels = min(512, original_hidden_channels // 4)
            
            original_layers = config_dict['model']['num_layers']
            optimized_layers = min(4, original_layers // 4)
            
            original_heads = config_dict['model']['heads']
            optimized_heads = min(8, original_heads // 8)
            
            logger.info(f"   🔧 AGGRESSIVE memory optimizations:")
            logger.info(f"      Batch size: {original_batch_size} → {optimized_batch_size}")
            logger.info(f"      Hidden channels: {original_hidden_channels} → {optimized_hidden_channels}")
            logger.info(f"      Layers: {original_layers} → {optimized_layers}")
            logger.info(f"      Heads: {original_heads} → {optimized_heads}")
            
            config_overrides = {
                'epochs': 300,
                'batch_size': optimized_batch_size,
                'hidden_channels': optimized_hidden_channels,
                'num_layers': optimized_layers,
                'heads': optimized_heads,
                'eda_frequency': 20,
                'use_mixed_precision': True,
                'save_dir': str(self.output_dir / component_name)
            }
            
            config_overrides = self._configure_data_loader_optimizations(config_overrides)
            
            start_time = time.time()
            logger.info(f"   Executando pipeline ultra-avançado...")
            
            model, history, evaluation_results = run_training_cell6_with_monitoring(
                graph_data=self.graph_data,
                scalers=self.scalers if self.scalers is not None else {},
                config_overrides=config_overrides,
                device=self.device,
                save_dir=str(self.output_dir / component_name),
                epochs=300,
                batch_size=optimized_batch_size,
                hidden_channels=optimized_hidden_channels,
                num_layers=optimized_layers,
                heads=optimized_heads,
                eda_frequency=20,
                use_mixed_precision=True
            )
            
            del model
            self._cleanup_memory(f"após {component_name}")
            
            training_time = time.time() - start_time
            
            final_metrics = {
                'component': component_name,
                'training_time': training_time,
                'final_loss': history.get('train_loss', [0])[-1] if history.get('train_loss') else 0,
                'best_error': evaluation_results.get('mean_error', 0),
                'test_metrics': evaluation_results,
                'history': history
            }
            
            logger.info(f"✅ Training finished in {training_time:.1f}s")
            logger.info(f"   MAE Final: {evaluation_results.get('mean_error', 0):.2f}m")
            
            return final_metrics
            
        except Exception as e:
            logger.error(f"❌ Error in the training of {component_name}: {e}")
            return {
                'component': component_name,
                'error': str(e),
                'training_time': 0,
                'final_loss': float('inf'),
                'best_error': float('inf'),
                'test_metrics': {},
                'history': {}
            }
    
    def run_ablation_study(self):
        logger.info("🚀 INICIANDO ABLATION STUDY COMPLETO")
        logger.info("="*50)
        
        self._cleanup_memory("inicial")
        
        logger.info('📊 Loading the ARPIA data...')
        self.graph_data, self.metadata, self.scalers = self._load_data()
        logger.info('✅ Data loaded successfully!')
        
        self._cleanup_memory("após carregar dados")
        
        start_time = time.time()
        
        for component_name, component_info in self.ablation_components.items():
            logger.info(f"\n🔬 Testando: {component_info['name']}")
            logger.info(f"   Descrição: {component_info['description']}")
            
            self._cleanup_memory(f"before the test {component_name}")
            
            config_dict = self._create_config_with_modifications(
                component_name, 
                component_info['modifications']
            )
            
            result = self._train_model(config_dict, component_name)
            
            self._cleanup_memory(f"após teste {component_name}")
            
            self.results[component_name] = result
            
            if 'error' in result:
                logger.error(f"❌ {component_name}: ERRO - {result['error']}")
            else:
                logger.info(f"✅ {component_name}: MAE = {result['test_metrics'].get('mean_error', 0):.2f}m")
        
        total_time = time.time() - start_time
        
        self._save_results()
        
        self._show_summary()
        
        logger.info(f"🏁 Ablation Study finished in {total_time:.1f}s")
    
    def _save_results(self):
        logger.info("💾 Salvando resultados...")
        
        results_file = self.output_dir / "ablation_results.json"
        
        json_results = {}
        for component, result in self.results.items():
            json_results[component] = {
                'component': result['component'],
                'training_time': result['training_time'],
                'final_loss': result['final_loss'],
                'best_error': result['best_error'],
                'test_metrics': result['test_metrics'],
                'error': result.get('error', None)
            }
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"✅ Results saved to {results_file}")
    
    def _show_summary(self):
        logger.info("\n📊 RESUMO DO ABLATION STUDY")
        logger.info("="*50)
        
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1].get('test_metrics', {}).get('mean_error', float('inf'))
        )
        
        for i, (component, result) in enumerate(sorted_results, 1):
            component_info = self.ablation_components[component]
            
            if 'error' in result:
                logger.info(f"{i:2d}. {component_info['name']}: ❌ ERRO")
            else:
                mae = result['test_metrics'].get('mean_error', 0)
                logger.info(f"{i:2d}. {component_info['name']}: {mae:.2f}m")
        
        logger.info("="*50)


def main():
    try:
        logger.info("🔬 Iniciando ARPIA V3 Ablation Study...")
        
        study = ARPIAAblationStudy()
        study.run_ablation_study()
        
        logger.info('✅ Ablation Study finished successfully!')
        
    except Exception as e:
        logger.error(f"❌ ERRO FATAL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
