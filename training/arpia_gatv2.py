#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ARPIA pipeline with GATv2 convolutions (architecture comparison)."""

INTERNAL_DATA_ROOT = __import__("os").environ.get("ARPIA_ROOT", ".")
import os
import sys
import json
import time
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import multiprocessing

if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    try:
        import io
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'buffer'):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

try:
    import psutil
    CPU_COUNT = multiprocessing.cpu_count()
    RAM_GB = psutil.virtual_memory().total / (1024**3)
    AVAILABLE_RAM_GB = RAM_GB * 0.8
    HAS_PSUTIL = True
except ImportError:
    CPU_COUNT = multiprocessing.cpu_count()
    RAM_GB = 128
    AVAILABLE_RAM_GB = 102
    HAS_PSUTIL = False

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    pass

try:
    from config_colab import create_default_config
    from data_loaders import prepare_graph_data
    from train_colab import run_training_cell6_with_monitoring
except ImportError as e:
    print(f"❌ Error while importing modules: {e}")
    print('Make sure you are in the correct directory of the ARPIA project')
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-15s | %(funcName)-15s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('arpia_gatv2.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class ARPIAGATv2:
    
    def __init__(self):
        logger.info("🧪 Inicializando teste ARPIA GATv2...")
        
        self._configure_memory_optimizations()
        
        if IN_COLAB:
            self.base_dir = Path(f"{INTERNAL_DATA_ROOT}/TREINAMENTO_REAL")
        else:
            self.base_dir = Path(__file__).parent
            
        self.output_dir = self.base_dir / "OUTPUT" / "gnn_architecture_comparison" / "gatv2"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cpu')
        logger.info(f"💻 Modo CPU ativado - Treinamento local otimizado")
        logger.info(f"🔧 Device configurado: {self.device}")
        logger.info(f"   🖥️ Hardware: Ryzen 9 5900X (24 threads) + 128GB RAM")
        
        self.base_config = create_default_config()
        
        logger.info("✅ ARPIA GATv2 inicializado")
    
    def _configure_memory_optimizations(self):
        logger.info('🔧 Configuring optimizations for local hardware...')
        
        logger.info(f"   🖥️ CPU cores detectados: {CPU_COUNT}")
        if HAS_PSUTIL:
            logger.info(f"   💾 RAM total: {RAM_GB:.1f}GB")
            logger.info(f"   💾 Available RAM (80%): {AVAILABLE_RAM_GB:.1f}GB")
        
        if not torch.cuda.is_available():
            torch.set_num_threads(CPU_COUNT)
            torch.set_num_interop_threads(CPU_COUNT // 2)
            
            os.environ['OMP_NUM_THREADS'] = str(CPU_COUNT)
            os.environ['MKL_NUM_THREADS'] = str(CPU_COUNT)
            os.environ['NUMEXPR_NUM_THREADS'] = str(CPU_COUNT)
            
            logger.info(f"   ✅ PyTorch threads configurados: {CPU_COUNT}")
            logger.info(f"   ✅ Inter-op threads: {CPU_COUNT // 2}")
        
        if torch.cuda.is_available():
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                logger.info("   ✅ Flash Attention habilitado")
            except:
                logger.info('   ⚠️ Flash Attention not available')
            torch.cuda.empty_cache()
        
        logger.info('   ✅ Hardware optimizations are configured')
    
    def _cleanup_memory(self, stage_name=""):
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            
            logger.info(f"   🧹 Memory cleanup {stage_name}:")
            logger.info(f"      📊 GPU Alocada: {allocated:.2f}GB")
            logger.info(f"      📊 GPU Reservada: {reserved:.2f}GB")
        else:
            if HAS_PSUTIL:
                ram_used = psutil.virtual_memory().used / (1024**3)
                ram_percent = psutil.virtual_memory().percent
                logger.info(f"   🧹 Memory cleanup {stage_name}:")
                logger.info(f"      📊 RAM Usada: {ram_used:.2f}GB ({ram_percent:.1f}%)")
    
    def _load_data(self):
        logger.info("📊 Carregando dados...")
        
        graph_path = self.base_dir / "arpia_hetero_graph_v3.pt"
        graph_data = torch.load(graph_path, map_location='cpu')
        graph_data = prepare_graph_data(graph_data, self.device)
        
        metadata_path = self.base_dir / "graph_metadata_v3.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        scalers_path = self.base_dir / "scalers.pkl"
        scalers = None
        
        try:
            scalers = torch.load(scalers_path, map_location='cpu')
            logger.info("✅ Scalers carregados com torch.load!")
        except Exception as e:
            logger.warning(f"⚠️ Error while loading scalers: {e}")
            try:
                import pickle
                with open(scalers_path, 'rb') as f:
                    scalers = pickle.load(f)
                logger.info("✅ Scalers carregados com pickle!")
            except Exception as e2:
                logger.error(f"❌ Error while loading scalers: {e2}")
                scalers = None
        
        logger.info('✅ Data loaded successfully')
        return graph_data, metadata, scalers
    
    def _create_gatv2_config(self):
        config_dict = self.base_config.to_dict()
        
        
        optimal_batch_size = min(1024, int(256 * (AVAILABLE_RAM_GB / 51)))
        
        optimal_num_neighbors = [64, 48, 32]
        
        optimal_num_workers = max(8, min(CPU_COUNT - 4, 18))
        
        logger.info(f"   📊 Batch size otimizado: {optimal_batch_size}")
        logger.info(f"   📊 Num neighbors otimizado: {optimal_num_neighbors}")
        logger.info(f"   📊 Num workers otimizado: {optimal_num_workers}")
        
        config_dict['model'].update({
            'architecture': 'GATv2',
            'hidden_channels': 512,
            'num_layers': 4,
            'heads': 8,
            'use_attention': True,
            'attention_version': 'v2',
            'concat_heads': True,
            'negative_slope': 0.2,
            'dropout': 0.1,
        })
        
        config_dict['data'].update({
            'batch_size': optimal_batch_size,
            'num_neighbors': optimal_num_neighbors,
        })
        
        config_dict['device'] = str(self.device)
        
        self.optimal_batch_size = optimal_batch_size
        self.optimal_num_neighbors = optimal_num_neighbors
        self.optimal_num_workers = optimal_num_workers
        
        return config_dict
    
    def run_test(self):
        logger.info('🚀 STARTING THE GATv2 TEST')
        logger.info("="*60)
        
        self._cleanup_memory("inicial")
        
        logger.info("📊 Carregando dados...")
        self.graph_data, self.metadata, self.scalers = self._load_data()
        logger.info("✅ Dados carregados!")
        
        self._cleanup_memory("após carregar dados")
        
        config_dict = self._create_gatv2_config()
        
        logger.info("🔧 Configuração GATv2:")
        logger.info(f"   Arquitetura: {config_dict['model'].get('architecture')}")
        logger.info(f"   Hidden channels: {config_dict['model'].get('hidden_channels')}")
        logger.info(f"   Layers: {config_dict['model'].get('num_layers')}")
        logger.info(f"   Heads: {config_dict['model'].get('heads')}")
        logger.info(f"   Attention version: {config_dict['model'].get('attention_version')}")
        logger.info(f"   Batch size: {config_dict['data'].get('batch_size')}")
        
        use_mixed_precision = torch.cuda.is_available()
        
        config_overrides = {
            'epochs': 300,
            'batch_size': config_dict['data']['batch_size'],
            'hidden_channels': config_dict['model']['hidden_channels'],
            'num_layers': config_dict['model']['num_layers'],
            'heads': config_dict['model']['heads'],
            'use_attention': True,
            'architecture': 'GATv2',
            'attention_version': 'v2',
            'eda_frequency': 20,
            'use_mixed_precision': use_mixed_precision,
            'save_dir': str(self.output_dir),
            'device': str(self.device)
        }
        
        config_overrides['num_neighbors'] = self.optimal_num_neighbors
        config_overrides['num_workers'] = self.optimal_num_workers
        config_overrides['pin_memory'] = False
        config_overrides['persistent_workers'] = True
        config_overrides['prefetch_factor'] = 4
        
        logger.info('🚀 Optimized settings for local hardware:')
        logger.info(f"   📦 Batch size: {config_overrides['batch_size']}")
        logger.info(f"   🔗 Num neighbors: {config_overrides['num_neighbors']}")
        logger.info(f"   👷 Num workers: {config_overrides['num_workers']}")
        logger.info(f"   🔄 Persistent workers: {config_overrides['persistent_workers']}")
        logger.info(f"   ⚡ Prefetch factor: {config_overrides['prefetch_factor']}")
        
        start_time = time.time()
        logger.info(f"🔄 Starting GATv2 training (300 epochs)...")
        
        try:
            self._cleanup_memory('before the GATv2 training')
            
            model, history, evaluation_results = run_training_cell6_with_monitoring(
                graph_data=self.graph_data,
                scalers=self.scalers if self.scalers is not None else {},
                config_overrides=config_overrides,
                device=self.device,
                save_dir=str(self.output_dir),
                epochs=300,
                batch_size=config_overrides['batch_size'],
                hidden_channels=config_overrides['hidden_channels'],
                num_layers=config_overrides['num_layers'],
                heads=config_overrides['heads'],
                eda_frequency=20,
                use_mixed_precision=use_mixed_precision
            )
            
            del model
            self._cleanup_memory("após treinamento GATv2")
            
            training_time = time.time() - start_time
            
            final_loss = history.get('train_loss', [float('inf')])[-1] if history.get('train_loss') else float('inf')
            best_error = evaluation_results.get('mae', float('inf')) if evaluation_results else float('inf')
            
            result = {
                'architecture': 'GATv2',
                'training_time': training_time / 60,
                'final_loss': final_loss,
                'best_error': best_error,
                'test_metrics': evaluation_results or {},
                'history': history,
                'status': 'SUCCESS',
                'config': config_overrides
            }
            
            logger.info(f"✅ GATv2 test is finished!")
            logger.info(f"   Tempo: {result['training_time']:.1f}min")
            logger.info(f"   Loss final: {final_loss:.4f}")
            logger.info(f"   Erro médio: {best_error:.1f}m")
            
            results_file = self.output_dir / "gatv2_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"✅ Results saved to {results_file}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in the GATv2 test: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return {
                'architecture': 'GATv2',
                'status': 'ERROR',
                'error': str(e)
            }

def main():
    try:
        logger.info('🧪 Starting the ARPIA GATv2 test...')
        
        test = ARPIAGATv2()
        result = test.run_test()
        
        logger.info('✅ GATv2 test is finished!')
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Erro fatal: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

if __name__ == "__main__":
    main()

