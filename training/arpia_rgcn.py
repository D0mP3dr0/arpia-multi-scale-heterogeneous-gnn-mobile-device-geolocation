#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ARPIA pipeline with RGCN convolutions (architecture comparison)."""

INTERNAL_DATA_ROOT = __import__("os").environ.get("ARPIA_ROOT", ".")
import os
import sys
import json
import time
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

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
        logging.FileHandler('arpia_rgcn.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class ARPIARGCN:
    
    def __init__(self):
        logger.info("🧪 Inicializando teste ARPIA RGCN...")
        
        self._configure_memory_optimizations()
        
        if IN_COLAB:
            self.base_dir = Path(f"{INTERNAL_DATA_ROOT}/TREINAMENTO_REAL")
        else:
            self.base_dir = Path(__file__).parent
            
        self.output_dir = self.base_dir / "OUTPUT" / "gnn_architecture_comparison" / "rgcn"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info(f"🚀 Usando GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device('cpu')
            logger.info(f"🔧 Device configurado: {self.device}")
        
        self.base_config = create_default_config()
        
        logger.info("✅ ARPIA RGCN inicializado")
    
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
    
    def _cleanup_memory(self, stage_name=""):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            import gc
            gc.collect()
            
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            
            logger.info(f"   🧹 Memory cleanup {stage_name}:")
            logger.info(f"      📊 GPU Alocada: {allocated:.2f}GB")
            logger.info(f"      📊 GPU Reservada: {reserved:.2f}GB")
    
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
    
    def _create_rgcn_config(self):
        config_dict = self.base_config.to_dict()
        
        config_dict['model'].update({
            'architecture': 'RGCN',
            'hidden_channels': 512,
            'num_layers': 3,
            'num_bases': 4,
            'num_blocks': 2,
            'use_attention': False,
            'normalize': True,
            'aggr': 'mean',
        })
        
        config_dict['data'].update({
            'batch_size': 256,
            'num_neighbors': [32, 24, 16],
        })
        
        config_dict['device'] = str(self.device)
        
        return config_dict
    
    def run_test(self):
        logger.info('🚀 STARTING THE RGCN TEST')
        logger.info("="*60)
        
        self._cleanup_memory("inicial")
        
        logger.info("📊 Carregando dados...")
        self.graph_data, self.metadata, self.scalers = self._load_data()
        logger.info("✅ Dados carregados!")
        
        self._cleanup_memory("após carregar dados")
        
        config_dict = self._create_rgcn_config()
        
        logger.info("🔧 Configuração RGCN:")
        logger.info(f"   Arquitetura: {config_dict['model'].get('architecture')}")
        logger.info(f"   Hidden channels: {config_dict['model'].get('hidden_channels')}")
        logger.info(f"   Layers: {config_dict['model'].get('num_layers')}")
        logger.info(f"   Num bases: {config_dict['model'].get('num_bases')}")
        logger.info(f"   Batch size: {config_dict['data'].get('batch_size')}")
        
        config_overrides = {
            'epochs': 300,
            'batch_size': config_dict['data']['batch_size'],
            'hidden_channels': config_dict['model']['hidden_channels'],
            'num_layers': config_dict['model']['num_layers'],
            'architecture': 'RGCN',
            'num_bases': config_dict['model'].get('num_bases', 4),
            'eda_frequency': 20,
            'use_mixed_precision': True,
            'save_dir': str(self.output_dir),
            'device': str(self.device)
        }
        
        config_overrides['num_neighbors'] = [32, 24, 16]
        config_overrides['num_workers'] = 2
        config_overrides['pin_memory'] = False
        config_overrides['persistent_workers'] = False
        
        start_time = time.time()
        logger.info(f"🔄 Starting RGCN training (300 epochs)...")
        
        try:
            self._cleanup_memory('before the RGCN training')
            
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
                heads=8,
                eda_frequency=20,
                use_mixed_precision=True
            )
            
            del model
            self._cleanup_memory("após treinamento RGCN")
            
            training_time = time.time() - start_time
            
            final_loss = history.get('train_loss', [float('inf')])[-1] if history.get('train_loss') else float('inf')
            best_error = evaluation_results.get('mae', float('inf')) if evaluation_results else float('inf')
            
            result = {
                'architecture': 'RGCN',
                'training_time': training_time / 60,
                'final_loss': final_loss,
                'best_error': best_error,
                'test_metrics': evaluation_results or {},
                'history': history,
                'status': 'SUCCESS',
                'config': config_overrides
            }
            
            logger.info(f"✅ RGCN test is finished!")
            logger.info(f"   Tempo: {result['training_time']:.1f}min")
            logger.info(f"   Loss final: {final_loss:.4f}")
            logger.info(f"   Erro médio: {best_error:.1f}m")
            
            results_file = self.output_dir / "rgcn_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"✅ Results saved to {results_file}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in the RGCN test: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return {
                'architecture': 'RGCN',
                'status': 'ERROR',
                'error': str(e)
            }

def main():
    try:
        logger.info('🧪 Starting the ARPIA RGCN test...')
        
        test = ARPIARGCN()
        result = test.run_test()
        
        logger.info('✅ RGCN test is finished!')
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Erro fatal: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

if __name__ == "__main__":
    main()


