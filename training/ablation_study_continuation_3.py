#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Continuation of the ablation study for the no_attention test."""

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
        logging.FileHandler('ablation_continuation_3.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class ARPIAAblationContinuation3:
    
    def __init__(self):
        logger.info("🔬 Inicializando ARPIA Ablation Study Continuation 3...")
        
        self._configure_memory_optimizations()
        
        if IN_COLAB:
            self.base_dir = Path(f"{INTERNAL_DATA_ROOT}/TREINAMENTO_REAL")
        else:
            self.base_dir = Path(__file__).parent
            
        self.output_dir = self.base_dir / "OUTPUT" / "ablation_results_arpia_v3"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info(f"🚀 Usando GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device('cpu')
            logger.info(f"🔧 Device configurado: {self.device}")
        
        self.base_config = create_default_config()
        
        self.ablation_components = self._define_pending_components()
        
        self.results = {}
        
        logger.info("✅ Ablation Study Continuation 3 inicializado")
    
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
    
    def _define_pending_components(self):
        COMPLETED_TESTS = ['baseline', 'no_local_precision', 'no_physics_loss']
        
        return {
            'no_attention': {
                'name': 'Sem Attention',
                'description': 'Removes the attention mechanism entirely',
                'modifications': {
                    'model': {
                        'heads': 1,
                        'use_attention': False
                    }
                }
            }
        }
    
    def _validate_no_completed_tests_in_components(self):
        COMPLETED_TESTS = ['baseline', 'no_local_precision', 'no_physics_loss']
        
        for component_name in self.ablation_components.keys():
            if component_name in COMPLETED_TESTS:
                raise ValueError(
                    f"❌ CRITICAL ERROR: '{component_name}' is an ALREADY COMPLETED test and CANNOT be run again!\n   Completed tests: {COMPLETED_TESTS}\n   Only the following tests can be run: {list(self.ablation_components.keys())}"
                )
    
    def _load_previous_results(self):
        logger.info('📂 Checking tests already finished (for reference only)...')
        previous_results = {}
        
        COMPLETED_TESTS = ['baseline', 'no_local_precision', 'no_physics_loss']
        
        for test_name in COMPLETED_TESTS:
            try:
                summary_path = self.output_dir / test_name / "final_summary.json"
                if summary_path.exists():
                    with open(summary_path, 'r') as f:
                        summary = json.load(f)
                    
                    previous_results[test_name] = {
                        'component': test_name,
                        'training_time': summary.get('total_time_minutes', 0),
                        'final_loss': summary.get('final_loss', float('inf')),
                        'best_error': summary.get('best_error', float('inf')),
                        'test_metrics': summary.get('test_metrics', {}),
                        'status': 'COMPLETED'
                    }
                    logger.info(f"   ✅ {test_name}: Loaded (already complete - will NOT be run)")
                else:
                    logger.warning(f"   ⚠️ {test_name}: File not found (but marked as complete)")
                    
            except Exception as e:
                logger.warning(f"   ⚠️ {test_name}: Error while loading - {e} (but marked as complete)")
        
        return previous_results
    
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
        
        logger.info('🔄 Loading the heterogeneous graph...')
        graph_path = self.base_dir / "arpia_hetero_graph_v3.pt"
        graph_data = torch.load(graph_path, map_location='cpu')
        
        graph_data = prepare_graph_data(graph_data, self.device)
        
        logger.info("🔄 Carregando metadata...")
        metadata_path = self.base_dir / "graph_metadata_v3.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info("🔄 Carregando scalers...")
        scalers_path = self.base_dir / "scalers.pkl"
        scalers = None
        
        try:
            scalers = torch.load(scalers_path, map_location='cpu')
            logger.info("✅ Scalers carregados com torch.load!")
        except Exception as e:
            logger.warning(f"⚠️ Error while loading scalers: {e}")
            logger.info("🔄 Tentando carregar com pickle...")
            try:
                import pickle
                with open(scalers_path, 'rb') as f:
                    scalers = pickle.load(f)
                logger.info("✅ Scalers carregados com pickle!")
            except Exception as e2:
                logger.error(f"❌ Error while loading scalers with pickle: {e2}")
                scalers = None
        
        logger.info('✅ Data loaded successfully')
        return graph_data, metadata, scalers
    
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
    
    def _create_config_with_modifications(self, component_name, modifications):
        config_dict = self.base_config.to_dict()
        
        for section, changes in modifications.items():
            if section in config_dict:
                config_dict[section].update(changes)
            else:
                config_dict[section] = changes
        
        config_dict['device'] = str(self.device)
        
        return config_dict
    
    def _train_model(self, config_dict, component_name):
        logger.info(f"\n🔄 Treinando: {component_name}")
        
        try:
            self._cleanup_memory(f"antes de {component_name}")
            
            ablation_heads = config_dict['model'].get('heads', 8)
            ablation_layers = config_dict['model'].get('num_layers', 4)
            ablation_use_attention = config_dict['model'].get('use_attention', True)
            ablation_local_gnn_layers = config_dict['model'].get('local_gnn_layers', 6)
            ablation_regional_gnn_layers = config_dict['model'].get('regional_gnn_layers', 5)
            ablation_global_context_layers = config_dict['model'].get('global_context_layers', 5)
            
            logger.info(f"   🔬 ABLATION modifications applied:")
            logger.info(f"      Heads: {ablation_heads} (use_attention: {ablation_use_attention})")
            logger.info(f"      Layers: {ablation_layers}")
            logger.info(f"      Local GNN Layers: {ablation_local_gnn_layers}")
            logger.info(f"      Regional GNN Layers: {ablation_regional_gnn_layers}")
            logger.info(f"      Global Context Layers: {ablation_global_context_layers}")
            
            original_batch_size = config_dict['data']['batch_size']
            optimized_batch_size = min(256, original_batch_size // 8)
            
            original_hidden_channels = config_dict['model']['hidden_channels']
            optimized_hidden_channels = min(512, original_hidden_channels // 4)
            
            if ablation_layers < 4:
                optimized_layers = ablation_layers
            else:
                optimized_layers = min(4, ablation_layers // 4)
            
            if ablation_heads < 8:
                optimized_heads = ablation_heads
            else:
                optimized_heads = min(8, ablation_heads // 8)
            
            logger.info(f"   🔧 AGGRESSIVE memory optimizations:")
            logger.info(f"      Batch size: {original_batch_size} → {optimized_batch_size}")
            logger.info(f"      Hidden channels: {original_hidden_channels} → {optimized_hidden_channels}")
            logger.info(f"      Layers: {ablation_layers} → {optimized_layers}")
            logger.info(f"      Heads: {ablation_heads} → {optimized_heads}")
            
            config_overrides = {
                'epochs': 300,
                'batch_size': optimized_batch_size,
                'hidden_channels': optimized_hidden_channels,
                'num_layers': optimized_layers,
                'heads': optimized_heads,
                'use_attention': ablation_use_attention,
                'local_gnn_layers': ablation_local_gnn_layers,
                'regional_gnn_layers': ablation_regional_gnn_layers,
                'global_context_layers': ablation_global_context_layers,
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
            
            final_loss = history.get('train_loss', [float('inf')])[-1] if history.get('train_loss') else float('inf')
            best_error = evaluation_results.get('mae', float('inf')) if evaluation_results else float('inf')
            
            return {
                'component': component_name,
                'training_time': training_time / 60,
                'final_loss': final_loss,
                'best_error': best_error,
                'test_metrics': evaluation_results or {},
                'history': history,
                'status': 'SUCCESS'
            }
            
        except Exception as e:
            logger.error(f"❌ Error in the training of {component_name}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return {
                'component': component_name,
                'training_time': 0,
                'final_loss': float('inf'),
                'best_error': float('inf'),
                'test_metrics': {},
                'history': {},
                'status': 'ERROR',
                'error': str(e)
            }
    
    def run_continuation(self):
        logger.info('🚀 STARTING CONTINUATION 3 OF THE ABLATION STUDY')
        logger.info("="*60)
        
        logger.info('🔒 Validating that only the no_attention test will be run...')
        self._validate_no_completed_tests_in_components()
        logger.info('✅ Validation passed - only no_attention will be run')
        
        self._cleanup_memory("inicial")
        
        logger.info('📊 Loading the ARPIA data...')
        self.graph_data, self.metadata, self.scalers = self._load_data()
        logger.info('✅ Data loaded successfully!')
        
        self._cleanup_memory("após carregar dados")
        
        previous_results = self._load_previous_results()
        self.results.update(previous_results)
        
        logger.info("📋 RESUMO DOS TESTES:")
        logger.info('   ✅ Tests already finished (will NOT be run):')
        COMPLETED_TESTS = ['baseline', 'no_local_precision', 'no_physics_loss']
        for test_name in COMPLETED_TESTS:
            if test_name in previous_results:
                status = previous_results[test_name].get('status', 'UNKNOWN')
                logger.info(f"      - {test_name}: {status} (BLOQUEADO)")
            else:
                logger.info(f"      - {test_name}: NOT FOUND (BLOCKED)")
        
        logger.info('   🔄 PENDING tests that WILL be run:')
        pending_tests = list(self.ablation_components.keys())
        if not pending_tests:
            logger.error('❌ ERROR: No pending test found!')
            raise ValueError('No pending test to run!')
        
        for component_name in pending_tests:
            logger.info(f"      - {component_name}: ✅ WILL BE RUN")
        
        start_time = time.time()
        
        for component_name, component_info in self.ablation_components.items():
            logger.info(f"\n🔬 Testando: {component_info['name']}")
            logger.info(f"   Descrição: {component_info['description']}")
            
            if component_name in COMPLETED_TESTS:
                logger.error(f"❌ CRITICAL ERROR: Attempt to run an already completed test: {component_name}")
                raise ValueError(f"Blocked attempt to run an already completed test: {component_name}")
            
            self._cleanup_memory(f"before the test {component_name}")
            
            config_dict = self._create_config_with_modifications(
                component_name, 
                component_info['modifications']
            )
            
            result = self._train_model(config_dict, component_name)
            
            self._cleanup_memory(f"após teste {component_name}")
            
            self.results[component_name] = result
            
            if result['status'] == 'SUCCESS':
                logger.info(f"✅ {component_name}: SUCESSO")
                logger.info(f"   Tempo: {result['training_time']:.1f}min")
                logger.info(f"   Loss final: {result['final_loss']:.4f}")
                logger.info(f"   Erro médio: {result['best_error']:.1f}m")
            else:
                logger.error(f"❌ {component_name}: ERRO - {result.get('error', 'Erro desconhecido')}")
        
        self._save_results()
        
        self._show_summary()
        
        total_time = time.time() - start_time
        logger.info(f"🏁 Continuation 3 of the Ablation Study finished in {total_time/60:.1f}min")
        
        return self.results
    
    def _save_results(self):
        logger.info("💾 Salvando resultados...")
        
        results_file = self.output_dir / "ablation_continuation_3_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Results saved to {results_file}")
    
    def _show_summary(self):
        logger.info('\n📊 SUMMARY OF CONTINUATION 3 OF THE ABLATION STUDY')
        logger.info("="*60)
        
        for i, (component_name, result) in enumerate(self.results.items(), 1):
            status = result.get('status', 'UNKNOWN')
            if status == 'SUCCESS':
                logger.info(f"{i}. {component_name}: ✅ SUCESSO")
                logger.info(f"   Tempo: {result['training_time']:.1f}min")
                logger.info(f"   Erro: {result['best_error']:.1f}m")
            elif status == 'COMPLETED':
                logger.info(f"{i}. {component_name}: ✅ ALREADY FINISHED")
                logger.info(f"   Tempo: {result['training_time']:.1f}min")
                logger.info(f"   Erro: {result['best_error']:.1f}m")
            else:
                logger.info(f"{i}. {component_name}: ❌ ERRO")
        
        logger.info("="*60)

def main():
    try:
        logger.info("🔬 Iniciando ARPIA V3 Ablation Study Continuation 3...")
        
        ablation_study = ARPIAAblationContinuation3()
        
        results = ablation_study.run_continuation()
        
        logger.info('✅ Ablation Study Continuation 3 finished successfully!')
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Erro fatal: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

if __name__ == "__main__":
    main()

