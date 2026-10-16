#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full V3 normalization pipeline: EDA, ID mapping, normalization and validation."""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import yaml
import json
import psutil
from concurrent.futures import ThreadPoolExecutor
import traceback

from normalizacao_v3 import NormalizadorGrafoV3
from eda_normalizacao_v3 import EDANormalizacaoV3
from id_mapeamento_v3 import MapeadorIDsV3


class PipelineNormalizacaoV3:
    
    def __init__(self, config_path: str, verbose: bool = False):
        self.config_path = config_path
        self.verbose = verbose
        self.config = self._load_config()
        self._setup_logging()
        
        self.start_time = time.time()
        self.steps_completed = []
        self.steps_failed = []
        self.system_stats = {}
        
        self.eda = None
        self.mapeador = None
        self.normalizador = None
        
        self.results = {
            'version': '3.0',
            'timestamp': datetime.now().isoformat(),
            'config_path': config_path,
            'steps': {},
            'statistics': {},
            'performance': {}
        }
        
        self.logger.info("Pipeline V3 inicializado")
    
    def _load_config(self) -> dict:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error while loading configuration: {str(e)}")
            sys.exit(1)
    
    def _setup_logging(self):
        log_dir = Path(self.config['paths']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger('pipeline_v3')
        self.logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        
        file_handler = logging.FileHandler(
            log_dir / 'pipeline_v3.log',
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        
        file_handler.setFormatter(detailed_formatter)
        console_handler.setFormatter(simple_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def executar_pipeline_completo(self, skip_eda: bool = False, only_eda: bool = False) -> bool:
        try:
            self.logger.info("=" * 80)
            self.logger.info("EXECUTANDO PIPELINE COMPLETO ARPIA V3")
            self.logger.info("=" * 80)
            
            self._log_system_info()
            
            success = True
            
            if not skip_eda:
                success = self._executar_eda() and success
            
            if only_eda:
                self._gerar_relatorio_final()
                return success
            
            success = self._executar_mapeamento_ids() and success
            
            success = self._executar_normalizacao() and success
            
            success = self._executar_validacao_final() and success
            
            self._gerar_relatorio_final()
            
            self._executar_limpeza()
            
            total_time = time.time() - self.start_time
            if success:
                self.logger.info(f"✓ V3 PIPELINE FINISHED SUCCESSFULLY in {total_time/60:.2f} minutes")
            else:
                self.logger.error(f"⚠ V3 PIPELINE FINISHED WITH FAILURES in {total_time/60:.2f} minutes")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Fatal error in the pipeline: {str(e)}", exc_info=True)
            return False
    
    def _log_system_info(self):
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage(str(Path(self.config['paths']['output_dir']).parent))
        
        self.system_stats = {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': memory.total / (1024**3),
            'memory_available_gb': memory.available / (1024**3),
            'disk_total_gb': disk.total / (1024**3),
            'disk_free_gb': disk.free / (1024**3),
            'python_version': sys.version.split()[0]
        }
        
        self.logger.info("Informações do Sistema:")
        self.logger.info(f"  CPU: {self.system_stats['cpu_count']} cores")
        self.logger.info(f"  Memory: {self.system_stats['memory_available_gb']:.1f}GB / {self.system_stats['memory_total_gb']:.1f}GB")
        self.logger.info(f"  Disco: {self.system_stats['disk_free_gb']:.1f}GB livres")
        self.logger.info(f"  Python: {self.system_stats['python_version']}")
    
    def _executar_eda(self) -> bool:
        try:
            self.logger.info("=" * 50)
            self.logger.info("ETAPA 1: EDA COMPLETA")
            self.logger.info("=" * 50)
            
            step_start = time.time()
            
            self.eda = EDANormalizacaoV3(self.config_path)
            
            success = self.eda.executar_eda_completa()
            
            step_time = time.time() - step_start
            self.results['steps']['eda'] = {
                'success': success,
                'duration_minutes': step_time / 60,
                'timestamp': datetime.now().isoformat()
            }
            
            if success:
                self.logger.info(f"✓ EDA finished in {step_time/60:.2f} minutes")
                self.steps_completed.append('eda')
            else:
                self.logger.error("✗ EDA falhou")
                self.steps_failed.append('eda')
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error in the EDA: {str(e)}", exc_info=True)
            self.steps_failed.append('eda')
            return False
    
    def _executar_mapeamento_ids(self) -> bool:
        try:
            self.logger.info("=" * 50)
            self.logger.info("ETAPA 2: MAPEAMENTO DE IDs")
            self.logger.info("=" * 50)
            
            step_start = time.time()
            
            self.mapeador = MapeadorIDsV3(self.config_path)
            
            success = self.mapeador.processar_mapeamentos_completos()
            
            step_time = time.time() - step_start
            self.results['steps']['mapeamento_ids'] = {
                'success': success,
                'duration_minutes': step_time / 60,
                'timestamp': datetime.now().isoformat()
            }
            
            if success:
                self.logger.info(f"✓ ID mapping finished in {step_time/60:.2f} minutes")
                self.steps_completed.append('mapeamento_ids')
                
                self.results['statistics']['id_mapping'] = self.mapeador.obter_estatisticas()
            else:
                self.logger.error("✗ Mapeamento de IDs falhou")
                self.steps_failed.append('mapeamento_ids')
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error in the mapping: {str(e)}", exc_info=True)
            self.steps_failed.append('mapeamento_ids')
            return False
    
    def _executar_normalizacao(self) -> bool:
        try:
            self.logger.info("=" * 50)
            self.logger.info('STEP 3: FULL NORMALIZATION')
            self.logger.info("=" * 50)
            
            step_start = time.time()
            
            self.normalizador = NormalizadorGrafoV3(self.config_path)
            
            success = self.normalizador.normalizar_grafo_completo()
            
            step_time = time.time() - step_start
            self.results['steps']['normalizacao'] = {
                'success': success,
                'duration_minutes': step_time / 60,
                'timestamp': datetime.now().isoformat()
            }
            
            if success:
                self.logger.info(f"✓ Normalization finished in {step_time/60:.2f} minutes")
                self.steps_completed.append('normalizacao')
                
                self.results['statistics']['normalization'] = {
                    'total_nodes': len(self.normalizador.nodes_df),
                    'total_edges': len(self.normalizador.edges_df),
                    'memory_usage': self.normalizador.check_memory_usage()
                }
            else:
                self.logger.error("✗ Normalização falhou")
                self.steps_failed.append('normalizacao')
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error in the normalization: {str(e)}", exc_info=True)
            self.steps_failed.append('normalizacao')
            return False
    
    def _executar_validacao_final(self) -> bool:
        try:
            self.logger.info("=" * 50)
            self.logger.info('STEP 4: FINAL VALIDATION')
            self.logger.info("=" * 50)
            
            step_start = time.time()
            
            output_dir = Path(self.config['paths']['output_dir'])
            arquivos_esperados = [
                self.config['output']['files']['nodes'],
                self.config['output']['files']['edges'],
                self.config['output']['files']['metadata'],
                self.config['output']['files']['scalers'],
                self.config['output']['files']['id_mapping']
            ]
            
            arquivos_existentes = []
            arquivos_ausentes = []
            
            for arquivo in arquivos_esperados:
                arquivo_path = output_dir / arquivo
                if arquivo_path.exists():
                    arquivos_existentes.append(arquivo)
                else:
                    arquivos_ausentes.append(arquivo)
            
            validacao_ok = len(arquivos_ausentes) == 0
            
            step_time = time.time() - step_start
            self.results['steps']['validacao'] = {
                'success': validacao_ok,
                'duration_minutes': step_time / 60,
                'timestamp': datetime.now().isoformat(),
                'arquivos_existentes': arquivos_existentes,
                'arquivos_ausentes': arquivos_ausentes
            }
            
            if validacao_ok:
                self.logger.info(f"✓ Validation finished in {step_time/60:.2f} minutes")
                self.logger.info(f"  All {len(arquivos_existentes)} expected files were generated")
                self.steps_completed.append('validacao')
            else:
                self.logger.error("✗ Validação falhou")
                self.logger.error(f"  Arquivos ausentes: {arquivos_ausentes}")
                self.steps_failed.append('validacao')
            
            return validacao_ok
            
        except Exception as e:
            self.logger.error(f"Error in the validation: {str(e)}", exc_info=True)
            self.steps_failed.append('validacao')
            return False
    
    def _gerar_relatorio_final(self):
        try:
            self.logger.info("=" * 50)
            self.logger.info("RELATÓRIO FINAL")
            self.logger.info("=" * 50)
            
            total_time = time.time() - self.start_time
            
            self.results['performance'] = {
                'total_duration_minutes': total_time / 60,
                'steps_completed': self.steps_completed,
                'steps_failed': self.steps_failed,
                'success_rate': len(self.steps_completed) / (len(self.steps_completed) + len(self.steps_failed)) if (self.steps_completed or self.steps_failed) else 0,
                'system_stats': self.system_stats
            }
            
            output_dir = Path(self.config['paths']['output_dir'])
            relatorio_path = output_dir / 'pipeline_report_v3.json'
            
            with open(relatorio_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Tempo total: {total_time/60:.2f} minutos")
            self.logger.info(f"Etapas concluídas: {len(self.steps_completed)}")
            self.logger.info(f"Etapas falhadas: {len(self.steps_failed)}")
            
            if self.steps_completed:
                self.logger.info(f"✓ Concluídas: {', '.join(self.steps_completed)}")
            
            if self.steps_failed:
                self.logger.info(f"✗ Falhadas: {', '.join(self.steps_failed)}")
            
            self.logger.info(f"Relatório salvo: {relatorio_path}")
            
        except Exception as e:
            self.logger.error(f"Error while generating the report: {str(e)}")
    
    def _executar_limpeza(self):
        try:
            self.logger.info("Executando limpeza final...")
            
            if self.normalizador:
                del self.normalizador
            
            if self.mapeador:
                del self.mapeador
            
            if self.eda:
                del self.eda
            
            import gc
            gc.collect()
            
            self.logger.info("✓ Limpeza concluída")
            
        except Exception as e:
            self.logger.warning(f"Error during the cleanup: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description='ARPIA V3 Normalization Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
    python pipeline_v3.py
    python pipeline_v3.py --config custom_config.yaml
    python pipeline_v3.py --skip-eda
    python pipeline_v3.py --only-eda
    python pipeline_v3.py --verbose
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        default='normalizacao_v3.yaml',
        help='Configuration file (default: normalizacao_v3.yaml)'
    )
    
    parser.add_argument(
        '--skip-eda',
        action='store_true',
        help='Skip the EDA and run only the normalization'
    )
    
    parser.add_argument(
        '--only-eda',
        action='store_true',
        help='Executar apenas EDA'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Saída detalhada'
    )
    
    args = parser.parse_args()
    
    if not Path(args.config).exists():
        print(f"Error: configuration file not found: {args.config}")
        return 1
    
    try:
        pipeline = PipelineNormalizacaoV3(args.config, args.verbose)
        
        success = pipeline.executar_pipeline_completo(
            skip_eda=args.skip_eda,
            only_eda=args.only_eda
        )
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print('\n⚠ Pipeline interrupted by the user')
        return 1
    
    except Exception as e:
        print(f"Erro fatal: {str(e)}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 
