#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Command-line runner of the V3 normalization."""

import sys
import os
import argparse
import time
import logging
from pathlib import Path
from datetime import datetime
import yaml
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from normalizacao_v3 import NormalizadorGrafoV3
from eda_normalizacao_v3 import EDANormalizacaoV3


def setup_logging(level='INFO'):
    log_dir = Path('logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / 'run_normalizacao_v3.log', encoding='utf-8')
        ]
    )
    return logging.getLogger('run_normalizacao_v3')


def executar_eda_completa(config_path: str, logger: logging.Logger) -> bool:
    try:
        logger.info("Iniciando EDA completa V3...")
        
        eda = EDANormalizacaoV3(config_path)
        success = eda.executar_eda_completa()
        
        if success:
            logger.info('✓ Full EDA finished successfully')
        else:
            logger.error("✗ EDA completa falhou")
            
        return success
        
    except Exception as e:
        logger.error(f"Error in the EDA: {str(e)}", exc_info=True)
        return False


def executar_normalizacao_completa(config_path: str, logger: logging.Logger) -> bool:
    try:
        logger.info('Starting the full V3 normalization...')
        
        normalizador = NormalizadorGrafoV3(config_path)
        success = normalizador.normalizar_grafo_completo()
        
        if success:
            logger.info('✓ Full normalization finished successfully')
        else:
            logger.error('✗ Full normalization has failed')
            
        return success
        
    except Exception as e:
        logger.error(f"Error in the normalization: {str(e)}", exc_info=True)
        return False


def gerar_relatorio_final(config_path: str, logger: logging.Logger):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        output_dir = Path(config['paths']['output_dir'])
        
        arquivos_gerados = []
        for arquivo in output_dir.glob("*"):
            if arquivo.is_file():
                arquivos_gerados.append({
                    'arquivo': arquivo.name,
                    'tamanho_mb': arquivo.stat().st_size / (1024 * 1024),
                    'modificado': datetime.fromtimestamp(arquivo.stat().st_mtime).isoformat()
                })
        
        relatorio = {
            'timestamp': datetime.now().isoformat(),
            'versao': '3.0',
            'config_utilizada': config_path,
            'arquivos_gerados': arquivos_gerados,
            'total_arquivos': len(arquivos_gerados),
            'tamanho_total_mb': sum(arq['tamanho_mb'] for arq in arquivos_gerados)
        }
        
        relatorio_path = output_dir / 'relatorio_final_v3.json'
        with open(relatorio_path, 'w', encoding='utf-8') as f:
            json.dump(relatorio, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Final report saved to {relatorio_path}")
        logger.info(f"Total number of generated files: {len(arquivos_gerados)}")
        logger.info(f"Tamanho total: {relatorio['tamanho_total_mb']:.1f} MB")
        
    except Exception as e:
        logger.error(f"Error while generating the report: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description='ARPIA V3 Normalization Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
    python run_normalizacao_v3.py
    python run_normalizacao_v3.py --config custom_config.yaml
    python run_normalizacao_v3.py --skip-eda
    python run_normalizacao_v3.py --only-eda
    python run_normalizacao_v3.py --verbose
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
    
    log_level = 'DEBUG' if args.verbose else 'INFO'
    logger = setup_logging(log_level)
    
    if not Path(args.config).exists():
        logger.error(f"Configuration file not found: {args.config}")
        return 1
    
    start_time = time.time()
    
    logger.info("=" * 80)
    logger.info('ARPIA V3 NORMALIZATION PIPELINE')
    logger.info("=" * 80)
    logger.info(f"Configuração: {args.config}")
    logger.info(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    success = True
    
    try:
        if not args.skip_eda:
            logger.info("\n1. Executando EDA completa...")
            if not executar_eda_completa(args.config, logger):
                success = False
                if not args.only_eda:
                    logger.error("EDA falhou - abortando pipeline")
                    return 1
        
        if not args.only_eda:
            logger.info('\n2. Running the full normalization...')
            if not executar_normalizacao_completa(args.config, logger):
                success = False
                logger.error("Normalização falhou")
                return 1
        
        if not args.only_eda:
            logger.info('\n3. Generating the final report...')
            gerar_relatorio_final(args.config, logger)
        
        total_time = time.time() - start_time
        
        logger.info("=" * 80)
        if success:
            logger.info('✓ V3 PIPELINE FINISHED SUCCESSFULLY')
        else:
            logger.info('⚠ V3 PIPELINE FINISHED WITH WARNINGS')
        logger.info(f"Tempo total: {total_time/60:.2f} minutos")
        logger.info("=" * 80)
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info('\n⚠ Pipeline interrupted by the user')
        return 1
    
    except Exception as e:
        logger.error(f"Fatal error in the pipeline: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main()) 
