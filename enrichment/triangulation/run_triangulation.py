# -*- coding: utf-8 -*-
"""Command-line runner of the CTC/CTNC triangulation."""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from triangulation_ctc_ctnc import TriangulationPipeline
except ImportError as e:
    print(f"Error while importing the triangulation module: {e}")
    sys.exit(1)


def setup_logging(debug: bool = False):
    level = logging.DEBUG if debug else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def validate_environment():
    logger = logging.getLogger(__name__)
    
    expected_files = [
        'erb_cps.csv',
        'erb_id_filtrado.csv',
        '../ALVO/PEAID_ERBID.csv',
        'triangulation_config.yaml'
    ]
    
    missing_files = []
    for file_path in expected_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        logger.error('Required files were not found:')
        for file_path in missing_files:
            logger.error(f"  - {file_path}")
        return False
        
    logger.info('Environment validation finished successfully')
    return True


def generate_execution_report(output_path: str, start_time: datetime, end_time: datetime, 
                            hardware_info=None):
    logger = logging.getLogger(__name__)
    
    duration = (end_time - start_time).total_seconds()
    
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f) - 1
        except Exception:
            line_count = "N/A"
    else:
        file_size = 0
        line_count = 0
    
    hw_section = ""
    if hardware_info:
        hw = hardware_info
        hw_section = f"\nHARDWARE CONFIGURATION:\n--------------------------------------------------\nCPU Cores: {hw.get('cpu_cores_logical', 'N/A')} logical ({hw.get('cpu_cores_physical', 'N/A')} physical)\nMemory: {hw.get('memory_gb', 'N/A'):.1f} GB total\nWorkers: {hw.get('optimal_workers', 'N/A')}\nChunk Size: {hw.get('optimal_chunk_size', 'N/A'):,}\nAcceleration: {'CUDA GPU' if hw.get('use_cuda') else 'Numba CPU' if hw.get('use_numba') else 'NumPy puro'}\n        "
    
    report = f"\n==================================================\nRUN REPORT - TRIANGULATION PIPELINE\n==================================================\nStart date/time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\nEnd date/time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\nDuration: {duration:.2f} seconds ({duration/60:.2f} minutes)\n{hw_section}\nGENERATED FILE:\n--------------------------------------------------\nPath: {output_path}\nSize: {file_size:,} bytes\nRecords: {line_count}\n\nSTATUS: {'SUCESSO' if os.path.exists(output_path) else 'FALHA'}\n==================================================\n    "
    
    report_path = f"triangulation_report_{start_time.strftime('%Y%m%d_%H%M%S')}.txt"
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Report saved to: {report_path}")
    except Exception as e:
        logger.warning(f"Error while saving report: {e}")
    
    print(report)


def main():
    parser = argparse.ArgumentParser(
        description='Runs the CTC/CTNC triangulation pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python run_triangulation.py
  python run_triangulation.py --config custom_config.yaml
  python run_triangulation.py --debug
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        default='triangulation_config.yaml',
        help='YAML configuration file (default: triangulation_config.yaml)'
    )
    
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Ativa modo debug com logging verboso'
    )
    
    parser.add_argument(
        '--validate-only', '-v',
        action='store_true',
        help='Only validates the environment, without running the pipeline'
    )
    
    args = parser.parse_args()
    
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)
    
    logger.info('=== STARTING THE TRIANGULATION PIPELINE RUN ===')
    logger.info(f"Configuration file in use: {args.config}")
    logger.info(f"Modo debug: {args.debug}")
    
    start_time = datetime.now()
    
    try:
        if not validate_environment():
            logger.error('Validation of the environment failed')
            return 1
        
        if args.validate_only:
            logger.info('Validation finished. --validate-only mode active, exiting.')
            return 0
        
        if not os.path.exists(args.config):
            logger.error(f"Configuration file not found: {args.config}")
            return 1
        
        logger.info('Starting the triangulation pipeline...')
        pipeline = TriangulationPipeline(args.config)
        output_path = pipeline.run_pipeline()
        
        end_time = datetime.now()
        
        hardware_info = getattr(pipeline, 'hardware_config', None)
        generate_execution_report(output_path, start_time, end_time, hardware_info)
        
        logger.info('=== RUN FINISHED SUCCESSFULLY ===')
        return 0
        
    except KeyboardInterrupt:
        logger.warning('Run interrupted by the user')
        return 130
        
    except Exception as e:
        end_time = datetime.now()
        logger.error(f"Error during the run: {str(e)}")
        
        if args.debug:
            import traceback
            logger.error("Traceback completo:")
            logger.error(traceback.format_exc())
        
        try:
            generate_execution_report("ERRO", start_time, end_time)
        except Exception:
            pass
            
        return 1


if __name__ == "__main__":
    sys.exit(main()) 
