#!/usr/bin/env python3
"""Optimized runner of the CTC/CTNC triangulation, writing the format read by the edge generator."""

INTERNAL_DATA_ROOT = __import__("os").environ.get("ARPIA_ROOT", ".")
import os
import sys
import argparse
import logging
import time
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

def validate_input_files():
    logger = logging.getLogger(__name__)
    
    input_files = {
        'PEAID_ERBID.csv': f"{INTERNAL_DATA_ROOT}/DATA_ENRICHED/ALVO/PEAID_ERBID.csv",
        'erb_cps.csv': f"{INTERNAL_DATA_ROOT}/DATA_ENRICHED/ANTENAS_ERB/erb_cps.csv"
    }
    
    missing_files = []
    for name, path in input_files.items():
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")
        else:
            size_mb = os.path.getsize(path) / (1024*1024)
            logger.info(f"✅ {name}: {size_mb:.2f}MB")
    
    if missing_files:
        logger.error('❌ Input files not found:')
        for file in missing_files:
            logger.error(f"   - {file}")
        return False
    
    return True

def check_output_directory():
    output_dir = f"{INTERNAL_DATA_ROOT}/DATA_ENRICHED/ANTENAS_ERB"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Output directory created: {output_dir}")
    
    return output_dir

def validate_output_file(output_file: str):
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(output_file):
        logger.error(f"❌ Output file was not created: {output_file}")
        return False
    
    file_size_mb = os.path.getsize(output_file) / (1024*1024)
    logger.info(f"📄 Arquivo criado: {file_size_mb:.2f}MB")
    
    try:
        import pandas as pd
        df = pd.read_csv(output_file, nrows=5)
        logger.info(f"📊 Primeiras linhas carregadas: {len(df)} registros")
        logger.info(f"📋 Colunas: {list(df.columns)}")
        
        required_columns = [
            'pea_id', 'pea_lat', 'pea_lon',
            'erb_id', 'lat_erb', 'lon_erb',
            'cps_id', 'lat_centroide', 'lon_centroide',
            'triangulation_id', 'triangulation_type',
            'triangulation_centroid_lat', 'triangulation_centroid_lon',
            'operadora', 'fator_punitivo'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"⚠️ Colunas faltantes: {missing_columns}")
        else:
            logger.info('✅ All required columns are present')
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error while validating file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Runs the OPTIMIZED triangulation pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='\n🔧 CORRECTED PIPELINE:\n  - Generates erb_cps_ctc_ctnc.csv in the correct format\n  - Creates unique triangulation IDs\n  - Maps columns for the edge pipeline\n  - Adds the default fator_punitivo\n\nUsage examples:\n  python run_triangulation_optimized.py\n  python run_triangulation_optimized.py --debug\n  python run_triangulation_optimized.py --validate-only\n        '
    )
    
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Ativa modo debug com logging verboso'
    )
    
    parser.add_argument(
        '--validate-only', '-v',
        action='store_true',
        help='Only validates the input files without running the pipeline'
    )
    
    parser.add_argument(
        '--config', '-c',
        default='triangulation_config.yaml',
        help='Path of the YAML configuration file'
    )
    
    args = parser.parse_args()
    
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)
    
    print('\n🔧 =============================================== 🔧\n   OPTIMIZED TRIANGULATION PIPELINE V3\n   Correct format for the edge pipeline\n🔧 =============================================== 🔧\n    ')
    
    logger.info('=== STARTING THE OPTIMIZED TRIANGULATION PIPELINE ===')
    logger.info(f"Configuração: {args.config}")
    logger.info(f"Modo debug: {args.debug}")
    
    start_time = datetime.now()
    
    try:
        logger.info("1️⃣ Validando arquivos de entrada...")
        if not validate_input_files():
            logger.error('❌ Validation of the files failed')
            return 1
        
        output_dir = check_output_directory()
        logger.info(f"📁 Output directory in use: {output_dir}")
        
        if args.validate_only:
            logger.info('✅ Validation finished successfully')
            return 0
        
        logger.info('2️⃣ Running the triangulation pipeline...')
        
        pipeline = TriangulationPipeline(args.config)
        output_file = pipeline.run_pipeline()
        
        end_time = datetime.now()
        
        logger.info('3️⃣ Validating the output file...')
        if not validate_output_file(output_file):
            logger.error('❌ Validation of the output file failed')
            return 1
        
        duration = (end_time - start_time).total_seconds()
        
        logger.info('🎉 === PIPELINE FINISHED SUCCESSFULLY ===')
        logger.info(f"⏱️ Tempo total: {duration:.2f}s ({duration/60:.2f}min)")
        logger.info(f"📄 Arquivo gerado: {output_file}")
        logger.info('🔧 Data in the correct format for the edge pipeline!')
        
        print(f"\n🎯 NEXT STEPS:\n=====================================\n1. Generated file: {output_file}\n2. Run the edge pipeline:\n   python run_arestas_sequencial.py\n3. Check the results in the folder:\n   $ARPIA_ROOT/DATA_PROCESSED/GRAPH_NODES_EDGES/\n        ")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning('⚠️ Run interrupted by the user')
        return 130
        
    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.error(f"❌ Error during the run: {str(e)}")
        logger.error(f"⏱️ Time until the error: {duration:.2f}s")
        
        if args.debug:
            import traceback
            logger.error("Traceback completo:")
            logger.error(traceback.format_exc())
        
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
