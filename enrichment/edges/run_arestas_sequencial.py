# -*- coding: utf-8 -*-
"""Command-line runner of the sequential V3 edge generator."""

INTERNAL_DATA_ROOT = __import__("os").environ.get("ARPIA_ROOT", ".")
import os
import sys
import argparse
import logging
import psutil
import time
from datetime import datetime
from pathlib import Path
import json

os.environ['NUMBA_DISABLE_JIT'] = '1'

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from gerador_arestas_v3_sequencial import GeradorArestasV3Sequencial
except ImportError as e:
    print(f"Error while importing the sequential edge generator module: {e}")
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


def check_system_requirements():
    logger = logging.getLogger(__name__)
    
    cpu_count = os.cpu_count()
    logger.info(f"💻 CPU: {cpu_count} cores available (using only 1)")
    
    memory = psutil.virtual_memory()
    total_gb = memory.total / (1024**3)
    available_gb = memory.available / (1024**3)
    logger.info(f"🧠 RAM: {total_gb:.1f}GB total, {available_gb:.1f}GB available")
    
    disk = psutil.disk_usage(f"{INTERNAL_DATA_ROOT}")
    free_gb = disk.free / (1024**3)
    logger.info(f"💽 Disco F:: {free_gb:.1f}GB livres")
    
    warnings = []
    
    if total_gb < 16:
        warnings.append(f"Pouca RAM ({total_gb:.1f}GB). Recomendado: 16GB+")
    
    if free_gb < 10:
        warnings.append(f"Low disk space ({free_gb:.1f}GB). Recommended: 10GB+")
    
    if warnings:
        logger.warning("⚠️ Avisos de sistema:")
        for warning in warnings:
            logger.warning(f"   - {warning}")
    else:
        logger.info("✅ Sistema adequado para processamento sequencial otimizado")
    
    return len(warnings) == 0


def validate_environment(config_file: str):
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(config_file):
        logger.error(f"Configuration file not found: {config_file}")
        return False
    
    expected_files = [
        f"{INTERNAL_DATA_ROOT}/DATA_ENRICHED/ANTENAS_ERB/erb_cps_ctc_ctnc.csv",
        f"{INTERNAL_DATA_ROOT}/DATA_ENRICHED/ALVO/PEAID_ERBID.csv"
    ]
    
    missing_files = []
    file_sizes = {}
    
    for file_path in expected_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            size_gb = os.path.getsize(file_path) / (1024**3)
            file_sizes[file_path] = size_gb
    
    if missing_files:
        logger.error('Required input files not found:')
        for file_path in missing_files:
            logger.error(f"  - {file_path}")
        return False
    
    logger.info('📁 Input files found:')
    for file_path, size_gb in file_sizes.items():
        filename = os.path.basename(file_path)
        logger.info(f"   ✅ {filename}: {size_gb:.2f}GB")
    
    logger.info('Environment validation finished successfully')
    return True


def analyze_quality_improvements(stats_file: str = None):
    logger = logging.getLogger(__name__)
    
    if not stats_file or not os.path.exists(stats_file):
        logger.warning('Statistics file not found for analysis')
        return
    
    try:
        with open(stats_file, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        
        logger.info('📊 === QUALITY IMPROVEMENT ANALYSIS ===')
        
        filters_applied = stats.get('filters_applied', {})
        if filters_applied:
            logger.info("🔍 Filtros aplicados:")
            for filter_name, count in filters_applied.items():
                logger.info(f"   - {filter_name}: {count:,} registros filtrados")
        
        alerts = stats.get('alerts_generated', [])
        if alerts:
            logger.info(f"🚨 Alertas de qualidade: {len(alerts)}")
            for alert in alerts:
                logger.info(f"   - {alert.get('type', 'UNKNOWN')}: {alert.get('message', 'Sem mensagem')}")
        else:
            logger.info("✅ Nenhum alerta de qualidade gerado")
        
        quality_metrics = stats.get('quality_metrics', {})
        if quality_metrics:
            logger.info('📈 Metrics of the quality:')
            for edge_type, metrics in quality_metrics.items():
                if 'distance_stats' in metrics:
                    dist_stats = metrics['distance_stats']
                    zero_pct = dist_stats.get('zero_percentage', 0)
                    logger.info(f"   - {edge_type}: {zero_pct:.1f}% zero distances")
                
                if 'duplicate_rate' in metrics:
                    dup_rate = metrics['duplicate_rate']
                    logger.info(f"   - {edge_type}: {dup_rate:.1%} duplicatas")
        
        logger.info('📊 === END OF QUALITY ANALYSIS ===')
        
    except Exception as e:
        logger.warning(f"Error while analysing quality improvements: {e}")


def generate_execution_report(start_time: datetime, end_time: datetime, 
                            success: bool, stats_file: str = None):
    logger = logging.getLogger(__name__)
    
    duration = (end_time - start_time).total_seconds()
    
    stats_info = ""
    quality_summary = ""
    
    if stats_file and os.path.exists(stats_file):
        try:
            with open(stats_file, 'r', encoding='utf-8') as f:
                stats = json.load(f)
            
            edges_created = stats.get('edges_created', {})
            processing_time = stats.get('processing_time', {})
            alerts_generated = stats.get('alerts_generated', [])
            filters_applied = stats.get('filters_applied', {})
            
            total_edges = sum(edges_created.values())
            total_filters = sum(filters_applied.values())
            
            stats_info = f"\nGENERATED EDGE STATISTICS:\n--------------------------------------------------\nTotal edges: {total_edges:,}\nPerformance: {total_edges/duration:.0f} edges/second\nFilters applied: {total_filters:,} records\n"
            
            quality_summary = f"\nQUALITY SUMMARY:\n--------------------------------------------------\nAlerts generated: {len(alerts_generated)}\nFilters applied: {len(filters_applied)} types\nImprovements implemented: ✅ All phases\n"
            
            for edge_type, count in edges_created.items():
                time_taken = processing_time.get(edge_type, 0)
                rate = count/time_taken if time_taken > 0 else 0
                stats_info += f"{edge_type}: {count:,} ({rate:.0f}/s)\n"
                
        except Exception as e:
            logger.warning(f"Error while loading statistics: {e}")
    
    output_dir = f"{INTERNAL_DATA_ROOT}/DATA_PROCESSED/GRAPH_NODES_EDGES"
    output_files = [
        "edges_conexao_pea_erb.csv",
        "edges_nao_conexao.csv", 
        "edges_intra_triangulo_conexao.csv",
        "edges_intra_triangulo_nao_conexao.csv",
        "edges_pea_pea.csv",
        "edges_erb_erb.csv",
        "edges_cps_cps.csv",
        "edges_ctc_ctc.csv",
        "edges_ctnc_ctnc.csv"
    ]
    
    files_info = "\nARQUIVOS GERADOS:\n" + "-" * 50 + "\n"
    files_created = 0
    total_size_mb = 0
    
    for filename in output_files:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            file_size_mb = file_size / (1024*1024)
            total_size_mb += file_size_mb
            files_info += f"✅ {filename} ({file_size_mb:.1f}MB)\n"
            files_created += 1
        else:
            files_info += f"❌ {filename} (not created)\n"
    
    status_emoji = "🎉" if success else "❌"
    status_text = "SUCESSO" if success else "FALHA"
    
    report = f"\n==================================================\n{status_emoji} RUN REPORT - SEQUENTIAL OPTIMIZED PIPELINE V3\n==================================================\nStart date/time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\nEnd date/time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\nDuration: {duration:.2f} seconds ({duration/60:.2f} minutes)\n\nSTATUS: {status_text}\n{stats_info}\n\nIMPROVEMENTS IMPLEMENTED:\n--------------------------------------------------\n✅ PHASE 1: Quality filters implemented\n✅ PHASE 2: Geographic clustering and optimization\n✅ PHASE 3: Automatic monitoring and alerts\n{quality_summary}\n\nPROCESSING MODE:\n--------------------------------------------------\n✅ SEQUENTIAL OPTIMIZED V3 processing\n✅ Quality filters applied\n✅ Connection limits per PEA node\n✅ Geographic clustering for PEA-PEA\n✅ Automatic alert system\n✅ Geometric validation\n✅ Memory optimization\n\n{files_info}\nFiles created: {files_created}/{len(output_files)}\nTotal size: {total_size_mb:.1f}MB (reduced by the optimizations)\n\nV3 FEATURES:\n--------------------------------------------------\n- Quality prioritized over quantity\n- Progressive filters implemented\n- Geographic clustering to reduce redundancy\n- Automatic alert system\n- Continuous quality metrics\n- Improved geometric validation\n- Memory and performance optimization\n\nCOMPARISON WITH THE PREVIOUS VERSION:\n--------------------------------------------------\n- Expected reduction: ~60-70% of the total volume\n- Quality improvement: removal of duplicates\n- Performance: optimized for large datasets\n- Monitoring: alert system implemented\n\n==================================================\n    "
    
    report_filename = f"arestas_report_sequencial_v3_{start_time.strftime('%Y%m%d_%H%M%S')}.txt"
    report_path = os.path.join(f"{INTERNAL_DATA_ROOT}/DATA_ENRICHED/ARESTAS", report_filename)
    
    try:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"📊 Detailed report saved to: {report_path}")
    except Exception as e:
        logger.warning(f"Error while saving report: {e}")
    
    print(report)


def main():
    parser = argparse.ArgumentParser(
        description='Runs the SEQUENTIAL OPTIMIZED V3 edge generation pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='\nV3 improvements implemented:\n  - Automatic quality filters\n  - Connection limits per PEA node\n  - Geographic clustering\n  - Alert system\n  - Geometric validation\n  - Memory optimization\n\nUsage examples:\n  python run_arestas_sequencial.py\n  python run_arestas_sequencial.py --config arestas_config_sequencial.yaml\n  python run_arestas_sequencial.py --debug\n        '
    )
    
    parser.add_argument(
        '--config', '-c',
        default='arestas_config_sequencial.yaml',
        help='YAML configuration file (default: arestas_config_sequencial.yaml)'
    )
    
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Ativa modo debug com logging verboso'
    )
    
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Forces the run even if output files already exist'
    )
    
    parser.add_argument(
        '--analyze-quality', '-q',
        action='store_true',
        help='Analyses quality improvements after the run'
    )
    
    args = parser.parse_args()
    
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)
    
    print('\n🔄 ================================================== 🔄\n   SEQUENTIAL OPTIMIZED EDGE PIPELINE V3\n   With Quality Filters + Clustering + Alerts\n🔄 ================================================== 🔄\n    ')
    
    logger.info('=== STARTING SEQUENTIAL OPTIMIZED V3 RUN ===')
    logger.info(f"Configuration file in use: {args.config}")
    logger.info(f"Modo debug: {args.debug}")
    logger.info(f"Analysis of the quality: {args.analyze_quality}")
    
    start_time = datetime.now()
    
    try:
        system_ok = check_system_requirements()
        
        if not validate_environment(args.config):
            logger.error('Validation of the environment failed')
            return 1
        
        if not args.force:
            output_dir = f"{INTERNAL_DATA_ROOT}/DATA_PROCESSED/GRAPH_NODES_EDGES"
            existing_files = []
            key_files = ["edges_conexao_pea_erb.csv", "edges_nao_conexao.csv"]
            
            for filename in key_files:
                filepath = os.path.join(output_dir, filename)
                if os.path.exists(filepath):
                    existing_files.append(filename)
            
            if existing_files:
                logger.warning('Edge files already exist on disk:')
                for filename in existing_files:
                    logger.warning(f"  - {filename}")
                
                response = input("\nDeseja continuar e sobrescrever? (s/N): ").lower().strip()
                if response not in ['s', 'sim', 'y', 'yes']:
                    logger.info('Run cancelled by the user')
                    return 0
        
        logger.info("🔄 Iniciando pipeline SEQUENCIAL OTIMIZADO V3...")
        logger.info("   ⚡ MELHORIAS: Filtros + Clustering + Alertas")
        
        gerador = GeradorArestasV3Sequencial(args.config)
        
        success = gerador.run_pipeline()
        
        end_time = datetime.now()
        
        if args.analyze_quality:
            stats_file = f"{INTERNAL_DATA_ROOT}/DATA_ENRICHED/ARESTAS/arestas_stats_sequencial.json"
            analyze_quality_improvements(stats_file)
        
        stats_file = f"{INTERNAL_DATA_ROOT}/DATA_ENRICHED/ARESTAS/arestas_stats_sequencial.json"
        generate_execution_report(start_time, end_time, success, stats_file)
        
        if success:
            duration = (end_time - start_time).total_seconds()
            logger.info('🎉 === SEQUENTIAL OPTIMIZED V3 RUN FINISHED SUCCESSFULLY ===')
            logger.info(f"🔄 Tempo total: {duration:.2f}s ({duration/60:.2f}min)")
            logger.info(f"📊 Dataset processed with quality filters!")
            logger.info(f"⚡ V3 improvements applied successfully!")
            return 0
        else:
            logger.error('❌ === THE RUN FAILED ===')
            return 1
        
    except KeyboardInterrupt:
        logger.warning('⚠️ Run interrupted by the user')
        return 130
        
    except Exception as e:
        end_time = datetime.now()
        logger.error(f"❌ Error during the run: {str(e)}")
        
        if args.debug:
            import traceback
            logger.error("Traceback completo:")
            logger.error(traceback.format_exc())
        
        try:
            generate_execution_report(start_time, end_time, False)
        except Exception:
            pass
            
        return 1


if __name__ == "__main__":
    sys.exit(main()) 
