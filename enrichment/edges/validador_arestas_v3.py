# -*- coding: utf-8 -*-
"""Validation of the edge files written by the V3 edge generator."""

INTERNAL_DATA_ROOT = __import__("os").environ.get("ARPIA_ROOT", ".")
import pandas as pd
import numpy as np
import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Tuple
import json

class ValidadorArestasV3:
    
    def __init__(self, input_dir: str = f"{INTERNAL_DATA_ROOT}/DATA_PROCESSED/GRAPH_NODES_EDGES"):
        self.input_dir = input_dir
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        self.expected_files = {
            'edges_conexao_pea_erb.csv': 'PEA-ERB connection edges (file)',
            'edges_nao_conexao.csv': 'Arestas de não-conexão',
            'edges_intra_triangulo_conexao.csv': 'Arestas intra-triângulo conexão',
            'edges_intra_triangulo_nao_conexao.csv': 'Arestas intra-triângulo não-conexão',
            'edges_pea_pea.csv': 'Arestas PEA-PEA',
            'edges_erb_erb.csv': 'Arestas ERB-ERB',
            'edges_cps_cps.csv': 'Arestas CPS-CPS',
            'edges_ctc_ctc.csv': 'Arestas CTC-CTC',
            'edges_ctnc_ctnc.csv': 'Arestas CTNC-CTNC'
        }
        
        self.validation_results = {
            'files_found': {},
            'data_quality': {},
            'consistency_checks': {},
            'statistics': {},
            'errors': [],
            'warnings': []
        }
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        
    def check_files_exist(self) -> bool:
        self.logger.info('🔍 Checking that the files exist...')
        
        all_exist = True
        
        for filename, description in self.expected_files.items():
            filepath = os.path.join(self.input_dir, filename)
            exists = os.path.exists(filepath)
            
            self.validation_results['files_found'][filename] = {
                'exists': exists,
                'description': description,
                'size_bytes': os.path.getsize(filepath) if exists else 0
            }
            
            if exists:
                self.logger.info(f"   ✅ {filename}: {description}")
            else:
                self.logger.warning(f"   ❌ {filename}: {description} (not found)")
                all_exist = False
        
        return all_exist
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        self.logger.info('📂 Loading the edge data...')
        
        data = {}
        
        for filename in self.expected_files.keys():
            filepath = os.path.join(self.input_dir, filename)
            
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    data[filename.replace('.csv', '')] = df
                    self.logger.info(f"   ✅ {filename}: {len(df)} registros carregados")
                    
                    self.validation_results['statistics'][filename] = {
                        'records': len(df),
                        'columns': list(df.columns),
                        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
                    }
                    
                except Exception as e:
                    self.logger.error(f"   ❌ Error while loading {filename}: {e}")
                    self.validation_results['errors'].append(f"Error while loading {filename}: {e}")
            
        return data
        
    def validate_id_formats(self, data: Dict[str, pd.DataFrame]) -> bool:
        self.logger.info("🔢 Validando formatos de IDs...")
        
        id_patterns = {
            'PEA': r'^PEA_PEA_\d+$',
            'ERB': r'^ERB_\d{6}$',
            'CPS': r'^CPS_\d+_C\d+$',
            'CTC': r'^CTC_PEA_\d+_\d+_\d+_C\d+$',
            'CTNC': r'^CTNC_PEA_\d+_\d+_\d+_C\d+$'
        }
        
        all_valid = True
        
        for file_key, df in data.items():
            if 'source_id' in df.columns and 'target_id' in df.columns:
                all_ids = pd.concat([df['source_id'], df['target_id']]).unique()
                
                invalid_ids = []
                for id_val in all_ids:
                    if pd.isna(id_val):
                        continue
                        
                    id_str = str(id_val)
                    is_valid = False
                    
                    for prefix, pattern in id_patterns.items():
                        if id_str.startswith(prefix):
                            import re
                            if re.match(pattern, id_str):
                                is_valid = True
                                break
                    
                    if not is_valid:
                        invalid_ids.append(id_str)
                
                if invalid_ids:
                    all_valid = False
                    self.logger.warning(f"   ⚠️ {file_key}: {len(invalid_ids)} invalid IDs found")
                    self.validation_results['warnings'].append(
                        f"{file_key}: {len(invalid_ids)} invalid IDs (e.g. {invalid_ids[:3]})"
                    )
                else:
                    self.logger.info(f"   ✅ {file_key}: all IDs are valid")
        
        self.validation_results['consistency_checks']['id_formats'] = all_valid
        return all_valid
        
    def check_duplicate_edges(self, data: Dict[str, pd.DataFrame]) -> bool:
        self.logger.info('🔄 Checking for duplicate edges...')
        
        all_clean = True
        
        for file_key, df in data.items():
            if 'source_id' in df.columns and 'target_id' in df.columns:
                duplicate_count = df.duplicated(subset=['source_id', 'target_id']).sum()
                
                if duplicate_count > 0:
                    all_clean = False
                    self.logger.warning(f"   ⚠️ {file_key}: {duplicate_count} arestas duplicadas")
                    self.validation_results['warnings'].append(
                        f"{file_key}: {duplicate_count} arestas duplicadas"
                    )
                else:
                    self.logger.info(f"   ✅ {file_key}: Sem duplicatas")
                
                self.validation_results['data_quality'][file_key] = {
                    'duplicate_edges': duplicate_count,
                    'unique_edges': len(df) - duplicate_count
                }
        
        self.validation_results['consistency_checks']['no_duplicates'] = all_clean
        return all_clean
        
    def validate_referential_integrity(self, data: Dict[str, pd.DataFrame]) -> bool:
        self.logger.info("🔗 Validando integridade referencial...")
        
        all_ids = {
            'pea': set(),
            'erb': set(),
            'cps': set(),
            'ctc': set(),
            'ctnc': set()
        }
        
        for file_key, df in data.items():
            if 'source_id' in df.columns and 'target_id' in df.columns:
                all_edge_ids = pd.concat([df['source_id'], df['target_id']]).unique()
                
                for id_val in all_edge_ids:
                    if pd.isna(id_val):
                        continue
                        
                    id_str = str(id_val)
                    if id_str.startswith('PEA_PEA_'):
                        all_ids['pea'].add(id_str)
                    elif id_str.startswith('ERB_'):
                        all_ids['erb'].add(id_str)
                    elif id_str.startswith('CPS_'):
                        all_ids['cps'].add(id_str)
                    elif id_str.startswith('CTC_'):
                        all_ids['ctc'].add(id_str)
                    elif id_str.startswith('CTNC_'):
                        all_ids['ctnc'].add(id_str)
        
        issues = []
        
        for triangle_type in ['ctc', 'ctnc']:
            for triangle_id in all_ids[triangle_type]:
                parts = triangle_id.split('_')
                if len(parts) >= 6:
                    pea_part = f"PEA_PEA_{parts[2]}"
                    erb_part = f"ERB_{parts[3]}"
                    cps_part = f"CPS_{parts[4]}_{parts[5]}"
                    
                    if pea_part not in all_ids['pea']:
                        issues.append(f"PEA {pea_part} referenced by {triangle_id} not found")
                    
                    if erb_part not in all_ids['erb']:
                        issues.append(f"ERB {erb_part} referenced by {triangle_id} not found")
                    
                    if cps_part not in all_ids['cps']:
                        issues.append(f"CPS {cps_part} referenced by {triangle_id} not found")
        
        if issues:
            self.logger.warning(f"   ⚠️ {len(issues)} referential integrity problems found")
            for issue in issues[:5]:
                self.logger.warning(f"      - {issue}")
            if len(issues) > 5:
                self.logger.warning(f"      ... and {len(issues) - 5} more problems")
            
            self.validation_results['warnings'].extend(issues)
            
        else:
            self.logger.info("   ✅ Integridade referencial validada")
        
        for node_type, ids in all_ids.items():
            self.logger.info(f"   📊 {node_type.upper()}: {len(ids)} unique IDs")
        
        self.validation_results['consistency_checks']['referential_integrity'] = len(issues) == 0
        self.validation_results['statistics']['node_counts'] = {k: len(v) for k, v in all_ids.items()}
        
        return len(issues) == 0
        
    def analyze_edge_distributions(self, data: Dict[str, pd.DataFrame]):
        self.logger.info('📊 Analysing the edge distributions...')
        
        distributions = {}
        
        for file_key, df in data.items():
            file_stats = {}
            
            file_stats['total_edges'] = len(df)
            
            if 'distancia_km' in df.columns:
                distances = df['distancia_km'].dropna()
                if len(distances) > 0:
                    file_stats['distance_stats'] = {
                        'mean': float(distances.mean()),
                        'median': float(distances.median()),
                        'std': float(distances.std()),
                        'min': float(distances.min()),
                        'max': float(distances.max())
                    }
            elif 'distancia_m' in df.columns:
                distances = df['distancia_m'].dropna() / 1000
                if len(distances) > 0:
                    file_stats['distance_stats'] = {
                        'mean': float(distances.mean()),
                        'median': float(distances.median()),
                        'std': float(distances.std()),
                        'min': float(distances.min()),
                        'max': float(distances.max())
                    }
            
            operadora_cols = [col for col in df.columns if 'operadora' in col.lower()]
            if operadora_cols:
                for col in operadora_cols:
                    if col in df.columns:
                        operadora_dist = df[col].value_counts().head(10).to_dict()
                        file_stats[f'{col}_distribution'] = operadora_dist
            
            if 'fator_punitivo' in df.columns:
                fatores = df['fator_punitivo'].dropna()
                if len(fatores) > 0:
                    file_stats['fator_punitivo_stats'] = {
                        'mean': float(fatores.mean()),
                        'median': float(fatores.median()),
                        'unique_values': fatores.unique().tolist()
                    }
            
            distributions[file_key] = file_stats
            
            if file_stats['total_edges'] > 0:
                self.logger.info(f"   📈 {file_key}: {file_stats['total_edges']:,} arestas")
                
                if 'distance_stats' in file_stats:
                    dist_stats = file_stats['distance_stats']
                    self.logger.info(f"      Distância média: {dist_stats['mean']:.2f}km")
        
        self.validation_results['statistics']['distributions'] = distributions
        
    def compare_with_v2_pattern(self, data: Dict[str, pd.DataFrame]) -> bool:
        self.logger.info('🔄 Comparing with the V2 standard...')
        
        expected_structures = {
            'edges_conexao_pea_erb': ['source_id', 'target_id', 'id_pea', 'id_erb'],
            'edges_nao_conexao': ['source_id', 'target_id', 'id_pea', 'id_erb_nc'],
            'edges_intra_triangulo_conexao': ['source_id', 'target_id', 'tipo_aresta', 'id_triangulo'],
            'edges_intra_triangulo_nao_conexao': ['source_id', 'target_id', 'tipo_aresta', 'id_triangulo'],
            'edges_pea_pea': ['source_id', 'target_id', 'tipo_aresta', 'distancia_km'],
            'edges_erb_erb': ['source_id', 'target_id', 'tipo_aresta', 'distancia_km'],
            'edges_cps_cps': ['source_id', 'target_id', 'tipo_aresta', 'distancia_km']
        }
        
        compatible = True
        
        for file_key, expected_cols in expected_structures.items():
            if file_key in data:
                df = data[file_key]
                missing_cols = [col for col in expected_cols if col not in df.columns]
                
                if missing_cols:
                    compatible = False
                    self.logger.warning(f"   ⚠️ {file_key}: Colunas faltantes {missing_cols}")
                    self.validation_results['warnings'].append(
                        f"{file_key}: Columns missing from the V2 standard: {missing_cols}"
                    )
                else:
                    self.logger.info(f"   ✅ {file_key}: Compatible with the V2 standard")
        
        self.validation_results['consistency_checks']['v2_compatibility'] = compatible
        return compatible
        
    def generate_report(self) -> str:
        self.logger.info('📋 Generating the validation report...')
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        total_files = len(self.expected_files)
        files_found = sum(1 for f in self.validation_results['files_found'].values() if f['exists'])
        total_edges = sum(
            self.validation_results['statistics'].get(f, {}).get('records', 0) 
            for f in self.expected_files.keys()
        )
        
        all_checks_passed = all(self.validation_results['consistency_checks'].values())
        status = "APROVADO" if all_checks_passed and len(self.validation_results['errors']) == 0 else "COM AVISOS"
        
        report = f"\n==============================================================\nVALIDATION REPORT - ARPIA V3 EDGES\n==============================================================\nDate/time: {timestamp}\nDirectory: {self.input_dir}\n\nOVERALL STATUS: {status}\n\nFILE SUMMARY:\n--------------------------------------------------------------\nExpected files: {total_files}\nFiles found: {files_found}\nTotal edges: {total_edges:,}\n\nPROCESSED FILES:\n--------------------------------------------------------------"
        
        for filename, info in self.validation_results['files_found'].items():
            status_icon = "✅" if info['exists'] else "❌"
            size_mb = info['size_bytes'] / 1024 / 1024 if info['size_bytes'] > 0 else 0
            records = self.validation_results['statistics'].get(filename, {}).get('records', 0)
            
            report += f"\n{status_icon} {filename}: {records:,} registros ({size_mb:.1f} MB)"
        
        report += f"\n\nCHECKS OF THE QUALITY:\n--------------------------------------------------------------"
        
        for check, passed in self.validation_results['consistency_checks'].items():
            status_icon = "✅" if passed else "❌"
            report += f"\n{status_icon} {check.replace('_', ' ').title()}"
        
        if self.validation_results['statistics'].get('node_counts'):
            report += f"\n\nSTATISTICS OF THE NODES:\n--------------------------------------------------------------"
            for node_type, count in self.validation_results['statistics']['node_counts'].items():
                report += f"\n{node_type.upper()}: {count:,} unique nodes"
        
        if self.validation_results['warnings']:
            report += f"""

AVISOS ({len(self.validation_results['warnings'])}):
--------------------------------------------------------------"""
            for warning in self.validation_results['warnings'][:10]:
                report += f"\n⚠️ {warning}"
            if len(self.validation_results['warnings']) > 10:
                report += f"\n... and {len(self.validation_results['warnings']) - 10} more warnings"
        
        if self.validation_results['errors']:
            report += f"""

ERROS ({len(self.validation_results['errors'])}):
--------------------------------------------------------------"""
            for error in self.validation_results['errors']:
                report += f"\n❌ {error}"
        
        report += f"\n\nDISTRIBUTIONS OF THE EDGES:\n--------------------------------------------------------------"
        
        distributions = self.validation_results['statistics'].get('distributions', {})
        for file_key, stats in distributions.items():
            if stats.get('total_edges', 0) > 0:
                report += f"\n{file_key}: {stats['total_edges']:,} arestas"
                if 'distance_stats' in stats:
                    dist = stats['distance_stats']
                    report += f" (dist. média: {dist['mean']:.2f}km)"
        
        report += f"""

==============================================================
"""
        
        return report
        
    def save_report(self, report: str, output_file: str = None):
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"validacao_arestas_v3_{timestamp}.txt"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            self.logger.info(f"📄 Report saved to: {output_file}")
        except Exception as e:
            self.logger.error(f"Error while saving report: {e}")
            
    def save_validation_json(self, output_file: str = None):
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"validacao_arestas_v3_{timestamp}.json"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.validation_results, f, indent=2, ensure_ascii=False)
            self.logger.info(f"📊 Validation data saved to: {output_file}")
        except Exception as e:
            self.logger.error(f"Error while saving JSON data: {e}")
        
    def run_validation(self, generate_report: bool = True) -> bool:
        self.logger.info('🚀 Starting validation of the V3 edges')
        
        files_exist = self.check_files_exist()
        
        if not files_exist:
            self.logger.warning('⚠️ Not all expected files were found')
        
        data = self.load_data()
        
        if not data:
            self.logger.error('❌ No edge file was loaded')
            return False
        
        self.validate_id_formats(data)
        self.check_duplicate_edges(data)
        self.validate_referential_integrity(data)
        self.analyze_edge_distributions(data)
        self.compare_with_v2_pattern(data)
        
        if generate_report:
            report = self.generate_report()
            print(report)
            
            self.save_report(report)
            self.save_validation_json()
        
        all_critical_passed = all(
            self.validation_results['consistency_checks'].get(check, False)
            for check in ['id_formats', 'no_duplicates']
        )
        
        no_critical_errors = len(self.validation_results['errors']) == 0
        
        success = all_critical_passed and no_critical_errors
        
        if success:
            self.logger.info('✅ Validation finished successfully')
        else:
            self.logger.warning('⚠️ Validation finished with warnings')
        
        return success


def main():
    parser = argparse.ArgumentParser(
        description='Validates the quality of the edges generated by the V3 pipeline'
    )
    
    parser.add_argument(
        '--input-dir', '-i',
        default=f"{INTERNAL_DATA_ROOT}/DATA_PROCESSED/GRAPH_NODES_EDGES",
        help='Directory with the edge files'
    )
    
    parser.add_argument(
        '--report', '-r',
        action='store_true',
        help='Gerar relatório detalhado'
    )
    
    args = parser.parse_args()
    
    validador = ValidadorArestasV3(args.input_dir)
    success = validador.run_validation(generate_report=args.report)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 
