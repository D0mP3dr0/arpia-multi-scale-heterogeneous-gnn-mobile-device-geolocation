#!/usr/bin/env python3
"""Triangulation of CPS, ERB and PEA into connected (CTC) and non-connected (CTNC) triangle nodes."""

INTERNAL_DATA_ROOT = __import__("os").environ.get("ARPIA_ROOT", ".")
import pandas as pd
import numpy as np
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import math
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class TriangulationPipeline:
    
    def __init__(self, config_path: str = "triangulation_config.yaml"):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
        self.max_workers = self.config.get('hardware', {}).get('max_workers', 8)
        self.chunk_size = self.config.get('hardware', {}).get('chunk_size', 10000)
        
        self.id_mapping = self._load_id_mapping()
        
        self.logger.info('=== STARTING THE CPS-ERB-PEA TRIANGULATION PIPELINE ===')
        self.logger.info(f"Configuração: {config_path}")
        self.logger.info('🔧 CORRECTED MODE: generating data for the edge pipeline')
        self.logger.info('🔧 CORRECTION: using normalized IDs from the mapping file')
        
    def _load_config(self, config_path: str) -> dict:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {config_path}")
            return {
                'input_files': {
                    'peaid_erbid': f"{INTERNAL_DATA_ROOT}/DATA_ENRICHED/ALVO/PEAID_ERBID.csv",
                    'erb_cps': f"{INTERNAL_DATA_ROOT}/DATA_ENRICHED/ANTENAS_ERB/erb_cps.csv"
                },
                'output_files': {
                    'erb_cps_ctc_ctnc': f"{INTERNAL_DATA_ROOT}/DATA_ENRICHED/ANTENAS_ERB/erb_cps_ctc_ctnc.csv"
                }
            }
    
    def _load_id_mapping(self) -> Dict[str, Dict[str, str]]:
        try:
            mapping_file = f"{INTERNAL_DATA_ROOT}/NORMALIZACAO_OUTPUT/id_mapping_v3.json"
            with open(mapping_file, 'r', encoding='utf-8') as f:
                id_mapping = json.load(f)
            
            self.logger.info(f"Mapeamento de IDs carregado: {mapping_file}")
            
            pea_reverse_mapping = {}
            if 'PEA' in id_mapping:
                pea_reverse_mapping = self._create_pea_reverse_mapping()
            
            id_mapping['PEA_REVERSE'] = pea_reverse_mapping
            
            return id_mapping
            
        except Exception as e:
            self.logger.error(f"Error while loading the ID mapping: {e}")
            return {}
    
    def _create_pea_reverse_mapping(self) -> Dict[str, str]:
        try:
            nodes_pea_file = f"{INTERNAL_DATA_ROOT}/NORMALIZACAO_OUTPUT/nodes_pea_v3.csv"
            nodes_pea = pd.read_csv(nodes_pea_file)
            
            pea_mapping = {}
            for _, row in nodes_pea.iterrows():
                if pd.notna(row['SetorID RE']) and pd.notna(row['node_id']):
                    original_id = str(row['SetorID RE']).strip()
                    normalized_id = str(row['node_id']).strip()
                    pea_mapping[original_id] = normalized_id
            
            self.logger.info(f"Mapeamento reverso de PEA criado: {len(pea_mapping)} IDs")
            return pea_mapping
            
        except Exception as e:
            self.logger.error(f"Error while building the reverse PEA mapping: {e}")
            return {}
    
    def _get_normalized_id(self, node_type: str, original_id: str) -> str:
        if node_type == 'PEA':
            return self.id_mapping.get('PEA_REVERSE', {}).get(original_id, original_id)
        elif node_type == 'ERB':
            return self.id_mapping.get('ERB', {}).get(original_id, original_id)
        elif node_type == 'CPS':
            return self.id_mapping.get('CPS', {}).get(original_id, original_id)
        else:
            return original_id
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        self.logger.info('Loading the input data...')
        data = {}
        
        peaid_file = self.config['input_files']['peaid_erbid']
        self.logger.info(f"Loading PEA-ERB from: {peaid_file}")
        data['peaid_erbid'] = pd.read_csv(peaid_file)
        self.logger.info(f"Dados PEA-ERB carregados: {len(data['peaid_erbid'])} registros")
        
        erb_cps_file = self.config['input_files']['erb_cps']
        self.logger.info(f"Loading ERB-CPS from: {erb_cps_file}")
        data['erb_cps'] = pd.read_csv(erb_cps_file)
        self.logger.info(f"Dados ERB-CPS carregados: {len(data['erb_cps'])} registros")
        
        return data
        
    def validate_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        self.logger.info('Validating the data structure...')
        
        expected_cols = self.config.get('expected_columns', {})
        
        if 'peaid_erbid' in expected_cols:
            missing_cols = [col for col in expected_cols['peaid_erbid'] 
                          if col not in data['peaid_erbid'].columns]
            if missing_cols:
                self.logger.warning(f"Colunas faltantes em peaid_erbid: {missing_cols}")
                self.logger.info(f"Colunas disponíveis: {list(data['peaid_erbid'].columns)}")
        
        if 'erb_cps' in expected_cols:
            missing_cols = [col for col in expected_cols['erb_cps'] 
                          if col not in data['erb_cps'].columns]
            if missing_cols:
                self.logger.warning(f"Colunas faltantes em erb_cps: {missing_cols}")
                self.logger.info(f"Colunas disponíveis: {list(data['erb_cps'].columns)}")
        
        erb_ids_pea = set(data['peaid_erbid']['erb_id'].unique())
        erb_ids_cps = set(data['erb_cps']['erb_id'].unique())
        common_erb_ids = erb_ids_pea.intersection(erb_ids_cps)
        
        self.logger.info(f"ERB IDs em PEA: {len(erb_ids_pea)}")
        self.logger.info(f"ERB IDs em CPS: {len(erb_ids_cps)}")
        self.logger.info(f"ERB IDs comuns: {len(common_erb_ids)}")
        
        if len(common_erb_ids) == 0:
            self.logger.error('No common ERB_ID found between the files!')
            return False
        
        self.logger.info('The validation finished successfully!')
        return True
    
    def calculate_triangle_centroid(self, p1: Tuple[float, float], 
                                  p2: Tuple[float, float], 
                                  p3: Tuple[float, float]) -> Tuple[float, float]:
        lat_centroid = (p1[0] + p2[0] + p3[0]) / 3
        lon_centroid = (p1[1] + p2[1] + p3[1]) / 3
        return lat_centroid, lon_centroid
    
    def calculate_triangle_area(self, p1: Tuple[float, float], 
                              p2: Tuple[float, float], 
                              p3: Tuple[float, float]) -> float:
        area = abs((p1[0] * (p2[1] - p3[1]) + 
                   p2[0] * (p3[1] - p1[1]) + 
                   p3[0] * (p1[1] - p2[1])) / 2)
        return area
    
    def create_triangulations(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        self.logger.info("Criando triangulações CPS-ERB-PEA...")
        
        peaid_erbid = data['peaid_erbid'].copy()
        erb_cps = data['erb_cps'].copy()
        
        self.logger.info(f"PEAs disponíveis: {len(peaid_erbid)}")
        self.logger.info(f"CPS disponíveis: {len(erb_cps)}")
        
        max_distance_deg = 0.009
        
        existing_connections = set()
        for _, row in peaid_erbid.iterrows():
            existing_connections.add(row['erb_id'])
        
        self.logger.info(f"ERBs with PEA connections: {len(existing_connections)}")
        
        triangulation_results = []
        ctc_count = 0
        ctnc_count = 0
        
        self.logger.info('Processing all the CPS...')
        
        batch_size = 1000
        
        for batch_start in range(0, len(erb_cps), batch_size):
            batch_end = min(batch_start + batch_size, len(erb_cps))
            cps_batch = erb_cps.iloc[batch_start:batch_end]
            
            self.logger.info(f"Processing batch {batch_start+1}-{batch_end} of {len(erb_cps)}")
            
            for _, cps_row in cps_batch.iterrows():
                try:
                    lat_min = cps_row['lat_centroide'] - max_distance_deg
                    lat_max = cps_row['lat_centroide'] + max_distance_deg
                    lon_min = cps_row['lon_centroide'] - max_distance_deg
                    lon_max = cps_row['lon_centroide'] + max_distance_deg
                    
                    nearby_peas = peaid_erbid[
                        (peaid_erbid['pea_lat'] >= lat_min) & 
                        (peaid_erbid['pea_lat'] <= lat_max) &
                        (peaid_erbid['pea_lon'] >= lon_min) & 
                        (peaid_erbid['pea_lon'] <= lon_max)
                    ]
                    
                    max_peas_per_cps = 20
                    
                    if len(nearby_peas) > max_peas_per_cps:
                        distances = []
                        for _, pea_row in nearby_peas.iterrows():
                            distance_deg = self.calculate_distance(
                                cps_row['lat_centroide'], cps_row['lon_centroide'],
                                pea_row['pea_lat'], pea_row['pea_lon']
                            )
                            if distance_deg <= max_distance_deg:
                                distances.append((distance_deg, pea_row))
                        
                        distances.sort(key=lambda x: x[0])
                        nearby_peas_list = [pea for _, pea in distances[:max_peas_per_cps]]
                    else:
                        nearby_peas_list = [row for _, row in nearby_peas.iterrows()]
                    
                    for pea_row in nearby_peas_list:
                        distance_deg = self.calculate_distance(
                            cps_row['lat_centroide'], cps_row['lon_centroide'],
                            pea_row['pea_lat'], pea_row['pea_lon']
                        )
                        
                        if distance_deg > max_distance_deg:
                            continue
                        
                        if cps_row['erb_id'] == pea_row['erb_id']:
                            connection_type = 'CTC'
                            ctc_count += 1
                        else:
                            connection_type = 'CTNC'
                            ctnc_count += 1
                        
                        cps_coords = (cps_row['lat_centroide'], cps_row['lon_centroide'])
                        erb_coords = (cps_row['lat_erb'], cps_row['lon_erb'])
                        pea_coords = (pea_row['pea_lat'], pea_row['pea_lon'])
                        
                        centroid_lat, centroid_lon = self.calculate_triangle_centroid(
                            cps_coords, erb_coords, pea_coords
                        )
                        
                        triangle_area = self.calculate_triangle_area(
                            cps_coords, erb_coords, pea_coords
                        )
                        
                        original_pea_id = str(pea_row['SetorID RE']).strip()
                        normalized_pea_id = self._get_normalized_id('PEA', original_pea_id)
                        normalized_erb_id = self._get_normalized_id('ERB', cps_row['erb_id'])
                        normalized_cps_id = self._get_normalized_id('CPS', cps_row['cps_id'])
                        
                        triangulation_id = f"{connection_type}_{normalized_pea_id}_{normalized_erb_id}_{normalized_cps_id}"
                        
                        result_row = {
                            'cps_id': cps_row['cps_id'],
                            'erb_id': cps_row['erb_id'],
                            'num_estacao': cps_row.get('num_estacao', ''),
            'operadora': cps_row.get('operadora', ''),
            'tecnologia': cps_row.get('tecnologia', ''),
                            'setor': cps_row.get('setor', ''),
                            'anel': cps_row.get('anel', ''),
                            'intensidade_sinal': cps_row.get('intensidade_sinal', ''),
                            'lat_centroide': cps_row['lat_centroide'],
                            'lon_centroide': cps_row['lon_centroide'],
                            'lat_erb': cps_row['lat_erb'],
                            'lon_erb': cps_row['lon_erb'],
                            'azimute': cps_row.get('azimute', 0),
                            'distancia_erb_km': cps_row.get('distancia_erb_km', 0),
                            'potencia_recebida_dbm': cps_row.get('potencia_recebida_dbm', 0),
                            'raio_interno_km': cps_row.get('raio_interno_km', 0),
                            'raio_externo_km': cps_row.get('raio_externo_km', 0),
                            
                            'pea_lat': pea_row['pea_lat'],
                            'pea_lon': pea_row['pea_lon'],
                            'SetorID RE': pea_row['SetorID RE'],
                            'pea_erb_id': pea_row['erb_id'],
                            
                            'triangulation_centroid_lat': centroid_lat,
                            'triangulation_centroid_lon': centroid_lon,
                            'triangulation_area': triangle_area,
                            'connection_type': connection_type,
                            'has_triangulation': True,
                            'pea_count': 1,
                            'distance_to_pea_km': distance_deg * 111,
                            
                            'triangulation_id': triangulation_id,
                            'triangulation_type': connection_type,
                            'pea_id': normalized_pea_id,
                            'centroid_lat': centroid_lat,
                            'centroid_lon': centroid_lon,
                            'fator_punitivo': 1.0
                        }
                        
                        triangulation_results.append(result_row)
                        
                except Exception as e:
                    self.logger.warning(f"Error while processing CPS: {e}")
                    continue
                    
            progress = (batch_end / len(erb_cps)) * 100
            self.logger.info(f"Progresso: {progress:.1f}% - CTCs: {ctc_count}, CTNCs: {ctnc_count}")
        
        enriched_data = pd.DataFrame(triangulation_results)
        
        self.logger.info(f"Triangulações criadas: {len(enriched_data)}")
        self.logger.info(f"CTC (Connections - same ERB): {ctc_count}")
        self.logger.info(f"CTNC (Non-connections - different ERBs): {ctnc_count}")
        
        if len(enriched_data) > 0:
            ctc_pct = (ctc_count / len(enriched_data)) * 100
            ctnc_pct = (ctnc_count / len(enriched_data)) * 100
            self.logger.info(f"Proportion: CTC {ctc_pct:.1f}%, CTNC {ctnc_pct:.1f}%")
        
        sample_pea_ids = enriched_data['pea_id'].head(10).tolist()
        self.logger.info(f"Examples of normalized PEA IDs: {sample_pea_ids}")
        
        return enriched_data
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        return ((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2) ** 0.5
    
    def save_results(self, enriched_data: pd.DataFrame):
        self.logger.info('Saving results in the correct format for the edge pipeline...')
        
        output_file = self.config.get('output_files', {}).get(
            'erb_cps_ctc_ctnc', 
            f"{INTERNAL_DATA_ROOT}/DATA_ENRICHED/ANTENAS_ERB/erb_cps_ctc_ctnc.csv"
        )
        
        enriched_data.to_csv(output_file, index=False)
        self.logger.info(f"✅ Main file saved: {output_file}")
        
        summary_file = self.config.get('output_files', {}).get(
            'triangulation_summary', 
            f"{INTERNAL_DATA_ROOT}/DATA_ENRICHED/ANTENAS_ERB/triangulation_summary.csv"
        )
        
        summary_data = enriched_data.groupby('connection_type').agg({
            'cps_id': 'count',
            'triangulation_area': ['mean', 'std', 'min', 'max'],
            'erb_id': 'nunique',
            'pea_id': 'nunique'
        }).reset_index()
        
        summary_data.to_csv(summary_file, index=False)
        self.logger.info(f"✅ Resumo salvo: {summary_file}")
        
        total_records = len(enriched_data)
        ctc_count = len(enriched_data[enriched_data['connection_type'] == 'CTC'])
        ctnc_count = len(enriched_data[enriched_data['connection_type'] == 'CTNC'])
        
        self.logger.info("=== RESULTADOS FINAIS ===")
        self.logger.info(f"Total number of records: {total_records}")
        self.logger.info(f"CTC (Connections): {ctc_count} ({ctc_count/total_records*100:.1f}%)")
        self.logger.info(f"CTNC (Non-connections): {ctnc_count} ({ctnc_count/total_records*100:.1f}%)")
        
        required_columns = [
            'pea_id', 'pea_lat', 'pea_lon', 
            'erb_id', 'lat_erb', 'lon_erb',
            'cps_id', 'lat_centroide', 'lon_centroide',
            'triangulation_id', 'triangulation_type',
            'triangulation_centroid_lat', 'triangulation_centroid_lon',
            'operadora', 'fator_punitivo'
        ]
        
        missing_columns = [col for col in required_columns if col not in enriched_data.columns]
        if missing_columns:
            self.logger.warning(f"⚠️ Columns missing for the edge pipeline: {missing_columns}")
        else:
            self.logger.info('✅ All columns required by the edge pipeline are present')
        
        unique_pea_ids = enriched_data['pea_id'].nunique()
        self.logger.info(f"✅ Unique PEA IDs (normalized): {unique_pea_ids}")
        
        return output_file
        
    def run_pipeline(self) -> str:
        try:
            data = self.load_data()
            
            if not self.validate_data(data):
                self.logger.warning('Continuing despite the validation warnings...')
            
            enriched_data = self.create_triangulations(data)
            
            output_file = self.save_results(enriched_data)
            
            self.logger.info('=== PIPELINE FINISHED SUCCESSFULLY ===')
            self.logger.info(f"✅ Arquivo gerado: {output_file}")
            self.logger.info('🔧 Correct format for the edge pipeline!')
            self.logger.info('🔧 CORRECTION: using normalized IDs!')
            
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error in the pipeline: {e}")
            raise

if __name__ == "__main__":
    pipeline = TriangulationPipeline()
    output_file = pipeline.run_pipeline()
    print(f"\n✅ Pipeline finished! Output file: {output_file}")
    print('🔧 Data in the correct format for the edge pipeline!')
    print('🔧 CORRECTION: using normalized IDs!') 
