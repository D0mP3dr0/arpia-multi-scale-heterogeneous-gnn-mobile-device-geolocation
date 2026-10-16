"""Target enrichment: matches GPRS events to antennas and builds the PEA records."""
import geopandas as gpd
import pandas as pd
import yaml
import os
from pyproj import Geod
import fiona

def criar_pea_id(indice, prefixo_numerico=9):
    return int(f"{prefixo_numerico}{indice:08d}")

def load_config(config_path="alvo_config.yaml"):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"Configuration '{config_path}' loaded successfully.")
        return config
    except FileNotFoundError:
        print(f"Error: configuration file '{config_path}' not found.")
        return None
    except Exception as e:
        print(f"Error while loading YAML configuration: {e}")
        return None

def list_gpkg_layers(gpkg_path):
    try:
        return fiona.listlayers(gpkg_path)
    except Exception as e:
        print(f"  ERROR while trying to list layers in '{gpkg_path}': {type(e).__name__} - {e}")
        return []

def ensure_crs(gdf, target_crs, layer_description, original_crs_fallback=None):
    if gdf is None or gdf.empty:
        return gdf
    current_crs = gdf.crs
    if current_crs is None and original_crs_fallback:
        print(f"Warning: layer '{layer_description}' has no CRS. Assuming {original_crs_fallback}.")
        try:
            gdf = gdf.set_crs(original_crs_fallback, allow_override=True)
            current_crs = gdf.crs
        except Exception as e:
            print(f"Error while setting fallback CRS {original_crs_fallback} for '{layer_description}': {e}.")
            return gdf
    if current_crs and current_crs != target_crs:
        print(f"Transforming '{layer_description}' from {current_crs} to {target_crs}.")
        try:
            gdf = gdf.to_crs(target_crs)
        except Exception as e:
            print(f"Error while transforming '{layer_description}' to CRS {target_crs}: {e}.")
    elif not current_crs and not original_crs_fallback:
        print(f"Error: '{layer_description}' has no CRS and no fallback was given.")
    return gdf

def read_gpkg_layer_alvo(gpkg_path_info, default_description="layer"):
    path = gpkg_path_info.get('path')
    layer_name = gpkg_path_info.get('layer_name')
    
    if not path or not os.path.exists(path):
        print(f"Error: file for {default_description} not found at '{path}'.")
        return gpd.GeoDataFrame()
    
    if path.lower().endswith('.csv'):
        try:
            df = pd.read_csv(path)
            print(f"CSV file '{path}' ({default_description}) read successfully.")
            
            lon_col = None
            lat_col = None
            
            for lon_name in ['Longitude', 'longitude', 'lon', 'lon_erb']:
                if lon_name in df.columns:
                    lon_col = lon_name
                    break
            
            for lat_name in ['Latitude', 'latitude', 'lat', 'lat_erb']:
                if lat_name in df.columns:
                    lat_col = lat_name
                    break
            
            if lon_col and lat_col:
                gdf = gpd.GeoDataFrame(
                    df, 
                    geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
                    crs="EPSG:4326"
                )
                print(f"Geometry created from the columns '{lon_col}' and '{lat_col}'.")
                return gdf
            else:
                print(f"Warning: the CSV has no recognizable latitude/longitude columns. Converting without geometry.")
                return gpd.GeoDataFrame(df)
        except Exception as e:
            print(f"Error while reading CSV '{path}' ({default_description}): {e}")
            return gpd.GeoDataFrame()
    
    available_layers = list_gpkg_layers(path)
    if not available_layers:
        print(f"Error: no layer found in '{path}' for {default_description}.")
        return gpd.GeoDataFrame()
    
    layer_name_to_read = layer_name
    if layer_name and layer_name not in available_layers:
        print(f"Warning: layer '{layer_name}' ({default_description}) not found. Reading the first one: '{available_layers[0]}'")
        layer_name_to_read = available_layers[0]
    elif not layer_name:
        layer_name_to_read = available_layers[0]
        print(f"Nenhuma camada especificada para {default_description}, lendo primeira: '{layer_name_to_read}'")
    
    try:
        gdf = gpd.read_file(path, layer=layer_name_to_read)
        print(f"Camada '{layer_name_to_read}' ({default_description}) lida de '{path}'.")
        return gdf
    except Exception as e:
        print(f"Error while reading '{layer_name_to_read}' ({default_description}) from '{path}': {e}")
        return gpd.GeoDataFrame()

def calculate_destination_point(lat_deg, lon_deg, bearing_deg, distance_m):
    if pd.isna(lat_deg) or pd.isna(lon_deg) or pd.isna(bearing_deg) or pd.isna(distance_m):
        return pd.NA, pd.NA
    geod = Geod(ellps="WGS84")
    lon_dest_deg, lat_dest_deg, _ = geod.fwd(lon_deg, lat_deg, bearing_deg, distance_m)
    return lat_dest_deg, lon_dest_deg

def enrich_gprs_with_pea_features(gdf_gprs_matched_and_filtered, config):
    if gdf_gprs_matched_and_filtered.empty:
        print('Input GeoDataFrame (gprs_matched_and_filtered) is empty. Skipping PEA enrichment.')
        pea_names_cfg_schema = config.get('pea_feature_names', {})
        schema_cols = list(pea_names_cfg_schema.values()) + ['geometry']
        empty_gdf = gpd.GeoDataFrame(columns=schema_cols, crs=config.get('output_crs', 'EPSG:4326'))
        for col, name in pea_names_cfg_schema.items():
            if name in empty_gdf.columns:
                if 'lat' in col or 'lon' in col or 'distancia' in col or 'duracao' in col:
                    empty_gdf[name] = pd.Series(dtype='float64')
                elif 'id' in col and 'pea_id' not in col :
                     empty_gdf[name] = pd.Series(dtype='object')
                elif 'pea_id' in col or 'raio' in col or 'azimute' in col:
                     empty_gdf[name] = pd.Series(dtype='int64')
                elif 'timestamp' in col:
                     empty_gdf[name] = pd.Series(dtype='datetime64[ms]')
                else:
                     empty_gdf[name] = pd.Series(dtype='object')
        return empty_gdf


    map_cols_gprs = config['input_gprs_gpkg']['map_columns']
    pea_names_cfg = config['pea_feature_names']
    temp_chosen_erb_id_col = 'temp_chosen_erb_id'

    required_original_gprs_keys = ['erb_ref_latitude', 'erb_ref_longitude', 'gprs_azimute',
                                   'gprs_raio_metros', 'gprs_timestamp', 'gprs_duracao']
    required_cols_from_gprs_data = [map_cols_gprs[k] for k in required_original_gprs_keys if k in map_cols_gprs]
    
    required_cols_from_matching = [
        pea_names_cfg['tipo_match_antena'],
        pea_names_cfg['operadora_evento_gprs'],
        pea_names_cfg['operadora_antena_escolhida'],
        pea_names_cfg['distancia_antena_metros'],
        temp_chosen_erb_id_col
    ]
    all_required_cols_for_enrichment = required_cols_from_gprs_data + required_cols_from_matching

    missing_actual_cols = [col for col in all_required_cols_for_enrichment if col not in gdf_gprs_matched_and_filtered.columns]
    if missing_actual_cols:
        print(f"Error: required columns missing in the input GDF for enrichment: {missing_actual_cols}")
        print(f"Colunas disponíveis: {gdf_gprs_matched_and_filtered.columns.tolist()}")
        return gpd.GeoDataFrame()

    gdf_output = gdf_gprs_matched_and_filtered.copy()
    
    print('Creating sequential PEA_IDs for the records with a match...')
    pea_ids = [criar_pea_id(i + 1) for i in range(len(gdf_output))]
    gdf_output[pea_names_cfg['pea_id']] = pea_ids
    if pea_ids:
        print(f"✅ PEA_IDs criados: {min(pea_ids)} - {max(pea_ids)}")
    else:
        print("✅ Nenhum PEA_ID criado (nenhum registro com match).")

    pea_id_col_name = pea_names_cfg['pea_id']
    gdf_output[pea_id_col_name] = gdf_output[pea_id_col_name].astype(str).apply(
        lambda x: f"PEA_{x}" if pd.notna(x) and not str(x).startswith("PEA_") else x
    )
    print(f"Column '{pea_id_col_name}' filled with the PEA ID (with prefix 'PEA_').")

    print('Computing the initial PEA coordinates...')
    try:
        lat_col_data = gdf_output[map_cols_gprs['erb_ref_latitude']].astype(float)
        lon_col_data = gdf_output[map_cols_gprs['erb_ref_longitude']].astype(float)
        az_col_data = gdf_output[map_cols_gprs['gprs_azimute']].astype(float)
        rad_col_data = gdf_output[map_cols_gprs['gprs_raio_metros']].astype(float)
    except Exception as e:
        print(f"Error while converting columns for the PEA computation: {e}.")
        return gpd.GeoDataFrame()

    pea_lats, pea_lons = [], []
    for i in range(len(gdf_output)):
        lat_pea, lon_pea = calculate_destination_point(
            lat_col_data.iloc[i], lon_col_data.iloc[i], az_col_data.iloc[i], rad_col_data.iloc[i]
        )
        pea_lats.append(lat_pea); pea_lons.append(lon_pea)
    gdf_output[pea_names_cfg['pea_lat']] = pd.to_numeric(pea_lats, errors='coerce')
    gdf_output[pea_names_cfg['pea_lon']] = pd.to_numeric(pea_lons, errors='coerce')
    print("Coordenadas iniciais do PEA calculadas.")

    print("Mapeando atributos do PEA...")
    gdf_output[pea_names_cfg['timestamp']] = gdf_output[map_cols_gprs['gprs_timestamp']]
    gdf_output[pea_names_cfg['raio_origem']] = gdf_output[map_cols_gprs['gprs_raio_metros']]
    gdf_output[pea_names_cfg['azimute_origem']] = gdf_output[map_cols_gprs['gprs_azimute']]
    gdf_output[pea_names_cfg['duracao_conexao']] = gdf_output[map_cols_gprs['gprs_duracao']]

    target_erb_id_col_name = pea_names_cfg.get('erb_id', 'erb_id')
    
    gdf_output[temp_chosen_erb_id_col] = gdf_output[temp_chosen_erb_id_col].astype(str)
    gdf_output[target_erb_id_col_name] = gdf_output[temp_chosen_erb_id_col].apply(
        lambda x: f"ERB_{x}" if pd.notna(x) and not str(x).startswith("ERB_") else x
    )
    
    print(f"Column '{target_erb_id_col_name}' filled with the ID of the chosen antenna (with prefix 'ERB_').")

    print("Atributos do PEA mapeados.")

    print("Limpando colunas desnecessárias...")
    colunas_para_remover = [temp_chosen_erb_id_col]
    for col in gdf_output.columns:
        if 'id_alvo_anonimo' in col.lower():
            colunas_para_remover.append(col)
    for col in set(colunas_para_remover):
        if col in gdf_output.columns:
            gdf_output = gdf_output.drop(columns=[col])
            print(f"   ❌ Removida coluna: {col}")

    output_crs_config = config.get('output_crs', config.get('input_crs_fallback', 'EPSG:4326'))
    if gdf_output.crs is None:
        input_fallback = config.get('input_crs_fallback', 'EPSG:4326')
        print(f"Critical warning: the PEA GDF has no CRS. Setting it to {input_fallback}.")
        gdf_output = gdf_output.set_crs(input_fallback, allow_override=True)
    if gdf_output.crs != output_crs_config:
        print(f"Converting GDF from {gdf_output.crs} to {output_crs_config}.")
        gdf_output = gdf_output.to_crs(output_crs_config)

    final_columns_ordered = []
    for key_config, name_config in pea_names_cfg.items():
        if name_config in gdf_output.columns:
            final_columns_ordered.append(name_config)
        else:
            print(f"Warning: configured column '{name_config}' (for key '{key_config}') not found in the final GDF.")

    if hasattr(gdf_output, 'geometry') and gdf_output.geometry.name not in final_columns_ordered and gdf_output.geometry.name in gdf_output.columns:
        final_columns_ordered.append(gdf_output.geometry.name)
    
    return gdf_output[final_columns_ordered]


def main_pea_enrichment_process(config_path="alvo_config.yaml"):
    config = load_config(config_path)
    if not config: return False

    input_gprs_info = config.get('input_gprs_gpkg', {})
    input_antenas_info = config.get('input_antenas_gpkg', {})
    output_pea_info = config.get('output_pea_gpkg', {})
    antenna_matching_cfg = config.get('antenna_matching', {})
    pea_names_cfg = config.get('pea_feature_names', {})
    input_crs_fallback = config.get('input_crs_fallback', 'EPSG:4326')

    gprs_op_def = input_gprs_info.get('map_columns', {}).get('gprs_operadora_definition', {})
    gprs_op_source_type = gprs_op_def.get('source_type')
    gprs_op_column_name = gprs_op_def.get('column_name')
    gprs_op_fixed_value = gprs_op_def.get('fixed_value')

    essential_paths = [input_gprs_info.get('path'), input_antenas_info.get('path'), output_pea_info.get('path')]
    essential_antennas_keys = [input_antenas_info.get('operadora_column_in_antenas'), input_antenas_info.get('id_column_in_antenas')]
    essential_matching_keys = [antenna_matching_cfg.get('max_distance_meters') is not None, antenna_matching_cfg.get('no_match_label')]
    essential_pea_names_keys_present = all(key in pea_names_cfg for key in [
        'pea_id', 'erb_id', 'pea_lat', 'pea_lon', 'timestamp', 'raio_origem', 
        'azimute_origem', 'duracao_conexao', 'tipo_match_antena', 
        'operadora_evento_gprs', 'operadora_antena_escolhida', 'distancia_antena_metros'
    ])

    if not all(essential_paths + essential_antennas_keys + essential_matching_keys) or not essential_pea_names_keys_present:
        print('Error: essential configuration incomplete. Check paths, antenna/matching definitions or pea_feature_names.')
        if not essential_pea_names_keys_present:
            print(f"  Keys missing or incorrect in pea_feature_names. Expected: {['pea_id', 'erb_id', 'pea_lat', 'pea_lon', 'timestamp', 'raio_origem', 'azimute_origem', 'duracao_conexao', 'tipo_match_antena', 'operadora_evento_gprs', 'operadora_antena_escolhida', 'distancia_antena_metros']}")
            print(f"  Presente em config: {list(pea_names_cfg.keys())}")
        return False
    if gprs_op_source_type == "COLUMN" and not gprs_op_column_name:
        print('Error: gprs_operadora_definition source_type is COLUMN, but column_name is not specified.')
        return False
    if gprs_op_source_type == "FIXED" and gprs_op_fixed_value is None:
        print('Error: gprs_operadora_definition source_type is FIXED, but fixed_value is not specified (it is None).')
        return False
    if gprs_op_source_type not in ["COLUMN", "FIXED"]:
        print(f"Error: invalid gprs_operadora_definition source_type: '{gprs_op_source_type}'.")
        return False

    MAX_DISTANCE_TOLERANCE_METERS = antenna_matching_cfg['max_distance_meters']
    priority_1_label = antenna_matching_cfg.get('priority_1_label', 'MESMA_OPERADORA')
    priority_2_label = antenna_matching_cfg.get('priority_2_label', 'MAIS_PROXIMA')
    no_match_label = antenna_matching_cfg['no_match_label']

    antena_id_col_name_source = input_antenas_info['id_column_in_antenas']
    antena_operadora_col_name = input_antenas_info['operadora_column_in_antenas']

    print("\n--- Lendo Dados GPRS ---")
    gdf_gprs = read_gpkg_layer_alvo(input_gprs_info, "Eventos GPRS")
    if gdf_gprs.empty: return False
    gdf_gprs = ensure_crs(gdf_gprs, input_crs_fallback, "Entrada GPRS", input_crs_fallback)
    if gdf_gprs.empty: return False

    temp_gprs_operator_col = "_temp_gprs_operator_val"
    if gprs_op_source_type == "COLUMN":
        if gprs_op_column_name not in gdf_gprs.columns:
            print(f"Error: GPRS operator column '{gprs_op_column_name}' not found. Columns: {gdf_gprs.columns.tolist()}")
            return False
        gdf_gprs[temp_gprs_operator_col] = gdf_gprs[gprs_op_column_name]
        print(f"Using column '{gprs_op_column_name}' for the GPRS operator.")
    elif gprs_op_source_type == "FIXED":
        gdf_gprs[temp_gprs_operator_col] = gprs_op_fixed_value
        print(f"Using fixed value '{gprs_op_fixed_value}' for the GPRS operator.")

    print('\n--- Reading Enriched Antenna Data ---')
    gdf_antenas = read_gpkg_layer_alvo(input_antenas_info, "Antenas Enriquecidas")
    if gdf_antenas.empty: return False
    gdf_antenas = ensure_crs(gdf_antenas, input_crs_fallback, "Entrada Antenas", input_crs_fallback)
    if gdf_antenas.empty: return False
    if antena_id_col_name_source not in gdf_antenas.columns or antena_operadora_col_name not in gdf_antenas.columns:
        print(f"Error: ID ('{antena_id_col_name_source}') or operator ('{antena_operadora_col_name}') columns not found in the antennas.")
        return False

    geod_calculator = Geod(ellps="WGS84")
    match_types_list, gprs_operators_list, chosen_antenna_operators_list = [], [], []
    chosen_antenna_ids_list, chosen_distances_list = [], []

    print(f"\nStarting antenna matching for {len(gdf_gprs)} GPRS events...")
    gdf_antenas_indexed = gdf_antenas.reset_index(drop=True)

    for i_gprs, gprs_event_row in gdf_gprs.iterrows():
        if i_gprs > 0 and i_gprs % 1000 == 0:
            print(f"  Processando evento GPRS {i_gprs+1}/{len(gdf_gprs)}...")

        gprs_geom = gprs_event_row.geometry
        gprs_operator_val = gprs_event_row[temp_gprs_operator_col]
        gprs_operators_list.append(gprs_operator_val)

        if gprs_geom is None or gprs_geom.is_empty:
            match_types_list.append(no_match_label)
            chosen_antenna_operators_list.append(pd.NA)
            chosen_antenna_ids_list.append(pd.NA)
            chosen_distances_list.append(pd.NA)
            continue

        candidate_antennas_details = []
        buffer_deg_approx = MAX_DISTANCE_TOLERANCE_METERS / 111000 * 1.5
        gprs_bounds_buffered = (
            gprs_geom.x - buffer_deg_approx, gprs_geom.y - buffer_deg_approx,
            gprs_geom.x + buffer_deg_approx, gprs_geom.y + buffer_deg_approx
        )
        possible_match_indices = list(gdf_antenas_indexed.sindex.intersection(gprs_bounds_buffered))

        for ant_idx in possible_match_indices:
            antena_row = gdf_antenas_indexed.iloc[ant_idx]
            antena_geom = antena_row.geometry
            if antena_geom is None or antena_geom.is_empty: continue
            
            _, _, distance_m = geod_calculator.inv(gprs_geom.x, gprs_geom.y, antena_geom.x, antena_geom.y)
            
            if distance_m <= MAX_DISTANCE_TOLERANCE_METERS:
                antena_id = antena_row[antena_id_col_name_source]
                if isinstance(antena_id, str) and antena_id.startswith('ERB_'):
                    antena_id = antena_id[4:]
                    
                candidate_antennas_details.append({
                    'id': antena_id,
                    'operator': antena_row[antena_operadora_col_name],
                    'distance': distance_m
                })

        current_match_type = no_match_label
        current_chosen_ant_op, current_chosen_ant_id, current_chosen_dist = pd.NA, pd.NA, pd.NA

        if candidate_antennas_details:
            same_operator_candidates = [
                ant for ant in candidate_antennas_details if ant['operator'] == gprs_operator_val and pd.notna(gprs_operator_val)
            ]
            if same_operator_candidates:
                best_match = min(same_operator_candidates, key=lambda x: x['distance'])
                current_match_type = priority_1_label
            else:
                best_match = min(candidate_antennas_details, key=lambda x: x['distance'])
                current_match_type = priority_2_label
            
            current_chosen_ant_op = best_match['operator']
            current_chosen_ant_id = best_match['id']
            current_chosen_dist = best_match['distance']

        match_types_list.append(current_match_type)
        chosen_antenna_operators_list.append(current_chosen_ant_op)
        chosen_antenna_ids_list.append(current_chosen_ant_id)
        chosen_distances_list.append(pd.to_numeric(current_chosen_dist, errors='coerce'))


    print('Antenna matching is now finished.')
    
    gdf_gprs_matched = gdf_gprs.copy()
    gdf_gprs_matched[pea_names_cfg['tipo_match_antena']] = match_types_list
    gdf_gprs_matched[pea_names_cfg['operadora_evento_gprs']] = gprs_operators_list
    gdf_gprs_matched[pea_names_cfg['operadora_antena_escolhida']] = chosen_antenna_operators_list
    gdf_gprs_matched[pea_names_cfg['distancia_antena_metros']] = pd.to_numeric(chosen_distances_list, errors='coerce')
    gdf_gprs_matched['temp_chosen_erb_id'] = chosen_antenna_ids_list
    
    if temp_gprs_operator_col in gdf_gprs_matched.columns:
        gdf_gprs_matched = gdf_gprs_matched.drop(columns=[temp_gprs_operator_col])

    print('\n--- Antenna Matching Statistics (Before Filtering SEM_MATCH) ---')
    match_counts_before_filter = gdf_gprs_matched[pea_names_cfg['tipo_match_antena']].value_counts(dropna=False)
    for match_type_stat, count_stat in match_counts_before_filter.items():
        percentage_stat = (count_stat / len(gdf_gprs_matched)) * 100
        print(f"  Tipo '{match_type_stat if pd.notna(match_type_stat) else 'N/A'}': {count_stat} eventos ({percentage_stat:.1f}%)")

    print(f"\nFiltrando eventos '{no_match_label}'...")
    gdf_gprs_final_match = gdf_gprs_matched[gdf_gprs_matched[pea_names_cfg['tipo_match_antena']] != no_match_label].copy()
    num_dropped = len(gdf_gprs_matched) - len(gdf_gprs_final_match)
    print(f"{num_dropped} eventos do tipo '{no_match_label}' foram removidos.")
    print(f"Total events with a match for PEA enrichment: {len(gdf_gprs_final_match)}")

    print('\n--- Enriching GPRS with PEA Attributes (after filtering SEM_MATCH) ---')
    gdf_enriched_pea = enrich_gprs_with_pea_features(gdf_gprs_final_match, config)
    
    if gdf_enriched_pea.empty and not gdf_gprs_final_match.empty :
        print('WARNING: the PEA enrichment process produced an empty GDF, although there was matched data.')
    elif gdf_enriched_pea.empty and gdf_gprs_final_match.empty:
        print('No matched data to enrich. The output file will be empty or will not be generated.')

    if not gdf_enriched_pea.empty:
        print(f"\n=== FINAL OUTPUT STATISTICS ({len(gdf_enriched_pea)} records) ===")
        if pea_names_cfg['pea_id'] in gdf_enriched_pea.columns:
            print(f"  Registros com PEA_ID: {gdf_enriched_pea[pea_names_cfg['pea_id']].notnull().sum()}")
        
        erb_id_col_final = pea_names_cfg.get('erb_id', 'erb_id')
        if erb_id_col_final in gdf_enriched_pea.columns:
            valid_erb_count = gdf_enriched_pea[erb_id_col_final].notnull().sum()
            print(f"  Records with '{erb_id_col_final}' (antenna ID): {valid_erb_count} ({ (valid_erb_count/len(gdf_enriched_pea)*100) if len(gdf_enriched_pea)>0 else 0 :.1f}%)")
            if valid_erb_count > 0:
                print(f"  '{erb_id_col_final}' únicos: {gdf_enriched_pea[erb_id_col_final].nunique()}")
                with_prefix = gdf_enriched_pea[erb_id_col_final].astype(str).str.startswith('ERB_').sum()
                print(f"  IDs com prefixo 'ERB_': {with_prefix} ({with_prefix/valid_erb_count*100:.1f}%)")
    else:
        print("\nNenhum registro PEA gerado no output final.")

    output_path = output_pea_info['path']
    output_layer_name = output_pea_info.get('layer_name', 'eventos_gprs_com_pea')
    output_dir = os.path.dirname(output_path)

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory created: {output_dir}")
    
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
            print(f"Arquivo existente '{output_path}' removido.")
        except Exception as e:
            print(f"Warning: could not remove '{output_path}': {e}")
    
    try:
        if not gdf_enriched_pea.empty:
            output_format = output_pea_info.get('format', '').upper()
            
            if output_format == 'CSV' or output_path.lower().endswith('.csv'):
                print(f"Salvando em formato CSV: {output_path}")
                if hasattr(gdf_enriched_pea, 'geometry') and not gdf_enriched_pea.geometry.isna().all():
                    if 'pea_lat' not in gdf_enriched_pea.columns or 'pea_lon' not in gdf_enriched_pea.columns:
                        gdf_enriched_pea['geometry_lat'] = gdf_enriched_pea.geometry.y
                        gdf_enriched_pea['geometry_lon'] = gdf_enriched_pea.geometry.x
                
                df_to_save = pd.DataFrame(gdf_enriched_pea.drop(columns=['geometry']) if 'geometry' in gdf_enriched_pea.columns else gdf_enriched_pea)
                df_to_save.to_csv(output_path, index=False)
                print(f"PEA data saved to '{output_path}' (CSV format)")
            else:
                print(f"Colunas finais para salvar: {gdf_enriched_pea.columns.tolist()}")
                gdf_enriched_pea.to_file(
                    output_path,
                    layer=output_layer_name,
                    driver="GPKG"
                )
                print(f"PEA data saved to '{output_path}' (layer: {output_layer_name})")
        else:
            print(f"The final GeoDataFrame is empty. No file saved to '{output_path}'.")

        return True
        
    except Exception as e:
        print(f"Error while saving PEA data: {type(e).__name__} - {e}")
        if hasattr(gdf_enriched_pea, 'columns'):
             print(f"  Colunas no gdf_enriched_pea: {gdf_enriched_pea.columns.tolist()}")
        if hasattr(gdf_enriched_pea, 'geometry') and gdf_enriched_pea.geometry is not None:
            print(f"  Nome da geometria ativa: {gdf_enriched_pea.geometry.name}, Tipo: {type(gdf_enriched_pea.geometry)}, CRS: {gdf_enriched_pea.geometry.crs}")
        else:
            print('  No active geometry in the GDF to save, or the GDF is not a GeoDataFrame.')
        return False

if __name__ == "__main__":
    if main_pea_enrichment_process():
        print('\nPEA enrichment process finished successfully.')
    else:
        print("\nProcesso de enriquecimento PEA falhou.")
