"""Distance from unmatched targets to the nearest base station (ERB)."""
import geopandas as gpd
import pandas as pd
import yaml
from shapely.ops import nearest_points
from pyproj import Geod
import os
import fiona

def load_alvo_config(config_path="alvo_config.yaml"):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: configuration file '{config_path}' not found.")
        return None
    except Exception as e:
        print(f"Error while loading YAML configuration: {e}")
        return None

def list_gpkg_layers(gpkg_path_str):
    try:
        return fiona.listlayers(gpkg_path_str)
    except Exception as e:
        print(f"  ERROR trying to list layers in '{gpkg_path_str}': {type(e).__name__} - {e}")
        return []

def read_gpkg_layer_robust(gpkg_path_info_dict, specified_layer_name_from_call=None, description="layer"):
    actual_path_str = gpkg_path_info_dict.get('path')

    if not actual_path_str or not os.path.exists(actual_path_str):
        print(f"Error: GPKG file for {description} not found at '{actual_path_str}'.")
        return gpd.GeoDataFrame()

    available_layers = list_gpkg_layers(actual_path_str)
    if not available_layers:
        print(f"Error: no layer found in the GPKG file '{actual_path_str}' for {description}.")
        return gpd.GeoDataFrame()

    layer_to_read = specified_layer_name_from_call
    if layer_to_read and layer_to_read not in available_layers:
        print(f"Warning: layer '{layer_to_read}' for {description} not found in '{actual_path_str}'. Available: {available_layers}. Reading the first layer: '{available_layers[0]}'.")
        layer_to_read = available_layers[0]
    elif not layer_to_read:
        layer_to_read = available_layers[0]
        print(f"No layer name specified for {description} in '{actual_path_str}', reading the first layer: '{layer_to_read}'.")
    
    try:
        gdf = gpd.read_file(actual_path_str, layer=layer_to_read)
        print(f"Layer '{layer_to_read}' ({description}) read successfully from '{actual_path_str}'.")
        return gdf
    except Exception as e:
        print(f"Error while reading layer '{layer_to_read}' ({description}) from '{actual_path_str}': {e}")
        return gpd.GeoDataFrame()

def ensure_gdf_crs(gdf, target_crs, description="GeoDataFrame", original_crs_fallback=None):
    if gdf.empty:
        return gdf
    if gdf.crs is None and original_crs_fallback:
        print(f"Warning: {description} has no CRS. Setting it to {original_crs_fallback}.")
        gdf = gdf.set_crs(original_crs_fallback, allow_override=True)
    elif gdf.crs is None and not original_crs_fallback:
        print(f"Error: {description} has no CRS and no fallback was given. The CRS cannot be guaranteed.")
        return gdf

    if gdf.crs != target_crs:
        print(f"Transforming {description} from {gdf.crs} to {target_crs}.")
        gdf = gdf.to_crs(target_crs)
    return gdf

def calculate_min_distances(gdf_alvo_nao_mapeado, gdf_antenas_todas, geod_calculator, antena_id_col):
    min_distances = []
    nearest_erb_ids = []

    if gdf_antenas_todas.empty:
        print('Error: the antenna GeoDataFrame is empty. Distances cannot be computed.')
        return [float('nan')] * len(gdf_alvo_nao_mapeado), [None] * len(gdf_alvo_nao_mapeado)

    todas_geometrias_antenas_gs = gdf_antenas_todas.geometry
    
    print(f"Computing the minimum distance for {len(gdf_alvo_nao_mapeado)} unmapped targets...")
    count = 0
    for idx, alvo_row in gdf_alvo_nao_mapeado.iterrows():
        alvo_geom = alvo_row.geometry
        if alvo_geom is None or alvo_geom.is_empty:
            min_distances.append(float('nan'))
            nearest_erb_ids.append(None)
            continue

        try:
            nearest_idx = todas_geometrias_antenas_gs.sindex.nearest(alvo_geom)[1][0]
            p2 = todas_geometrias_antenas_gs.iloc[nearest_idx]
            p1 = alvo_geom
        except Exception as e:
            print(f"Warning: sindex.nearest failed for target index {idx} (trying nearest_points): {e}")
            try:
                unary_union_antenas = todas_geometrias_antenas_gs.unary_union
                if unary_union_antenas is None or unary_union_antenas.is_empty:
                    min_distances.append(float('nan')); nearest_erb_ids.append(None); continue
                p1, p2 = nearest_points(alvo_geom, unary_union_antenas)
            except Exception as e_np:
                print(f"Error in nearest_points for target index {idx}: {e_np}")
                min_distances.append(float('nan')); nearest_erb_ids.append(None); continue
            
        _, _, distance_m = geod_calculator.inv(p1.x, p1.y, p2.x, p2.y)
        min_distances.append(distance_m)
        
        try:
            nearest_erb_ids.append(gdf_antenas_todas.iloc[nearest_idx][antena_id_col])
        except Exception as e_id:
            print(f"Warning: could not obtain erb_id directly via the index after nearest_points for target {idx}. Trying a geometry search (may be inaccurate/slow).")
            try:
                matched_antena = gdf_antenas_todas[gdf_antenas_todas.geometry.equals_exact(p2, tolerance=1e-9)]
                if not matched_antena.empty:
                    nearest_erb_ids.append(matched_antena[antena_id_col].iloc[0])
                else:
                    nearest_erb_ids.append(None)
            except Exception as e_fallback_id:
                print(f"Error in the fallback to identify erb_id for target {idx}: {e_fallback_id}")
                nearest_erb_ids.append(None)


        count += 1
        if count % 100 == 0:
            print(f"  Distances computed for {count}/{len(gdf_alvo_nao_mapeado)} targets...")
            
    return min_distances, nearest_erb_ids

def main():
    print('Starting the ERB proximity analysis...')
    config = load_alvo_config(config_path="alvo_config.yaml")
    if not config:
        return

    input_gprs_info = config.get('input_gprs_gpkg')
    input_antenas_info = config.get('input_antenas_gpkg')
    crs_fallback = config.get('input_crs_fallback', 'EPSG:4326')

    if not input_gprs_info or not input_gprs_info.get('path') or \
       not input_antenas_info or not input_antenas_info.get('path'):
        print('Paths to the GPRS/Target or Antenna GPKG not specified in the config.')
        return

    print('\nLoading the Target/GPRS data...')
    gdf_gprs = read_gpkg_layer_robust(input_gprs_info, input_gprs_info.get('layer_name'), "Eventos GPRS")
    if gdf_gprs.empty: return
    gdf_gprs = ensure_gdf_crs(gdf_gprs, crs_fallback, "Eventos GPRS", crs_fallback)
    if gdf_gprs.empty or gdf_gprs.geometry.name not in gdf_gprs.columns or gdf_gprs.geometry is None:
        print(f"Error: the GPRS event layer ('{input_gprs_info.get('path')}') has no valid active geometry column or is empty after CRS handling.")
        return

    print('\nLoading the Enriched Antenna data...')
    gdf_antenas = read_gpkg_layer_robust(input_antenas_info, input_antenas_info.get('layer_name'), "Antenas Enriquecidas")
    if gdf_antenas.empty: return
    gdf_antenas = ensure_gdf_crs(gdf_antenas, crs_fallback, "Antenas Enriquecidas", crs_fallback)
    antena_id_col = input_antenas_info.get('id_column_in_antenas', 'erb_id')
    if antena_id_col not in gdf_antenas.columns:
        print(f"Error: ID column '{antena_id_col}' not found in the Antenna layer. Available columns: {gdf_antenas.columns.tolist()}")
        return
    if gdf_antenas.empty or gdf_antenas.geometry.name not in gdf_antenas.columns or gdf_antenas.geometry is None:
        print(f"Error: the Antenna layer ('{input_antenas_info.get('path')}') has no valid active geometry column or is empty after CRS handling.")
        return

    print("\nTentando join espacial exato (sem buffer)...")
    if gdf_gprs.crs != gdf_antenas.crs:
        print(f"Aligning CRS for the join: GPRS ({gdf_gprs.crs}) and Antennas ({gdf_antenas.crs}).")
        gdf_antenas = gdf_antenas.to_crs(gdf_gprs.crs)

    gdf_gprs_copy = gdf_gprs.copy()
    gdf_gprs_copy['original_index'] = gdf_gprs_copy.index
    
    gdf_antenas_for_join = gdf_antenas[[antena_id_col, gdf_antenas.geometry.name]].copy()
    
    gdf_gprs_sjoined = gpd.sjoin(gdf_gprs_copy, gdf_antenas_for_join, 
                                how='left', predicate='intersects', lsuffix='gprs', rsuffix='ant')

    idx_col_from_sjoin = 'index_ant' if 'index_ant' in gdf_gprs_sjoined.columns else 'index_right'

    if idx_col_from_sjoin not in gdf_gprs_sjoined.columns:
        print(f"Warning: join index column ('{idx_col_from_sjoin}') not found. Assuming that no target was mapped, for the proximity analysis.")
        gdf_alvo_nao_mapeado = gdf_gprs.copy()
        num_mapeados_exato = 0
    else:
        mapeados_mask = gdf_gprs_sjoined[idx_col_from_sjoin].notna()
        num_mapeados_exato = mapeados_mask.sum()
        indices_nao_mapeados = gdf_gprs_sjoined.loc[~mapeados_mask, 'original_index']
        gdf_alvo_nao_mapeado = gdf_gprs.loc[indices_nao_mapeados].copy()


    print(f"{num_mapeados_exato} of {len(gdf_gprs)} GPRS events mapped with an exact spatial join.")
    
    if gdf_alvo_nao_mapeado.empty:
        print('All GPRS events were mapped with an exact spatial join. No additional proximity analysis needed.')
        return

    print(f"{len(gdf_alvo_nao_mapeado)} GPRS events were not mapped. Computing the distance to the nearest antenna...")

    target_crs_for_distance = "EPSG:4326"
    gdf_alvo_nao_mapeado_geo = ensure_gdf_crs(gdf_alvo_nao_mapeado, target_crs_for_distance, 'Unmapped Targets (for the distance)', crs_fallback)
    gdf_antenas_geo = ensure_gdf_crs(gdf_antenas, target_crs_for_distance, 'All Antennas (for the distance)', crs_fallback)
    
    original_count_nao_mapeado = len(gdf_alvo_nao_mapeado_geo)
    gdf_alvo_nao_mapeado_geo = gdf_alvo_nao_mapeado_geo[gdf_alvo_nao_mapeado_geo.geometry.notna() & ~gdf_alvo_nao_mapeado_geo.geometry.is_empty]
    if len(gdf_alvo_nao_mapeado_geo) < original_count_nao_mapeado:
        print(f"Removed {original_count_nao_mapeado - len(gdf_alvo_nao_mapeado_geo)} rows with null/empty geometry from the unmapped target layer.")

    if gdf_alvo_nao_mapeado_geo.empty:
        print('No unmapped target with valid geometry to compute distances.')
        return

    geod = Geod(ellps="WGS84")
    distancias_min_metros, ids_erb_mais_proxima = calculate_min_distances(
        gdf_alvo_nao_mapeado_geo, 
        gdf_antenas_geo,
        geod,
        antena_id_col
    )

    gdf_alvo_nao_mapeado_geo['dist_min_antena_m'] = distancias_min_metros
    gdf_alvo_nao_mapeado_geo['erb_id_mais_proxima'] = ids_erb_mais_proxima

    print('\n--- Statistics of the Minimum Distances to the ERBs for Unmapped Targets (metres) ---')
    if not gdf_alvo_nao_mapeado_geo['dist_min_antena_m'].empty and gdf_alvo_nao_mapeado_geo['dist_min_antena_m'].notna().any():
        print(gdf_alvo_nao_mapeado_geo['dist_min_antena_m'].describe(percentiles=[.1, .25, .5, .75, .9, .95, .99]))
        
        mapeados_por_proximidade = gdf_alvo_nao_mapeado_geo['erb_id_mais_proxima'].notna().sum()
        print(f"\nOf the {len(gdf_alvo_nao_mapeado_geo)} unmapped targets analysed by proximity, {mapeados_por_proximidade} had a nearest ERB identified.")

        print('\nSample of unmapped targets with the distance to the nearest ERB:')
        cols_originais_relevantes = [col for col in ['latitude', 'longitude', 'id_alvo_anonimo'] if col in gdf_alvo_nao_mapeado_geo.columns]
        print(gdf_alvo_nao_mapeado_geo[cols_originais_relevantes + ['dist_min_antena_m', 'erb_id_mais_proxima']].head(20))
        
        output_distancias_path = os.path.join(os.path.dirname(config['output_pea_gpkg']['path']), "alvos_nao_mapeados_com_distancias.gpkg")
        try:
            gdf_alvo_nao_mapeado_geo.to_file(output_distancias_path, layer="distancias_erbs", driver="GPKG")
            print(f"\nUnmapped target data with distances saved to: {output_distancias_path}")
        except Exception as e:
            print(f"\nError while saving the distance GDF: {e}")
    else:
        print('No distance could be computed (possibly because of null geometries or because all antennas are invalid).')

if __name__ == "__main__":
    main()

