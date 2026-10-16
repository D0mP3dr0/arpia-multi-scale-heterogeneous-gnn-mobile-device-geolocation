"""Exploratory analysis of the enriched target records."""
import geopandas as gpd
import pandas as pd
import yaml
import os
import fiona

def load_config_for_eda(config_path="alvo_config.yaml"):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Warning: Could not load '{config_path}' for EDA: {e}")
        return {}

def analyze_operator_matching_choices(gdf, config):
    print(f"\n🎯 === DETAILED ANALYSIS OF MATCHING CHOICES (ONLY RECORDS WITH A MATCH) ===")
    
    antenna_config = config.get('antenna_matching', {})
    priority_1_label = antenna_config.get('priority_1_label', 'MESMA_OPERADORA')
    priority_2_label = antenna_config.get('priority_2_label', 'MAIS_PROXIMA')
    
    pea_names = config.get('pea_feature_names', {})
    tipo_match_col = pea_names.get('tipo_match_antena', 'tipo_match_antena')
    operadora_gprs_col = pea_names.get('operadora_evento_gprs', 'operadora_evento_gprs')
    operadora_antena_col = pea_names.get('operadora_antena_escolhida', 'operadora_antena_escolhida')
    distancia_col = pea_names.get('distancia_antena_metros', 'distancia_antena_metros')
    
    required_cols = [tipo_match_col, operadora_gprs_col, operadora_antena_col, distancia_col]
    missing_cols = [col for col in required_cols if col not in gdf.columns]
    
    if missing_cols:
        print(f"⚠️  Matching analysis columns not found: {missing_cols}")
        print('   Detailed analysis not available.')
        return
    
    if gdf.empty:
        print('📊 The GeoDataFrame is empty. No matching analysis to run.')
        return

    total_records = len(gdf)
    print(f"📊 Total records WITH A MATCH analysed: {total_records:,}")
    
    print(f"\n1. 📋 DISTRIBUTION BY MATCH TYPE:")
    if tipo_match_col in gdf.columns:
        match_counts = gdf[tipo_match_col].value_counts(dropna=False)
        for match_type, count in match_counts.items():
            percentage = (count / total_records) * 100 if total_records > 0 else 0
            print(f"   🔹 {match_type if pd.notna(match_type) else 'N/A (Tipo Match)'}: {count:,} registros ({percentage:.1f}%)")
            
            if pd.notna(match_type):
                subset = gdf[gdf[tipo_match_col] == match_type]
                if not subset.empty and distancia_col in subset.columns:
                    numeric_distances = pd.to_numeric(subset[distancia_col], errors='coerce').dropna()
                    if not numeric_distances.empty:
                        dist_stats = numeric_distances.describe()
                        if 'mean' in dist_stats: print(f"      → Mean distance: {dist_stats['mean']:.1f}m")
                        if 'max' in dist_stats: print(f"      → Maximum distance: {dist_stats['max']:.1f}m")
                        if 'min' in dist_stats: print(f"      → Minimum distance: {dist_stats['min']:.1f}m")
                    else:
                        print(f"      → Distance statistics not available for '{match_type}' (no valid numeric data).")
    
    print(f"\n2. 🏢 ANALYSIS BY OPERATOR OF THE GPRS EVENT:")
    if operadora_gprs_col in gdf.columns:
        op_counts = gdf[operadora_gprs_col].value_counts(dropna=False)
        for operadora, count in op_counts.items():
            operadora_label = operadora if pd.notna(operadora) else "N/A (Op. GPRS)"
            percentage = (count / total_records) * 100 if total_records > 0 else 0
            print(f"   🔸 {operadora_label}: {count:,} eventos ({percentage:.1f}%)")
            
            if pd.notna(operadora):
                op_subset = gdf[gdf[operadora_gprs_col] == operadora]
                if not op_subset.empty and tipo_match_col in op_subset.columns:
                    match_breakdown = op_subset[tipo_match_col].value_counts(dropna=False)
                    for match_type_op, match_count_op in match_breakdown.items():
                        match_pct_op = (match_count_op / count) * 100 if count > 0 else 0
                        match_type_label = match_type_op if pd.notna(match_type_op) else "N/A (Tipo Match)"
                        print(f"      → {match_type_label}: {match_count_op:,} ({match_pct_op:.1f}%)")
    
    print(f"\n3. 🔄 COMPATIBILITY ANALYSIS BETWEEN OPERATORS:")
    if operadora_gprs_col in gdf.columns and operadora_antena_col in gdf.columns:
        gdf_copy = gdf.copy() 
        gdf_copy[operadora_gprs_col] = gdf_copy[operadora_gprs_col].fillna("N/A_GPRS_EDA")
        gdf_copy[operadora_antena_col] = gdf_copy[operadora_antena_col].fillna("N/A_Antena_EDA")

        mesma_operadora = gdf_copy[gdf_copy[operadora_gprs_col] == gdf_copy[operadora_antena_col]]
        operadora_diferente = gdf_copy[gdf_copy[operadora_gprs_col] != gdf_copy[operadora_antena_col]]
        
        print(f"   ✅ Same operator (or both N/A in the EDA): {len(mesma_operadora):,} ({ (len(mesma_operadora)/total_records*100) if total_records > 0 else 0 :.1f}%)")
        print(f"   🔄 Operadora diferente (na EDA): {len(operadora_diferente):,} ({ (len(operadora_diferente)/total_records*100) if total_records > 0 else 0 :.1f}%)")
        
        if not operadora_diferente.empty:
            print(f"\n   📊 MATRIX OF OPERATOR SWITCHES (GPRS → Antenna):")
            cross_tab = pd.crosstab(
                operadora_diferente[operadora_gprs_col], 
                operadora_diferente[operadora_antena_col],
                margins=True,
                dropna=False
            )
            print(cross_tab)
    
    print(f"\n4. 📏 ANALYSIS OF OVERALL DISTANCES (all records are WITH A MATCH):")
    if distancia_col in gdf.columns:
        valid_distances = gdf[distancia_col].dropna()

        if not valid_distances.empty:
            dist_stats = valid_distances.describe()
            print(f"   📊 Distance statistics (metres):")
            if 'count' in dist_stats: print(f"      → No. of records with a distance: {dist_stats['count']:.0f}")
            if 'mean' in dist_stats: print(f"      → Média: {dist_stats['mean']:.1f}m")
            if '50%' in dist_stats: print(f"      → Mediana: {dist_stats['50%']:.1f}m")
            if 'std' in dist_stats: print(f"      → Standard deviation: {dist_stats['std']:.1f}m")
            if 'min' in dist_stats: print(f"      → Mínima: {dist_stats['min']:.1f}m")
            if 'max' in dist_stats: print(f"      → Máxima: {dist_stats['max']:.1f}m")
            
            print(f"\n   📈 DISTRIBUTION BY DISTANCE BAND:")
            max_config_dist = antenna_config.get('max_distance_meters', 50)
            bins = [0, 10, 25, max_config_dist, float('inf')]
            labels = ['0-10m', '10-25m', f'25-{max_config_dist}m', f'>{max_config_dist}m']
            
            temp_faixa_distancia_col = 'faixa_distancia_temp_eda'
            gdf[temp_faixa_distancia_col] = pd.cut(gdf[distancia_col], bins=bins, labels=labels, right=False, include_lowest=True)
            
            faixa_counts = gdf[temp_faixa_distancia_col].value_counts().sort_index()
            if total_records > 0:
                for faixa, count in faixa_counts.items():
                    percentage = (count / total_records) * 100
                    print(f"      → {faixa}: {count:,} ({percentage:.1f}%)")
            
            if temp_faixa_distancia_col in gdf.columns:
                gdf.drop(temp_faixa_distancia_col, axis=1, inplace=True)
        else:
            print(f"   📊 No valid distance found for analysis.")
    
    print(f"\n5. 💡 INSIGHTS AND RECOMMENDATIONS (based on {total_records} records WITH A MATCH):")
    if tipo_match_col in gdf.columns and total_records > 0:
        match_counts_for_insights = gdf[tipo_match_col].value_counts(dropna=False) 
        
        if priority_1_label in match_counts_for_insights.index:
            count_priority_1 = match_counts_for_insights[priority_1_label]
            pct_mesma_op = (count_priority_1 / total_records) * 100
            if pct_mesma_op > 70:
                print(f"   ✅ Excellent rate of connection to the own operator ({pct_mesma_op:.1f}%) among the matches.")
            elif pct_mesma_op > 50:
                print(f"   ⚠️  Good rate of connection to the own operator ({pct_mesma_op:.1f}%) among the matches.")
            else:
                print(f"   🔴 Rate of connection to the own operator below expectation ({pct_mesma_op:.1f}%) among the matches.")
        else:
            print(f"   ℹ️ No match found for '{priority_1_label}'.")

        if priority_2_label in match_counts_for_insights.index:
            count_priority_2 = match_counts_for_insights[priority_2_label]
            pct_outras_op = (count_priority_2 / total_records) * 100
            if pct_outras_op > 0:
                print(f"   🔄 Use of antennas of other operators ({pct_outras_op:.1f}%) among the matches.")
                if pct_outras_op > 10:
                    print(f"      → Consider roaming agreements or an expansion analysis in these areas.")
        else:
             print(f"   ℹ️ No match found for '{priority_2_label}'.")
        
        print(f"\n   Reminder: the analysis above is based only on the records that had an antenna match. ")
        print(f"   Registros classificados como '{antenna_config.get('no_match_label', 'SEM_MATCH')}' foram removidos antes desta EDA.")


def run_alvo_eda(gpkg_file_path, config_path="alvo_config.yaml"):
    print(f"\n======= Starting EDA for Enriched ALVO/PEA File: {gpkg_file_path} =======")

    if not os.path.exists(gpkg_file_path):
        print(f"Error: Enriched file '{gpkg_file_path}' not found for EDA.")
        return

    config = load_config_for_eda(config_path)
    if not config:
        print("EDA configuration could not be loaded. Analysis might be limited.")
        return

    output_pea_info = config.get('output_pea_gpkg', {})
    layer_name_to_analyze = output_pea_info.get('layer_name', 'eventos_gprs_com_pea') 
    pea_feature_names_config = config.get('pea_feature_names', {})

    available_layers = []
    try:
        available_layers = fiona.listlayers(gpkg_file_path)
        print(f"\nAvailable layers in '{gpkg_file_path}': {available_layers}")
    except Exception as e:
        print(f"Error listing layers in '{gpkg_file_path}': {e}.")
        if layer_name_to_analyze not in available_layers : 
             print(f"Cannot confirm if configured layer '{layer_name_to_analyze}' exists. Attempting to read anyway.")

    if not available_layers and os.path.exists(gpkg_file_path):
        print(f"No layers found by Fiona in '{gpkg_file_path}'. File might be empty or not a valid GPKG. Cannot perform EDA.")
        return
    elif not available_layers:
        print(f"Could not list layers (file '{gpkg_file_path}' might be missing or invalid). Cannot perform EDA.")
        return


    if layer_name_to_analyze not in available_layers:
        print(f"Warning: Configured layer '{layer_name_to_analyze}' not found in '{available_layers}'. Analyzing first available layer: '{available_layers[0]}'")
        if available_layers:
            layer_name_to_analyze = available_layers[0]
        else:
            print(f"No layers available to analyze in '{gpkg_file_path}'. Aborting EDA.")
            return


    print(f"\nReading layer '{layer_name_to_analyze}' for EDA...")
    try:
        gdf = gpd.read_file(gpkg_file_path, layer=layer_name_to_analyze)
        if gdf.empty:
            print(f"Layer '{layer_name_to_analyze}' was read but is empty. The EDA will be limited.")
        
        distancia_col_name_global = pea_feature_names_config.get('distancia_antena_metros', 'distancia_antena_metros')
        if distancia_col_name_global in gdf.columns:
            gdf[distancia_col_name_global] = pd.to_numeric(gdf[distancia_col_name_global], errors='coerce')
            print(f"Column '{distancia_col_name_global}' converted to numeric type for EDA.")

    except Exception as e:
        print(f"Error reading layer '{layer_name_to_analyze}' from '{gpkg_file_path}': {e}")
        return

    print(f"\n--- EDA for Layer: {layer_name_to_analyze} ---")
    if gdf.empty:
        print('The GeoDataFrame is empty. The subsequent analyses may produce no results.')
    
    print(f"\n1. DataFrame Head (first 5 rows):")
    print(gdf.head())

    print(f"\n2. DataFrame Info:")
    gdf.info(verbose=True)

    print(f"\n3. Descriptive Statistics (numerical columns):")
    numeric_cols = gdf.select_dtypes(include=['number']).columns
    if not numeric_cols.empty and not gdf.empty:
        print(gdf[numeric_cols].describe())
    elif gdf.empty:
        print("GeoDataFrame is empty, no descriptive statistics.")
    else:
        print("No purely numerical columns found for descriptive statistics.")

    print(f"\n4. Null Value Counts per Column:")
    print(gdf.isnull().sum())

    print(f"\n5. CRS (Coordinate Reference System):")
    print(gdf.crs)

    if hasattr(gdf, 'geometry') and gdf.geometry is not None and hasattr(gdf.geometry, 'name') and gdf.geometry.name in gdf.columns:
        active_geometry_col = gdf.geometry.name
        print(f"\n6. Active Geometry Column: '{active_geometry_col}'")
        if not gdf.empty:
            geom_types = gdf[active_geometry_col].geom_type.unique()
            print(f"   Geometry Types: {[g_type for g_type in geom_types if pd.notna(g_type)]}")
        else:
            print("   GeoDataFrame is empty, no geometry types to display.")
    else:
        print("\n6. No active geometry column found, or GDF is not a GeoDataFrame.")

    print(f"\n7. Checking for PEA feature columns (as per config):")
    if pea_feature_names_config:
        erb_id_col_name_cfg = pea_feature_names_config.get('erb_id', 'erb_id')

        for key, col_name in pea_feature_names_config.items():
            if col_name in gdf.columns:
                print(f"  - PEA Column '{col_name}' (for {key}) found. Non-nulls: {gdf[col_name].notnull().sum()}, Dtype: {gdf[col_name].dtype}")
                
                if not gdf.empty:
                    if key == 'tipo_match_antena' and gdf[col_name].dtype == 'object':
                        print(f"    └─ Unique values: {gdf[col_name].unique().tolist()}")
                    elif col_name == erb_id_col_name_cfg and gdf[col_name].dtype == 'object':
                         print(f"    └─ Unique values of '{erb_id_col_name_cfg}': {gdf[col_name].nunique()} (sample: {gdf[col_name].unique()[:5].tolist()}...)")
                    elif key == 'distancia_antena_metros' and pd.api.types.is_numeric_dtype(gdf[col_name].dtype):
                        if gdf[col_name].notnull().any():
                             print(f"    └─ Mean distance: {gdf[col_name].mean():.1f}m, Max: {gdf[col_name].max():.1f}m (overall)")
                        else:
                             print(f"    └─ Distance: all values are null.")
                    elif key in ['operadora_evento_gprs', 'operadora_antena_escolhida'] and gdf[col_name].dtype == 'object':
                        print(f"    └─ Operadoras: {gdf[col_name].value_counts(dropna=False).to_dict()}")
            else:
                print(f"  - PEA Column '{col_name}' (for {key}) NOT FOUND.")
    else:
        print("  PEA feature names not found in EDA configuration (pea_feature_names).")

    analyze_operator_matching_choices(gdf, config)

    print(f"\n--- End of EDA for Layer: {layer_name_to_analyze} ---")
    print("\n======= EDA for Enriched ALVO/PEA File Completed =======")

if __name__ == "__main__":
    cfg_main_eda = load_config_for_eda("alvo_config.yaml") 
    if cfg_main_eda:
        output_gpkg_path_from_config_eda = cfg_main_eda.get('output_pea_gpkg', {}).get('path')
        if output_gpkg_path_from_config_eda:
            run_alvo_eda(output_gpkg_path_from_config_eda, "alvo_config.yaml")
        else:
            print("Output ALVO/PEA GPKG path not configured in alvo_config.yaml for direct EDA run.")
    else:
        print("EDA aborted due to configuration loading failure.")
