"""Runs the target enrichment followed by its exploratory analysis."""
import alvo_enriquecimento
import alvo_eda
import os
import pandas as pd
import geopandas as gpd

CONFIG_FILE = "alvo_config.yaml"

def main():
    print("==============================================================")
    print("=== STARTING ALVO/GPRS (PEA CREATION) ENRICHMENT PIPELINE ===")
    print("==============================================================")

    print("\n--- Running PEA Enrichment Process ---")
    success = alvo_enriquecimento.main_pea_enrichment_process(CONFIG_FILE)

    if not success:
        print("\nPEA Enrichment process failed. Aborting EDA.")
        print("======================================================")
        print("====== PIPELINE EXECUTION FAILED =====")
        print("======================================================")
        return

    print("\nPEA Enrichment process completed.")

    print("\n--- Running Exploratory Data Analysis (EDA) on PEA Enriched File ---")
    config_for_eda = alvo_enriquecimento.load_config(CONFIG_FILE) 
    
    if config_for_eda and config_for_eda.get('output_pea_gpkg', {}).get('path'):
        output_path = config_for_eda['output_pea_gpkg']['path']
        if os.path.exists(output_path):
            if output_path.lower().endswith('.csv'):
                print(f"CSV file found at: {output_path}")
                try:
                    df = pd.read_csv(output_path)
                    if 'pea_lat' in df.columns and 'pea_lon' in df.columns:
                        gdf = gpd.GeoDataFrame(
                            df, 
                            geometry=gpd.points_from_xy(df.pea_lon, df.pea_lat),
                            crs="EPSG:4326"
                        )
                        temp_gpkg = output_path.replace('.csv', '_temp.gpkg')
                        gdf.to_file(temp_gpkg, driver="GPKG")
                        alvo_eda.run_alvo_eda(temp_gpkg, CONFIG_FILE)
                        if os.path.exists(temp_gpkg):
                            os.remove(temp_gpkg)
                    else:
                        print('The CSV has no pea_lat/pea_lon columns to build geometry for the EDA.')
                except Exception as e:
                    print(f"Error while preparing the CSV for the EDA: {e}")
            else:
                alvo_eda.run_alvo_eda(output_path, CONFIG_FILE)
        else:
            print(f"Error: Enriched PEA file '{output_path}' not found for EDA.")
    else:
        print("Error: Could not load configuration or output PEA path for EDA.")

    print("\n======================================================")
    print("====== PIPELINE EXECUTION FINISHED =====")
    print("======================================================")

if __name__ == "__main__":
    main()
