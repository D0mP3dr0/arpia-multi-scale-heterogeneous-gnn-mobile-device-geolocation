# ARPIA: multi-scale heterogeneous GNN for mobile device geolocation

Code of the article

> L. F. C. Seelig and R. M. Salles, "ARPIA: A multi-scale heterogeneous graph neural network for mobile device geolocation with explicit non-connection modeling," *Ad Hoc Networks*, vol. 189, art. 104263, 2026. doi: [10.1016/j.adhoc.2026.104263](https://doi.org/10.1016/j.adhoc.2026.104263)

ARPIA (Advanced Radio Positioning Intelligence Architecture) locates mobile devices (PEA nodes) on a heterogeneous graph that also holds base stations (ERB), coverage points (CPS) and connected and non-connected triangle nodes (CTC, CTNC). The absence of a device-to-base-station connection enters the graph as an explicit relation and the loss as a physics-informed force term.

## What the code does

- **Enrichment.** Matches device events to antennas and builds the device records; triangulates CPS, ERB and PEA into the CTC and CTNC nodes; generates the connection, non-connection, same-type proximity and intra-triangle edges, and validates them.
- **Normalization.** Normalizes the node and edge tables (coordinates, identifiers and features) and maps the node IDs across tables.
- **Graph construction.** Builds the heterogeneous graph as a PyTorch Geometric `HeteroData` object.
- **Model.** The ARPIA multi-scale heterogeneous GNN, with local, regional and global components and a local precision refinement.
- **Loss.** A physics-informed loss in which connections and non-connections act as a vector force system with relation-specific parameters, trained in phases with adaptive weights.
- **Training.** Data loaders, phased trainer with automatic evaluation and checkpointing, the ablation study and the sensitivity analysis of the force-system parameters.
- **Baselines.** GNN variants with GATv2, HAN and RGCN convolutions, and the classical RSSI, TOA and TDOA localization methods.
- **Computational cost.** Parameter count, inference time and GPU memory of ARPIA and of the GNN variants, with the figures.

## Layout

| folder | content |
| --- | --- |
| `enrichment/target/` | matching of device events to antennas and construction of the device records (`alvo_enriquecimento.py`, `alvo_config.yaml`), with exploratory and nearest-ERB analyses |
| `enrichment/triangulation/` | CTC and CTNC triangulation (`triangulation_ctc_ctnc.py`, runners, `triangulation_config.yaml`) |
| `enrichment/edges/` | edge generation and validation (`gerador_arestas_v3_sequencial.py`, `run_arestas_sequencial.py`, `validador_arestas_v3.py`, `arestas_config_sequencial.yaml`) |
| `normalization/` | normalization, ID mapping and exploratory analysis (`run_normalizacao_v3.py`, `pipeline_v3.py`, `normalizacao_v3.yaml`) |
| `graph_construction/` | heterogeneous graph construction (`run_grafo_v3.py`, `grafo_v3.py`, `grafo_v3.yaml`) |
| `training/` | model (`models.py`), loss (`losses.py`), trainer (`trainers.py`), data loaders (`data_loaders.py`), configuration (`config_colab.py`), training entry points (`train_colab.py`), utilities (`utils.py`); ablation (`ablation_study_*.py`, `run_ablation_arpia.py`, `run_continuation.py`); sensitivity (`ARPIA_Sensitivity_Parallel.py`); GNN baselines (`arpia_gatv2.py`, `arpia_han.py`, `arpia_rgcn.py`, `baseline_models.py`) |
| `baselines/classical/` | RSSI, TOA and TDOA baselines (`run_classical_methods_rigorous.py`, `run_classical_methods_enhanced.py`) |
| `analysis/computational_cost/` | computational cost measurement and figures |
| `verify_code.py`, `CODE_AST_SHA256`, `PROVENANCE.md`, `SHA256SUMS` | integrity check of the published scripts against the scripts as run, and file digests |

File and module names are those of the original scripts, because the scripts import each other by module name.

## How to run

Install the packages listed in `environment.txt` (Python 3.11; training needs a CUDA GPU).

1. **Data root.** Set the folder that holds the input and output data:

   ```bash
   export ARPIA_ROOT=/path/to/arpia_data
   ```

   The scripts read and write under `ARPIA_ROOT` in the sub-folders `DATA_ENRICHED/`, `DATA_PROCESSED/GRAPH_NODES_EDGES/`, `NORMALIZACAO_OUTPUT/`, `GRAFO_OUTPUT/`, `TREINAMENTO_REAL/` and `RESULTADOS/`. The YAML files carry the same root as the literal prefix `${ARPIA_ROOT}` (and `${ARPIA_V2_ROOT}` for the inputs of `alvo_config.yaml`); replace it before running, for example with `envsubst < grafo_v3.yaml > grafo_v3.local.yaml` and `--config grafo_v3.local.yaml`, or by editing the file. `alvo_config.yaml` also reads `${ARPIA_OPERATOR}`, the name of the operator of the input records; replace it in the same way.

2. **Data preparation.** Run each script from its own folder, so that the modules of that folder import each other:

   ```bash
   cd enrichment/target        && python run_alvo_pipeline.py
   cd enrichment/triangulation && python run_triangulation_optimized.py
   cd enrichment/edges         && python run_arestas_sequencial.py --config arestas_config_sequencial.yaml
   cd normalization            && python run_normalizacao_v3.py --config normalizacao_v3.yaml
   cd graph_construction       && python run_grafo_v3.py --config grafo_v3.yaml
   ```

3. **Training.** The training code was run on Google Colab, with the contents of `training/` and the graph file `arpia_hetero_graph_v3.pt` in `$ARPIA_ROOT/TREINAMENTO_REAL`. The entry points are the functions of `train_colab.py`. From `training/`, the ablation study runs with `python run_ablation_arpia.py`, the GNN baselines with `python arpia_gatv2.py`, `python arpia_han.py` and `python arpia_rgcn.py`, and the sensitivity analysis with `SENSITIVITY_GROUP=<0..3> python ARPIA_Sensitivity_Parallel.py`. A few functions of `train_colab.py` and `baseline_models.py` try optional helper modules of the Colab session inside `try`/`except`; they are skipped when absent.

4. **Classical baselines and computational cost.** The scripts in `baselines/classical/` and `analysis/computational_cost/` read the graph and the normalized tables from paths relative to the script or under `ARPIA_ROOT`.

5. **Integrity check** (Python 3.11 with PyYAML):

   ```bash
   python3.11 verify_code.py
   ```

## Data

The data are confidential operator records and are not distributed with this repository. No data, checkpoint or per-device result is included.

## License

MIT (`LICENSE`).

## Citation

Please cite the article above (doi: [10.1016/j.adhoc.2026.104263](https://doi.org/10.1016/j.adhoc.2026.104263)); `CITATION.cff` carries the metadata.

## Contact

Corresponding author: R. M. Salles, <rmo@estg.ipp.pt>.
