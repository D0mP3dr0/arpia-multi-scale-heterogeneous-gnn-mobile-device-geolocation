#!/usr/bin/env python3
"""Loading of the heterogeneous graph, data splits and neighbour loaders."""

import os
import torch
import pickle
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import RandomNodeSplit
from config_colab import ColabConfig
import logging

logger = logging.getLogger(__name__)

class ARPIADataLoader:
    
    def __init__(self, config: ColabConfig):
        self.config = config
        self.graph_data = None
        self.metadata = None
        self.scalers = None
        self.splits = None
        self.loaders = None
        
        print(f"📊 INICIALIZANDO ARPIA DATA LOADER")
        print(f"   Drive root: {config.drive_root}")
        print(f"   Batch size: {config.batch_size}")
        print(f"   Num neighbors: {config.num_neighbors}")
    
    def load_data(self) -> bool:
        
        print('\n🔄 LOADING DATA FROM GOOGLE DRIVE')
        print("=" * 50)
        
        try:
            if not self._check_drive_mounted():
                print('❌ Google Drive is not mounted!')
                return False
            
            if not self._check_files_exist():
                print('❌ Required files were not found!')
                return False
            
            print('🔄 Loading the main graph...')
            self.graph_data = self._load_graph_data()
            if self.graph_data is None:
                return False
            
            print("🔄 Carregando metadados...")
            self.metadata = self._load_metadata()
            if self.metadata is None:
                return False
            
            print("🔄 Carregando scalers...")
            self.scalers = self._load_scalers()
            if self.scalers is None:
                return False
            
            print("🔄 Validando dados...")
            if not self._validate_data():
                return False
            
            print("🔄 Preprocessando dados...")
            self._preprocess_data()
            
            print('✅ ALL DATA LOADED SUCCESSFULLY!')
            self._print_data_summary()
            
            return True
            
        except Exception as e:
            print(f"❌ ERROR while loading: {str(e)}")
            logger.error(f"Error while loading data: {str(e)}")
            return False
    
    def _check_drive_mounted(self) -> bool:
        
        drive_path = Path("/content/drive")
        return drive_path.exists() and drive_path.is_dir()
    
    def _check_files_exist(self) -> bool:
        
        required_files = [
            self.config.graph_path,
            self.config.metadata_path,
            self.config.scalers_path
        ]
        
        missing_files = []
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            print("❌ Arquivos em falta:")
            for file_path in missing_files:
                print(f"   {file_path}")
            return False
        
        for file_path in required_files:
            size = Path(file_path).stat().st_size
            print(f"✅ {Path(file_path).name}: {size:,} bytes ({size / 1024 / 1024:.1f} MB)")
        
        return True
    
    def _load_graph_data(self) -> Optional[HeteroData]:
        
        try:
            graph_data = torch.load(
                self.config.graph_path,
                map_location=self.config.device,
                weights_only=False
            )
            
            print(f"✅ Grafo carregado:")
            print(f"   Node types in use: {graph_data.node_types}")
            print(f"   Edge types in use: {graph_data.edge_types}")
            
            for node_type in graph_data.node_types:
                num_nodes = graph_data[node_type].num_nodes
                print(f"   {node_type}: {num_nodes:,} nós")
            
            return graph_data
            
        except Exception as e:
            print(f"❌ Error while loading the graph: {str(e)}")
            return None
    
    def _load_metadata(self) -> Optional[Dict]:
        
        try:
            with open(self.config.metadata_path, 'r') as f:
                metadata = json.load(f)
            
            print(f"✅ Metadados carregados:")
            print(f"   Chaves: {list(metadata.keys())}")
            
            return metadata
            
        except Exception as e:
            print(f"❌ Error while loading metadata: {str(e)}")
            return None
    
    def _load_scalers(self) -> Optional[Dict]:
        
        try:
            with open(self.config.scalers_path, 'rb') as f:
                scalers = pickle.load(f)
            
            print(f"✅ Scalers carregados:")
            if isinstance(scalers, dict):
                print(f"   Chaves: {list(scalers.keys())}")
            else:
                print(f"   Tipo: {type(scalers)}")
            
            return scalers
            
        except Exception as e:
            print(f"❌ Error while loading scalers: {str(e)}")
            return None
    
    def _validate_data(self) -> bool:
        
        if 'PEA' not in self.graph_data.node_types:
            print("❌ Node type 'PEA' not found!")
            return False
        
        if not hasattr(self.graph_data['PEA'], 'x') or self.graph_data['PEA'].x is None:
            print('❌ PEA coordinates not found!')
            return False
        
        pea_features = self.graph_data['PEA'].x
        if pea_features.shape[1] < 2:
            print("❌ PEAs devem ter pelo menos 2 features (x, y)!")
            return False
        
        if len(self.graph_data.edge_types) == 0:
            print('❌ No edge was found!')
            return False
        
        print('✅ Validation of the data: OK')
        return True
    
    def _preprocess_data(self):
        
        self.graph_data = self.graph_data.to(self.config.device)
        
        if self.config.data_augmentation.get('enable', False):
            self._apply_data_augmentation()
        
        self._add_missing_features()
        
        print("✅ Preprocessamento concluído")
    
    def _apply_data_augmentation(self):
        
        noise_std = self.config.data_augmentation.get('noise_std', 0.01)
        
        for node_type in self.graph_data.node_types:
            if hasattr(self.graph_data[node_type], 'x') and self.graph_data[node_type].x is not None:
                features = self.graph_data[node_type].x
                if features.shape[1] >= 2:
                    noise = torch.randn_like(features[:, :2]) * noise_std
                    features[:, :2] += noise
        
        print(f"✅ Augmentation applied: noise σ={noise_std}")
    
    def _add_missing_features(self):
        
        for node_type in self.graph_data.node_types:
            if not hasattr(self.graph_data[node_type], 'x') or self.graph_data[node_type].x is None:
                num_nodes = self.graph_data[node_type].num_nodes
                dummy_features = torch.zeros(num_nodes, 2, device=self.config.device)
                self.graph_data[node_type].x = dummy_features
                print(f"✅ Dummy features created for {node_type}: {num_nodes} nodes")
    
    def _print_data_summary(self):
        
        print('\n📊 SUMMARY OF THE LOADED DATA')
        print("=" * 50)
        
        total_nodes = 0
        total_edges = 0
        
        for node_type in self.graph_data.node_types:
            num_nodes = self.graph_data[node_type].num_nodes
            total_nodes += num_nodes
            print(f"   {node_type}: {num_nodes:,} nós")
        
        for edge_type in self.graph_data.edge_types:
            num_edges = self.graph_data[edge_type].num_edges
            total_edges += num_edges
            print(f"   {edge_type}: {num_edges:,} arestas")
        
        print(f"   TOTAL: {total_nodes:,} nodes, {total_edges:,} edges")
        
        if 'PEA' in self.graph_data.node_types:
            pea_features = self.graph_data['PEA'].x
            print(f"   PEAs: {pea_features.shape[0]:,} nodes, {pea_features.shape[1]} features")
        
        if self.config.device == 'cuda':
            memory_used = torch.cuda.memory_allocated() / 1024**3
            print(f"   GPU memory in use: {memory_used:.1f} GB")
    
    def create_splits(self) -> bool:
        
        print('\n🔄 CREATING THE DATA SPLITS')
        print("=" * 50)
        
        try:
            num_peas = self.graph_data['PEA'].num_nodes
            
            print(f"📊 USING ALL {num_peas:,} AVAILABLE PEAs")
            
            if num_peas < 3:
                print(f"⚠️ Few PEAs ({num_peas}) - using minimal splits")
                self.splits = {
                    'train': torch.arange(max(1, num_peas - 2), device=self.config.device),
                    'val': torch.tensor([max(0, num_peas - 2)], device=self.config.device) if num_peas > 1 else torch.tensor([0], device=self.config.device),
                    'test': torch.tensor([num_peas - 1], device=self.config.device) if num_peas > 0 else torch.tensor([0], device=self.config.device)
                }
            else:
                indices = torch.randperm(num_peas, device=self.config.device)
                
                train_size = int(self.config.data_split['train'] * num_peas)
                val_size = int(self.config.data_split['val'] * num_peas)
                test_size = num_peas - train_size - val_size
                
                if train_size == 0:
                    train_size = 1
                if val_size == 0:
                    val_size = 1
                if test_size == 0:
                    test_size = 1
                    
                total_needed = train_size + val_size + test_size
                if total_needed > num_peas:
                    train_size = max(1, int(train_size * num_peas / total_needed))
                    val_size = max(1, int(val_size * num_peas / total_needed))
                    test_size = num_peas - train_size - val_size
                
                self.splits = {
                    'train': indices[:train_size],
                    'val': indices[train_size:train_size + val_size],
                    'test': indices[train_size + val_size:train_size + val_size + test_size]
                }
            
            print(f"✅ Splits created with ALL the data:")
            print(f"   TOTAL PEAs: {num_peas:,}")
            print(f"   Treino: {len(self.splits['train']):,} PEAs ({len(self.splits['train'])/num_peas*100:.1f}%)")
            print(f"   Validation: {len(self.splits['val']):,} PEAs ({len(self.splits['val'])/num_peas*100:.1f}%)")
            print(f"   Teste: {len(self.splits['test']):,} PEAs ({len(self.splits['test'])/num_peas*100:.1f}%)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error while creating splits: {str(e)}")
            return False
    
    def create_loaders(self) -> bool:
        
        print("\n🔄 CRIANDO DATA LOADERS")
        print("=" * 50)
        
        try:
            if self.splits is None:
                print('❌ Splits were not created!')
                return False
            
            for split_name, split_indices in self.splits.items():
                if len(split_indices) == 0:
                    print(f"❌ Split '{split_name}' is empty!")
                    return False
                
                if split_indices.dtype != torch.long:
                    self.splits[split_name] = split_indices.long()
                
                if split_indices.device != self.config.device:
                    self.splits[split_name] = split_indices.to(self.config.device)
                
                max_idx = self.graph_data['PEA'].num_nodes
                if torch.any(split_indices >= max_idx) or torch.any(split_indices < 0):
                    print(f"❌ Invalid indices in split '{split_name}'")
                    return False
            
            for edge_type in self.graph_data.edge_types:
                if hasattr(self.graph_data[edge_type], 'edge_index'):
                    edge_index = self.graph_data[edge_type].edge_index
                    
                    if edge_index.dtype != torch.long:
                        self.graph_data[edge_type].edge_index = edge_index.long()
                    
                    if edge_index.device != self.config.device:
                        self.graph_data[edge_type].edge_index = edge_index.to(self.config.device)
            
            loader_config = {
                'num_neighbors': self.config.num_neighbors,
                'batch_size': self.config.batch_size,
                'num_workers': self.config.num_workers,
                'persistent_workers': False,
                'pin_memory': True if self.config.device == 'cuda' else False
            }
            
            print(f"   Configurações:")
            for key, value in loader_config.items():
                print(f"     {key}: {value}")
            
            self.loaders = {}
            
            print('🔄 Creating the training loader...')
            try:
                self.loaders['train'] = NeighborLoader(
                    self.graph_data,
                    input_nodes=('PEA', self.splits['train']),
                    shuffle=True,
                    **loader_config
                )
            except Exception as e:
                print(f"❌ Error while creating the training loader: {str(e)}")
                return False
            
            print('🔄 Creating the validation loader...')
            try:
                self.loaders['val'] = NeighborLoader(
                    self.graph_data,
                    input_nodes=('PEA', self.splits['val']),
                    shuffle=False,
                    **loader_config
                )
            except Exception as e:
                print(f"❌ Error while creating the validation loader: {str(e)}")
                return False
            
            print('🔄 Creating the test loader...')
            try:
                self.loaders['test'] = NeighborLoader(
                    self.graph_data,
                    input_nodes=('PEA', self.splits['test']),
                    shuffle=False,
                    **loader_config
                )
            except Exception as e:
                print(f"❌ Error while creating the test loader: {str(e)}")
                return False
            
            num_train_batches = len(self.loaders['train'])
            num_val_batches = len(self.loaders['val'])
            num_test_batches = len(self.loaders['test'])
            
            print(f"✅ Loaders criados:")
            print(f"   Treino: {num_train_batches} batches")
            print(f"   Validação: {num_val_batches} batches")
            print(f"   Teste: {num_test_batches} batches")
            
            return True
            
        except Exception as e:
            print(f"❌ Error while creating loaders: {str(e)}")
            return False
    
    def get_data_info(self) -> Dict:
        
        if self.graph_data is None:
            return {}
        
        info = {
            'node_types': list(self.graph_data.node_types),
            'edge_types': list(self.graph_data.edge_types),
            'num_nodes': {},
            'num_edges': {},
            'device': str(self.graph_data['PEA'].x.device) if 'PEA' in self.graph_data.node_types else 'unknown'
        }
        
        total_nodes = 0
        for node_type in self.graph_data.node_types:
            num_nodes = self.graph_data[node_type].num_nodes
            info['num_nodes'][node_type] = num_nodes
            total_nodes += num_nodes
        
        info['total_nodes'] = total_nodes
        
        total_edges = 0
        for edge_type in self.graph_data.edge_types:
            num_edges = self.graph_data[edge_type].num_edges
            info['num_edges'][edge_type] = num_edges
            total_edges += num_edges
        
        info['total_edges'] = total_edges
        
        if self.splits is not None:
            info['splits'] = {
                'train': len(self.splits['train']),
                'val': len(self.splits['val']),
                'test': len(self.splits['test'])
            }
        
        return info
    
    def validate_batch(self, batch: HeteroData) -> bool:
        
        try:
            for node_type in batch.node_types:
                if not hasattr(batch[node_type], 'x') or batch[node_type].x is None:
                    return False
            
            if 'PEA' not in batch.node_types:
                return False
            
            if batch['PEA'].x.shape[1] < 2:
                return False
            
            if len(batch.edge_types) == 0:
                return False
            
            for edge_type in batch.edge_types:
                if hasattr(batch[edge_type], 'edge_index') and batch[edge_type].edge_index is not None:
                    edge_index = batch[edge_type].edge_index
                    
                    if edge_index.shape[0] != 2:
                        return False
                    
                    if edge_index.shape[1] == 0:
                        continue
                    
                    src_type, _, dst_type = edge_type
                    if src_type not in batch.node_types or dst_type not in batch.node_types:
                        return False
                    
                    max_src = batch[src_type].num_nodes
                    max_dst = batch[dst_type].num_nodes
                    
                    if max_src == 0 or max_dst == 0:
                        return False
                    
                    if torch.any(edge_index < 0):
                        return False
                    
                    if torch.any(edge_index[0] >= max_src) or torch.any(edge_index[1] >= max_dst):
                        return False
            
            return True
            
        except Exception:
            return False
    
    def get_sample_batch(self, split: str = 'train') -> Optional[HeteroData]:
        
        if self.loaders is None or split not in self.loaders:
            return None
        
        try:
            for batch in self.loaders[split]:
                return batch
        except Exception:
            return None
    
    def cleanup(self):
        
        if self.config.device == 'cuda':
            torch.cuda.empty_cache()
        
        print('✅ Memory cleanup is finished')

class DemoDataLoader:
    
    def __init__(self, config: ColabConfig):
        self.config = config
        self.graph_data = None
        self.splits = None
        self.loaders = None
    
    def create_demo_data(self, num_peas: int = 1000) -> bool:
        
        print(f"🎲 CREATING DEMONSTRATION DATA ({num_peas} PEAs)")
        
        self.graph_data = HeteroData()
        
        self.graph_data['PEA'].x = torch.randn(num_peas, 4, device=self.config.device)
        
        num_erbs = num_peas // 5
        self.graph_data['ERB'].x = torch.randn(num_erbs, 6, device=self.config.device)
        
        num_cps = num_peas // 3
        self.graph_data['CPS'].x = torch.randn(num_cps, 3, device=self.config.device)
        
        self.graph_data['PEA', 'connects', 'ERB'].edge_index = torch.randint(
            0, min(num_peas, num_erbs), (2, num_peas * 2), device=self.config.device
        )
        
        self.graph_data['PEA', 'connects', 'CPS'].edge_index = torch.randint(
            0, min(num_peas, num_cps), (2, num_peas), device=self.config.device
        )
        
        indices = torch.randperm(num_peas, device=self.config.device)
        train_size = int(0.7 * num_peas)
        val_size = int(0.15 * num_peas)
        
        self.splits = {
            'train': indices[:train_size],
            'val': indices[train_size:train_size + val_size],
            'test': indices[train_size + val_size:]
        }
        
        print(f"✅ Dados demo criados: {num_peas} PEAs, {num_erbs} ERBs, {num_cps} CPS")
        
        return True

def mount_google_drive():
    
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print('✅ Google Drive mounted successfully!')
        return True
    except ImportError:
        print('❌ Not running in Google Colab!')
        return False
    except Exception as e:
        print(f"❌ Error while mounting Google Drive: {str(e)}")
        return False

def check_data_files(config: ColabConfig) -> Dict[str, bool]:
    
    files_status = {}
    
    required_files = {
        'graph': config.graph_path,
        'metadata': config.metadata_path,
        'scalers': config.scalers_path
    }
    
    for name, path in required_files.items():
        exists = Path(path).exists()
        files_status[name] = exists
        
        if exists:
            size = Path(path).stat().st_size
            print(f"✅ {name}: {Path(path).name} ({size:,} bytes)")
        else:
            print(f"❌ {name}: {Path(path).name} (not found)")
    
    return files_status

def create_data_loader(config: ColabConfig, use_demo: bool = False) -> Union[ARPIADataLoader, DemoDataLoader]:
    
    if use_demo:
        return DemoDataLoader(config)
    else:
        return ARPIADataLoader(config)

def create_data_loaders(graph_data, batch_size=2048, num_neighbors=[128, 96, 64], 
                       num_workers=4, device='cuda'):
    from torch_geometric.loader import NeighborLoader
    
    if isinstance(device, torch.device):
        is_cpu = (device.type == 'cpu')
        device_str = str(device)
    else:
        is_cpu = (device == 'cpu')
        device_str = device
    
    if is_cpu:
        print(f"🔄 Creating data loaders optimized for local CPU (batch={batch_size}, workers={num_workers})...")
    else:
        print('🔄 Creating MASSIVE data loaders for A100...')
    
    graph_data = prepare_graph_data(graph_data, device)
    
    num_peas = graph_data['PEA'].num_nodes
    
    indices = torch.randperm(num_peas, device=device)
    train_size = int(0.7 * num_peas)
    val_size = int(0.2 * num_peas)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    loader_config = {
        'num_neighbors': num_neighbors,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'persistent_workers': num_workers > 0,
        'pin_memory': not is_cpu,
        'prefetch_factor': 4 if is_cpu else 16,
        'subgraph_type': 'bidirectional'
    }
    
    try:
        loaders = {}
        
        print(f"   Creating the training loader...")
        loaders['train'] = NeighborLoader(
            graph_data,
            input_nodes=('PEA', train_idx),
            shuffle=True,
            **loader_config
        )
        
        print(f"   Creating the validation loader...")
        loaders['val'] = NeighborLoader(
            graph_data,
            input_nodes=('PEA', val_idx),
            shuffle=False,
            **loader_config
        )
        
        print(f"   Creating the test loader...")
        loaders['test'] = NeighborLoader(
            graph_data,
            input_nodes=('PEA', test_idx),
            shuffle=False,
            **loader_config
        )
        
        print(f"✅ Data loaders criados:")
        print(f"   Train: {len(train_idx):,} PEAs")
        print(f"   Val: {len(val_idx):,} PEAs") 
        print(f"   Test: {len(test_idx):,} PEAs")
        print(f"   Batch size: {loader_config['batch_size']}")
        
        return loaders
        
    except Exception as e:
        print(f"❌ Error while creating data loaders: {e}")
        print('🔄 Trying an even more conservative configuration...')
        
        fallback_config = {
            'num_neighbors': [20, 15, 10],
            'batch_size': 128,
            'num_workers': 0,
            'persistent_workers': False,
            'pin_memory': True
        }
        
        loaders = {
            'train': NeighborLoader(
                graph_data,
                input_nodes=('PEA', train_idx),
                shuffle=True,
                **fallback_config
            ),
            'val': NeighborLoader(
                graph_data,
                input_nodes=('PEA', val_idx),
                shuffle=False,
                **fallback_config
            ),
            'test': NeighborLoader(
                graph_data,
                input_nodes=('PEA', test_idx),
                shuffle=False,
                **fallback_config
            )
        }
        
        print(f"✅ Data loaders created (conservative configuration):")
        print(f"   Train: {len(train_idx):,} PEAs")
        print(f"   Val: {len(val_idx):,} PEAs") 
        print(f"   Test: {len(test_idx):,} PEAs")
        print(f"   Batch size: {fallback_config['batch_size']}")
        
        return loaders


def prepare_graph_data(graph_data, device='cuda'):
    print('🔄 Preparing the graph data...')
    
    graph_data = graph_data.to(device)
    
    if 'PEA' not in graph_data.node_types:
        raise ValueError("Graph must contain nodes of type 'PEA'")
    
    if not hasattr(graph_data['PEA'], 'x') or graph_data['PEA'].x is None:
        raise ValueError("PEAs devem ter features (coordenadas)")
    
    for node_type in graph_data.node_types:
        if not hasattr(graph_data[node_type], 'x') or graph_data[node_type].x is None:
            num_nodes = graph_data[node_type].num_nodes
            graph_data[node_type].x = torch.zeros(num_nodes, 2, device=device)
            print(f"✅ Features created for {node_type}: {num_nodes} nodes")
    
    print('✅ Graph data are prepared')
    return graph_data


if __name__ == "__main__":
    config = ColabConfig()
    
    files_status = check_data_files(config)
    
    if all(files_status.values()):
        data_loader = create_data_loader(config, use_demo=False)
        success = data_loader.load_data()
        
        if success:
            data_loader.create_splits()
            data_loader.create_loaders()
            
            info = data_loader.get_data_info()
            print(f"\n📊 Data loaded: {info['total_nodes']:,} nodes, {info['total_edges']:,} edges")
    else:
        print('⚠️ Using the demonstration data')
        demo_loader = create_data_loader(config, use_demo=True)
        demo_loader.create_demo_data() 
