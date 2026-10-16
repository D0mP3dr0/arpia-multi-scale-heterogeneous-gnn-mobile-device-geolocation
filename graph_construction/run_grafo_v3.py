#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runs the V3 graph construction and its exploratory analysis."""

import os
import sys
import time
import argparse
import subprocess
import logging
import json
from datetime import datetime
from pathlib import Path

DEFAULT_CONFIG = "grafo_v3.yaml"
CONSTRUCTION_SCRIPT = "grafo_v3.py"
EDA_SCRIPT = "eda_grafo_v3.py"

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    HAS_COLORS = True
except ImportError:
    HAS_COLORS = False
    class Fore:
        GREEN = BLUE = RED = YELLOW = CYAN = MAGENTA = ""
    class Style:
        BRIGHT = ""

class ARPIAPipelineRunnerV3:
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path).resolve()
        if not self.config_path.exists():
            self._fatal_error(f"Configuration file not found: {self.config_path}")
        
        self.base_dir = self.config_path.parent
        self.start_time = datetime.now()
        self.logger = self._setup_logging()
        self.config = self._load_config()
        self.output_dir = Path(self.config['paths']['output']['base_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {
            'start_time': self.start_time.isoformat(),
            'version': '3.0',
            'config_file': str(self.config_path),
            'steps': [],
            'status': 'pending'
        }
    
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('ARPIA_Pipeline_V3')
        logger.setLevel(logging.INFO)
        
        if logger.handlers:
            logger.handlers.clear()
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def _setup_file_logging(self):
        log_dir = self.output_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"pipeline_v3_{self.start_time.strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        logger = logging.getLogger('ARPIA_Pipeline_V3')
        logger.addHandler(file_handler)
        logger.info(f"V3 log being saved to: {log_file}")
    
    def _load_config(self) -> dict:
        try:
            import yaml
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            self._fatal_error(f"Error while loading the configuration file: {str(e)}")
    
    def _print_header(self):
        header = "=" * 80
        title = f"ARPIA V3 - GRAPH PIPELINE v{self.config['project']['version']} (GNN-READY)"
        
        if HAS_COLORS:
            print(f"\n{Fore.CYAN}{Style.BRIGHT}{header}")
            print(f"{Fore.CYAN}{Style.BRIGHT}{title.center(80)}")
            print(f"{Fore.CYAN}{Style.BRIGHT}{header}{Style.RESET_ALL}\n")
        else:
            print(f"\n{header}")
            print(title.center(80))
            print(f"{header}\n")
        
        print(f"Data/Hora: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Configuração: {self.config_path}")
        print(f"Output directory in use: {self.output_dir}")
        
        if HAS_COLORS:
            print(f"\n{Fore.GREEN}{Style.BRIGHT}🚀 NOVIDADES DA V3:{Style.RESET_ALL}")
            print(f"   • New node types: {Fore.YELLOW}PEA, ERB, CPS, CTC, CTNC{Style.RESET_ALL}")
            print(f"   • Proximity connections: {Fore.YELLOW}Nodes of the same type{Style.RESET_ALL}")
            print(f"   • Expanded analysis: {Fore.YELLOW}Proximity patterns{Style.RESET_ALL}")
            print(f"   • Estrutura triangular: {Fore.YELLOW}Contexto espacial ampliado{Style.RESET_ALL}")
            print(f"\n{Fore.GREEN}{Style.BRIGHT}✅ CORRECTIONS KEPT FOR GNN TRAINING:{Style.RESET_ALL}")
            print(f"   • Force model: {Fore.RED}DISABLED{Style.RESET_ALL} (avoids bias)")
            print(f"   • Features calculadas: {Fore.RED}REMOVIDAS{Style.RESET_ALL} (GNN aprende)")
            print(f"   • Coordenadas: {Fore.GREEN}PADRONIZADAS{Style.RESET_ALL} (x_norm, y_norm)")
        else:
            print(f"\n🚀 NOVIDADES DA V3:")
            print(f"   • New node types: PEA, ERB, CPS, CTC, CTNC")
            print(f"   • Proximity connections: Nodes of the same type")
            print(f"   • Expanded analysis: Proximity patterns")
            print(f"   • Estrutura triangular: Contexto espacial ampliado")
            print(f"\n✅ CORRECTIONS KEPT FOR GNN TRAINING:")
            print(f"   • Force model: DISABLED (avoids bias)")
            print(f"   • Features calculadas: REMOVIDAS (GNN aprende)")
            print(f"   • Coordenadas: PADRONIZADAS (x_norm, y_norm)")
        print("\n")
    
    def _fatal_error(self, message: str):
        if HAS_COLORS:
            print(f"\n{Fore.RED}{Style.BRIGHT}ERRO FATAL V3: {message}{Style.RESET_ALL}")
        else:
            print(f"\nERRO FATAL V3: {message}")
        sys.exit(1)
    
    def _run_step(self, step_name: str, script_path: str, args: list = None) -> bool:
        if args is None:
            args = []
        
        full_script_path = self.base_dir / script_path
        if not full_script_path.exists():
            self.logger.error(f"Script was not found: {full_script_path}")
            return False
        
        cmd = [sys.executable, str(full_script_path)] + args
        cmd_str = " ".join(cmd)
        
        step_start = datetime.now()
        self.logger.info(f"Iniciando etapa V3: {step_name}")
        self.logger.info(f"Comando: {cmd_str}")
        
        if HAS_COLORS:
            print(f"\n{Fore.YELLOW}{Style.BRIGHT}[V3 - {step_name}] Iniciando...{Style.RESET_ALL}")
        else:
            print(f"\n[V3 - {step_name}] Iniciando...")
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=str(self.base_dir)
            )
            
            output_lines = []
            for line in iter(process.stdout.readline, ''):
                line = line.rstrip()
                output_lines.append(line)
                
                if HAS_COLORS:
                    if "erro" in line.lower() or "error" in line.lower():
                        print(f"{Fore.RED}{line}")
                    elif "aviso" in line.lower() or "warning" in line.lower():
                        print(f"{Fore.YELLOW}{line}")
                    elif "info" in line.lower():
                        print(f"{Fore.CYAN}{line}")
                    elif "v3" in line.lower():
                        print(f"{Fore.MAGENTA}{line}")
                    else:
                        print(line)
                else:
                    print(line)
                
                self.logger.info(f"[V3 - {step_name}] {line}")
            
            return_code = process.wait()
            
            step_end = datetime.now()
            duration = (step_end - step_start).total_seconds()
            
            step_result = {
                'name': step_name,
                'start_time': step_start.isoformat(),
                'end_time': step_end.isoformat(),
                'duration_seconds': duration,
                'command': cmd_str,
                'return_code': return_code,
                'status': 'success' if return_code == 0 else 'error',
                'output_lines': len(output_lines),
                'version': '3.0'
            }
            self.results['steps'].append(step_result)
            
            if return_code == 0:
                if HAS_COLORS:
                    print(f"\n{Fore.GREEN}{Style.BRIGHT}[V3 - {step_name}] ✓ Finished successfully in {duration:.2f}s{Style.RESET_ALL}")
                else:
                    print(f"\n[V3 - {step_name}] ✓ Finished successfully in {duration:.2f}s")
                self.logger.info(f"V3 step '{step_name}' finished successfully in {duration:.2f}s")
                return True
            else:
                if HAS_COLORS:
                    print(f"\n{Fore.RED}{Style.BRIGHT}[V3 - {step_name}] ✗ Failed with code {return_code} after {duration:.2f}s{Style.RESET_ALL}")
                else:
                    print(f"\n[V3 - {step_name}] ✗ Failed with code {return_code} after {duration:.2f}s")
                self.logger.error(f"V3 step '{step_name}' failed with code {return_code} after {duration:.2f}s")
                return False
                
        except Exception as e:
            self.logger.error(f"Error while running V3 step '{step_name}': {str(e)}", exc_info=True)
            if HAS_COLORS:
                print(f"\n{Fore.RED}{Style.BRIGHT}[V3 - {step_name}] ✗ Erro: {str(e)}{Style.RESET_ALL}")
            else:
                print(f"\n[V3 - {step_name}] ✗ Erro: {str(e)}")
            
            self.results['steps'].append({
                'name': step_name,
                'start_time': step_start.isoformat(),
                'end_time': datetime.now().isoformat(),
                'command': cmd_str,
                'status': 'error',
                'error': str(e),
                'version': '3.0'
            })
            return False
    
    def run(self, args):
        self._print_header()
        self._setup_file_logging()
        
        self.logger.info(f"Iniciando pipeline ARPIA V3 v{self.config['project']['version']}")
        
        try:
            if not args.skip_build:
                success = self._run_step(
                    'Construction of the V3 Graph', 
                    CONSTRUCTION_SCRIPT, 
                    [str(self.config_path)]
                )
                if not success and not args.continue_on_error:
                    self.logger.error('Construction of the V3 graph failed, pipeline interrupted.')
                    self.results['status'] = 'error'
                    self._save_results()
                    self._print_summary()
                    return
            else:
                self.logger.info('V3 graph construction step skipped as requested.')
                if HAS_COLORS:
                    print(f"{Fore.YELLOW}[V3 - Graph Construction] Skipped as requested.")
                else:
                    print('[V3 - Graph Construction] Skipped as requested.')
            
            if not args.skip_eda:
                graph_file = self.output_dir / self.config['paths']['output']['graph_file']
                metadata_file = self.output_dir / self.config['paths']['output']['metadata_file']
                
                if not graph_file.exists():
                    self.logger.error(f"V3 graph file not found: {graph_file}")
                    if not args.continue_on_error:
                        self.results['status'] = 'error'
                        self._save_results()
                        self._print_summary()
                        return
                
                success = self._run_step(
                    "Análise Exploratória V3", 
                    EDA_SCRIPT, 
                    [str(graph_file), str(metadata_file), str(self.output_dir / "eda_v3")]
                )
                if not success and not args.continue_on_error:
                    self.logger.error('V3 exploratory analysis failed, pipeline interrupted.')
                    self.results['status'] = 'error'
                    self._save_results()
                    self._print_summary()
                    return
            else:
                self.logger.info('V3 exploratory analysis step skipped as requested.')
                if HAS_COLORS:
                    print(f"{Fore.YELLOW}[V3 - Exploratory Analysis] Skipped as requested.")
                else:
                    print('[V3 - Exploratory Analysis] Skipped as requested.')
            
            self.results['status'] = 'success'
            self._save_results()
            self._print_summary()
            
        except KeyboardInterrupt:
            self.logger.warning('V3 pipeline interrupted by the user.')
            self.results['status'] = 'interrupted'
            self._save_results()
            self._print_summary()
        except Exception as e:
            self.logger.error(f"Unhandled error during the V3 pipeline run: {str(e)}", exc_info=True)
            self.results['status'] = 'error'
            self.results['error'] = str(e)
            self._save_results()
            self._print_summary()
    
    def _save_results(self):
        self.results['end_time'] = datetime.now().isoformat()
        self.results['duration_seconds'] = (datetime.fromisoformat(self.results['end_time']) - 
                                          datetime.fromisoformat(self.results['start_time'])).total_seconds()
        
        results_file = self.output_dir / f"pipeline_results_v3_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
        
        self.logger.info(f"V3 pipeline results saved to: {results_file}")
    
    def _print_summary(self):
        status_text = self.results['status'].upper()
        duration = self.results.get('duration_seconds', 0)
        
        header = "=" * 80
        title = "RESUMO DO PIPELINE V3"
        
        if HAS_COLORS:
            status_color = Fore.GREEN if self.results['status'] == 'success' else Fore.RED
            print(f"\n{Fore.CYAN}{header}")
            print(f"{Fore.CYAN}{title.center(80)}")
            print(f"{Fore.CYAN}{header}")
            print(f"\nStatus: {status_color}{Style.BRIGHT}{status_text}{Style.RESET_ALL}")
        else:
            print(f"\n{header}")
            print(title.center(80))
            print(f"{header}")
            print(f"\nStatus: {status_text}")
        
        print(f"Total duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        print(f"Versão: {self.results.get('version', '3.0')}")
        print(f"\nEtapas executadas: {len(self.results['steps'])}")
        
        for i, step in enumerate(self.results['steps'], 1):
            step_status = step['status']
            step_duration = step.get('duration_seconds', 0)
            
            if HAS_COLORS:
                status_color = Fore.GREEN if step_status == 'success' else Fore.RED
                print(f"  {i}. {step['name']}: {status_color}{step_status.upper()}{Style.RESET_ALL} ({step_duration:.2f}s)")
            else:
                print(f"  {i}. {step['name']}: {step_status.upper()} ({step_duration:.2f}s)")
        
        print(f"\nOutput directory in use: {self.output_dir}")
        
        if self.results['status'] == 'success':
            if HAS_COLORS:
                print(f"\n{Fore.GREEN}{Style.BRIGHT}🎉 V3 PIPELINE FINISHED SUCCESSFULLY!{Style.RESET_ALL}")
                print(f"   • Expanded heterogeneous graph with 5 node types")
                print(f"   • Proximity connections between nodes of the same type")
                print(f"   • Estruturas triangulares ampliadas")
                print(f"   • Ready for advanced GNN training")
            else:
                print(f"\n🎉 V3 PIPELINE FINISHED SUCCESSFULLY!")
                print(f"   • Expanded heterogeneous graph with 5 node types")
                print(f"   • Proximity connections between nodes of the same type")
                print(f"   • Estruturas triangulares ampliadas")
                print(f"   • Ready for advanced GNN training")
        
        print(header)

def main():
    parser = argparse.ArgumentParser(description='ARPIA V3 - GNN Graph Pipeline')
    parser.add_argument("--config", "-c", default=DEFAULT_CONFIG,
                      help=f"Path to the YAML configuration file (default: {DEFAULT_CONFIG})")
    parser.add_argument("--skip-build", action="store_true",
                      help='Skip the V3 graph construction step')
    parser.add_argument("--skip-eda", action="store_true",
                      help='Skip the V3 exploratory analysis step')
    parser.add_argument("--continue-on-error", action="store_true",
                      help='Continue the run even if a step fails')
    
    args = parser.parse_args()
    
    try:
        runner = ARPIAPipelineRunnerV3(args.config)
        runner.run(args)
    except Exception as e:
        print(f"\nERRO FATAL V3: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 
