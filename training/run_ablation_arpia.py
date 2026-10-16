#!/usr/bin/env python3
"""Runs the ablation study."""

import sys
import logging
from pathlib import Path

current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from ablation_study_arpia_v3 import ARPIAAblationStudy

def main():
    print("="*80)
    print("🚀 INICIANDO ABLATION STUDY ARPIA V3")
    print("="*80)
    print()
    print("Este estudo irá:")
    print('1. Train the baseline model with the full configuration')
    print("2. Treinar 6 modelos ablados, removendo:")
    print("   - Multi-Scale GNN")
    print("   - Physics-Informed Loss")
    print("   - Connection Modeling")
    print("   - Spatial Constraints")  
    print("   - Precision Refinement")
    print("   - Progressive Training")
    print('3. Compare and rank the importance of the components')
    print()
    print("⏱️  Tempo estimado: 30-60 minutos (dependendo do hardware)")
    print("="*80)
    
    try:
        study = ARPIAAblationStudy()
        study.run_ablation_study()
        
        print('\n✅ Ablation Study finished successfully!')
        print(f"📁 Results saved to: {study.output_dir}")
        
    except KeyboardInterrupt:
        print('\n\n⚠️  Ablation Study interrupted by the user')
        sys.exit(1)
        
    except Exception as e:
        print(f"\n\n❌ ERRO: {str(e)}")
        logging.error('Error while running the ablation study', exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
