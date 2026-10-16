#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runs the continuation of the ablation study."""

from ablation_study_continuation import main

if __name__ == "__main__":
    print('🚀 Starting the continuation of the ablation study...')
    print('📋 Running only the pending tests:')
    print("   - Sem Attention (no_attention)")
    print("   - Camadas Reduzidas (reduced_layers)")
    print('📂 Loading the results of the tests already finished...')
    print("="*60)
    
    results = main()
    
    if results:
        print('\n🎉 Continuation finished successfully!')
        print('📊 Check the results in: ablation_results_arpia_v3/')
    else:
        print('\n❌ Error while running the continuation')


