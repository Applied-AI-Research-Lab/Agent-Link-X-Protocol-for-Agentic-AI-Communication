#!/usr/bin/env python3
"""
Master script to run all three experiments sequentially.
This script orchestrates the complete experimental pipeline.
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def check_dependencies():
    """Check if required packages are installed."""
    print("Checking dependencies...")
    missing = []
    
    try:
        import openai
        print("openai")
    except ImportError:
        missing.append("openai")
        print("openai")
    
    try:
        import transformers
        print("transformers")
    except ImportError:
        missing.append("transformers")
        print("transformers")
    
    try:
        import torch
        print("torch")
    except ImportError:
        missing.append("torch")
        print("torch (optional, but recommended)")
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("All dependencies installed\n")
    return True

def check_api_key():
    """Check if OpenAI API key is set."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        print("OpenAI API key not found!")
        print("\nPlease set your API key:")
        print("export OPENAI_API_KEY='your-key-here'")
        print("\nOr edit the API_KEY variable in the experiment scripts.")
        return False
    print(f"API key found: {api_key[:10]}...\n")
    return True

def run_experiment_3():
    """Run Experiment 3 (no API key needed)."""
    print("="*80)
    print("RUNNING EXPERIMENT 3: Cross-Linguistic Tokenization")
    print("="*80)
    print("This experiment requires NO API calls (free)\n")
    
    try:
        import experiment3_cross_linguistic_tokenization as exp3
        result = exp3.run_experiment_3()
        if result:
            print("\nExperiment 3 completed successfully")
            return True
        else:
            print("\nExperiment 3 failed")
            return False
    except Exception as e:
        print(f"\nExperiment 3 error: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_experiment_1():
    """Run Experiment 1."""
    print("\n" + "="*80)
    print("RUNNING EXPERIMENT 1: Human-AI Message Cost Analysis")
    print("="*80)
    print("This experiment will make ~20 API calls")
    print("Estimated cost: $0.05-0.10 with gpt-4o-mini\n")
    
    try:
        import experiment1_human_ai_message_cost as exp1
        result = exp1.run_experiment_1(exp1.SAMPLE_TEXTS, exp1.API_KEY)
        if result:
            print("\nExperiment 1 completed successfully")
            return True
        else:
            print("\nExperiment 1 failed")
            return False
    except Exception as e:
        print(f"\nExperiment 1 error: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_experiment_2():
    """Run Experiment 2."""
    print("\n" + "="*80)
    print("RUNNING EXPERIMENT 2: Multi-Agent Communication")
    print("="*80)
    print("This experiment will make ~32 API calls")
    print("Estimated cost: $0.20-0.40 with gpt-4o-mini\n")
    
    try:
        import experiment2_multi_agent_communication as exp2
        result = exp2.run_experiment_2(exp2.API_KEY)
        if result:
            print("\nExperiment 2 completed successfully")
            return True
        else:
            print("\nExperiment 2 failed")
            return False
    except Exception as e:
        print(f"\nExperiment 2 error: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_final_report(results):
    """Generate a final summary report."""
    print("\n" + "="*80)
    print("EXPERIMENTAL PIPELINE COMPLETE")
    print("="*80)
    
    print("\nResults Summary:")
    print(f"  Experiment 1 (Message Cost): {'Success' if results['exp1'] else 'Failed/Skipped'}")
    print(f"  Experiment 2 (Multi-Agent): {'Success' if results['exp2'] else 'Failed/Skipped'}")
    print(f"  Experiment 3 (Cross-Linguistic): {'Success' if results['exp3'] else 'Failed/Skipped'}")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Check the 'results/' directory for all output files")
    print("\n2. LaTeX tables are ready to copy:")
    print(" - experiment1_table_*.tex")
    print(" - experiment2_table_*.tex")
    print(" - experiment3_table_*.tex")
    print("\n3. JSON files contain detailed data for further analysis:")
    print(" - *_detailed_*.json (raw data)")
    print(" - *_summary_*.json (statistics)")
    print("\n4. Review the results")
    print("\n" + "="*80)

def main():
    """Main entry point."""
    print("\n" + "="*80)
    print("BITSTRINGS EXPERIMENTAL PIPELINE")
    print("Machine-Native Communication Efficiency Experiments")
    print("="*80)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check API key (only for experiments 1 and 2)
    api_key_available = check_api_key()
    
    # Track results
    results = {
        'exp1': False,
        'exp2': False,
        'exp3': False
    }
    
    # Run experiments
    print("="*80)
    print("EXECUTION PLAN")
    print("="*80)
    print("\nExperiment 3 will run first (no API calls, free)")
    if api_key_available:
        print("Experiments 1 and 2 will prompt for confirmation before making API calls")
    else:
        print("Experiments 1 and 2 will be SKIPPED (no API key)")
    print("\n" + "="*80)
    
    input("\nPress Enter to begin...")
    
    # Experiment 3 (always run, no API needed)
    results['exp3'] = run_experiment_3()
    
    # Experiments 1 and 2 (need API key)
    if api_key_available:
        results['exp1'] = run_experiment_1()
        results['exp2'] = run_experiment_2()
    else:
        print("\nSkipping Experiments 1 and 2 (API key required)")
    
    # Generate final report
    generate_final_report(results)
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "="*80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        print("Results from completed experiments are saved in the 'results/' directory")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
