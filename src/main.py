import argparse
import yaml
import json
import os
from pathlib import Path

from .train import UniRISKTrainer
from .evaluate import evaluate_model
from .preprocess import prepare_datasets

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="UniRISK-T: Unified, Trustworthy, Token-Aware Risk Transformer")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test with minimal configuration")
    parser.add_argument("--full-experiment", action="store_true", help="Run full experiment")
    
    args = parser.parse_args()
    
    if args.smoke_test:
        config_path = "config/smoke_test.yaml"
        print("=== SMOKE TEST EXECUTION ===")
    elif args.full_experiment:
        config_path = "config/full_experiment.yaml"
        print("=== FULL EXPERIMENT EXECUTION ===")
    else:
        print("Please specify either --smoke-test or --full-experiment")
        return
    
    config = load_config(config_path)
    
    os.makedirs(config['output']['save_dir'], exist_ok=True)
    os.makedirs(f"{config['output']['save_dir']}/images", exist_ok=True)
    
    print(f"\nExperiment: {config['experiment']['name']}")
    print(f"Configuration loaded from: {config_path}")
    print(f"Output directory: {config['output']['save_dir']}")
    
    print("\n=== PHASE 1: DATA PREPARATION ===")
    train_dataset, val_dataset, calib_dataset = prepare_datasets(config)
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Calibration samples: {len(calib_dataset)}")
    
    print("\n=== PHASE 2: MODEL TRAINING ===")
    trainer = UniRISKTrainer(config)
    model, training_results = trainer.train(train_dataset, val_dataset)
    
    print(f"Training completed in {training_results['epochs']} epochs")
    print(f"Final training loss: {training_results['final_loss']:.4f}")
    print(f"Best validation score: {training_results['best_val_score']:.4f}")
    
    print("\n=== PHASE 3: EVALUATION ===")
    eval_results = evaluate_model(model, val_dataset, calib_dataset, config)
    
    print("\n=== EXPERIMENT RESULTS ===")
    print(f"MT-Bench Score: {eval_results['mt_bench_score']:.2f}")
    print(f"Worst-axis CVaR: {eval_results['worst_cvar']:.4f}")
    print(f"Certificate Violation Rate: {eval_results['cert_violation_rate']:.2%}")
    print(f"Span Repair Success: {eval_results['span_repair_success']:.2%}")
    
    for k, hazard in enumerate(eval_results['per_dim_hazards']):
        print(f"Dimension {k+1} 99th percentile hazard: {hazard:.4f}")
    
    results_file = f"{config['output']['save_dir']}/experiment_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'config': config,
            'training_results': training_results,
            'evaluation_results': eval_results
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Visualization saved to: {eval_results['figure_path']}")
    
    print(f"\nJSON Results Contents:")
    with open(results_file, 'r') as f:
        print(f.read())
    
    if args.smoke_test:
        print("\n=== SMOKE TEST COMPLETED SUCCESSFULLY ===")
        print("Proceeding to full experiment...")
        
        full_config = load_config("config/full_experiment.yaml")
        print(f"\nFull experiment would run with:")
        print(f"- {full_config['experiment']['epochs']} epochs")
        print(f"- {full_config['data']['train_samples']} training samples")
        print(f"- Batch size: {full_config['experiment']['batch_size']}")
        print("(Full experiment not executed in smoke test mode)")

if __name__ == "__main__":
    main()
