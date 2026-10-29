"""
main.py — Smart Traffic Signal Control using Neuro-Fuzzy System
================================================================
Main entry point that orchestrates the entire pipeline:

  1. Generate synthetic traffic data
  2. Initialize Fuzzy Logic Controller
  3. Train Neural Network
  4. Build Neuro-Fuzzy hybrid system
  5. Evaluate all systems
  6. Generate comprehensive visualizations
  7. Output metrics report

Usage:
  python main.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_generator import TrafficDataGenerator
from src.fuzzy_system import FuzzyTrafficController
from src.neural_network import TrafficNeuralNetwork
from src.neuro_fuzzy import NeuroFuzzyTrafficSystem
from src.visualizations import TrafficVisualizer


def print_banner():
    """Display project banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   🚦  SMART TRAFFIC SIGNAL CONTROL                          ║
    ║       using Neuro-Fuzzy System                               ║
    ║                                                              ║
    ║   Fuzzy Logic + Neural Network = Adaptive Intelligence       ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_section(title, char='═'):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    """Execute the complete Neuro-Fuzzy Traffic Control pipeline."""
    print_banner()

    # Create output directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('outputs/plots', exist_ok=True)
    os.makedirs('outputs/metrics', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # ============================================================
    # PHASE 1: Data Generation
    # ============================================================
    print_section("PHASE 1: Generating Synthetic Traffic Data")
    
    generator = TrafficDataGenerator(seed=42)
    df = generator.generate(n_samples=5000, save_csv=True, output_path='data/traffic_data.csv')
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"Features: {list(df.columns)}")
    print(f"\nDataset Statistics:")
    print(df.describe().round(2).to_string())
    
    # ============================================================
    # PHASE 2: Fuzzy Logic Controller
    # ============================================================
    print_section("PHASE 2: Initializing Fuzzy Logic Controller")
    
    fuzzy_controller = FuzzyTrafficController()
    print(f"\n✅ Fuzzy Controller initialized")
    print(f"   → {len(fuzzy_controller.rules)} fuzzy rules defined")
    print(f"   → Defuzzification: Centroid method")
    print(f"   → MF Types: Triangular & Trapezoidal")
    
    # Test fuzzy system
    test_scenarios = [
        {"num_vehicles": 10, "traffic_density": 15, "waiting_time": 20, "emergency_vehicle": 0,
         "desc": "Low Traffic (Quiet Road)"},
        {"num_vehicles": 50, "traffic_density": 55, "waiting_time": 90, "emergency_vehicle": 0,
         "desc": "Medium Traffic (Normal)"},
        {"num_vehicles": 85, "traffic_density": 90, "waiting_time": 150, "emergency_vehicle": 0,
         "desc": "High Traffic (Rush Hour)"},
        {"num_vehicles": 40, "traffic_density": 45, "waiting_time": 60, "emergency_vehicle": 1,
         "desc": "Emergency Vehicle Present"},
    ]
    
    print("\n  Fuzzy System Test Results:")
    print("  " + "-" * 75)
    for sc in test_scenarios:
        desc = sc.pop('desc')
        result = fuzzy_controller.predict(**sc)
        print(f"  Scenario: {desc}")
        print(f"    Input: V={sc['num_vehicles']}, D={sc['traffic_density']}%, "
              f"W={sc['waiting_time']}s, E={sc['emergency_vehicle']}")
        print(f"    → Green: {result['green_duration']}s, Priority: {result['signal_priority']}, "
              f"Congestion: {result['congestion_level']}%")
        sc['desc'] = desc
    
    # ============================================================
    # PHASE 3: Neural Network Training
    # ============================================================
    print_section("PHASE 3: Training Neural Network")
    
    nn = TrafficNeuralNetwork()
    print(f"\nModel Architecture:")
    print(nn.get_model_summary())
    
    nn.prepare_data(df)
    history = nn.train(epochs=150, batch_size=32, verbose=1)
    
    # ============================================================
    # PHASE 4: Neural Network Evaluation
    # ============================================================
    print_section("PHASE 4: Neural Network Evaluation")
    
    nn_metrics = nn.evaluate()
    
    print("\n  Neural Network Performance Metrics:")
    print("  " + "-" * 60)
    for target, metrics in nn_metrics.items():
        print(f"\n  {target.upper()}:")
        for metric, value in metrics.items():
            print(f"    {metric}: {value}")
    
    # Save metrics
    with open('outputs/metrics/nn_metrics.json', 'w') as f:
        json.dump(nn_metrics, f, indent=2)
    
    # ============================================================
    # PHASE 5: Neuro-Fuzzy Integration
    # ============================================================
    print_section("PHASE 5: Neuro-Fuzzy System Integration")
    
    nf_system = NeuroFuzzyTrafficSystem()
    nf_system.fuzzy_controller = fuzzy_controller
    nf_system.neural_network = nn
    nf_system.is_trained = True
    
    system_info = nf_system.get_system_info()
    print(f"\n  System Configuration:")
    print(f"  Fuzzy: {system_info['fuzzy']['type']}")
    print(f"  NN: {system_info['neural_network']['type']}")
    print(f"  Integration: {system_info['integration']['method']}")
    
    # Test neuro-fuzzy system
    print("\n  Neuro-Fuzzy Test Results:")
    print("  " + "-" * 75)
    for sc in test_scenarios:
        desc = sc.pop('desc')
        result = nf_system.predict(
            sc['num_vehicles'], sc['traffic_density'], 
            sc['waiting_time'], sc['emergency_vehicle']
        )
        print(f"  Scenario: {desc}")
        print(f"    Combined → Green: {result['green_duration']}s, "
              f"Priority: {result['signal_priority']}, "
              f"Congestion: {result['congestion_level']}%")
        print(f"    Weights: Fuzzy={result['weights']['fuzzy']}, NN={result['weights']['nn']}")
        sc['desc'] = desc
    
    # ============================================================
    # PHASE 6: System Comparison
    # ============================================================
    print_section("PHASE 6: Comparing All Systems")
    
    comparison_df = nf_system.compare_systems(df)
    
    print("\n  Performance Comparison Table:")
    print(comparison_df.to_string(index=False))
    
    comparison_df.to_csv('outputs/metrics/system_comparison.csv', index=False)
    
    # ============================================================
    # PHASE 7: Generate Batch Predictions for Visualization
    # ============================================================
    print_section("PHASE 7: Generating Batch Predictions")
    
    sample_df = df.sample(200, random_state=42)
    nf_batch_results = nf_system.predict_batch(sample_df)
    
    print(f"  Generated predictions for {len(sample_df)} samples")
    
    # ============================================================
    # PHASE 8: Generate Visualizations
    # ============================================================
    print_section("PHASE 8: Generating Visualizations")
    
    visualizer = TrafficVisualizer(output_dir='outputs/plots')
    
    visualizer.generate_all(
        fuzzy_controller=fuzzy_controller,
        nn_history=nn.history,
        y_true=nn.y_true,
        y_pred=nn.y_pred,
        comparison_df=comparison_df,
        df=df,
        nf_results=nf_batch_results
    )
    
    # ============================================================
    # PHASE 9: Save Model & Generate Summary
    # ============================================================
    print_section("PHASE 9: Saving Model & Generating Summary")
    
    nn.save(model_dir='models')
    
    # Generate comprehensive summary
    summary = {
        'project': 'Smart Traffic Signal Control using Neuro-Fuzzy System',
        'dataset': {
            'total_samples': len(df),
            'features': 4,
            'targets': 3,
            'emergency_percentage': f"{df['emergency_vehicle'].mean()*100:.1f}%"
        },
        'fuzzy_system': {
            'type': 'Mamdani FIS',
            'rules': len(fuzzy_controller.rules),
            'defuzzification': 'Centroid'
        },
        'neural_network': nn_metrics,
        'system_comparison': comparison_df.to_dict('records'),
        'system_info': system_info
    }
    
    with open('outputs/metrics/full_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # ============================================================
    # FINAL OUTPUT
    # ============================================================
    print_section("PIPELINE COMPLETE ✅")
    
    print(f"""
    📁 Output Files:
    ├── data/traffic_data.csv          — Synthetic dataset (5000 samples)
    ├── models/
    │   ├── traffic_nn_model.keras     — Trained neural network
    │   ├── scaler_X.pkl               — Feature scaler
    │   ├── scaler_y.pkl               — Target scaler
    │   └── metrics.json               — NN evaluation metrics
    ├── outputs/
    │   ├── metrics/
    │   │   ├── nn_metrics.json        — Neural network metrics
    │   │   ├── system_comparison.csv  — Fuzzy vs NN vs Neuro-Fuzzy
    │   │   └── full_summary.json      — Complete system summary
    │   └── plots/
    │       ├── membership_functions.png
    │       ├── training_curves.png
    │       ├── actual_vs_predicted.png
    │       ├── residual_distributions.png
    │       ├── system_comparison.png
    │       ├── traffic_heatmap.png
    │       ├── correlation_matrix.png
    │       ├── data_distributions.png
    │       ├── emergency_comparison.png
    │       ├── fuzzy_surface_3d.png
    │       ├── time_patterns.png
    │       └── neuro_fuzzy_breakdown.png
    """)
    
    print("  🎓 Ready for report generation and viva preparation!")
    print("=" * 60)


if __name__ == "__main__":
    main()
