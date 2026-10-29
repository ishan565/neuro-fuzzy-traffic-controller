"""
neuro_fuzzy.py — Neuro-Fuzzy Integration System
=================================================
Combines the Fuzzy Logic Controller with the Neural Network
into an Adaptive Neuro-Fuzzy Inference System (ANFIS-inspired).

Integration Strategy:
  1. The Fuzzy system provides rule-based expert reasoning
  2. The Neural Network learns data-driven patterns
  3. The Neuro-Fuzzy system combines both via weighted ensemble
  4. Weights are optimized based on prediction confidence

This hybrid approach ensures:
  - Interpretability from fuzzy rules
  - Generalization from neural network learning
  - Robustness through ensemble combination
"""

import numpy as np
import pandas as pd
from src.fuzzy_system import FuzzyTrafficController
from src.neural_network import TrafficNeuralNetwork


class NeuroFuzzyTrafficSystem:
    """
    Hybrid Neuro-Fuzzy system combining fuzzy logic and neural networks.
    
    The system uses a weighted ensemble approach:
      output = α × fuzzy_output + (1-α) × nn_output
    
    where α is adaptive based on input characteristics:
      - For normal traffic: α=0.4 (trust NN more — better generalization)
      - For emergency scenarios: α=0.7 (trust fuzzy rules — explicit handling)
      - For extreme conditions: α=0.5 (balanced)
    
    Attributes:
        fuzzy_controller (FuzzyTrafficController): The fuzzy logic component.
        neural_network (TrafficNeuralNetwork): The neural network component.
        fuzzy_weight (float): Default weight for fuzzy component.
        nn_weight (float): Default weight for neural network component.
    """

    def __init__(self, fuzzy_weight=0.4, nn_weight=0.6):
        """
        Initialize both subsystems.
        
        Args:
            fuzzy_weight (float): Weight for fuzzy system output (default 0.4).
            nn_weight (float): Weight for neural network output (default 0.6).
        """
        self.fuzzy_controller = FuzzyTrafficController()
        self.neural_network = TrafficNeuralNetwork()
        self.default_fuzzy_weight = fuzzy_weight
        self.default_nn_weight = nn_weight
        self.comparison_results = None
        self.is_trained = False

    def train(self, df, epochs=150, batch_size=32, verbose=1):
        """
        Train the neural network component on the dataset.
        
        The fuzzy system doesn't require training (rule-based),
        but the neural network learns from the data.
        
        Args:
            df (pd.DataFrame): Training dataset.
            epochs (int): Maximum training epochs.
            batch_size (int): Batch size.
            verbose (int): Verbosity level.
        
        Returns:
            dict: Training history.
        """
        print("\n" + "=" * 60)
        print("  NEURO-FUZZY SYSTEM — Training Phase")
        print("=" * 60)
        print("\n[Phase 1] Fuzzy Logic Controller initialized (rule-based, no training needed)")
        print(f"  → {len(self.fuzzy_controller.rules)} fuzzy rules loaded")

        print(f"\n[Phase 2] Training Neural Network...")
        self.neural_network.prepare_data(df)
        history = self.neural_network.train(epochs=epochs, batch_size=batch_size, verbose=verbose)
        
        self.is_trained = True
        print("\n[Phase 3] Neuro-Fuzzy integration ready!")
        
        return history

    def _adaptive_weight(self, num_vehicles, traffic_density, waiting_time, emergency):
        """
        Compute adaptive weights based on input characteristics.
        
        The system dynamically adjusts the contribution of each subsystem:
          - Emergency: Trust fuzzy rules more (explicit emergency handling)
          - Extreme traffic: Balanced approach
          - Normal traffic: Trust NN more (learned patterns)
        
        Args:
            num_vehicles, traffic_density, waiting_time, emergency: Input features.
        
        Returns:
            tuple: (fuzzy_weight, nn_weight) that sum to 1.0
        """
        if emergency == 1:
            # Emergency: fuzzy rules have explicit emergency handling
            return 0.7, 0.3
        
        # Compute how "extreme" the situation is
        extremity = max(
            num_vehicles / 100,
            traffic_density / 100,
            waiting_time / 180
        )
        
        if extremity > 0.85:
            # Very extreme: balanced approach
            return 0.5, 0.5
        elif extremity > 0.6:
            # Moderate: slightly favor NN
            return 0.4, 0.6
        else:
            # Normal: trust NN's learned patterns
            return 0.35, 0.65

    def predict(self, num_vehicles, traffic_density, waiting_time, emergency_vehicle):
        """
        Make a prediction using the combined neuro-fuzzy system.
        
        Process:
          1. Get fuzzy system prediction
          2. Get neural network prediction
          3. Compute adaptive weights
          4. Combine predictions via weighted average
        
        Args:
            num_vehicles, traffic_density, waiting_time, emergency_vehicle: Inputs.
        
        Returns:
            dict: Combined prediction with individual subsystem outputs.
        """
        # Get individual predictions
        fuzzy_pred = self.fuzzy_controller.predict(
            num_vehicles, traffic_density, waiting_time, emergency_vehicle
        )
        
        nn_pred = self.neural_network.predict(
            num_vehicles, traffic_density, waiting_time, emergency_vehicle
        )

        # Compute adaptive weights
        fw, nw = self._adaptive_weight(
            num_vehicles, traffic_density, waiting_time, emergency_vehicle
        )

        # Weighted combination
        combined = {
            'green_duration': round(
                fw * fuzzy_pred['green_duration'] + nw * nn_pred['green_duration'], 2
            ),
            'signal_priority': round(
                fw * fuzzy_pred['signal_priority'] + nw * nn_pred['signal_priority'], 2
            ),
            'congestion_level': round(
                fw * fuzzy_pred['congestion_level'] + nw * nn_pred['congestion_level'], 2
            ),
            'fuzzy_output': fuzzy_pred,
            'nn_output': nn_pred,
            'weights': {'fuzzy': fw, 'nn': nw},
            'emergency_mode': emergency_vehicle == 1
        }

        return combined

    def predict_batch(self, df):
        """
        Batch prediction using the neuro-fuzzy system.
        
        Args:
            df (pd.DataFrame): Input data.
        
        Returns:
            dict: Arrays of predictions for each output.
        """
        results = {
            'green_duration': [],
            'signal_priority': [],
            'congestion_level': [],
            'fuzzy_green': [],
            'nn_green': [],
            'fuzzy_priority': [],
            'nn_priority': [],
            'fuzzy_congestion': [],
            'nn_congestion': []
        }

        for _, row in df.iterrows():
            pred = self.predict(
                row['num_vehicles'],
                row['traffic_density'],
                row['waiting_time'],
                row['emergency_vehicle']
            )
            results['green_duration'].append(pred['green_duration'])
            results['signal_priority'].append(pred['signal_priority'])
            results['congestion_level'].append(pred['congestion_level'])
            results['fuzzy_green'].append(pred['fuzzy_output']['green_duration'])
            results['nn_green'].append(pred['nn_output']['green_duration'])
            results['fuzzy_priority'].append(pred['fuzzy_output']['signal_priority'])
            results['nn_priority'].append(pred['nn_output']['signal_priority'])
            results['fuzzy_congestion'].append(pred['fuzzy_output']['congestion_level'])
            results['nn_congestion'].append(pred['nn_output']['congestion_level'])

        return {k: np.array(v) for k, v in results.items()}

    def compare_systems(self, df):
        """
        Compare Fuzzy-only, NN-only, and Neuro-Fuzzy performance.
        
        Computes MSE, MAE, RMSE, R² for each system against ground truth.
        
        Args:
            df (pd.DataFrame): Dataset with ground truth targets.
        
        Returns:
            pd.DataFrame: Comparison table.
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        # Sample for faster comparison
        sample = df.sample(min(500, len(df)), random_state=42)
        
        y_true_green = sample['green_duration'].values
        y_true_priority = sample['signal_priority'].values
        y_true_congestion = sample['congestion_level'].values

        # Get predictions from all three systems
        fuzzy_green, fuzzy_priority, fuzzy_congestion = \
            self.fuzzy_controller.predict_batch(sample)
        nn_green, nn_priority, nn_congestion = \
            self.neural_network.predict_batch(sample)
        
        nf_results = self.predict_batch(sample)
        nf_green = nf_results['green_duration']
        nf_priority = nf_results['signal_priority']
        nf_congestion = nf_results['congestion_level']

        comparisons = []
        
        for name, y_true, fuzzy_y, nn_y, nf_y in [
            ('Green Duration', y_true_green, fuzzy_green, nn_green, nf_green),
            ('Signal Priority', y_true_priority, fuzzy_priority, nn_priority, nf_priority),
            ('Congestion Level', y_true_congestion, fuzzy_congestion, nn_congestion, nf_congestion)
        ]:
            for system, y_pred in [('Fuzzy', fuzzy_y), ('Neural Network', nn_y), ('Neuro-Fuzzy', nf_y)]:
                mse = mean_squared_error(y_true, y_pred)
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_true, y_pred)
                
                comparisons.append({
                    'Output': name,
                    'System': system,
                    'MSE': round(mse, 4),
                    'RMSE': round(rmse, 4),
                    'MAE': round(mae, 4),
                    'R²': round(r2, 4)
                })

        self.comparison_results = pd.DataFrame(comparisons)
        return self.comparison_results

    def get_system_info(self):
        """
        Get comprehensive information about the neuro-fuzzy system.
        
        Returns:
            dict: System architecture details.
        """
        return {
            'fuzzy': {
                'type': 'Mamdani Fuzzy Inference System',
                'num_rules': len(self.fuzzy_controller.rules),
                'inputs': 4,
                'outputs': 3,
                'defuzzification': 'Centroid',
                'mf_types': 'Triangular & Trapezoidal'
            },
            'neural_network': {
                'type': 'Multi-Output Feedforward Neural Network',
                'architecture': '4 → 64 → 128 → 64 → 32 → 3',
                'activation': 'ReLU (hidden), Linear (output)',
                'optimizer': 'Adam',
                'loss': 'Mean Squared Error',
                'regularization': 'BatchNormalization + Dropout(0.3)'
            },
            'integration': {
                'method': 'Adaptive Weighted Ensemble',
                'default_weights': {
                    'fuzzy': self.default_fuzzy_weight,
                    'nn': self.default_nn_weight
                },
                'adaptive_modes': {
                    'emergency': '70% fuzzy, 30% NN',
                    'extreme': '50% fuzzy, 50% NN',
                    'moderate': '40% fuzzy, 60% NN',
                    'normal': '35% fuzzy, 65% NN'
                }
            }
        }


if __name__ == "__main__":
    from src.data_generator import TrafficDataGenerator
    
    gen = TrafficDataGenerator(seed=42)
    df = gen.generate(3000, save_csv=False)
    
    system = NeuroFuzzyTrafficSystem()
    system.train(df, epochs=50, verbose=0)
    
    # Test prediction
    result = system.predict(75, 80, 120, 0)
    print("\nNeuro-Fuzzy Prediction:")
    print(f"  Green Duration: {result['green_duration']}s")
    print(f"  Signal Priority: {result['signal_priority']}")
    print(f"  Congestion Level: {result['congestion_level']}%")
    print(f"  Weights: Fuzzy={result['weights']['fuzzy']}, NN={result['weights']['nn']}")
