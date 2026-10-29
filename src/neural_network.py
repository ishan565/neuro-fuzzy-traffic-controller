"""
neural_network.py — Neural Network for Traffic Pattern Learning
================================================================
Implements a multi-output feedforward neural network using TensorFlow/Keras
to learn traffic patterns from historical/simulated data.

Architecture:
  Input Layer  → 4 neurons (num_vehicles, traffic_density, waiting_time, emergency)
  Hidden Layer 1 → 64 neurons (ReLU, BatchNorm, Dropout)
  Hidden Layer 2 → 128 neurons (ReLU, BatchNorm, Dropout)
  Hidden Layer 3 → 64 neurons (ReLU, BatchNorm, Dropout)
  Hidden Layer 4 → 32 neurons (ReLU)
  Output Layer → 3 neurons (green_duration, signal_priority, congestion_level)

Training:
  - Optimizer: Adam with learning rate scheduling
  - Loss: Mean Squared Error
  - Regularization: Dropout + BatchNormalization
  - Early stopping with patience
"""

import numpy as np
import pandas as pd
import os
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle


class TrafficNeuralNetwork:
    """
    Multi-output neural network for traffic signal prediction.
    
    Learns the mapping:
      (num_vehicles, traffic_density, waiting_time, emergency) 
        → (green_duration, signal_priority, congestion_level)
    
    Attributes:
        model (keras.Model): The compiled Keras model.
        scaler_X (StandardScaler): Feature scaler.
        scaler_y (StandardScaler): Target scaler.
        history (dict): Training history metrics.
    """

    def __init__(self, input_dim=4, output_dim=3):
        """
        Initialize the neural network.
        
        Args:
            input_dim (int): Number of input features (default 4).
            output_dim (int): Number of output targets (default 3).
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = self._build_model()
        self.history = None
        self.training_metrics = {}

    def _build_model(self):
        """
        Build the neural network architecture.
        
        Architecture Details:
          - 4 hidden layers with decreasing width
          - BatchNormalization for training stability
          - Dropout (0.3) for regularization
          - He normal initialization for ReLU layers
        
        Returns:
            keras.Model: Compiled model.
        """
        model = keras.Sequential([
            # Input layer
            layers.InputLayer(input_shape=(self.input_dim,)),
            
            # Hidden Layer 1
            layers.Dense(64, kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.3),
            
            # Hidden Layer 2
            layers.Dense(128, kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.3),
            
            # Hidden Layer 3
            layers.Dense(64, kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.3),
            
            # Hidden Layer 4
            layers.Dense(32, kernel_initializer='he_normal'),
            layers.Activation('relu'),
            
            # Output Layer
            layers.Dense(self.output_dim, activation='linear')
        ])

        # Compile with Adam optimizer
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )

        return model

    def prepare_data(self, df, test_size=0.2, val_size=0.1):
        """
        Prepare and split data for training.
        
        Performs:
          1. Feature/target separation
          2. Train/validation/test split (70/10/20)
          3. StandardScaler normalization
        
        Args:
            df (pd.DataFrame): Full dataset with input and output columns.
            test_size (float): Fraction for test set.
            val_size (float): Fraction for validation set (from remaining).
        
        Returns:
            dict: Contains X_train, X_val, X_test, y_train, y_val, y_test
                  (all as numpy arrays, scaled).
        """
        feature_cols = ['num_vehicles', 'traffic_density', 'waiting_time', 'emergency_vehicle']
        target_cols = ['green_duration', 'signal_priority', 'congestion_level']

        X = df[feature_cols].values
        y = df[target_cols].values

        # Split: train+val / test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Split: train / val
        val_fraction = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_fraction, random_state=42
        )

        # Scale features and targets
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_val_scaled = self.scaler_X.transform(X_val)
        X_test_scaled = self.scaler_X.transform(X_test)

        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_val_scaled = self.scaler_y.transform(y_val)
        y_test_scaled = self.scaler_y.transform(y_test)

        self.data = {
            'X_train': X_train_scaled, 'y_train': y_train_scaled,
            'X_val': X_val_scaled, 'y_val': y_val_scaled,
            'X_test': X_test_scaled, 'y_test': y_test_scaled,
            'X_train_raw': X_train, 'y_train_raw': y_train,
            'X_val_raw': X_val, 'y_val_raw': y_val,
            'X_test_raw': X_test, 'y_test_raw': y_test,
        }

        print(f"[NeuralNetwork] Data split:")
        print(f"  Training:   {X_train_scaled.shape[0]} samples")
        print(f"  Validation: {X_val_scaled.shape[0]} samples")
        print(f"  Test:       {X_test_scaled.shape[0]} samples")

        return self.data

    def train(self, epochs=150, batch_size=32, verbose=1):
        """
        Train the neural network with early stopping and LR scheduling.
        
        Callbacks:
          - EarlyStopping: patience=15, monitors val_loss
          - ReduceLROnPlateau: factor=0.5, patience=7
        
        Args:
            epochs (int): Maximum training epochs.
            batch_size (int): Batch size.
            verbose (int): Verbosity level.
        
        Returns:
            keras.callbacks.History: Training history object.
        """
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )

        lr_scheduler = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        )

        history = self.model.fit(
            self.data['X_train'], self.data['y_train'],
            validation_data=(self.data['X_val'], self.data['y_val']),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, lr_scheduler],
            verbose=verbose
        )

        self.history = history.history
        return history

    def evaluate(self):
        """
        Evaluate the model on the test set and compute comprehensive metrics.
        
        Computes:
          - MSE, RMSE, MAE, R² for each output
          - Overall metrics
          - MAPE (Mean Absolute Percentage Error)
        
        Returns:
            dict: Comprehensive evaluation metrics.
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        y_pred_scaled = self.model.predict(self.data['X_test'], verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        y_true = self.data['y_test_raw']

        target_names = ['green_duration', 'signal_priority', 'congestion_level']
        metrics = {}

        for i, name in enumerate(target_names):
            mse = mean_squared_error(y_true[:, i], y_pred[:, i])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
            r2 = r2_score(y_true[:, i], y_pred[:, i])
            
            # MAPE (avoid division by zero)
            mask = y_true[:, i] != 0
            if mask.sum() > 0:
                mape = np.mean(np.abs((y_true[mask, i] - y_pred[mask, i]) / y_true[mask, i])) * 100
            else:
                mape = 0.0

            metrics[name] = {
                'MSE': round(mse, 4),
                'RMSE': round(rmse, 4),
                'MAE': round(mae, 4),
                'R2': round(r2, 4),
                'MAPE': round(mape, 2)
            }

        # Overall metrics
        overall_mse = mean_squared_error(y_true, y_pred)
        overall_mae = mean_absolute_error(y_true, y_pred)
        overall_r2 = r2_score(y_true, y_pred)

        metrics['overall'] = {
            'MSE': round(overall_mse, 4),
            'RMSE': round(np.sqrt(overall_mse), 4),
            'MAE': round(overall_mae, 4),
            'R2': round(overall_r2, 4)
        }

        self.training_metrics = metrics
        self.y_pred = y_pred
        self.y_true = y_true

        return metrics

    def predict(self, num_vehicles, traffic_density, waiting_time, emergency_vehicle):
        """
        Predict outputs for a single input.
        
        Args:
            num_vehicles, traffic_density, waiting_time, emergency_vehicle: Input values.
        
        Returns:
            dict: Predicted green_duration, signal_priority, congestion_level.
        """
        X = np.array([[num_vehicles, traffic_density, waiting_time, emergency_vehicle]])
        X_scaled = self.scaler_X.transform(X)
        y_scaled = self.model.predict(X_scaled, verbose=0)
        y = self.scaler_y.inverse_transform(y_scaled)[0]

        return {
            'green_duration': round(float(np.clip(y[0], 10, 120)), 2),
            'signal_priority': round(float(np.clip(y[1], 1, 5)), 2),
            'congestion_level': round(float(np.clip(y[2], 0, 100)), 2)
        }

    def predict_batch(self, df):
        """
        Predict outputs for a DataFrame of inputs.
        
        Args:
            df (pd.DataFrame): Input data.
        
        Returns:
            tuple: (green_durations, signal_priorities, congestion_levels)
        """
        feature_cols = ['num_vehicles', 'traffic_density', 'waiting_time', 'emergency_vehicle']
        X = df[feature_cols].values
        X_scaled = self.scaler_X.transform(X)
        y_scaled = self.model.predict(X_scaled, verbose=0)
        y = self.scaler_y.inverse_transform(y_scaled)

        return (
            np.clip(y[:, 0], 10, 120),
            np.clip(y[:, 1], 1, 5),
            np.clip(y[:, 2], 0, 100)
        )

    def get_model_summary(self):
        """
        Get model architecture summary as a string.
        
        Returns:
            str: Model summary.
        """
        summary_lines = []
        self.model.summary(print_fn=lambda x: summary_lines.append(x))
        return '\n'.join(summary_lines)

    def save(self, model_dir="models"):
        """
        Save the trained model and scalers.
        
        Args:
            model_dir (str): Directory to save model files.
        """
        os.makedirs(model_dir, exist_ok=True)
        self.model.save(os.path.join(model_dir, 'traffic_nn_model.keras'))
        
        with open(os.path.join(model_dir, 'scaler_X.pkl'), 'wb') as f:
            pickle.dump(self.scaler_X, f)
        with open(os.path.join(model_dir, 'scaler_y.pkl'), 'wb') as f:
            pickle.dump(self.scaler_y, f)
        
        if self.training_metrics:
            with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
                json.dump(self.training_metrics, f, indent=2)
        
        print(f"[NeuralNetwork] Model saved to {model_dir}/")

    def load(self, model_dir="models"):
        """
        Load a previously saved model and scalers.
        
        Args:
            model_dir (str): Directory containing saved model files.
        """
        self.model = keras.models.load_model(os.path.join(model_dir, 'traffic_nn_model.keras'))
        
        with open(os.path.join(model_dir, 'scaler_X.pkl'), 'rb') as f:
            self.scaler_X = pickle.load(f)
        with open(os.path.join(model_dir, 'scaler_y.pkl'), 'rb') as f:
            self.scaler_y = pickle.load(f)
        
        print(f"[NeuralNetwork] Model loaded from {model_dir}/")


if __name__ == "__main__":
    from data_generator import TrafficDataGenerator
    
    # Generate data
    gen = TrafficDataGenerator(seed=42)
    df = gen.generate(5000, save_csv=False)
    
    # Build and train
    nn = TrafficNeuralNetwork()
    nn.prepare_data(df)
    nn.train(epochs=100, verbose=1)
    
    # Evaluate
    metrics = nn.evaluate()
    print("\n" + "=" * 50)
    print("Neural Network Evaluation Metrics")
    print("=" * 50)
    for key, val in metrics.items():
        print(f"\n{key}:")
        for metric, value in val.items():
            print(f"  {metric}: {value}")
