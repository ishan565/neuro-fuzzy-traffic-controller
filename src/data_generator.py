"""
data_generator.py — Synthetic Traffic Data Generator
=====================================================
Generates realistic simulated traffic data for training
the neural network component of the neuro-fuzzy system.

Features:
  - Time-of-day based traffic patterns (rush hour, off-peak, night)
  - Emergency vehicle random injection
  - Correlated feature generation (density ~ vehicles / capacity)
  - Noise injection for realism
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class TrafficDataGenerator:
    """
    Generates synthetic traffic data simulating real-world patterns.
    
    The generator creates data with four input features:
      1. num_vehicles: Number of vehicles at the intersection (0–100)
      2. traffic_density: Traffic density as a percentage (0–100%)
      3. waiting_time: Average waiting time in seconds (0–180s)
      4. emergency_vehicle: Binary flag for emergency vehicle presence (0 or 1)
    
    And three output targets:
      1. green_duration: Optimal green light duration in seconds (10–120s)
      2. signal_priority: Priority level (1=Low, 2=Medium, 3=High, 4=Very High, 5=Emergency)
      3. congestion_level: Congestion severity (0–100%)
    """

    def __init__(self, seed=42):
        """
        Initialize the data generator.
        
        Args:
            seed (int): Random seed for reproducibility.
        """
        self.rng = np.random.RandomState(seed)
        self.road_capacity = 100  # Maximum vehicles per road segment

    def _time_pattern(self, hour):
        """
        Returns a traffic multiplier based on time of day.
        Models typical urban traffic patterns:
          - Morning rush: 7–9 AM (high)
          - Midday: 10 AM–3 PM (moderate)
          - Evening rush: 4–7 PM (very high)
          - Night: 8 PM–6 AM (low)
        
        Args:
            hour (float): Hour of the day (0–24).
        
        Returns:
            float: Traffic intensity multiplier (0.1–1.0).
        """
        if 7 <= hour < 9:
            return 0.8 + 0.2 * self.rng.random()   # Morning rush
        elif 9 <= hour < 16:
            return 0.4 + 0.2 * self.rng.random()   # Midday moderate
        elif 16 <= hour < 19:
            return 0.85 + 0.15 * self.rng.random()  # Evening rush
        elif 19 <= hour < 22:
            return 0.3 + 0.2 * self.rng.random()   # Evening moderate
        else:
            return 0.1 + 0.15 * self.rng.random()  # Night low

    def _compute_green_duration(self, num_vehicles, traffic_density, waiting_time, emergency):
        """
        Compute the ideal green-light duration using expert-derived rules.
        
        This function acts as a 'ground truth' label generator, combining
        domain knowledge about traffic signal timing:
          - Base duration proportional to vehicle count
          - Bonus for high density
          - Bonus for long waiting times
          - Emergency override
        
        Args:
            num_vehicles (float): Number of vehicles (0–100).
            traffic_density (float): Density percentage (0–100).
            waiting_time (float): Waiting time in seconds (0–180).
            emergency (int): Emergency vehicle flag (0 or 1).
        
        Returns:
            float: Green light duration in seconds (clamped 10–120).
        """
        base = 15 + (num_vehicles / 100) * 45
        density_bonus = (traffic_density / 100) * 25
        wait_bonus = (waiting_time / 180) * 20
        emergency_bonus = 30 * emergency

        duration = base + density_bonus + wait_bonus + emergency_bonus
        noise = self.rng.normal(0, 3)
        duration = np.clip(duration + noise, 10, 120)
        return round(duration, 2)

    def _compute_signal_priority(self, num_vehicles, traffic_density, waiting_time, emergency):
        """
        Compute signal priority level.
        
        Priority levels:
          1 = Low, 2 = Medium, 3 = High, 4 = Very High, 5 = Emergency
        
        Args:
            num_vehicles, traffic_density, waiting_time, emergency: Input features.
        
        Returns:
            int: Priority level (1–5).
        """
        if emergency:
            return 5
        score = (num_vehicles / 100) * 0.3 + (traffic_density / 100) * 0.4 + (waiting_time / 180) * 0.3
        if score > 0.75:
            return 4
        elif score > 0.55:
            return 3
        elif score > 0.3:
            return 2
        else:
            return 1

    def _compute_congestion(self, num_vehicles, traffic_density, waiting_time):
        """
        Compute congestion level as a percentage.
        
        Uses a weighted combination of all non-emergency inputs.
        
        Args:
            num_vehicles, traffic_density, waiting_time: Input features.
        
        Returns:
            float: Congestion level (0–100, rounded).
        """
        congestion = (
            0.35 * (num_vehicles / 100) +
            0.40 * (traffic_density / 100) +
            0.25 * (waiting_time / 180)
        ) * 100
        noise = self.rng.normal(0, 4)
        return round(np.clip(congestion + noise, 0, 100), 2)

    def generate(self, n_samples=5000, save_csv=True, output_path="data/traffic_data.csv"):
        """
        Generate n_samples of synthetic traffic data.
        
        The data follows realistic patterns:
          - Traffic density is correlated with vehicle count
          - Waiting time is correlated with density
          - Emergency vehicles appear ~5% of the time
        
        Args:
            n_samples (int): Number of data samples to generate.
            save_csv (bool): Whether to save the data to CSV.
            output_path (str): Path for the CSV output file.
        
        Returns:
            pd.DataFrame: Generated traffic dataset.
        """
        records = []

        for i in range(n_samples):
            # Simulate time-of-day
            hour = self.rng.uniform(0, 24)
            multiplier = self._time_pattern(hour)

            # Generate correlated features
            num_vehicles = int(np.clip(
                self.rng.normal(multiplier * 60, 15), 0, 100
            ))
            traffic_density = np.clip(
                (num_vehicles / self.road_capacity) * 100 + self.rng.normal(0, 8),
                0, 100
            )
            traffic_density = round(traffic_density, 2)

            waiting_time = np.clip(
                traffic_density * 1.5 + self.rng.normal(0, 15), 0, 180
            )
            waiting_time = round(waiting_time, 2)

            emergency = 1 if self.rng.random() < 0.05 else 0

            # Compute targets
            green_duration = self._compute_green_duration(
                num_vehicles, traffic_density, waiting_time, emergency
            )
            signal_priority = self._compute_signal_priority(
                num_vehicles, traffic_density, waiting_time, emergency
            )
            congestion_level = self._compute_congestion(
                num_vehicles, traffic_density, waiting_time
            )

            records.append({
                'hour': round(hour, 2),
                'num_vehicles': num_vehicles,
                'traffic_density': traffic_density,
                'waiting_time': waiting_time,
                'emergency_vehicle': emergency,
                'green_duration': green_duration,
                'signal_priority': signal_priority,
                'congestion_level': congestion_level
            })

        df = pd.DataFrame(records)

        if save_csv:
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"[DataGenerator] Saved {n_samples} samples to {output_path}")

        return df


if __name__ == "__main__":
    gen = TrafficDataGenerator(seed=42)
    df = gen.generate(n_samples=5000)
    print(df.describe())
    print(f"\nShape: {df.shape}")
    print(f"\nEmergency vehicle percentage: {df['emergency_vehicle'].mean()*100:.1f}%")
