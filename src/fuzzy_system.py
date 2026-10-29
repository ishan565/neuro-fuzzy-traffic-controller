"""
fuzzy_system.py — Fuzzy Logic Controller for Traffic Signal
============================================================
Implements a Mamdani-type fuzzy inference system (FIS) with:
  - 4 input variables (num_vehicles, traffic_density, waiting_time, emergency)
  - 3 output variables (green_duration, signal_priority, congestion_level)
  - 27+ fuzzy rules based on expert knowledge

Membership functions:
  - Triangular and trapezoidal shapes for smooth transitions
  - 3–5 linguistic terms per variable

Defuzzification: Centroid method
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzyTrafficController:
    """
    Mamdani-type Fuzzy Inference System for traffic signal control.
    
    This controller uses fuzzy logic to determine:
      - How long the green light should stay on
      - The priority level of the signal
      - The estimated congestion level
    
    Based on:
      - Number of vehicles at the intersection
      - Traffic density (%)
      - Average waiting time (seconds)
      - Presence of emergency vehicles
    """

    def __init__(self):
        """Initialize fuzzy variables, membership functions, and rules."""
        self._define_variables()
        self._define_membership_functions()
        self._define_rules()
        self._build_system()

    def _define_variables(self):
        """
        Define fuzzy input and output universe of discourse.
        
        Universes:
          - num_vehicles: [0, 100] — count of vehicles
          - traffic_density: [0, 100] — percentage
          - waiting_time: [0, 180] — seconds
          - emergency: [0, 1] — boolean-like
          - green_duration: [10, 120] — seconds
          - signal_priority: [1, 5] — ordinal scale
          - congestion_level: [0, 100] — percentage
        """
        # Inputs
        self.num_vehicles = ctrl.Antecedent(np.arange(0, 101, 1), 'num_vehicles')
        self.traffic_density = ctrl.Antecedent(np.arange(0, 101, 1), 'traffic_density')
        self.waiting_time = ctrl.Antecedent(np.arange(0, 181, 1), 'waiting_time')
        self.emergency = ctrl.Antecedent(np.arange(0, 2, 0.1), 'emergency')

        # Outputs
        self.green_duration = ctrl.Consequent(np.arange(10, 121, 1), 'green_duration')
        self.signal_priority = ctrl.Consequent(np.arange(1, 6, 0.1), 'signal_priority')
        self.congestion_level = ctrl.Consequent(np.arange(0, 101, 1), 'congestion_level')

    def _define_membership_functions(self):
        """
        Define membership functions for all fuzzy variables.
        
        Input MFs:
          - num_vehicles: Low, Medium, High (triangular)
          - traffic_density: Low, Medium, High (triangular)
          - waiting_time: Short, Medium, Long (triangular)
          - emergency: No, Yes (trapezoidal)
        
        Output MFs:
          - green_duration: Very_Short, Short, Medium, Long, Very_Long (triangular)
          - signal_priority: Low, Medium, High, Very_High, Emergency (triangular)
          - congestion_level: Low, Moderate, High, Severe (triangular)
        """
        # --- Input: Number of Vehicles ---
        self.num_vehicles['low'] = fuzz.trimf(self.num_vehicles.universe, [0, 0, 35])
        self.num_vehicles['medium'] = fuzz.trimf(self.num_vehicles.universe, [20, 50, 80])
        self.num_vehicles['high'] = fuzz.trimf(self.num_vehicles.universe, [65, 100, 100])

        # --- Input: Traffic Density ---
        self.traffic_density['low'] = fuzz.trimf(self.traffic_density.universe, [0, 0, 35])
        self.traffic_density['medium'] = fuzz.trimf(self.traffic_density.universe, [20, 50, 80])
        self.traffic_density['high'] = fuzz.trimf(self.traffic_density.universe, [65, 100, 100])

        # --- Input: Waiting Time ---
        self.waiting_time['short'] = fuzz.trimf(self.waiting_time.universe, [0, 0, 60])
        self.waiting_time['medium'] = fuzz.trimf(self.waiting_time.universe, [30, 90, 150])
        self.waiting_time['long'] = fuzz.trimf(self.waiting_time.universe, [120, 180, 180])

        # --- Input: Emergency Vehicle ---
        self.emergency['no'] = fuzz.trapmf(self.emergency.universe, [0, 0, 0.3, 0.5])
        self.emergency['yes'] = fuzz.trapmf(self.emergency.universe, [0.5, 0.7, 1, 1])

        # --- Output: Green Duration ---
        self.green_duration['very_short'] = fuzz.trimf(self.green_duration.universe, [10, 10, 30])
        self.green_duration['short'] = fuzz.trimf(self.green_duration.universe, [20, 35, 50])
        self.green_duration['medium'] = fuzz.trimf(self.green_duration.universe, [40, 60, 80])
        self.green_duration['long'] = fuzz.trimf(self.green_duration.universe, [70, 85, 100])
        self.green_duration['very_long'] = fuzz.trimf(self.green_duration.universe, [90, 120, 120])

        # --- Output: Signal Priority ---
        self.signal_priority['low'] = fuzz.trimf(self.signal_priority.universe, [1, 1, 2])
        self.signal_priority['medium'] = fuzz.trimf(self.signal_priority.universe, [1.5, 2.5, 3.5])
        self.signal_priority['high'] = fuzz.trimf(self.signal_priority.universe, [2.5, 3.5, 4.5])
        self.signal_priority['very_high'] = fuzz.trimf(self.signal_priority.universe, [3.5, 4.5, 5])
        self.signal_priority['emergency_p'] = fuzz.trimf(self.signal_priority.universe, [4.5, 5, 5])

        # --- Output: Congestion Level ---
        self.congestion_level['low'] = fuzz.trimf(self.congestion_level.universe, [0, 0, 30])
        self.congestion_level['moderate'] = fuzz.trimf(self.congestion_level.universe, [20, 40, 60])
        self.congestion_level['high'] = fuzz.trimf(self.congestion_level.universe, [50, 70, 85])
        self.congestion_level['severe'] = fuzz.trimf(self.congestion_level.universe, [75, 100, 100])

    def _define_rules(self):
        """
        Define the fuzzy rule base.
        
        27 rules covering all combinations of {low, medium, high} for
        vehicles × density × waiting_time, plus emergency overrides.
        
        Example rules:
          R1: IF density=high AND wait=long → green=very_long, priority=very_high, congestion=severe
          R2: IF density=low AND wait=short → green=very_short, priority=low, congestion=low
          R3: IF emergency=yes → green=very_long, priority=emergency, congestion=severe
        """
        self.rules = []

        # ===== Emergency Override Rules =====
        self.rules.append(ctrl.Rule(
            self.emergency['yes'],
            (self.green_duration['very_long'],
             self.signal_priority['emergency_p'],
             self.congestion_level['severe'])
        ))

        # ===== Non-Emergency Rules (3×3×3 = 27 combinations) =====
        
        # --- High density scenarios ---
        self.rules.append(ctrl.Rule(
            self.traffic_density['high'] & self.waiting_time['long'] & self.num_vehicles['high'],
            (self.green_duration['very_long'],
             self.signal_priority['very_high'],
             self.congestion_level['severe'])
        ))
        self.rules.append(ctrl.Rule(
            self.traffic_density['high'] & self.waiting_time['long'] & self.num_vehicles['medium'],
            (self.green_duration['long'],
             self.signal_priority['very_high'],
             self.congestion_level['severe'])
        ))
        self.rules.append(ctrl.Rule(
            self.traffic_density['high'] & self.waiting_time['long'] & self.num_vehicles['low'],
            (self.green_duration['long'],
             self.signal_priority['high'],
             self.congestion_level['high'])
        ))
        self.rules.append(ctrl.Rule(
            self.traffic_density['high'] & self.waiting_time['medium'] & self.num_vehicles['high'],
            (self.green_duration['long'],
             self.signal_priority['very_high'],
             self.congestion_level['severe'])
        ))
        self.rules.append(ctrl.Rule(
            self.traffic_density['high'] & self.waiting_time['medium'] & self.num_vehicles['medium'],
            (self.green_duration['long'],
             self.signal_priority['high'],
             self.congestion_level['high'])
        ))
        self.rules.append(ctrl.Rule(
            self.traffic_density['high'] & self.waiting_time['medium'] & self.num_vehicles['low'],
            (self.green_duration['medium'],
             self.signal_priority['high'],
             self.congestion_level['high'])
        ))
        self.rules.append(ctrl.Rule(
            self.traffic_density['high'] & self.waiting_time['short'] & self.num_vehicles['high'],
            (self.green_duration['long'],
             self.signal_priority['high'],
             self.congestion_level['high'])
        ))
        self.rules.append(ctrl.Rule(
            self.traffic_density['high'] & self.waiting_time['short'] & self.num_vehicles['medium'],
            (self.green_duration['medium'],
             self.signal_priority['high'],
             self.congestion_level['high'])
        ))
        self.rules.append(ctrl.Rule(
            self.traffic_density['high'] & self.waiting_time['short'] & self.num_vehicles['low'],
            (self.green_duration['medium'],
             self.signal_priority['medium'],
             self.congestion_level['moderate'])
        ))

        # --- Medium density scenarios ---
        self.rules.append(ctrl.Rule(
            self.traffic_density['medium'] & self.waiting_time['long'] & self.num_vehicles['high'],
            (self.green_duration['long'],
             self.signal_priority['high'],
             self.congestion_level['high'])
        ))
        self.rules.append(ctrl.Rule(
            self.traffic_density['medium'] & self.waiting_time['long'] & self.num_vehicles['medium'],
            (self.green_duration['medium'],
             self.signal_priority['high'],
             self.congestion_level['high'])
        ))
        self.rules.append(ctrl.Rule(
            self.traffic_density['medium'] & self.waiting_time['long'] & self.num_vehicles['low'],
            (self.green_duration['medium'],
             self.signal_priority['medium'],
             self.congestion_level['moderate'])
        ))
        self.rules.append(ctrl.Rule(
            self.traffic_density['medium'] & self.waiting_time['medium'] & self.num_vehicles['high'],
            (self.green_duration['medium'],
             self.signal_priority['high'],
             self.congestion_level['high'])
        ))
        self.rules.append(ctrl.Rule(
            self.traffic_density['medium'] & self.waiting_time['medium'] & self.num_vehicles['medium'],
            (self.green_duration['medium'],
             self.signal_priority['medium'],
             self.congestion_level['moderate'])
        ))
        self.rules.append(ctrl.Rule(
            self.traffic_density['medium'] & self.waiting_time['medium'] & self.num_vehicles['low'],
            (self.green_duration['short'],
             self.signal_priority['medium'],
             self.congestion_level['moderate'])
        ))
        self.rules.append(ctrl.Rule(
            self.traffic_density['medium'] & self.waiting_time['short'] & self.num_vehicles['high'],
            (self.green_duration['medium'],
             self.signal_priority['medium'],
             self.congestion_level['moderate'])
        ))
        self.rules.append(ctrl.Rule(
            self.traffic_density['medium'] & self.waiting_time['short'] & self.num_vehicles['medium'],
            (self.green_duration['short'],
             self.signal_priority['medium'],
             self.congestion_level['moderate'])
        ))
        self.rules.append(ctrl.Rule(
            self.traffic_density['medium'] & self.waiting_time['short'] & self.num_vehicles['low'],
            (self.green_duration['short'],
             self.signal_priority['low'],
             self.congestion_level['low'])
        ))

        # --- Low density scenarios ---
        self.rules.append(ctrl.Rule(
            self.traffic_density['low'] & self.waiting_time['long'] & self.num_vehicles['high'],
            (self.green_duration['medium'],
             self.signal_priority['medium'],
             self.congestion_level['moderate'])
        ))
        self.rules.append(ctrl.Rule(
            self.traffic_density['low'] & self.waiting_time['long'] & self.num_vehicles['medium'],
            (self.green_duration['medium'],
             self.signal_priority['medium'],
             self.congestion_level['moderate'])
        ))
        self.rules.append(ctrl.Rule(
            self.traffic_density['low'] & self.waiting_time['long'] & self.num_vehicles['low'],
            (self.green_duration['short'],
             self.signal_priority['low'],
             self.congestion_level['low'])
        ))
        self.rules.append(ctrl.Rule(
            self.traffic_density['low'] & self.waiting_time['medium'] & self.num_vehicles['high'],
            (self.green_duration['medium'],
             self.signal_priority['medium'],
             self.congestion_level['moderate'])
        ))
        self.rules.append(ctrl.Rule(
            self.traffic_density['low'] & self.waiting_time['medium'] & self.num_vehicles['medium'],
            (self.green_duration['short'],
             self.signal_priority['low'],
             self.congestion_level['low'])
        ))
        self.rules.append(ctrl.Rule(
            self.traffic_density['low'] & self.waiting_time['medium'] & self.num_vehicles['low'],
            (self.green_duration['very_short'],
             self.signal_priority['low'],
             self.congestion_level['low'])
        ))
        self.rules.append(ctrl.Rule(
            self.traffic_density['low'] & self.waiting_time['short'] & self.num_vehicles['high'],
            (self.green_duration['short'],
             self.signal_priority['medium'],
             self.congestion_level['moderate'])
        ))
        self.rules.append(ctrl.Rule(
            self.traffic_density['low'] & self.waiting_time['short'] & self.num_vehicles['medium'],
            (self.green_duration['very_short'],
             self.signal_priority['low'],
             self.congestion_level['low'])
        ))
        self.rules.append(ctrl.Rule(
            self.traffic_density['low'] & self.waiting_time['short'] & self.num_vehicles['low'],
            (self.green_duration['very_short'],
             self.signal_priority['low'],
             self.congestion_level['low'])
        ))

    def _build_system(self):
        """Build the fuzzy control system and simulation from defined rules."""
        self.control_system = ctrl.ControlSystem(self.rules)
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)

    def predict(self, num_vehicles, traffic_density, waiting_time, emergency_vehicle):
        """
        Run fuzzy inference for a single data point.
        
        Args:
            num_vehicles (float): Number of vehicles (0–100).
            traffic_density (float): Density percentage (0–100).
            waiting_time (float): Waiting time in seconds (0–180).
            emergency_vehicle (int): Emergency flag (0 or 1).
        
        Returns:
            dict: {
                'green_duration': float,
                'signal_priority': float,
                'congestion_level': float
            }
        """
        # Clamp inputs to valid ranges
        num_vehicles = np.clip(num_vehicles, 0, 100)
        traffic_density = np.clip(traffic_density, 0, 100)
        waiting_time = np.clip(waiting_time, 0, 180)
        emergency_vehicle = np.clip(emergency_vehicle, 0, 1)

        self.simulation.input['num_vehicles'] = num_vehicles
        self.simulation.input['traffic_density'] = traffic_density
        self.simulation.input['waiting_time'] = waiting_time
        self.simulation.input['emergency'] = emergency_vehicle

        try:
            self.simulation.compute()
            return {
                'green_duration': round(self.simulation.output['green_duration'], 2),
                'signal_priority': round(self.simulation.output['signal_priority'], 2),
                'congestion_level': round(self.simulation.output['congestion_level'], 2)
            }
        except Exception as e:
            # Fallback for edge cases where rules don't fire
            return {
                'green_duration': 30.0,
                'signal_priority': 2.0,
                'congestion_level': 30.0
            }

    def predict_batch(self, dataframe):
        """
        Run fuzzy inference on a pandas DataFrame.
        
        Args:
            dataframe (pd.DataFrame): Must contain columns:
                num_vehicles, traffic_density, waiting_time, emergency_vehicle
        
        Returns:
            tuple: (green_durations, signal_priorities, congestion_levels) as numpy arrays
        """
        green_durations = []
        signal_priorities = []
        congestion_levels = []

        for _, row in dataframe.iterrows():
            result = self.predict(
                row['num_vehicles'],
                row['traffic_density'],
                row['waiting_time'],
                row['emergency_vehicle']
            )
            green_durations.append(result['green_duration'])
            signal_priorities.append(result['signal_priority'])
            congestion_levels.append(result['congestion_level'])

        return (
            np.array(green_durations),
            np.array(signal_priorities),
            np.array(congestion_levels)
        )

    def get_membership_data(self):
        """
        Return membership function data for visualization.
        
        Returns:
            dict: Structured data for plotting membership functions.
        """
        return {
            'inputs': {
                'num_vehicles': {
                    'universe': self.num_vehicles.universe,
                    'terms': {name: mf.mf for name, mf in self.num_vehicles.terms.items()},
                    'label': 'Number of Vehicles',
                    'unit': 'vehicles'
                },
                'traffic_density': {
                    'universe': self.traffic_density.universe,
                    'terms': {name: mf.mf for name, mf in self.traffic_density.terms.items()},
                    'label': 'Traffic Density',
                    'unit': '%'
                },
                'waiting_time': {
                    'universe': self.waiting_time.universe,
                    'terms': {name: mf.mf for name, mf in self.waiting_time.terms.items()},
                    'label': 'Waiting Time',
                    'unit': 'seconds'
                },
                'emergency': {
                    'universe': self.emergency.universe,
                    'terms': {name: mf.mf for name, mf in self.emergency.terms.items()},
                    'label': 'Emergency Vehicle',
                    'unit': ''
                }
            },
            'outputs': {
                'green_duration': {
                    'universe': self.green_duration.universe,
                    'terms': {name: mf.mf for name, mf in self.green_duration.terms.items()},
                    'label': 'Green Duration',
                    'unit': 'seconds'
                },
                'signal_priority': {
                    'universe': self.signal_priority.universe,
                    'terms': {name: mf.mf for name, mf in self.signal_priority.terms.items()},
                    'label': 'Signal Priority',
                    'unit': 'level'
                },
                'congestion_level': {
                    'universe': self.congestion_level.universe,
                    'terms': {name: mf.mf for name, mf in self.congestion_level.terms.items()},
                    'label': 'Congestion Level',
                    'unit': '%'
                }
            }
        }


if __name__ == "__main__":
    controller = FuzzyTrafficController()
    
    # Test scenarios
    test_cases = [
        {"num_vehicles": 10, "traffic_density": 15, "waiting_time": 20, "emergency_vehicle": 0},
        {"num_vehicles": 50, "traffic_density": 55, "waiting_time": 90, "emergency_vehicle": 0},
        {"num_vehicles": 85, "traffic_density": 90, "waiting_time": 150, "emergency_vehicle": 0},
        {"num_vehicles": 40, "traffic_density": 45, "waiting_time": 60, "emergency_vehicle": 1},
    ]
    
    print("=" * 70)
    print("Fuzzy Traffic Controller — Test Results")
    print("=" * 70)
    for i, tc in enumerate(test_cases):
        result = controller.predict(**tc)
        print(f"\nScenario {i+1}: {tc}")
        print(f"  → Green Duration: {result['green_duration']}s")
        print(f"  → Signal Priority: {result['signal_priority']}")
        print(f"  → Congestion Level: {result['congestion_level']}%")
