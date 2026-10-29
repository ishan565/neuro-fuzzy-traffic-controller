"""
visualizations.py — Comprehensive Visualization Module
========================================================
Generates all graphs and plots for the Neuro-Fuzzy Traffic System:

  1. Membership Function Plots (all inputs and outputs)
  2. Neural Network Training Curves (loss, MAE)
  3. Prediction Scatter Plots (actual vs predicted)
  4. Residual Distribution Plots
  5. System Comparison Bar Charts
  6. Traffic Pattern Heatmaps
  7. Feature Correlation Matrix
  8. Confusion Matrix for Priority Classification
  9. Surface Plots (3D fuzzy response surfaces)
  10. Data Distribution Histograms
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os


# Set global style
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'axes.edgecolor': '#e94560',
    'axes.labelcolor': '#eee',
    'text.color': '#eee',
    'xtick.color': '#aaa',
    'ytick.color': '#aaa',
    'grid.color': '#333',
    'grid.alpha': 0.3,
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.titlesize': 16,
})

# Color palettes
COLORS = {
    'primary': '#e94560',
    'secondary': '#0f3460',
    'accent': '#533483',
    'success': '#00d2d3',
    'warning': '#ff9f43',
    'info': '#54a0ff',
    'gradient': ['#e94560', '#ff6b6b', '#feca57', '#00d2d3', '#54a0ff', '#5f27cd'],
    'fuzzy_mf': ['#e94560', '#feca57', '#00d2d3', '#54a0ff', '#5f27cd'],
    'systems': ['#e94560', '#54a0ff', '#00d2d3']
}


class TrafficVisualizer:
    """
    Generates all visualizations for the Neuro-Fuzzy Traffic System.
    
    Attributes:
        output_dir (str): Directory to save generated plots.
    """

    def __init__(self, output_dir="outputs/plots"):
        """
        Initialize visualizer with output directory.
        
        Args:
            output_dir (str): Path to save generated plots.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _save_fig(self, fig, filename):
        """Save figure with tight layout."""
        fig.savefig(
            os.path.join(self.output_dir, filename),
            dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor()
        )
        plt.close(fig)
        print(f"  → Saved: {filename}")

    # ================================================================
    # 1. MEMBERSHIP FUNCTION PLOTS
    # ================================================================
    def plot_membership_functions(self, fuzzy_controller):
        """
        Plot all membership functions for inputs and outputs.
        
        Creates a 2×4 grid showing triangular/trapezoidal MFs for:
          Row 1: num_vehicles, traffic_density, waiting_time, emergency
          Row 2: green_duration, signal_priority, congestion_level
        
        Args:
            fuzzy_controller: FuzzyTrafficController instance.
        """
        mf_data = fuzzy_controller.get_membership_data()
        
        fig, axes = plt.subplots(2, 4, figsize=(22, 10))
        fig.suptitle('Membership Functions — Fuzzy Traffic Controller', 
                     fontsize=18, fontweight='bold', color='#e94560', y=0.98)

        # Plot inputs
        for idx, (key, data) in enumerate(mf_data['inputs'].items()):
            ax = axes[0, idx]
            for i, (term_name, mf) in enumerate(data['terms'].items()):
                color = COLORS['fuzzy_mf'][i % len(COLORS['fuzzy_mf'])]
                ax.plot(data['universe'], mf, linewidth=2.5, label=term_name.capitalize(), color=color)
                ax.fill_between(data['universe'], mf, alpha=0.15, color=color)
            ax.set_title(f"INPUT: {data['label']}", fontweight='bold')
            ax.set_xlabel(f"{data['label']} ({data['unit']})" if data['unit'] else data['label'])
            ax.set_ylabel('Membership Degree')
            ax.legend(loc='upper right', fontsize=9, framealpha=0.8)
            ax.set_ylim(-0.05, 1.15)
            ax.grid(True, alpha=0.2)

        # Plot outputs
        for idx, (key, data) in enumerate(mf_data['outputs'].items()):
            ax = axes[1, idx]
            for i, (term_name, mf) in enumerate(data['terms'].items()):
                color = COLORS['fuzzy_mf'][i % len(COLORS['fuzzy_mf'])]
                ax.plot(data['universe'], mf, linewidth=2.5, label=term_name.replace('_', ' ').capitalize(), color=color)
                ax.fill_between(data['universe'], mf, alpha=0.15, color=color)
            ax.set_title(f"OUTPUT: {data['label']}", fontweight='bold')
            ax.set_xlabel(f"{data['label']} ({data['unit']})")
            ax.set_ylabel('Membership Degree')
            ax.legend(loc='upper right', fontsize=9, framealpha=0.8)
            ax.set_ylim(-0.05, 1.15)
            ax.grid(True, alpha=0.2)

        # Hide the empty subplot
        axes[1, 3].set_visible(False)
        
        plt.tight_layout()
        self._save_fig(fig, 'membership_functions.png')

    # ================================================================
    # 2. NEURAL NETWORK TRAINING CURVES
    # ================================================================
    def plot_training_curves(self, history):
        """
        Plot training and validation loss/MAE curves.
        
        Args:
            history (dict): Keras training history dictionary.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Neural Network Training Progress', 
                     fontsize=16, fontweight='bold', color='#e94560')

        epochs = range(1, len(history['loss']) + 1)

        # Loss curve
        ax = axes[0]
        ax.plot(epochs, history['loss'], color=COLORS['primary'], linewidth=2, label='Training Loss', alpha=0.9)
        ax.plot(epochs, history['val_loss'], color=COLORS['success'], linewidth=2, label='Validation Loss', alpha=0.9)
        ax.fill_between(epochs, history['loss'], alpha=0.1, color=COLORS['primary'])
        ax.fill_between(epochs, history['val_loss'], alpha=0.1, color=COLORS['success'])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title('Training & Validation Loss')
        ax.legend(framealpha=0.8)
        ax.grid(True, alpha=0.2)

        # MAE curve
        ax = axes[1]
        ax.plot(epochs, history['mae'], color=COLORS['warning'], linewidth=2, label='Training MAE', alpha=0.9)
        ax.plot(epochs, history['val_mae'], color=COLORS['info'], linewidth=2, label='Validation MAE', alpha=0.9)
        ax.fill_between(epochs, history['mae'], alpha=0.1, color=COLORS['warning'])
        ax.fill_between(epochs, history['val_mae'], alpha=0.1, color=COLORS['info'])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAE')
        ax.set_title('Training & Validation MAE')
        ax.legend(framealpha=0.8)
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        self._save_fig(fig, 'training_curves.png')

    # ================================================================
    # 3. ACTUAL vs PREDICTED SCATTER PLOTS
    # ================================================================
    def plot_predictions(self, y_true, y_pred):
        """
        Scatter plot of actual vs predicted values for all outputs.
        
        Args:
            y_true (np.ndarray): Ground truth values (N×3).
            y_pred (np.ndarray): Predicted values (N×3).
        """
        target_names = ['Green Duration (s)', 'Signal Priority', 'Congestion Level (%)']
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle('Neural Network — Actual vs Predicted', 
                     fontsize=16, fontweight='bold', color='#e94560')

        for i, (name, ax) in enumerate(zip(target_names, axes)):
            ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.4, s=20, 
                      color=COLORS['gradient'][i], edgecolors='none')
            
            # Perfect prediction line
            lims = [min(y_true[:, i].min(), y_pred[:, i].min()),
                    max(y_true[:, i].max(), y_pred[:, i].max())]
            ax.plot(lims, lims, '--', color=COLORS['primary'], linewidth=2, label='Perfect Prediction')
            
            ax.set_xlabel(f'Actual {name}')
            ax.set_ylabel(f'Predicted {name}')
            ax.set_title(name, fontweight='bold')
            ax.legend(framealpha=0.8)
            ax.grid(True, alpha=0.2)

        plt.tight_layout()
        self._save_fig(fig, 'actual_vs_predicted.png')

    # ================================================================
    # 4. RESIDUAL DISTRIBUTION PLOTS
    # ================================================================
    def plot_residuals(self, y_true, y_pred):
        """
        Plot residual (error) distributions for each output.
        
        Args:
            y_true, y_pred: Ground truth and predicted values.
        """
        target_names = ['Green Duration', 'Signal Priority', 'Congestion Level']
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle('Prediction Residual Distributions', 
                     fontsize=16, fontweight='bold', color='#e94560')

        for i, (name, ax) in enumerate(zip(target_names, axes)):
            residuals = y_true[:, i] - y_pred[:, i]
            color = COLORS['gradient'][i]
            
            ax.hist(residuals, bins=50, alpha=0.7, color=color, edgecolor='white', linewidth=0.5)
            ax.axvline(x=0, color=COLORS['primary'], linewidth=2, linestyle='--', label='Zero Error')
            ax.axvline(x=residuals.mean(), color=COLORS['success'], linewidth=2, 
                      linestyle='-', label=f'Mean: {residuals.mean():.3f}')
            
            ax.set_xlabel('Residual (Actual - Predicted)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{name}\nStd: {residuals.std():.3f}', fontweight='bold')
            ax.legend(framealpha=0.8)
            ax.grid(True, alpha=0.2)

        plt.tight_layout()
        self._save_fig(fig, 'residual_distributions.png')

    # ================================================================
    # 5. SYSTEM COMPARISON BAR CHARTS
    # ================================================================
    def plot_system_comparison(self, comparison_df):
        """
        Bar chart comparing Fuzzy, NN, and Neuro-Fuzzy performance.
        
        Args:
            comparison_df (pd.DataFrame): Comparison metrics table.
        """
        metrics = ['RMSE', 'MAE', 'R²']
        outputs = comparison_df['Output'].unique()
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        fig.suptitle('System Performance Comparison — Fuzzy vs NN vs Neuro-Fuzzy', 
                     fontsize=16, fontweight='bold', color='#e94560')

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            x = np.arange(len(outputs))
            width = 0.25
            
            for j, system in enumerate(['Fuzzy', 'Neural Network', 'Neuro-Fuzzy']):
                values = []
                for output in outputs:
                    row = comparison_df[(comparison_df['Output'] == output) & 
                                       (comparison_df['System'] == system)]
                    values.append(row[metric].values[0])
                
                bars = ax.bar(x + j * width, values, width, 
                            label=system, color=COLORS['systems'][j], 
                            alpha=0.85, edgecolor='white', linewidth=0.5)
                
                # Value labels on bars
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=8, color='#eee')
            
            ax.set_xlabel('Output Variable')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} Comparison', fontweight='bold')
            ax.set_xticks(x + width)
            ax.set_xticklabels(outputs, fontsize=9)
            ax.legend(framealpha=0.8, fontsize=9)
            ax.grid(True, alpha=0.2, axis='y')

        plt.tight_layout()
        self._save_fig(fig, 'system_comparison.png')

    # ================================================================
    # 6. TRAFFIC PATTERN HEATMAP
    # ================================================================
    def plot_traffic_heatmap(self, df):
        """
        Heatmap showing green duration across vehicle count and density.
        
        Args:
            df (pd.DataFrame): Traffic dataset.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle('Traffic Signal Response Heatmap', 
                     fontsize=16, fontweight='bold', color='#e94560')

        # Create bins
        df_copy = df.copy()
        df_copy['veh_bin'] = pd.cut(df_copy['num_vehicles'], bins=10, labels=False)
        df_copy['dens_bin'] = pd.cut(df_copy['traffic_density'], bins=10, labels=False)
        
        pivot = df_copy.groupby(['veh_bin', 'dens_bin'])['green_duration'].mean().unstack()
        
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
                   linewidths=0.5, linecolor='#333',
                   cbar_kws={'label': 'Avg Green Duration (s)'})
        
        ax.set_xlabel('Traffic Density Bin')
        ax.set_ylabel('Vehicle Count Bin')
        ax.set_title('Average Green Light Duration\n(Vehicles × Density)', fontweight='bold')
        
        self._save_fig(fig, 'traffic_heatmap.png')

    # ================================================================
    # 7. FEATURE CORRELATION MATRIX
    # ================================================================
    def plot_correlation_matrix(self, df):
        """
        Plot correlation heatmap between all features and targets.
        
        Args:
            df (pd.DataFrame): Full dataset.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.suptitle('Feature-Target Correlation Matrix', 
                     fontsize=16, fontweight='bold', color='#e94560')

        cols = ['num_vehicles', 'traffic_density', 'waiting_time', 
                'emergency_vehicle', 'green_duration', 'signal_priority', 'congestion_level']
        
        corr = df[cols].corr()
        
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', center=0,
                   cmap='RdYlGn', ax=ax, square=True,
                   linewidths=1, linecolor='#333',
                   vmin=-1, vmax=1,
                   cbar_kws={'label': 'Correlation Coefficient'})
        
        ax.set_title('Pearson Correlation Matrix', fontweight='bold')
        
        labels = ['Vehicles', 'Density', 'Wait Time', 'Emergency', 
                  'Green Dur.', 'Priority', 'Congestion']
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels, rotation=0)
        
        self._save_fig(fig, 'correlation_matrix.png')

    # ================================================================
    # 8. DATA DISTRIBUTION PLOTS
    # ================================================================
    def plot_data_distributions(self, df):
        """
        Plot distributions of all input and output variables.
        
        Args:
            df (pd.DataFrame): Full dataset.
        """
        cols = ['num_vehicles', 'traffic_density', 'waiting_time',
                'green_duration', 'signal_priority', 'congestion_level']
        titles = ['Number of Vehicles', 'Traffic Density (%)', 'Waiting Time (s)',
                  'Green Duration (s)', 'Signal Priority', 'Congestion Level (%)']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Data Distribution Analysis', 
                     fontsize=16, fontweight='bold', color='#e94560')

        for idx, (col, title, ax) in enumerate(zip(cols, titles, axes.flatten())):
            color = COLORS['gradient'][idx % len(COLORS['gradient'])]
            
            ax.hist(df[col], bins=40, alpha=0.7, color=color, 
                   edgecolor='white', linewidth=0.5)
            
            # Statistics
            mean_val = df[col].mean()
            std_val = df[col].std()
            ax.axvline(mean_val, color='white', linewidth=2, linestyle='--',
                      label=f'Mean: {mean_val:.1f}')
            
            ax.set_title(f'{title}\n(σ={std_val:.1f})', fontweight='bold')
            ax.set_xlabel(title)
            ax.set_ylabel('Count')
            ax.legend(framealpha=0.8, fontsize=9)
            ax.grid(True, alpha=0.2)

        plt.tight_layout()
        self._save_fig(fig, 'data_distributions.png')

    # ================================================================
    # 9. EMERGENCY vs NORMAL COMPARISON
    # ================================================================
    def plot_emergency_comparison(self, df):
        """
        Box plots comparing outputs for emergency vs normal scenarios.
        
        Args:
            df (pd.DataFrame): Full dataset.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Emergency vs Normal Traffic — Output Comparison', 
                     fontsize=16, fontweight='bold', color='#e94560')

        targets = ['green_duration', 'signal_priority', 'congestion_level']
        titles = ['Green Duration (s)', 'Signal Priority', 'Congestion Level (%)']

        for ax, target, title in zip(axes, targets, titles):
            data = [df[df['emergency_vehicle'] == 0][target].values,
                    df[df['emergency_vehicle'] == 1][target].values]
            
            try:
                bp = ax.boxplot(data, tick_labels=['Normal', 'Emergency'], 
                              patch_artist=True, widths=0.5)
            except TypeError:
                bp = ax.boxplot(data, labels=['Normal', 'Emergency'], 
                              patch_artist=True, widths=0.5)
            
            bp['boxes'][0].set_facecolor(COLORS['info'])
            bp['boxes'][1].set_facecolor(COLORS['primary'])
            bp['boxes'][0].set_alpha(0.7)
            bp['boxes'][1].set_alpha(0.7)
            
            for median in bp['medians']:
                median.set_color('white')
                median.set_linewidth(2)
            
            ax.set_title(title, fontweight='bold')
            ax.set_ylabel(title)
            ax.grid(True, alpha=0.2, axis='y')

        plt.tight_layout()
        self._save_fig(fig, 'emergency_comparison.png')

    # ================================================================
    # 10. FUZZY SURFACE PLOT
    # ================================================================
    def plot_fuzzy_surface(self, fuzzy_controller):
        """
        3D surface plot showing fuzzy system response.
        
        Shows green duration as a function of (traffic_density, waiting_time)
        with fixed num_vehicles=50 and no emergency.
        
        Args:
            fuzzy_controller: FuzzyTrafficController instance.
        """
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        density_range = np.linspace(5, 95, 25)
        wait_range = np.linspace(5, 175, 25)
        D, W = np.meshgrid(density_range, wait_range)
        
        Z = np.zeros_like(D)
        for i in range(D.shape[0]):
            for j in range(D.shape[1]):
                result = fuzzy_controller.predict(50, D[i, j], W[i, j], 0)
                Z[i, j] = result['green_duration']
        
        surf = ax.plot_surface(D, W, Z, cmap='plasma', alpha=0.85,
                              edgecolor='none', antialiased=True)
        
        ax.set_xlabel('Traffic Density (%)', labelpad=10)
        ax.set_ylabel('Waiting Time (s)', labelpad=10)
        ax.set_zlabel('Green Duration (s)', labelpad=10)
        ax.set_title('Fuzzy System Response Surface\n(Vehicles=50, No Emergency)', 
                     fontweight='bold', color='#e94560', fontsize=14)
        
        fig.colorbar(surf, shrink=0.5, aspect=10, label='Green Duration (s)')
        ax.view_init(elev=25, azim=45)
        
        self._save_fig(fig, 'fuzzy_surface_3d.png')

    # ================================================================
    # 11. TIME-OF-DAY TRAFFIC PATTERN
    # ================================================================
    def plot_time_patterns(self, df):
        """
        Plot traffic patterns across different times of day.
        
        Args:
            df (pd.DataFrame): Dataset with 'hour' column.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Traffic Patterns by Time of Day', 
                     fontsize=16, fontweight='bold', color='#e94560')

        df_sorted = df.sort_values('hour')
        
        plots = [
            ('num_vehicles', 'Number of Vehicles', COLORS['primary']),
            ('traffic_density', 'Traffic Density (%)', COLORS['success']),
            ('waiting_time', 'Waiting Time (s)', COLORS['warning']),
            ('green_duration', 'Green Duration (s)', COLORS['info'])
        ]

        for ax, (col, title, color) in zip(axes.flatten(), plots):
            # Bin by hour and compute mean
            df_sorted['hour_bin'] = df_sorted['hour'].round()
            hourly = df_sorted.groupby('hour_bin')[col].agg(['mean', 'std']).reset_index()
            
            ax.plot(hourly['hour_bin'], hourly['mean'], color=color, linewidth=2.5)
            ax.fill_between(hourly['hour_bin'], 
                          hourly['mean'] - hourly['std'],
                          hourly['mean'] + hourly['std'],
                          alpha=0.2, color=color)
            
            # Highlight rush hours
            ax.axvspan(7, 9, alpha=0.08, color=COLORS['primary'], label='Morning Rush')
            ax.axvspan(16, 19, alpha=0.08, color=COLORS['warning'], label='Evening Rush')
            
            ax.set_xlabel('Hour of Day')
            ax.set_ylabel(title)
            ax.set_title(title, fontweight='bold')
            ax.legend(framealpha=0.8, fontsize=9)
            ax.grid(True, alpha=0.2)
            ax.set_xlim(0, 24)
            ax.set_xticks(range(0, 25, 3))

        plt.tight_layout()
        self._save_fig(fig, 'time_patterns.png')

    # ================================================================
    # 12. NEURO-FUZZY BREAKDOWN PLOT
    # ================================================================
    def plot_neuro_fuzzy_breakdown(self, nf_results, sample_size=100):
        """
        Show how fuzzy and NN outputs combine in the neuro-fuzzy system.
        
        Args:
            nf_results (dict): Results from neuro_fuzzy.predict_batch().
            sample_size (int): Number of samples to display.
        """
        fig, axes = plt.subplots(3, 1, figsize=(16, 14))
        fig.suptitle('Neuro-Fuzzy System Breakdown — Component Contributions', 
                     fontsize=16, fontweight='bold', color='#e94560')

        n = min(sample_size, len(nf_results['green_duration']))
        x = np.arange(n)

        targets = [
            ('green_duration', 'fuzzy_green', 'nn_green', 'Green Duration (s)'),
            ('signal_priority', 'fuzzy_priority', 'nn_priority', 'Signal Priority'),
            ('congestion_level', 'fuzzy_congestion', 'nn_congestion', 'Congestion Level (%)')
        ]

        for ax, (combined_key, fuzzy_key, nn_key, title) in zip(axes, targets):
            ax.plot(x, nf_results[fuzzy_key][:n], '--', color=COLORS['primary'], 
                   alpha=0.6, linewidth=1.5, label='Fuzzy Output')
            ax.plot(x, nf_results[nn_key][:n], '--', color=COLORS['info'], 
                   alpha=0.6, linewidth=1.5, label='NN Output')
            ax.plot(x, nf_results[combined_key][:n], '-', color=COLORS['success'], 
                   linewidth=2.5, label='Neuro-Fuzzy (Combined)')
            
            ax.set_xlabel('Sample Index')
            ax.set_ylabel(title)
            ax.set_title(title, fontweight='bold')
            ax.legend(framealpha=0.8, loc='upper right')
            ax.grid(True, alpha=0.2)

        plt.tight_layout()
        self._save_fig(fig, 'neuro_fuzzy_breakdown.png')

    # ================================================================
    # GENERATE ALL PLOTS
    # ================================================================
    def generate_all(self, fuzzy_controller, nn_history, y_true, y_pred,
                     comparison_df, df, nf_results):
        """
        Generate all visualization plots.
        
        Args:
            fuzzy_controller: FuzzyTrafficController instance.
            nn_history: Neural network training history dict.
            y_true: Ground truth test values.
            y_pred: Predicted test values.
            comparison_df: System comparison DataFrame.
            df: Full traffic dataset.
            nf_results: Neuro-fuzzy batch prediction results.
        """
        print("\n" + "=" * 50)
        print("  Generating Visualizations")
        print("=" * 50)
        
        print("\n[1/12] Membership Functions...")
        self.plot_membership_functions(fuzzy_controller)
        
        print("[2/12] Training Curves...")
        self.plot_training_curves(nn_history)
        
        print("[3/12] Actual vs Predicted...")
        self.plot_predictions(y_true, y_pred)
        
        print("[4/12] Residual Distributions...")
        self.plot_residuals(y_true, y_pred)
        
        print("[5/12] System Comparison...")
        self.plot_system_comparison(comparison_df)
        
        print("[6/12] Traffic Heatmap...")
        self.plot_traffic_heatmap(df)
        
        print("[7/12] Correlation Matrix...")
        self.plot_correlation_matrix(df)
        
        print("[8/12] Data Distributions...")
        self.plot_data_distributions(df)
        
        print("[9/12] Emergency Comparison...")
        self.plot_emergency_comparison(df)
        
        print("[10/12] Fuzzy Surface Plot...")
        self.plot_fuzzy_surface(fuzzy_controller)
        
        print("[11/12] Time Patterns...")
        self.plot_time_patterns(df)
        
        print("[12/12] Neuro-Fuzzy Breakdown...")
        self.plot_neuro_fuzzy_breakdown(nf_results)
        
        print(f"\n✅ All 12 plots saved to {self.output_dir}/")
