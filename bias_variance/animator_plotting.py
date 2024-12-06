import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import norm
import matplotlib.gridspec as gridspec

class PredictionAnimator:
    def __init__(self, true_func, x_range=(-2*np.pi, 2*np.pi), num_points=100):
        # Create figure with subplots
        self.fig = plt.figure(figsize=(15, 12))
        gs = gridspec.GridSpec(3, 2, height_ratios=[3, 1, 1])
        
        # Main plot
        self.ax_main = plt.subplot(gs[0, :])
        # Error distribution plot
        self.ax_error = plt.subplot(gs[1, 0])
        # Running metrics plot
        self.ax_metrics = plt.subplot(gs[1, 1])
        # Bias-variance decomposition plot
        self.ax_bias_var = plt.subplot(gs[2, :])
        
        self.true_func = true_func
        self.x_range = x_range
        
        # Generate true function line
        self.x_continuous = np.linspace(x_range[0], x_range[1], 500)
        self.y_continuous = self.true_func(self.x_continuous)
        
        # Plot true function
        self.true_line, = self.ax_main.plot(
            self.x_continuous, 
            self.y_continuous, 
            'b-', label='True Function', 
            alpha=0.8
        )
        
        # Initialize storage for predictions and statistics
        self.pred_lines = []
        self.max_lines = 10
        self.alpha_decay = 0.7
        self.all_x_preds = []
        self.all_y_preds = []
        
        # Storage for statistical elements
        self.mean_line = None
        self.std_band = None
        self.percentile_band = None
        
        # Storage for metrics
        self.rmse_history = []
        self.mae_history = []
        self.max_error_history = []
        self.metrics_lines = []
        
        # Storage for bias-variance decomposition
        self.bias_history = []
        self.variance_history = []
        self.total_error_history = []
        
        # Initialize error distribution plot
        self.error_hist = None
        self.error_kde = None
        
        self._setup_plots()
        
    def _setup_plots(self):
        """Set up the appearance of all plots"""
        # Main plot
        self.ax_main.set_xlabel('x')
        self.ax_main.set_ylabel('y')
        self.ax_main.set_title('Function Predictions with Statistics')
        self.ax_main.grid(True, alpha=0.3)
        
        # Error distribution plot
        self.ax_error.set_title('Error Distribution')
        self.ax_error.set_xlabel('Error')
        self.ax_error.set_ylabel('Density')
        
        # Metrics plot
        self.ax_metrics.set_title('Running Metrics')
        self.ax_metrics.set_xlabel('Prediction Number')
        self.ax_metrics.grid(True, alpha=0.3)
        
        # Bias-variance plot
        self.ax_bias_var.set_title('Bias-Variance Decomposition')
        self.ax_bias_var.set_xlabel('Prediction Number')
        self.ax_bias_var.set_ylabel('Error Component')
        self.ax_bias_var.grid(True, alpha=0.3)
        
        plt.tight_layout()

    def calculate_bias_variance(self, x_common, interpolated_y, y_true):
        """Calculate bias-variance decomposition of the error"""
        # Calculate mean prediction
        mean_prediction = np.mean(interpolated_y, axis=0)
        
        # Calculate bias (squared difference between mean prediction and true function)
        bias = np.mean((mean_prediction - y_true) ** 2)
        
        # Calculate variance (average squared deviation from mean prediction)
        variance = np.mean(np.var(interpolated_y, axis=0))
        
        # Calculate total error (MSE)
        total_error = np.mean(np.mean((interpolated_y - y_true.reshape(1, -1)) ** 2, axis=0))
        
        # Irreducible error can be estimated as: total_error - (bias + variance)
        irreducible_error = total_error - (bias + variance)
        
        return bias, variance, total_error, irreducible_error

    def calculate_statistics(self):
        """Calculate comprehensive statistics of predictions"""
        if not self.all_x_preds:
            return None, None, None, None, None, None
        
        x_common = np.linspace(min(np.min(x) for x in self.all_x_preds),
                             max(np.max(x) for x in self.all_x_preds),
                             200)
        
        interpolated_y = []
        for x, y in zip(self.all_x_preds, self.all_y_preds):
            interpolated_y.append(np.interp(x_common, x, y))
        
        interpolated_y = np.array(interpolated_y)
        
        # Basic statistics
        y_mean = np.mean(interpolated_y, axis=0)
        y_std = np.std(interpolated_y, axis=0)
        
        # Percentiles
        y_percentile_25 = np.percentile(interpolated_y, 25, axis=0)
        y_percentile_75 = np.percentile(interpolated_y, 75, axis=0)
        
        # Calculate errors
        y_true = self.true_func(x_common)
        prediction_errors = interpolated_y - y_true
        
        # Calculate bias-variance decomposition
        bias, variance, total_error, irreducible_error = self.calculate_bias_variance(
            x_common, interpolated_y, y_true)
        
        return (x_common, y_mean, y_std, (y_percentile_25, y_percentile_75), 
                prediction_errors, (bias, variance, total_error, irreducible_error))

    def update(self, frame):
        x_pred, y_pred = frame
        elements = []
        
        # Store predictions
        self.all_x_preds.append(x_pred)
        self.all_y_preds.append(y_pred)
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((y_pred - self.true_func(x_pred))**2))
        mae = np.mean(np.abs(y_pred - self.true_func(x_pred)))
        max_error = np.max(np.abs(y_pred - self.true_func(x_pred)))
        
        self.rmse_history.append(rmse)
        self.mae_history.append(mae)
        self.max_error_history.append(max_error)
        
        # Keep only recent predictions
        if len(self.all_x_preds) > self.max_lines:
            self.all_x_preds.pop(0)
            self.all_y_preds.pop(0)
        
        # Calculate statistics
        stats = self.calculate_statistics()
        if stats[0] is not None:
            x_common, y_mean, y_std, percentiles, errors, bias_var = stats
            
            # Update main plot statistics
            if self.mean_line:
                self.mean_line.remove()
            if self.std_band:
                self.std_band.remove()
            if self.percentile_band:
                self.percentile_band.remove()
            
            # Plot statistical elements
            self.mean_line, = self.ax_main.plot(x_common, y_mean, 'g-', 
                                              label='Mean Prediction', alpha=0.8)
            self.std_band = self.ax_main.fill_between(x_common, 
                                                    y_mean - y_std, 
                                                    y_mean + y_std, 
                                                    color='g', alpha=0.2,
                                                    label='±1 Std Dev')
            self.percentile_band = self.ax_main.fill_between(x_common, 
                                                           percentiles[0], 
                                                           percentiles[1], 
                                                           color='y', alpha=0.2,
                                                           label='25-75 Percentile')
            elements.extend([self.mean_line, self.std_band, self.percentile_band])
            
            # Store bias-variance decomposition
            bias, variance, total_error, irreducible_error = bias_var
            self.bias_history.append(bias)
            self.variance_history.append(variance)
            self.total_error_history.append(total_error)
            
            # Update error distribution plot
            self.ax_error.clear()
            self.ax_error.set_title('Error Distribution')
            self.ax_error.set_xlabel('Error')
            self.ax_error.hist(errors.flatten(), bins=30, density=True, 
                             alpha=0.6, color='blue')
            
            # Fit and plot normal distribution
            mu, std = norm.fit(errors.flatten())
            x_error = np.linspace(errors.min(), errors.max(), 100)
            p = norm.pdf(x_error, mu, std)
            self.ax_error.plot(x_error, p, 'r-', lw=2, 
                             label=f'Normal(μ={mu:.2f}, σ={std:.2f})')
            self.ax_error.legend()
            
            # Update metrics plot
            self.ax_metrics.clear()
            self.ax_metrics.set_title('Running Metrics')
            self.ax_metrics.plot(self.rmse_history, label='RMSE', color='red')
            self.ax_metrics.plot(self.mae_history, label='MAE', color='blue')
            self.ax_metrics.plot(self.max_error_history, label='Max Error', 
                               color='green')
            self.ax_metrics.legend()
            self.ax_metrics.grid(True, alpha=0.3)
            
            # Update bias-variance plot
            self.ax_bias_var.clear()
            self.ax_bias_var.set_title('Bias-Variance Decomposition')
            self.ax_bias_var.plot(self.bias_history, label='Bias²', color='blue')
            self.ax_bias_var.plot(self.variance_history, label='Variance', color='red')
            self.ax_bias_var.plot(self.total_error_history, label='Total Error', 
                                color='purple', linestyle='--')
            self.ax_bias_var.plot([irreducible_error] * len(self.bias_history), 
                                label='Irreducible Error', color='gray', 
                                linestyle=':')
            self.ax_bias_var.legend()
            self.ax_bias_var.grid(True, alpha=0.3)
            
            # Add annotations for current values
            self.ax_bias_var.text(0.02, 0.98, 
                                f'Current Bias²: {bias:.4f}\n'
                                f'Current Variance: {variance:.4f}\n'
                                f'Total Error: {total_error:.4f}\n'
                                f'Irreducible Error: {irreducible_error:.4f}',
                                transform=self.ax_bias_var.transAxes,
                                verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add new prediction line
        new_line, = self.ax_main.plot(x_pred, y_pred, 'r.', alpha=0.8, 
                                    label='Current Prediction' if not self.pred_lines else "")
        self.pred_lines.append(new_line)
        elements.append(new_line)
        
        # Update alpha values for old lines
        for i, line in enumerate(self.pred_lines[:-1]):
            line.set_alpha(self.alpha_decay ** (len(self.pred_lines) - i - 1))
            elements.append(line)
        
        # Remove oldest lines if we exceed max_lines
        while len(self.pred_lines) > self.max_lines:
            old_line = self.pred_lines.pop(0)
            old_line.remove()
        
        # Update legend
        if len(self.pred_lines) == 1:
            handles = [self.true_line, self.mean_line, new_line]
            labels = ['True Function', 'Mean Prediction', 'Current Prediction']
            self.ax_main.legend(handles=handles, labels=labels)
        
        return elements

    def animate(self, predictions, interval=500):
        anim = FuncAnimation(self.fig, self.update, frames=predictions,
                           interval=interval, blit=True)
        plt.show()
        return anim