from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt

app = Flask(__name__)

class LogisticRegressionExperiment:
    def __init__(self):
        # Ensure results directory exists
        self.results_dir = 'results'
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def generate_data(self, n_samples=100, shift_distance=1.0, random_state=42):
        """Generate two clusters with specified shift distance."""
        np.random.seed(random_state)
        
        # Generate first cluster centered at origin
        cluster1_x = np.random.normal(0, 1, n_samples//2)
        cluster1_y = np.random.normal(0, 1, n_samples//2)
        cluster1 = np.column_stack((cluster1_x, cluster1_y))
        
        # Generate second cluster with shift
        cluster2_x = np.random.normal(shift_distance, 1, n_samples//2)
        cluster2_y = np.random.normal(shift_distance, 1, n_samples//2)
        cluster2 = np.column_stack((cluster2_x, cluster2_y))
        
        # Combine data and create labels
        X = np.vstack((cluster1, cluster2))
        y = np.array([0] * (n_samples//2) + [1] * (n_samples//2))
        
        return X, y

    def fit_model(self, X, y):
        """Fit logistic regression and return model parameters."""
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        
        return {
            'beta0': model.intercept_[0],
            'beta1': model.coef_[0][0],
            'beta2': model.coef_[0][1],
            'loss': -model.score(X, y)  # Using negative log likelihood as loss
        }

    def plot_decision_boundary(self, X, y, model_params, shift_distance):
        """Plot dataset and decision boundary."""
        plt.figure(figsize=(10, 8))
        
        # Plot data points
        plt.scatter(X[y==0, 0], X[y==0, 1], color='blue', label='Class 0')
        plt.scatter(X[y==1, 0], X[y==1, 1], color='red', label='Class 1')
        
        # Calculate and plot decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x_points = np.linspace(x_min, x_max, 100)
        y_points = -(model_params['beta0'] + model_params['beta1'] * x_points) / model_params['beta2']
        
        plt.plot(x_points, y_points, 'k--', label='Decision Boundary')
        plt.title(f'Logistic Regression Decision Boundary (Shift = {shift_distance:.2f})')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        
        # Save plot
        plt.savefig(os.path.join(self.results_dir, 'dataset.png'))
        plt.close()

    def plot_parameters(self, shift_distances, parameters):
        """Plot how parameters change with shift distance."""
        plt.figure(figsize=(15, 10))
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot beta parameters
        ax1.plot(shift_distances, [p['beta0'] for p in parameters], label='β₀')
        ax1.plot(shift_distances, [p['beta1'] for p in parameters], label='β₁')
        ax1.plot(shift_distances, [p['beta2'] for p in parameters], label='β₂')
        ax1.set_title('Model Parameters vs. Shift Distance')
        ax1.set_xlabel('Shift Distance')
        ax1.set_ylabel('Parameter Value')
        ax1.legend()
        
        # Plot loss
        ax2.plot(shift_distances, [p['loss'] for p in parameters])
        ax2.set_title('Logistic Loss vs. Shift Distance')
        ax2.set_xlabel('Shift Distance')
        ax2.set_ylabel('Loss')
        
        # Plot margin width
        margin_widths = [2/np.sqrt(p['beta1']**2 + p['beta2']**2) for p in parameters]
        ax3.plot(shift_distances, margin_widths)
        ax3.set_title('Margin Width vs. Shift Distance')
        ax3.set_xlabel('Shift Distance')
        ax3.set_ylabel('Margin Width')
        
        # Plot decision boundary slope
        slopes = [-p['beta1']/p['beta2'] for p in parameters]
        ax4.plot(shift_distances, slopes)
        ax4.set_title('Decision Boundary Slope vs. Shift Distance')
        ax4.set_xlabel('Shift Distance')
        ax4.set_ylabel('Slope')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'parameters_vs_shift_distance.png'))
        plt.close()

def do_experiments(start, end, step_num):
    """Run experiments for different shift distances."""
    try:
        experiment = LogisticRegressionExperiment()
        
        # Generate shift distances
        shift_distances = np.linspace(start, end, step_num)
        parameters = []
        
        # Run experiments for each shift distance
        for distance in shift_distances:
            # Generate data
            X, y = experiment.generate_data(shift_distance=distance)
            
            # Fit model and get parameters
            model_params = experiment.fit_model(X, y)
            parameters.append(model_params)
            
            # Plot dataset and decision boundary for the last distance
            if distance == shift_distances[-1]:
                experiment.plot_decision_boundary(X, y, model_params, distance)
        
        # Plot parameter changes
        experiment.plot_parameters(shift_distances, parameters)
        
        return True
    except Exception as e:
        print(f"Error in do_experiments: {str(e)}")
        raise

# Define the main route
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle experiment parameters and trigger the experiment
@app.route('/run_experiment', methods=['POST'])
def run_experiment():
    try:
        # Ensure results directory exists
        if not os.path.exists('results'):
            os.makedirs('results')
        
        # Get and validate input data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data received'}), 400
        
        # Extract parameters
        start = float(data.get('start', 0))
        end = float(data.get('end', 0))
        step_num = int(data.get('step_num', 0))
        
        # Additional validation
        if start >= end:
            return jsonify({'error': 'Start must be less than end'}), 400
        if step_num <= 0:
            return jsonify({'error': 'Step number must be positive'}), 400
        
        # Run the experiment
        do_experiments(start, end, step_num)
        
        # Check if files were created
        dataset_path = "results/dataset.png"
        parameters_path = "results/parameters_vs_shift_distance.png"
        
        if not os.path.exists(dataset_path) or not os.path.exists(parameters_path):
            return jsonify({'error': 'Failed to generate images'}), 500
        
        # Return paths without leading slash
        return jsonify({
            "dataset_img": dataset_path,
            "parameters_img": parameters_path
        })
    
    except Exception as e:
        print(f"Error in run_experiment: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Route to serve result images
@app.route('/results/<filename>')
def results(filename):
    return send_from_directory('results', filename)

if __name__ == '__main__':
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    app.run(debug=True, port=5000)