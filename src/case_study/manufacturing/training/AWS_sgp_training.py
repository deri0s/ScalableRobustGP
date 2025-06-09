# Set before importing any libraries
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from pathlib import Path
import argparse
import os
import torch
import gpytorch
import numpy as np
import traceback # For detailed error printing
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
# GPyTorch imports
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import ScaleKernel
from gpytorch.kernels import RBFKernel as RBF, RQKernel as RQ, PeriodicKernel as Per, LinearKernel as Lin
from gpytorch.constraints import GreaterThan
from models.svgp_auto_model_construction import GPTraining, SVGP

def model_fn(model_dir):
    """ Load the trained model from the model directory """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(os.path.join(model_dir, "sgpr_model.pth"), map_location=device)
    model.eval()
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Hyperparameters sent by SageMaker are passed as command-line arguments to the scrip
    parser.add_argument('--outputscale', type=float, default=1.0)
    parser.add_argument('--se_lengthscale', type=float, default=1.0)
    parser.add_argument('--rq_lengthscale', type=float, default=1.0)
    parser.add_argument('--rq_alpha', type=float, default=1.0)
    parser.add_argument('--per_period_length', type=float, default=7.0)
    parser.add_argument('--per_lengthscale', type=float, default=2)
    parser.add_argument('--noise_variance', type=float, default=0.026)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--training_iterations', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.009)
    parser.add_argument('--levels', type=int, default=1)
    parser.add_argument('--N_sim', type=int, default=5)
    # SageMaker specific arguments
    ROOT_PATH = Path(__file__).resolve()
    print('crrent-path: ', ROOT_PATH.parent)
    # parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ[str(ROOT_PATH)])
    # parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    floating_point = torch.float64
    # Load data
    train_file = os.path.join(args.train, "train_data.npz")
    test_file = os.path.join(args.test, "test_data.npz")
    train_data = np.load(train_file)
    test_data = np.load(test_file)
    X_train_np = train_data['X']
    y_train_nonstand = train_data['y']
    X_test_np = test_data['X']
    y_test_nonstand = test_data['y']

    # Standardise outputs
    y_train_reshape = y_train_nonstand.reshape(-1,1)
    scaler = StandardScaler()
    scaler.fit(y_train_reshape)
    y_stand_np = scaler.transform(y_train_reshape)
    y_test_reshape = y_test_nonstand.reshape(-1,1)
    y_test_stand = scaler.transform(y_test_reshape)

    # Convert data to torch tensors
    X_train = torch.tensor(X_train_np, dtype=floating_point).to(device)
    y_train = torch.tensor(y_stand_np, dtype=floating_point).squeeze().to(device)
    X_test = torch.tensor(X_test_np, dtype=floating_point).to(device)
    y_test = torch.tensor(y_test_stand, dtype=floating_point).squeeze().to(device)

    # Define the model dimensions
    N_train, D = X_train.shape

    import warnings
    M = 120 # Number of inducing points
    init_ip_method = 'kmeans++'

    if init_ip_method == 'random':
        indices = np.random.choice(N_train, min(M, N_train), replace=False)
        inducing_points =  X_train[indices, :]
        inducing_points = X_train[np.random.choice(N_train, M, replace=False), :]
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kmeans = KMeans(n_clusters=M, init='k-means++', n_init=10)
            kmeans.fit(X_train_np)
            inducing_points = torch.tensor(kmeans.cluster_centers_, dtype=floating_point)

    # Create initial model
    likelihood = GaussianLikelihood(noise_constraint=GreaterThan(1e-5))
    k = ScaleKernel(RQ(ard_num_dims=D, dtype=floating_point))
    gp0 = SVGP(inducing_points, D, k)
    gp0.likelihood = likelihood
    gp0.likelihood.noise = torch.tensor(args.noise_variance, dtype=floating_point)
    gp0.to(floating_point).to(device)

    # Define parameter limits
    limits = {
        'outputscale': [0.1, 10.0],
        'se_lengthscale': [0.05, 100.0],
        'rq_lengthscale': [0.05, 100.0],
        'rq_alpha': [0.05, 5],
        'per_period_length': [6, 10],
        'per_lengthscale': [0.05, 100.0],
        'lin_variance': [0.1*0.025, 0.025],
        'noise_variance': [0.025, 0.028]
    }
    # Automatic Model Construction
    auto_trainer = GPTraining(gp0, X_train, y_train, X_test, y_test)
    gp_gs = auto_trainer.auto_model_cons(
                        levels=args.levels,
                        N_sim=args.N_sim,
                        param_limits=limits,
                        mse_stop=0.05,
                        lr=args.lr,
                        training_iterations=args.training_iterations,
                        batch_size=args.batch_size
    )
    # Enhanced kernel parameter extraction for complex models
    def extract_kernel_info(model):
        kernel_info = {}
        base_kernel = model.covar_module.base_kernel
        
        print('\n--Estimated Model and kernel parameters--')
        print(f'Outputscale: {model.covar_module.outputscale.item():.4f}')
        
        # Handle different kernel types and combinations
        if hasattr(base_kernel, 'kernels'):  # Additive or Multiplicative kernel
            print(f"Complex kernel with {len(base_kernel.kernels)} components:")
            for i, kernel in enumerate(base_kernel.kernels):
                print(f"  Component {i+1}: {type(kernel).__name__}")
                if isinstance(kernel, (RBF, RQ)):
                    lengthscales = kernel.lengthscale.squeeze()
                    kernel_info[f'component_{i}_lengthscales'] = lengthscales
                    print(f"    Lengthscales: {lengthscales.tolist()}")
                    if isinstance(kernel, RQ):
                        print(f"    Alpha: {kernel.alpha.item():.4f}")
                elif isinstance(kernel, Per):
                    print(f"    Period: {kernel.period_length.squeeze().tolist()}")
                    print(f"    Lengthscales: {kernel.lengthscale.squeeze().tolist()}")
                elif isinstance(kernel, Lin):
                    print(f"    Variance: {kernel.variance.squeeze().tolist()}")
        else:  # Single kernel
            if isinstance(base_kernel, RBF):
                lengthscales = base_kernel.lengthscale.squeeze()
                kernel_info['lengthscales'] = lengthscales
                print(f'RBF Lengthscales: {lengthscales.tolist()}')
            elif isinstance(base_kernel, RQ):
                lengthscales = base_kernel.lengthscale.squeeze()
                kernel_info['lengthscales'] = lengthscales
                print(f'RQ Lengthscales: {lengthscales.tolist()}')
                print(f'RQ Alpha: {base_kernel.alpha.item():.4f}')
            elif isinstance(base_kernel, Per):
                print(f'Periodic Period: {base_kernel.period_length.squeeze().tolist()}')
                print(f'Periodic Lengthscales: {base_kernel.lengthscale.squeeze().tolist()}')
            elif isinstance(base_kernel, Lin):
                print(f'Linear Variance: {base_kernel.variance.squeeze().tolist()}')
        
        print(f'Noise variance: {model.likelihood.noise.item():.6f}')
        return kernel_info

    kernel_info = extract_kernel_info(gp_gs)

    def feature_selection(kernel_info, ls_threshold=50.0):
        """
        Identify features with lengthscales <= threshold for feature selection
        
        Args:
            kernel_info: Dictionary containing kernel lengthscale information
            lengthscale_threshold: Threshold for lengthscale filtering
        
        Returns:
            relevant_indices: List of feature indices to keep
            removed_indices: List of feature indices to remove
        """
        relevant_indices = []
        removed_indices = []
        all_lengthscales = []
        
        for key, lengthscales in kernel_info.items():
            if 'lengthscales' in key and torch.is_tensor(lengthscales):
                if lengthscales.dim() == 0:  # Scalar
                    all_lengthscales.append(lengthscales.item())
                else:  # Vector
                    all_lengthscales.extend(lengthscales.tolist())
        
        # If no lengthscales found in complex kernel, try to extract from simple kernel
        if not all_lengthscales and 'lengthscales' in kernel_info:
            ls = kernel_info['lengthscales']
            if torch.is_tensor(ls):
                if ls.dim() == 0:
                    all_lengthscales = [ls.item()]
                else:
                    all_lengthscales = ls.tolist()
        
        # If still no lengthscales, extract directly from model
        if not all_lengthscales:
            print("Warning: Could not extract lengthscales from kernel_info. Extracting directly from model...")
            try:
                base_kernel = gp_gs.covar_module.base_kernel
                if hasattr(base_kernel, 'lengthscale'):
                    ls = base_kernel.lengthscale.squeeze()
                    all_lengthscales = ls.tolist() if ls.dim() > 0 else [ls.item()]
            except:
                print("Could not extract lengthscales. Keeping all features.")
                return list(range(D)), []
        
        # Ensure we have the right number of lengthscales
        if len(all_lengthscales) < D:
            print(f"Warning: Found {len(all_lengthscales)} lengthscales but have {D} features.")
            print("Using available lengthscales and keeping remaining features.")
            all_lengthscales.extend([1.0] * (D - len(all_lengthscales)))  # Default to 1.0 for missing
        elif len(all_lengthscales) > D:
            print(f"Warning: Found {len(all_lengthscales)} lengthscales but only have {D} features.")
            all_lengthscales = all_lengthscales[:D]  # Truncate
        
        # Identify relevant and irrelevant features
        for i, ls in enumerate(all_lengthscales):
            if ls <= ls_threshold:
                relevant_indices.append(i)
            else:
                removed_indices.append(i)
        
        print(f"\nFeature Selection Results: with threshold={ls_threshold}")
        print(f"  Features to keep: {len(relevant_indices)}/{D}")
        print(f"  Features to remove: {len(removed_indices)}/{D}")
        print(f"  Removed feature indices: {removed_indices}")
        
        return relevant_indices, removed_indices

    # Apply feature selection
    relevant_indices, removed_indices = feature_selection(kernel_info,
                                                          ls_threshold=50.0)

    # Create filtered datasets if features were removed
    if removed_indices:
        print(f"\n🔥 Removing {len(removed_indices)} features with high lengthscales...")
        
        # Filter training and test data
        X_train_reduced = X_train[:, relevant_indices]
        X_test_reduced = X_test[:, relevant_indices] 
        
        # Update dimensions
        D_filtered = len(relevant_indices)
        print(f"Reduced feature dimensions: {D} -> {D_filtered}")
        
        # Re-initialize inducing points with filtered data
        if init_ip_method == 'random':
            indices = np.random.choice(N_train, min(M, N_train), replace=False)
            inducing_points_filtered = X_train_reduced[indices, :]
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kmeans = KMeans(n_clusters=M, init='k-means++', n_init=10)
                kmeans.fit(X_train_reduced.numpy())
                inducing_points_filtered = torch.tensor(kmeans.cluster_centers_, dtype=floating_point)
        
        # Create new kernel with filtered dimensions
        k_filtered = ScaleKernel(RQ(ard_num_dims=D_filtered, dtype=floating_point))
        
        # Create filtered model for tuning
        gp_reduced = SVGP(inducing_points_filtered, D_filtered, k_filtered)
        gp_reduced.likelihood = GaussianLikelihood(noise_constraint=GreaterThan(1e-5))
        gp_reduced.likelihood.noise = torch.tensor(0.028, dtype=floating_point)
        gp_reduced.to(floating_point)        
    else:
        print("\n✓ No features removed (all lengthscales <= 50). Using original data for tuning.")
        X_train_filtered = X_train
        X_test_filtered = X_test
        D_filtered = D
        gp_reduced = gp_gs

    # Fine-tuning
    def generate_stds(lengthscales, base_std_dev):
        if not torch.is_tensor(lengthscales):
            lengthscales = torch.tensor(lengthscales)
            min_lengthscale = torch.min(lengthscales)
            std_devs = base_std_dev * torch.exp((lengthscales - min_lengthscale)/6)
            std_devs = torch.tensor([300 if std == torch.inf else std for std in std_devs])
            return std_devs.tolist()
    # Extract lengthscales for tuning (from the reduced model if applicable)
    try:
        if hasattr(gp_reduced.covar_module.base_kernel, 'lengthscale'):
            tuning_lengthscales = gp_reduced.covar_module.base_kernel.lengthscale.squeeze()
        else:
            tuning_lengthscales = gp_reduced.covar_module.base_kernel.kernels[0].lengthscale.squeeze()
        
        ls_stds = generate_stds(tuning_lengthscales, base_std_dev=1e-4)
    except Exception as e:
        print(f"Warning: Could not extract lengthscales for tuning: {e}")
        ls_stds = [1e-4] * D_filtered  # Default std devs

    if gp_reduced is not gp0: # Only tune if we have a model to tune
        stds = {'outputscale': 1e-3,
                'se_lengthscale': ls_stds,
                'rq_lengthscale': ls_stds,
                'per_lengthscale': ls_stds,
                'rq_alpha': 1e-2,
                'lin_variance': 1e-3,
                'noise_variance': 1e-4}
        
        auto_tuner = GPTraining(gp_reduced, X_train_filtered, y_train, X_test_filtered, y_test)
        auto_tuner.param_stds = stds

        """ Hyperparameter tunning """
        try:
            print(f"\n🔧 Starting hyperparameter tuning with {D_filtered} features...")
            tuned_gp = auto_tuner.tune(
                gp_to_tune=gp_reduced,
                N_sim=5,
                mse_stop=0.003,
                lr=0.005,
                training_iterations=100,
                batch_size=256)
            print("✓ Hyperparameter tuning completed successfully")
            
            # Make predictions with tuned model
            tuned_gp.eval()
            tuned_gp.likelihood.eval()

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred_tuned = tuned_gp.likelihood(tuned_gp(X_test))

            # Unormalise predictions
            pred_mean_tuned = observed_pred_tuned.mean
            mu_tuned = scaler.inverse_transform(pred_mean_tuned.unsqueeze(1))[:,0]
            stds_tuned = scaler.inverse_transform(observed_pred_tuned.stddev.unsqueeze(1))[:,0]
            lower_stand_tuned, upper_stand_tuned = observed_pred_tuned.confidence_region()
            lower_tuned = scaler.inverse_transform(lower_stand_tuned.unsqueeze(1))[:,0]
            upper_tuned = scaler.inverse_transform(upper_stand_tuned.unsqueeze(1))[:,0]

            print(f'MSE-tuned: {mean_squared_error(mu_tuned, y_test_nonstand):.6f}')
            
        except Exception as e:
            print(f"❌ Error during tuning: {e}")
            traceback.print_exc()
            print("Using non-tuned model for final predictions")
    else:
        print("No tuning performed (using original model)")
        tuned_gp = gp_reduced

    # Save the model
    # torch.save(tuned_gp, os.path.join(args.model_dir, "sgpr_model.pth"))
    # # Also save the scaler for inference
    # torch.save(scaler, os.path.join(args.model_dir, "scaler.pth"))