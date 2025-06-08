import argparse
import os
import torch
import gpytorch
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
# GPyTorch imports
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import ScaleKernel
from gpytorch.kernels import RQKernel as RQ
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
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--levels', type=int, default=1)
    parser.add_argument('--N_sim', type=int, default=5)
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
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
    M = min(150, N_train) # Number of inducing points

    # Initialise the inducing-points using kmeans++
    kmeans = KMeans(n_clusters=M, init='k-means++', n_init=10)
    kmeans.fit(X_train)
    inducing_points = torch.tensor(kmeans.cluster_centers_, dtype=X_train)

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
    # Fine-tuning
    def generate_stds(lengthscales, base_std_dev):
        if not torch.is_tensor(lengthscales):
            lengthscales = torch.tensor(lengthscales)
            min_lengthscale = torch.min(lengthscales)
            std_devs = base_std_dev * torch.exp((lengthscales - min_lengthscale)/6)
            std_devs = torch.tensor([300 if std == torch.inf else std for std in std_devs])
            return std_devs.tolist()
    try:
        ls_stds = generate_stds(gp_gs.covar_module.base_kernel.lengthscale.squeeze(),
                                base_std_dev=1e-4)
        stds = {
        'outputscale': 1e-3,
        'se_lengthscale': ls_stds,
        'rq_lengthscale': ls_stds,
        'rq_alpha': 1e-2,
        'noise_variance': 1e-4
        }
        auto_trainer.param_stds = stds
        tuned_gp = auto_trainer.tune(gp_to_tune=gp_gs,
                                     N_sim=10,
                                     mse_stop=0.003,
                                     lr=0.005,
                                     training_iterations=100,
                                     batch_size=256)
    except Exception as e:
        print(f"Error during tuning: {e}")
        tuned_gp = gp_gs # Fall back to untuned model

    # Evaluate final model
    tuned_gp.eval()
    tuned_gp.likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = tuned_gp.likelihood(tuned_gp(X_test))

    # Calculate and print MSE
    pred_mean = observed_pred.mean
    mu_test = scaler.inverse_transform(pred_mean.cpu().unsqueeze(1))[:,0]
    mse = ((mu_test - y_test_nonstand) ** 2).mean()
    print(f"Final MSE: {mse}")

    # Save the model
    # torch.save(tuned_gp, os.path.join(args.model_dir, "sgpr_model.pth"))
    # # Also save the scaler for inference
    # torch.save(scaler, os.path.join(args.model_dir, "scaler.pth"))