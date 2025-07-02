import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import boto3
import argparse
import torch
import gpytorch
import numpy as np
import traceback
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# GPyTorch imports
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import ScaleKernel
from gpytorch.kernels import RBFKernel as RBF, RQKernel as RQ, PeriodicKernel as Per, LinearKernel as Lin
from gpytorch.constraints import GreaterThan

# Import custom modules - handle import errors gracefully
try:
    from models.svgp_auto_model_construction import GPTraining, SVGP
    logger.info("✅ Successfully imported custom modules")
except ImportError as e:
    logger.error(f"❌ Failed to import custom modules: {e}")
    logger.error("Make sure your models directory is properly structured in the source_dir")
    raise


def validate_data_files(train_path, test_path, index):
    """Validate that data files exist and are readable"""
    train_file = os.path.join(train_path, f"train_data{index}.npz")
    test_file = os.path.join(test_path, f"test_data{index}.npz")

    logger.info(f"Looking for training data: {train_file}")
    logger.info(f"Looking for test data: {test_file}")

    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training data file not found: {train_file}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test data file not found: {test_file}")

    # Test loading files
    try:
        train_data = np.load(train_file)
        test_data = np.load(test_file)
        logger.info("✅ Successfully loaded data files")
        logger.info(f"Training data shape: X={train_data['X'].shape}, y={train_data['y'].shape}")
        logger.info(f"Test data shape: X={test_data['X'].shape}, y={test_data['y'].shape}")
        return train_data, test_data
    except Exception as e:
        logger.error(f"❌ Error loading data files: {e}")
        raise

if __name__ == "__main__":
    try:
        logger.info("🚀 Starting SGP training script")
        
        parser = argparse.ArgumentParser()
        # Hyperparameters sent by SageMaker are passed as command-line arguments
        parser.add_argument('--outputscale', type=float, default=1.0)
        parser.add_argument('--se_lengthscale', type=float, default=1.0)
        parser.add_argument('--rq_lengthscale', type=float, default=1.0)
        parser.add_argument('--rq_alpha', type=float, default=1.0)
        parser.add_argument('--per_period_length', type=float, default=7.0)
        parser.add_argument('--per_lengthscale', type=float, default=2)
        parser.add_argument('--noise_variance', type=float, default=0.022)
        parser.add_argument('--batch_size', type=int, default=512)
        parser.add_argument('--training_iterations', type=int, default=100)
        parser.add_argument('-M', '--M', type=int, default=40)
        parser.add_argument('--lr', type=float, default=0.009)
        parser.add_argument('--levels', type=int, default=1)
        parser.add_argument('--N_sim', type=int, default=2)
        parser.add_argument('--index', type=int, default=0)

        # SageMaker specific arguments
        parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
        parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
        parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

        args = parser.parse_args()

        logger.info("📊 Training parameters:")
        logger.info(f"  Index: {args.index}")
        logger.info(f"  Training iterations: {args.training_iterations}")
        logger.info(f"  Learning rate: {args.lr}")
        logger.info(f"  Batch size: {args.batch_size}")
        logger.info(f"  Train path: {args.train}")
        logger.info(f"  Test path: {args.test}")
        logger.info(f"  Model dir: {args.model_dir}")

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        floating_point = torch.float32
        logger.info(f"🖥️  Using device: {device}")

        index = args.index

        # Validate and load data
        logger.info(f"📁 Loading data for partition {index}")
        train_data, test_data = validate_data_files(args.train, args.test, index)
        
        X_train_np = train_data['X']
        y_train_nonstand = train_data['y']
        X_test_np = test_data['X']
        y_test_nonstand = test_data['y']

        logger.info("🔧 Preprocessing data...")
        # Standardise outputs
        y_train_reshape = y_train_nonstand.reshape(-1, 1)
        scaler = StandardScaler()
        scaler.fit(y_train_reshape)

        y_stand_np = scaler.transform(y_train_reshape)
        y_test_reshape = y_test_nonstand.reshape(-1, 1)
        y_test_stand = scaler.transform(y_test_reshape)

        # Convert data to torch tensors
        X_train = torch.tensor(X_train_np, dtype=floating_point).to(device)
        y_train = torch.tensor(y_stand_np, dtype=floating_point).squeeze().to(device)
        X_test = torch.tensor(X_test_np, dtype=floating_point).to(device)
        y_test = torch.tensor(y_test_stand, dtype=floating_point).squeeze().to(device)

        # Define the model dimensions
        N_train, D = X_train.shape
        logger.info(f"📐 Data dimensions: N_train={N_train}, D={D}")

        import warnings
        M = args.M  # Number of inducing points
        init_ip_method = 'kmeans++'

        logger.info(f"🎯 Initializing inducing points using {init_ip_method}")
        if init_ip_method == 'random':
            indices = np.random.choice(N_train, min(M, N_train), replace=False)
            inducing_points = X_train[indices, :].to(device)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kmeans = KMeans(n_clusters=M, init='k-means++', n_init=10, random_state=42)
                kmeans.fit(X_train_np)
                inducing_points = torch.tensor(kmeans.cluster_centers_,
                                               dtype=floating_point).to(device)

        logger.info("🏗️  Creating initial model...")
        # Create initial model
        likelihood = GaussianLikelihood(noise_constraint=GreaterThan(1e-5)).to(device)
        k = ScaleKernel(RQ(ard_num_dims=D, dtype=floating_point)).to(device)
        gp0 = SVGP(inducing_points, D, k)
        gp0.likelihood = likelihood
        gp0.likelihood.noise = torch.tensor(args.noise_variance,
                                            dtype=floating_point,
                                            device=device)
        # Move model to device (should be redundant, but safe)
        gp0 = gp0.to(device)

        # Define parameter limits
        limits = {
            'outputscale': [0.1, 10.0],
            'se_lengthscale': [0.1, 100.0],
            'rq_lengthscale': [0.08, 100.0],
            'rq_alpha': [0.1, 5],
            'per_period_length': [6, 10],
            'per_lengthscale': [0.1, 100.0],
            'lin_variance': [0.1*0.020, 0.020],
            'noise_variance': [0.01, 0.028]
        }

        logger.info(f"\nWhat is gp0: {type(gp0)}")
        logger.info("🤖 Starting automatic model construction...")
        # Automatic Model Construction
        auto_trainer = GPTraining(gp0, X_train, y_train, X_test, y_test)

        gp_gs = auto_trainer.auto_model_cons(
                            levels=args.levels,
                            N_sim=args.N_sim,
                            param_limits=limits,
                            mse_stop=0.003,
                            lr=args.lr,
                            training_iterations=args.training_iterations,
                            batch_size=args.batch_size)

        logger.info("🎛️  Preparing for fine-tuning...")

        """ Fine tuning """
        def generate_stds(lengthscales, base_std_dev):
            """Penalise lengthscales that are high using a greater std"""
            if not torch.is_tensor(lengthscales):
                lengthscales = torch.tensor(lengthscales)

            # Handle both scalar and ARD cases
            if lengthscales.numel() == 1:
                # Scalar lengthscale case
                return base_std_dev
            else:
                # ARD case - return list with correct length
                min_lengthscale = torch.min(lengthscales)
                std_devs = base_std_dev * torch.exp((lengthscales - min_lengthscale)/6)
                std_devs = torch.tensor([300 if std == torch.inf else std for std in std_devs])
                return std_devs.tolist()

        try:
            if hasattr(gp_gs.covar_module.base_kernel, 'lengthscale'):
                lengthscales = gp_gs.covar_module.base_kernel.lengthscale.squeeze()
                logger.info('Lengthscales in base kernel: %s', lengthscales)
            else:
                lengthscales = gp_gs.covar_module.base_kernel.kernels[0].lengthscale.squeeze()
                logger.info('Lengthscales in kernel[0]: %s', lengthscales)

            ls_stds = generate_stds(lengthscales, base_std_dev=1e-4)
            logger.info('Generated standard deviations: %s', ls_stds)

        except Exception as e:
            logger.warning(f"Could not extract lengthscales for tuning: {e}")
            ls_stds = [1e-4] * D

        if gp_gs is not gp0:  # Only tune if we have a model to tune
            logger.info("🔧 Starting hyperparameter tuning...")
            stds = {'outputscale': 1e-3,
                    'se_lengthscale': ls_stds,
                    'rq_lengthscale': ls_stds,
                    'per_lengthscale': ls_stds,
                    'rq_alpha': 1e-2,
                    'lin_variance': 1e-3,
                    'noise_variance': 1e-4}

            auto_tuner = GPTraining(gp_gs, X_train, y_train, X_test, y_test)
            auto_tuner.param_stds = stds

            try:
                tuned_gp = auto_tuner.tune(
                    gp_to_tune=gp_gs,
                    N_sim=5,
                    mse_stop=0.001,
                    lr=0.001,
                    training_iterations=200,
                    batch_size=256)

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

                mse_score = mean_squared_error(mu_tuned, y_test_nonstand)
                logger.info(f'✅ Final MSE (tuned): {mse_score:.6f}')

            except Exception as e:
                logger.error(f"❌ Error during tuning: {e}")
                traceback.print_exc()
                logger.info("Using non-tuned model for final predictions")
                tuned_gp = gp_gs
        else:
            logger.info("No tuning performed (using original model)")
            tuned_gp = gp_gs

        logger.info("💾 Saving models...")
        # Save models to the model directory (SageMaker will handle S3 upload)
        model_path = os.path.join(args.model_dir, f"expert{index}.pth")
        scaler_path = os.path.join(args.model_dir, f"scaler{index}.pth")

        # Ensure model directory exists
        os.makedirs(args.model_dir, exist_ok=True)

        torch.save(tuned_gp, model_path)
        torch.save(scaler, scaler_path)

        logger.info("✅ Models saved to:")
        logger.info(f"  - {model_path}")
        logger.info(f"  - {scaler_path}")

        logger.info("🎉 Training completed successfully!")

    except Exception as e:
        logger.error(f"❌ Fatal error in training script: {e}")
        traceback.print_exc()
        raise  # Re-raise to ensure SageMaker marks the job as failed