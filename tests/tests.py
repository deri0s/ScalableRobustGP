import numpy as np
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from models.DPGP import DirichletProcessGaussianProcess as DPGP


# Generate 'noise' to corrupt signals
np.random.seed(42)
noise = np.random.randn(100)

# Generate always the same random numbers
from numpy.random import RandomState
prob = RandomState(123)

# general sine function
sine = lambda x: np.sin(x)

# general kernel
se = 1**2 * RBF(length_scale=0.5, length_scale_bounds=(0.07, 0.9))
wn = WhiteKernel(noise_level=0.5**2, noise_level_bounds=(1e-6,0.7))

kernel = se + wn

def test_DPGP_1D():

    # Define the domain of the function and the number of observations
    N = 200
    x = np.linspace(0, 10, N)

    # Corrupt samples with noise generated from 2 independent sources
    f = sine(x)
    y = np.zeros(N)

    # Noise structure
    std1 = 0.1
    std2 = 1.5
    pi1 = 0.5
    pi2 = 1 - pi1

    # proportionalities vector
    u = prob.uniform(0,1,N)

    for i in range(N):
        if u[i] < pi1:
            y[i] = f[i] + std1*prob.randn()
        else:
            y[i] = f[i] + std2*prob.randn()

    # training
    N_GPs = 5
    model = DPGP(x, y, init_K=5, kernel=kernel, normalise_y=True)
    model.train(pseudo_sparse=True)

    # predictions
    mu, std = model.predict(x)

    # Average mean prediction should be within 0.1 of true
    # (element wise)
    assert np.allclose(mu, np.vstack(f), atol=0.1)

    # Noise std should be within 0.02 of true
    std = np.sqrt(np.exp(model.kernel_.theta))
    assert np.allclose(model.K == 2,
                       'Estimated K different from original')
    assert np.allclose(model.stds[0], std1, atol=0.02)
    assert np.allclose(model.stds[1], std2, atol=0.02)
    assert np.allclose(model.pies[0], pi1, atol=0.02)
    assert np.allclose(model.pies[1], pi2, atol=0.02)


# Create 2D training inputs
N = 100
x_range = np.linspace(0, 10, 10)
X = np.zeros([N, 2])
n = 0
for i in range(10):
    for j in range(10):
        x = np.array([x_range[i], x_range[j]])
        X[n, :] = x
        n += 1

# Create 2D testing inputs
N_star = 400
x_star_range = np.linspace(0, 10, 20)
X_star = np.zeros([N_star, 2])
n = 0
for i in range(20):
    for j in range(20):
        x = np.array([x_star_range[i], x_star_range[j]])
        X_star[n, :] = x
        n += 1


def test_standard_2D_GP():
    """ Test our standard GP on some data with 2D inputs

    """

    # True 2D model
    def f(x):
        return np.sin(x[0]) + np.sin(x[1])
    
    f = f(X)
    
    # Noise structure
    std1 = 0.1
    std2 = 1.5
    pi1 = 0.5
    pi2 = 1 - pi1


    # proportionalities vector
    u = prob.uniform(0,1,N)

    y = np.zeros(np.shape(f))

    for i in range(N):
        if u[i] < pi1:
            y[i] = f[i] + std1*prob.randn()
        else:
            y[i] = f[i] + std2*prob.randn()

    # training
    N_GPs = 5
    model = DPGP(x, y, init_K=5, kernel=kernel, normalise_y=True)
    model.train(pseudo_sparse=True)

    # predictions
    mu, std = model.predict(x)

    # Average mean prediction should be within 0.1 of true
    # (element wise)
    assert np.allclose(mu, np.vstack(f), atol=0.1)

    # Noise std should be within 0.02 of true
    std = np.sqrt(np.exp(model.kernel_.theta))
    assert np.allclose(model.K == 2,
                       'Estimated K different from original')
    assert np.allclose(model.stds[0], std1, atol=0.02)
    assert np.allclose(model.stds[1], std2, atol=0.02)
    assert np.allclose(model.pies[0], pi1, atol=0.02)
    assert np.allclose(model.pies[1], pi2, atol=0.02)