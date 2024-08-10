from .dgp import DistributedGP
from .dpgp import DirichletProcessGaussianProcess
from .dpsgp_gpytorch import DirichletProcessSparseGaussianProcess
from .ddpgp import DistributedDPGP

__all__ = ["DistributedGP",
           "DirichletProcessGaussianProcess",
           "DirichletProcessSparseGaussianProcess",
           "DistributedDPGP"]