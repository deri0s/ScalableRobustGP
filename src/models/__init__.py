from .dgp import DistributedGP
from .dpgp import DirichletProcessGaussianProcess
from .dpsgp_torch import DirichletProcessSparseGaussianProcess
from .ddpgp import DistributedDPGP

__all__ = ["DistributedGP",
           "DirichletProcessGaussianProcess",
           "DirichletProcessSparseGaussianProcess",
           "DistributedDPGP"]