import torch
import gpytorch

kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

# Initialize scale (essential for ScaleKernel)
kernel.scale = torch.nn.Parameter(torch.tensor(2.0))

# Initialize base kernel parameters (RBF, Matern, etc.)
kernel.base_kernel.initialize(lengthscale=torch.tensor(0.5))  # Correct!

# If you have a simple kernel (not a scaled one):
# simple_kernel = gpytorch.kernels.RBFKernel()
# simple_kernel.initialize(lengthscale=torch.tensor(1.0))

def pretty_print_kernel(covar_module):
    """Prints a kernel formula with parameter values."""

    kernel_name = covar_module.__class__.__name__

    if isinstance(covar_module, gpytorch.kernels.ScaleKernel):
        if isinstance(covar_module.scale, torch.nn.Parameter): #Check if it is a nn.Parameter
          scale = covar_module.scale.item()
        else:
          scale = covar_module.scale # If it is not a nn.Parameter, it is a simple float.

        base_kernel = covar_module.base_kernel
        base_kernel_name = base_kernel.__class__.__name__

        if isinstance(base_kernel, gpytorch.kernels.RBFKernel):
            lengthscale = base_kernel.lengthscale.item()
            print(f"{scale:.2f} * exp(-0.5 * (x - x')^2 / {lengthscale:.2f}^2)")
        elif isinstance(base_kernel, gpytorch.kernels.MaternKernel):
          lengthscale = base_kernel.lengthscale.item()
          nu = base_kernel.nu
          print(f"{scale:.2f} * Matern({nu}, {lengthscale:.2f})") # Simplified representation
        # Add more cases for other kernels as needed (Linear, Periodic, etc.)
        else:
            print(f"{kernel_name} (Base Kernel: {base_kernel_name}) - Formula not implemented yet")

    elif isinstance(covar_module, gpytorch.kernels.RBFKernel): #Handle RBF directly
        lengthscale = covar_module.lengthscale.item()
        print(f"exp(-0.5 * (x - x')^2 / {lengthscale:.2f}^2)")
    elif isinstance(covar_module, gpytorch.kernels.MaternKernel):
        lengthscale = covar_module.lengthscale.item()
        nu = covar_module.nu
        print(f"Matern({nu}, {lengthscale:.2f})")
    elif isinstance(covar_module, gpytorch.kernels.LinearKernel):
        variance = covar_module.variance.item()
        print(f"{variance:.2f} * x * x'")
    elif isinstance(covar_module, gpytorch.kernels.PeriodicKernel):
      lengthscale = covar_module.lengthscale.item()
      period = covar_module.period.item()
      print(f"exp(-2 * sin^2(pi * (x - x') / {period:.2f}) / {lengthscale:.2f}^2)")
    # ... (Add more kernel types as needed)
    else:
        print(f"{kernel_name} - Formula not implemented yet")

pretty_print_kernel(kernel)

# Example with a sum of kernels (using the initialize method):
kernel_sum = gpytorch.kernels.RBFKernel() + gpytorch.kernels.LinearKernel()
kernel_sum.kernels[0].initialize(lengthscale=torch.tensor(1.0))
kernel_sum.kernels[1].initialize(variance=torch.tensor(0.5))
print('\n')
for k in kernel_sum.kernels:
  pretty_print_kernel(k) # Print each component