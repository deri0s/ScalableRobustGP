# ScalableRobustGP
Implementation of the Machine Learning model contributed in the paper

[Predicting product quality in continuous manufacturing processes using a scalable robust Gaussian Process approach](https://pdf.sciencedirectassets.com/271095/1-s2.0-S0952197623X00141/1-s2.0-S0952197623014173/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEDoaCXVzLWVhc3QtMSJHMEUCIHHFoXVN0H5gDGyu%2B3XV0SYoUb2fnJx6wI8GEZAuC4qOAiEA1G4TzbaE4TaLBopnzVV%2B5V2scYYUNIFAU4PK6EYmfpcqswUIMhAFGgwwNTkwMDM1NDY4NjUiDJnjDhIj1srtWMJsPSqQBeYHO1WDu9tpJX6zUH7PpGvk%2BmCsGCSgDVpwV8FBZ8Dsju5pK9bMrZXv5gF4IrEj0US0gJ9K814oT8qTGVGjAYAch8I8SSm4edVHDh71Vy5DDGrDCbceRBXZ%2BNb3eDHuZOcruM2fVhMhtYPOIbgh38tQYkDJu5ezsIZjArGT3c14Nz6x6fEBHezDyh12Xk2i1shT7gNwiqdjOTQBHbZDrpIMisDY1GWhIZpnZ5gZ%2BLqWQ%2FV8UMOLwRFFZWFxQ%2FcpsCvqcVAX9h65oHXTZ1HkE13BHUIbKeZg9sLa14u55c1Ps7puODrxIvz0u6hueb4ag14vKduMS%2FanV6E9IDKDqrv%2FU1frm97AG7adc2VjyKw%2FCb9iDifL1KQwn%2F8yIhByCjAzy1W4U3PqHSM8MibFb%2FwwlUuXnu7jmXauHt%2FaXJgA9O7QhqAVJgOelUwlzJ7o8T7Ka4s%2FrN1P4JT%2BzHlLj5ae3HNp8I5nWbtNHMVfCQ98rr%2Fhc3GaMR4bYXWOA7sFW8jJG5cEitv5ffRvRGiFGnOdK8WBdLzZx4JqT2Cqg8PrDyUFWwBPtMTi9cjpoJKStna23B5qdst2sIa93K4Io4E4ijZ8H4iYbiLPyrutXaJClpyMQUAWb8QAwMBXNmPOv5Vdjqg9acAzgXgd2L%2FYP2Q0es10s0ZUAOeMU3nE92lkpfrtVvdmCOa6WMgsT%2Bp5z31Y1GWVhCcVd%2FPI668x3NDI1c8P9RvGFc2qAVi%2F5UIb3zZnltjl4O%2BjjSGSeXHkrQZRRDRx18Y5Yo2iXJTP3ogyUNnM21HiRokys%2BD9HLpSsFdOvY9RgqL1UBh6EvQjQfEVaHPrGsSUtM1jkvI1ctX7jYBWPffjyUNf5G6XfcmAMJ%2BciK8GOrEBSOnjJ4CEuT4f95TniLxz8urNyaNvYk6ZU2LU0AIzWwNxRYs80Ctraeew5EnltYQOffdWDfVa8%2Bs9RNDKcqJNoC9gmRa0owJSEdkzaYTR6XlLBWu%2Fp07SVNBDhH5%2BaHcozumqdvkr%2FFQcpFvflH1PVR82360Velm3mjFPHIQ2QAkK%2F%2FNspATfJ5VWQ18T%2FIERSocqJwhdTHRoloT9eFYnGzHh062EtZtS7XTs1aU3zErZ&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240301T172212Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY52V6WBOF%2F20240301%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=cc997cab55c014e046b4025c07e5d1454c7946999f9b88c06bc568a02672c88c&hash=f8149a2ee2f540b33a54018eda25547590cd005a28ff3b1edd5d508d2591c780&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0952197623014173&tid=spdf-197986cb-b02b-4563-b3f5-9521cd107bbd&sid=e775532e4af775468119e0802ac9152cd7bagxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=1d005a510b0107500304&rr=85dad3ec9aa47726&cc=gb)

The model uses a Bayesian Nonparametric approach based on the iterative application of Dirichlet Process (DP) clustering and Gaussian Process (GP) regression to address the two main issues in GP regression:

1) Data corrupted with noise generated from a non-Gaussian distribution.
2) The cubic complexity training time $\mathcal{O}(N^3)$.


The model automatically ignores data generated by independent noise sources when estimating the latent function as it uses an infinite mixture of Gaussian distributions to describe the measurement noise. In addition, it uses a distributed approach based on the product of robust GP experts to alleviate the computation memory and time effort.

# Development
The current version was developed on top of the `scikit-learn` packages `gaussian_process` and `mixture.BayesianGaussianMixture`.

Distributed computations were implemented using the Generalised Product of GPs detailed in the following [paper](https://proceedings.mlr.press/v37/deisenroth15.pdf).


# Example using synthetic data

Let us import some dependencies and simulate a toy dataset $\mathcal{D}$.

```python
import numpy as np
import matplotlib.pyplot as plt
from models.DGP import DistributedGP as DGP
from models.DDPGP import DistributedDPGP as DDPGP

# Generate always the same random numbers
from numpy.random import RandomState
prob = RandomState(123)

# Define the domain of the function and the number of observations
N = 2000
x = np.linspace(0, 10, N)

# Corrupt samples with noise generated from 2 independent sources
sine = lambda x: np.sin(x)
f = sine(x)
y = np.zeros(N)

# Noise structure
std1 = 0.1
std2 = 1
pi1 = 0.5
pi2 = 1 - pi1

# proportionalities vector
u = prob.uniform(0,1,N)

for i in range(N):
    if u[i] < pi1:
        y[i] = f[i] + std1*prob.randn()
    else:
        y[i] = f[i] + std2*prob.randn()

# define the covariance function
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

se = 1**2 * RBF(length_scale=0.5, length_scale_bounds=(0.07, 0.9))
wn = WhiteKernel(noise_level=0.5**2, noise_level_bounds=(1e-6,0.7))

kernel = se + wn

# training
N_GPs = 5
model = DDPGP(x, y, N_GPs=N_GPs, init_K=7, kernel=kernel, normalise_y=True)
model.train()

# predictions
mu, std, betas = model.predict(x)
```

The following figure shows the regression performance of the standard Distributed GP (red line) and the proposed DDPGP (green line) when estimating the function, $f(x) = sin(x)$, using data that has been corrupted by two independent sources of noise. The Distributed GP (DGP) is affected by the data that has been corrupted with a noise source whose variance is "large". On the other hand, the DDPGP predictions successfully ignoreed observations corrupted with large noise levels.

<div align="right">Zoomed view</div>
<div class="row">
    <img src="figures/sine/sine_DDPGP.png" alt="Snow" style="width:49%">
    <img src="figures/sine/sine_zoom.png" alt="Forest" style="width:49%">
</div>

# Real applications
## Continous manufacturing
We show the capability of the proposed model to estimate relationships using data obtained from industrial processes. The folder `real applications` contains a case study in continuous manufacturing where we estimate the relationship between the furnace parameters and product quality from measurements generated by noise and non-furnace processes.


### Process identification - Signal Processing - Anomaly Detection
The following image shows the raw fault density data (used to determine product quality) extracted from scanners located at the end of the production line. Notice that the data shows high peaks, noise bursts, and slow and rapid trends.

#### The Challenge:
Identify the relationship between the furnace parameters and product quality (fault density). The model inputs are obtained from sensors located at the furnace. The model targets are obtained from the scanners readings.

One way to address such a problem is to use process information to develop pre-processing pipelines to condition the fault density data and then, use a Machine Learning approach to estimate the latent furnace process.

Alternatiely, in the following case study, we test the DDPGP <b>process identification</b> capabilities to <b>automatically</b> ignore data that has been generated by non-furnace processes. The results are shown in the figure below.

![process identification](/figures/NSG/NSG_clustering.png?raw=true "Employee Data title")

We speficically selected the region between Day 0 and Day 14 as manufacturing experts have identify that the fault density increasees located at Day 6 and Day 10 were exclusively generated by the furnace process.

## Product Quality Forecast
The DDPGP model can be used as a decision tool as it has the following characteristics:

- <b>Modularity</b>: New experts trained with data corresponding to a new product can be added without affecting the models structure.
- <b>Uncertainty Quantification:</b> Quantifies the confidence of its preditions using the estiated (furnace) process variance as confidence bounds.
- <b>New data alerts:</b> Alerts when it is operating on data whose features are considerably different from the training data. On such scenarios the model predictions must not be use to influence decisions.

We now test the DDPGP capability to generalise outside training data.

The following video shows a zoomed in view of the raw furnace faults (grey line), conditioned furnace faults (blue line), the DDPGP predictions (red line) and its confidence bounds (yellow area).

<div align="center">
    <img src="figures/NSG/forecast.gif" alt="Snow" style="width:90%">
</div>

Notice that the conditioned furnace faults were obtained using large pre-processing pipelines developed using manufacturing experts knowledge. The processed data, shown with the blue line, is just another estimate of what we believe the number of faults generated by the furnace looks like, and by no means it should be treated as the furnace process ground truth.

In addition, the model alerts the user when is it at a region where the data is different from the one observed in the training stage by eliminating the confidence bounds. In other words, when there is no yellow area, then the user must not use the model's predictions for decision making.


## Application
Copy the `models` folder into your working directory and follow the steps detailed in Example. 

## Issues
If you find an error in your application, please raise it as an issue on the github page of the mian repository.

## Testing
We have developed some tests (`tests` folder) to validate the model on different scenarios. To run the test, first make sure you have `pytest` installed:

`python -m pip install pytest`

Then, in the command window, change to the Machine_Learning directory and simply type `pytest` in the command window. Note that these tests will be also be run automatically when somebody submits a pull request to the master branch.

## Next steps
Apply the DDPGP framework using the GPJax libraries to implement Sparse Robust GP experts.