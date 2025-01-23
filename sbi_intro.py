import numpy as np
import matplotlib.pyplot as plt

import torch
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer

num_dim = 3
prior = utils.BoxUniform(low=-2 * torch.ones(num_dim), high=2 * torch.ones(num_dim))

def simulator(parameter_set):
    return 1.0 + parameter_set + torch.randn(parameter_set.shape) * 0.1

posterior = infer(simulator, prior, method="SNPE", num_simulations=1000)

observation = torch.zeros(3)
samples = posterior.sample((10000,), x=observation)
log_probability = posterior.log_prob(samples, x=observation)
print(log_probability)
_ = analysis.pairplot(samples, limits=[[-2, 2], [-2, 2], [-2, 2]], figsize=(6, 6))
#_ = analysis.pairplot(samples, limits=[[-25, -2], [-26, -2], [-27, -2]], figsize=(6, 6))
plt.show()

