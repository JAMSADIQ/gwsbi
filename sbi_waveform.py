import numpy as np
import matplotlib.pyplot as plt
from pycbc import types, fft, waveform

import torch
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi

num_dim = 3
# Mtot, q, switch 
prior = utils.BoxUniform(low=[10, 0.8, 0], high=[80, 1, 1])

#def simulator(parameter_set):

def hp(parameter_set):
    M, q, switch =  np.asarray(parameter_set.cpu())
    m1 = M*q/(1+q)
    m2 = M/(1+q)
    hp, hc = waveform.get_td_waveform(approximant='IMRPhenomD', mass1=m1, mass2=m2, delta_t=1.0/4096, f_lower=40)
    Tvals = hp.sample_times
    start_Time, end_Time = np.searchsorted(Tvals, (-1, 0.01)) 
    hp_cut = hp[start_Time : end_Time]
    hc_cut = hc[start_Time : end_Time]
    T_cut = Tvals[start_Time : end_Time]
    return hp_cut 

simulator, prior = prepare_for_sbi(hp, prior)
posterior = infer(simulator, prior, method="SNPE", num_simulations=1)
#posterior = infer(simulator, prior, method="SNLE", num_simulations=10)

print(posterior)
observation = torch.zeros(3)
samples = posterior.sample((10,), x=observation)
log_probability = posterior.log_prob(samples, x=observation)
print(log_probability)
_ = analysis.pairplot(samples, limits=[[-2, 2], [-2, 2], [-2, 2]], figsize=(6, 6))
#_ = analysis.pairplot(samples, limits=[[-25, -2], [-26, -2], [-27, -2]], figsize=(6, 6))
plt.show()

