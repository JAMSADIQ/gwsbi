import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.mlab as mlab

# sbi
import sbi.utils as utils
from sbi.inference.base import infer
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn
from sbi.analysis import pairplot
from torch.utils.tensorboard import SummaryWriter
#from sbi import analysis as analysis

#### 
from scipy.signal import tukey
from scipy.interpolate import interp1d
from scipy.linalg import toeplitz, inv
import lal

from pycbc import types, fft, waveform


#### can use pycbc?
from FDnoise import noise_from_PSD


#Define the prior: here we want 3 paramters as input to generate waveforms

#M,  chi, C220, P220, C221, P221   #assumin chi=Xeff C and P? [0-5] and [0-2pi] 

#params m1, m2, fmin, switch
prior_min = [20.0,  40.0, 0.0]
prior_max = [100.0, 100., 1.0]


prior = utils.torchutils.BoxUniform(
    low=torch.as_tensor(prior_min), 
    high=torch.as_tensor(prior_max),
    device='cpu'
)



#time of merger?
tgps = lal.LIGOTimeGPS(1126259466.4085458)
gmst = lal.GreenwichMeanSiderealTime(tgps)

#antenna pattern
Fp, Fc = lal.ComputeDetAMResponse(lal.CachedDetectors[lal.LALDetectorIndexLHODIFF].response, 1.95, -1.27, 0.82, gmst)

###  
def wvf_simp(h_p, h_c, fp, fc):
    """
    related to detector?
    """
    return fp*h_p + fc*h_c



def hphc(params, trigtime=1):
    M, fmin, switch  = np.asarray(params.cpu()) 
    hp, hc = waveform.get_td_waveform(approximant='SEOBNRv4',mass1=M, mass2=M, delta_t=1.0/4096, f_lower=60)
    lenh = len(hp)
    if switch <=0.6:
        swtich = 0.0
        hp = np.zeros(lenh)
        hc = np.zeros(lenh)
    else:
        switch = 1.0

    return  wvf_simp(hp, hc, Fp, Fc)#, wvf_simp(hp, hc, Fp, Fc)))




num_noise_rel_x_theta = 50#100
num_sim = 50 #100#1000

#SBI
# Given a observation x_o, we can then sample from the posterior p(θ|x), evaluate its log-probability, or plot it.
#observation
htp, htc = waveform.get_td_waveform(approximant='SEOBNRv4',mass1=20, mass2=20, delta_t=1.0/4096, f_lower=50)
x_o = wvf_simp(htp, htc, Fp, Fc)#torch.load('x_H1_SNRx01.pt').cpu()



def three_sigma_interval(dim, post_samples, new_prior_min, new_prior_max):  
    hist, bins = np.histogram(post_samples[:, dim].cpu(), bins=1000, range=(new_prior_min[dim],new_prior_max[dim]))
    bins_del_last=np.delete(bins,-1) #remove final val
    hist_array = np.stack([hist, bins_del_last], axis=-1)
    hist_array_sort=hist_array[hist_array[:, 0].argsort()]
    num_post_sample = len(post_samples[:,dim])
    s=0
    j=0
    while s < 0.9999:
        s += hist_array_sort[-1-j,0]/num_post_sample
        j += 1
    hist_array_sort_del=np.delete(hist_array_sort,np.s_[0:1000-j],0)
    hist_array_del=hist_array_sort_del[hist_array_sort_del[:, 1].argsort()]
    sig_min=hist_array_del[0,1]
    sig_max=hist_array_del[-1,1]+(new_prior_max[dim]-new_prior_min[dim])/1000
    return sig_min, sig_max


def new_priors():    
    new_min = []
    new_max = []
    for dim in range(3):
        threesig = three_sigma_interval(dim, posterior_samples, new_prior_min, new_prior_max)
        new_min.append(threesig[0])
        new_max.append(threesig[1])
    return utils.torchutils.BoxUniform(low=torch.as_tensor(new_min), high=torch.as_tensor(new_max), device='cpu') , new_min , new_max


posteriors = []
priors_min = []
priors_max = []
new_prior_min = prior_min
new_prior_max = prior_max
proposal = prior
cond=1
trcount=1



neural_posterior = utils.posterior_nn(model="maf", hidden_features=80)
#Returns a function that builds a density estimator for learning the posterior (for SNPE model can be "made" "mdn" "nsf" as well)

#here we start
while(cond):    
    priors_min.append(new_prior_min)
    priors_max.append(new_prior_max)
    #proposal is prior limits  hphc will generate simlautions?
    simulator, prior = prepare_for_sbi(hphc, proposal)
    
    inference = SNPE(prior=prior, density_estimator=neural_posterior, device='cpu', summary_writer=SummaryWriter("./runs_SNRx01_ROB/run_{i}".format(i=trcount)))
    print(prior)
    trcount=trcount+1
    print("trcount = ", trcount) 
    theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=num_sim, num_workers=1)
    
    ##########
    
    theta = torch.repeat_interleave(theta, torch.tensor(num_noise_rel_x_theta).repeat(num_sim), dim=0)
    x = torch.reshape(x, (num_noise_rel_x_theta*num_sim, 112))
    shufford = torch.randperm(num_noise_rel_x_theta*num_sim)
    theta = theta[shufford]
    x = x[shufford]
    
    ##########
    
    inference = inference.append_simulations(theta, x)
    ### this is the PDF from simulations?
    density_estimator = inference.train(stop_after_epochs=10, show_train_summary=True)
    posterior = inference.build_posterior(density_estimator)
    posteriors.append(posterior)
    #observation is given in x_0
    #posterior_samples = posterior.sample((100000,), x=x_o)
    posterior_samples = posterior.sample((1000, ), x=x_o)
    log_probability = posterior.log_prob(posterior_samples, x=x_o)
    #_=
    #pairplot(posterior_samples, limits=[[20,70], [15, 70], [40, 4096], [0, 1]], figsize=(8, 8))
    #plt.show()    

    proposal , new_prior_min , new_prior_max = new_priors()
    
    cond = 1 if np.any((np.asarray(new_prior_max) - np.asarray(new_prior_min))/(np.asarray(priors_max[-1]) - np.asarray(priors_min[-1])) < 0.5) else 0

torch.save(posteriors, 'posterior-OT-MRI-HM-SNRx01-ROB.pt')
pp = priors_min , priors_max
np.save('priors-SNRx01-ROB.npy', pp)
print(posterior)
