import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

# sbi
import sbi.utils as utils
from sbi.inference.base import infer
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn
from sbi.analysis import pairplot
from torch.utils.tensorboard import SummaryWriter

from scipy.signal import tukey
from scipy.interpolate import interp1d
from scipy.linalg import toeplitz, inv

import lal

from FDnoise import noise_from_PSD

#Define the prior:
# M, chi, C220, P220, C221, P221
prior_min = [50., 0., 0., 0., 0., 0.]
prior_max = [100., 0.9999, 5., 2*np.pi, 5., 2*np.pi]
prior = utils.torchutils.BoxUniform(
    low=torch.as_tensor(prior_min), 
    high=torch.as_tensor(prior_max),
    device='cuda'
)

#Define the simulator:

psdfile = './PSD_H1_1126259447_32_2.0_4096.0.txt'
ligo_psd  = np.genfromtxt(psdfile, delimiter='')
psd = interp1d(ligo_psd[:,0], ligo_psd[:,1], fill_value='extrapolate', bounds_error=False)

berti_fit_data = np.genfromtxt('./l2/n1l2m2.dat')
omega_r = interp1d(berti_fit_data[:,0], berti_fit_data[:,1])
omega_i = interp1d(berti_fit_data[:,0], berti_fit_data[:,2])
berti_fit_data_OT = np.genfromtxt('./l2/n2l2m2.dat')
omega_OT_r = interp1d(berti_fit_data_OT[:,0], berti_fit_data_OT[:,1])
omega_OT_i = interp1d(berti_fit_data_OT[:,0], berti_fit_data_OT[:,2])

tgps = lal.LIGOTimeGPS(1126259466.4085458)
gmst = lal.GreenwichMeanSiderealTime(tgps)
Fp, Fc = lal.ComputeDetAMResponse(lal.CachedDetectors[lal.LALDetectorIndexLHODIFF].response, 1.95, -1.27, 0.82, gmst)

def wvf_simp(t, f, gamma, A, fp, fc):

    h_tmp = np.array(1/2. * np.sqrt(5./np.pi) * np.conjugate(A) * np.exp(-gamma*t) * np.exp(-1j*2*np.pi*f*t), dtype=complex)
    h_p = np.real(h_tmp)
    h_c = np.imag(h_tmp)
    return fp*h_p + fc*h_c

def hphc(params, trigtime=1):
    
    M, chi, C220, P220, C221, P221 = np.asarray(params.cpu())
    
    prefactor_omega_r = (lal.C_SI*lal.C_SI*lal.C_SI/(2.*np.pi*lal.G_SI*M*lal.MSUN_SI))
    prefactor_omega_i = (lal.C_SI*lal.C_SI*lal.C_SI/(lal.G_SI*M*lal.MSUN_SI))
    
    A220 = C220*np.exp(1j*P220)*reference_amplitude
    A221 = C221*np.exp(1j*P221)*reference_amplitude
    
    A = np.array([A220,A221])
    
    f = np.array([prefactor_omega_r*omega_r(chi),prefactor_omega_r*omega_OT_r(chi)])
    gamma = np.array([-prefactor_omega_i*omega_i(chi),-prefactor_omega_i*omega_OT_i(chi)])
    
    T = 2
    srate = 4096
    dt = 1/srate
    
    rawstrains = np.zeros((num_noise_rel_x_theta, len(noise_from_PSD(psdfile, srate=4096, T=2, fmin=20, fmax=4096))))
    for i in range(num_noise_rel_x_theta):
        rawstrains[i] = noise_from_PSD(psdfile, srate=4096, T=2, fmin=20, fmax=4096)
    
    times = np.arange(0.0, T, dt)
    
    idx = np.argwhere(times>=trigtime)[0,0]
    
    times = times[idx:] - trigtime
    rawstrains = rawstrains[:,idx:]
    
    waveform = np.zeros(len(times))
    for j in range(nmode):
        waveform += wvf_simp(times, f[j], gamma[j], A[j], Fp, Fc)
        
    rawstrains += waveform

    idx_en = np.argwhere(times>0.1)[0,0]
    
    window_len = len(times[:idx_en])
    dwindow = tukey(window_len, alpha=1/70)
    
    dt = np.diff(times[:idx_en])[0]
    freq = np.fft.rfftfreq(len(times[:idx_en]),d = dt)
    
    ssfs = np.zeros((num_noise_rel_x_theta, len(freq)), dtype = np.complex64)
    for i in range(num_noise_rel_x_theta):
        ssfs[i] = np.fft.rfft(rawstrains[i,:idx_en]*dwindow)*dt / np.sqrt(psd(freq))
    
    mm = np.where((freq >= 100) & (freq <=650))[0][0]
    MM = np.where((freq >= 100) & (freq <=650))[0][-1]
    
    return np.concatenate((np.real(ssfs[:,mm-1:MM+1]), np.imag(ssfs[:,mm-1:MM+1])),axis=1)

reference_amplitude = 1e-20
nmode = 2

num_noise_rel_x_theta = 100
num_sim = 1000

#SBI

x_o = torch.load('x_H1_SNRx01.pt').cuda()

def three_sigma_interval(dim, post_samples, new_prior_min, new_prior_max):

    hist, bins = np.histogram(post_samples[:,dim].cpu(), bins=1000, range=(new_prior_min[dim],new_prior_max[dim]))

    bins_del_last=np.delete(bins,-1)

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
    
    for dim in range(6):
        threesig = three_sigma_interval(dim, posterior_samples, new_prior_min, new_prior_max)
        new_min.append(threesig[0])
        new_max.append(threesig[1])

    return utils.torchutils.BoxUniform(low=torch.as_tensor(new_min), high=torch.as_tensor(new_max), device='cuda') , new_min , new_max

posteriors = []
priors_min = []
priors_max = []
new_prior_min = prior_min
new_prior_max = prior_max
proposal = prior

cond=1
trcount=1

neural_posterior = utils.posterior_nn(model="maf", hidden_features=120)

while(cond):
    
    priors_min.append(new_prior_min)
    priors_max.append(new_prior_max)
    
    simulator, prior = prepare_for_sbi(hphc, proposal)
    
    inference = SNPE(prior=prior, density_estimator=neural_posterior, device='cuda', summary_writer=SummaryWriter("./runs_SNRx01_ROB/run_{i}".format(i=trcount)))
    
    trcount=trcount+1
    
    theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=num_sim, num_workers=50)
    
    ##########
    
    theta = torch.repeat_interleave(theta, torch.tensor(num_noise_rel_x_theta).repeat(num_sim), dim=0)
    x = torch.reshape(x, (num_noise_rel_x_theta*num_sim, 112))
    shufford = torch.randperm(num_noise_rel_x_theta*num_sim)
    theta = theta[shufford]
    x = x[shufford]
    
    ##########
    
    inference = inference.append_simulations(theta, x)
    
    density_estimator = inference.train(stop_after_epochs=20, show_train_summary=True)
    
    posterior = inference.build_posterior(density_estimator)

    posteriors.append(posterior)
    
    posterior_samples = posterior.sample((100000,), x=x_o)
    proposal , new_prior_min , new_prior_max = new_priors()
    
    cond = 1 if np.any((np.asarray(new_prior_max) - np.asarray(new_prior_min))/(np.asarray(priors_max[-1]) - np.asarray(priors_min[-1])) < 0.5) else 0

torch.save(posteriors, 'posterior-OT-MRI-HM-SNRx01-ROB.pt')

pp = priors_min , priors_max
np.save('priors-SNRx01-ROB.npy', pp)

print(posterior)
