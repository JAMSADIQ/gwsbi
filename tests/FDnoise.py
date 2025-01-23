#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.linalg import toeplitz
import matplotlib.mlab as mlab
from scipy.signal import tukey
from scipy.interpolate import interp1d


# In[11]:


# psd_H1 = np.genfromtxt('../data/psd-check-files/PSD_H1_1126259447_32_2.0_4096.0.txt', delimiter='')
# psd_L1 = np.genfromtxt('../data/psd-check-files/PSD_L1_1126259447_32_2.0_4096.0.txt', delimiter='')


# # In[3]:


# psd_H1 = interp1d(psd_H1[:,0], psd_H1[:,1], fill_value='extrapolate', bounds_error=False)
# psd_L1 = interp1d(psd_L1[:,0], psd_L1[:,1], fill_value='extrapolate', bounds_error=False)


# # In[4]:


# srate = 4096
# T = 32
# dt = 1/srate
# N_points = int(T*srate)
# freqs_cgn = np.fft.rfftfreq(N_points, d = dt)
# df_cgn    = np.diff(freqs_cgn)[0]


# In[7]:


def noise_from_PSD(psd, srate=4096, T=32, fmin=40.0, fmax=4096):
    
    #psd = np.genfromtxt(psdfile, delimiter='')
    
    #psd = interp1d(psd[:,0], psd[:,1], fill_value='extrapolate', bounds_error=False)
    
    dt = 1/srate
    N_points = int(T*srate)
    freqs_cgn = np.fft.rfftfreq(N_points, d = dt)
    df_cgn    = np.diff(freqs_cgn)[0]
    
    f_min = np.max([fmin, freqs_cgn.min()])
    f_max = np.min([fmax, freqs_cgn.max()])

    kmin = int(f_min/df_cgn)
    kmax = int(f_max/df_cgn)

    frequencies      = df_cgn*np.arange(0,N_points/2.+1)
    frequency_strain = np.zeros(len(frequencies), dtype = np.complex64)

    for i in range(kmin, kmax+1):
        sigma_cgn = 0.5*np.sqrt(psd(frequencies[i])/df_cgn) ## from pyRing
        #sigma_cgn = np.sqrt(0.5*psd(frequencies[i])/df_cgn)
        frequency_strain[i] = np.random.normal(0.0,sigma_cgn)+1j*np.random.normal(0.0,sigma_cgn) ## from pyRing
        #frequency_strain[i] = sigma_cgn*np.exp(1j*np.random.rand()*2*np.pi)
    
    rawstrain = np.fft.irfft(frequency_strain)/dt
    
    return rawstrain


def noise_from_ASD(asdfile, srate=4096, T=32, fmin=11.0, fmax=4096):
    
    asd = np.genfromtxt(asdfile, delimiter='')
    
    psd = interp1d(asd[:,0], asd[:,1]**2, fill_value='extrapolate', bounds_error=False)
    
    dt = 1/srate
    N_points = int(T*srate)
    freqs_cgn = np.fft.rfftfreq(N_points, d = dt)
    df_cgn    = np.diff(freqs_cgn)[0]
    
    f_min = np.max([fmin, freqs_cgn.min()])
    f_max = np.min([fmax, freqs_cgn.max()])

    kmin = np.int(f_min/df_cgn)
    kmax = np.int(f_max/df_cgn)

    frequencies      = df_cgn*np.arange(0,N_points/2.+1)
    frequency_strain = np.zeros(len(frequencies), dtype = np.complex64)

    for i in range(kmin, kmax+1):

    #     sigma_cgn = 0.5*np.sqrt(psd_H1(frequencies[i])/df_cgn) ## from pyRing
    #     The extra factor of 1/2 in the variance comes from the fact that we need to 
    #     sample a complex random variable, where the sigma^2_real = sigma^2_img = 0.5*sigma^2.
    #     See https://dsp.stackexchange.com/questions/40306/what-would-be-the-variance-for-complex-number for details.

        sigma_cgn = np.sqrt(0.5*psd(frequencies[i])/df_cgn)
    #     frequency_strain[i] = np.random.normal(0.0,sigma_cgn)+1j*np.random.normal(0.0,sigma_cgn) ## from pyRing
        frequency_strain[i] = sigma_cgn*np.exp(1j*np.random.rand()*2*np.pi)
    
    rawstrain = np.real(np.fft.irfft(frequency_strain))*df_cgn*N_points
    
    return rawstrain


# In[8]:


# rawst = noise_from_PSD('../data/psd-check-files/PSD_H1_1126259447_32_2.0_4096.0.txt')
# plt.plot(rawst)


# # In[9]:


# noise_chunk_size = 2.0
# alpha_window = 0.1
# noise_seglen      = np.int(4096*noise_chunk_size)
# psd_window        = tukey(noise_seglen, alpha_window)
# psd_welch, freqs_welch = mlab.psd(rawst,
#                                       Fs     = 4096,
#                                       NFFT   = noise_seglen,
#                                       window = psd_window,
#                                       sides  = 'onesided')


# # In[12]:


# plt.loglog(freqs_welch, psd_welch, '--')
# plt.loglog(psd_H1[:,0], psd_H1[:,1])
# # plt.loglog(freqs_welch, psd_H1(freqs_welch), '--')
# plt.show()


# # In[5]:


# f_min = np.max([11.0, freqs_cgn.min()])
# f_max = np.min([4096., freqs_cgn.max()])

# kmin = np.int(f_min/df_cgn)
# kmax = np.int(f_max/df_cgn)

# frequencies      = df_cgn*np.arange(0,N_points/2.+1)
# frequency_strain = np.zeros(len(frequencies), dtype = np.complex64)

# for i in range(kmin, kmax+1):

# #     sigma_cgn = 0.5*np.sqrt(psd_H1(frequencies[i])/df_cgn) ## from pyRing
# #     The extra factor of 1/2 in the variance comes from the fact that we need to 
# #     sample a complex random variable, where the sigma^2_real = sigma^2_img = 0.5*sigma^2.
# #     See https://dsp.stackexchange.com/questions/40306/what-would-be-the-variance-for-complex-number for details.

#     sigma_cgn = np.sqrt(0.5*psd_H1(frequencies[i])/df_cgn)
# #     frequency_strain[i] = np.random.normal(0.0,sigma_cgn)+1j*np.random.normal(0.0,sigma_cgn) ## from pyRing
#     frequency_strain[i] = sigma_cgn*np.exp(1j*np.random.rand()*2*np.pi)
    

# rawstrain = np.real(np.fft.irfft(frequency_strain))*df_cgn*N_points


# # In[6]:


# plt.plot(rawstrain)
# plt.show()


# # In[7]:


# noise_chunk_size = 2.0
# alpha_window = 0.1
# noise_seglen      = np.int(srate*noise_chunk_size)
# psd_window        = tukey(noise_seglen, alpha_window)
# psd_welch, freqs_welch = mlab.psd(rawstrain,
#                                       Fs     = 4096,
#                                       NFFT   = noise_seglen,
#                                       window = psd_window,
#                                       sides  = 'onesided')


# # In[8]:


# plt.loglog(freqs_welch, psd_welch, '--')
# plt.loglog(freqs_welch, psd_H1(freqs_welch), '--')
# plt.show()


# # In[9]:


# psd_ET = np.genfromtxt('curves_Jan_2020/et_d.txt', delimiter='')


# # In[10]:


# plt.loglog(psd_ET[:,0], psd_ET[:,1]**2)
# plt.loglog(frequencies, psd_H1(frequencies))
# plt.show()


# # In[11]:


# psd_ET = interp1d(psd_ET[:,0], psd_ET[:,1]**2, fill_value='extrapolate', bounds_error=False)


# # In[12]:


# f_min = np.max([2.0, freqs_cgn.min()])
# f_max = np.min([4096., freqs_cgn.max()])

# kmin = np.int(f_min/df_cgn)
# kmax = np.int(f_max/df_cgn)

# frequencies      = df_cgn*np.arange(0,N_points/2.+1)
# frequency_strain = np.zeros(len(frequencies), dtype = np.complex64)

# for i in range(kmin, kmax+1):
#     sigma_cgn = np.sqrt(0.5 * psd_ET(frequencies[i])/df_cgn)
#     frequency_strain[i] = sigma_cgn*np.exp(1j*2*np.pi*np.random.rand())

# rawstrain_ET = np.real(np.fft.irfft(frequency_strain))*df_cgn*N_points


# # In[13]:


# # plt.plot(rawstrain)
# plt.plot(rawstrain_ET)
# plt.show()


# # In[14]:


# noise_chunk_size = 2.0
# alpha_window = 0.1
# noise_seglen      = np.int(srate*noise_chunk_size)
# psd_window        = tukey(noise_seglen, alpha_window)
# psd_ET_welch, freqs_ET_welch = mlab.psd(rawstrain_ET,
#                                       Fs     = 4096,
#                                       NFFT   = noise_seglen,
#                                       window = psd_window,
#                                       sides  = 'onesided')


# # In[15]:


# plt.loglog(freqs_ET_welch, psd_ET(freqs_welch), '--')
# plt.loglog(freqs_ET_welch, psd_ET_welch, '--')
# plt.show()


# # In[ ]:




