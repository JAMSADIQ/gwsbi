import numpy as np
import matplotlib.pyplot as plt
import lal
import lalsimulation as lalsim



##we need psd lets take gw150915 one
from urllib.request import urlretrieve
from pycbc.frame import read_frame
from pycbc.filter import highpass_fir, matched_filter
from pycbc.waveform import get_fd_waveform
from pycbc.psd import welch, interpolate

# Read data and remove low frequency content
#fname = 'H-H1_LOSC_4_V2-1126259446-32.gwf'
#url = "https://www.gwosc.org/GW150914data/" + fname
#urlretrieve(url, filename=fname)
h1 = read_frame('H-H1_LOSC_4_V2-1126259446-32.gwf', 'H1:LOSC-STRAIN')
h1 = highpass_fir(h1, 15, 8)
# Calculate the noise spectrum
psd = interpolate(welch(h1), 1.0 / h1.duration)
#plt.plot(psd)
#plt.show()
#quit()
# Read data and remove low frequency content
def noise_from_PSD(psd, srate=4096, T=32, fmin=40.0, fmax=4096):
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
        frequency_strain[i] = np.random.normal(0.0,sigma_cgn)+1j*np.random.normal(0.0,sigma_cgn) ## from pyRing
    rawstrain = np.fft.irfft(frequency_strain)/dt

    return rawstrain


M, chi, C220, P220, C221, P221 = np.asarray([10, 0.5, 2, np.pi, 2, np.pi])
prefactor_omega_r = (lal.C_SI*lal.C_SI*lal.C_SI/(2.*np.pi*lal.G_SI*M*lal.MSUN_SI))
prefactor_omega_i = (lal.C_SI*lal.C_SI*lal.C_SI/(lal.G_SI*M*lal.MSUN_SI))
reference_amplitude = 1e-20
A220 = C220*np.exp(1j*P220)*reference_amplitude
A221 = C221*np.exp(1j*P221)*reference_amplitude
#A = np.array([A220, A221])
#f = np.array([prefactor_omega_r*omega_r(chi),prefactor_omega_r*omega_OT_r(chi)])
#gamma = np.array([-prefactor_omega_i*omega_i(chi),-prefactor_omega_i*omega_OT_i(chi)])
T = 2
srate = 4096
dt = 1/srate
num_noise_rel_x_theta = 2
num_sim = 2
rawstrains = np.zeros((num_noise_rel_x_theta, len(noise_from_PSD(psd, srate=4096, T=2, fmin=20, fmax=4096))))

print(rawstrains.shape)
for i in range(num_noise_rel_x_theta):
    rawstrains[i] = noise_from_PSD(psd, srate=4096, T=2, fmin=20, fmax=4096)
    times = np.arange(0.0, T, dt) # time from 0 to 2
    idx = np.argwhere(times>=1)[0,0] #
    print("index of time> trigger?", idx)
    times = times[idx:] - trigtime # trigger time  ?
    rawstrains = rawstrains[:,idx:]
  

