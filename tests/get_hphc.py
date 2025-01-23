import matplotlib.pyplot as pp
from pycbc import types, fft, waveform
import numpy as np

# Get a time domain waveform
hp, hc = waveform.get_td_waveform(approximant='IMRPhenomD', mass1=16, mass2=6, delta_t=1.0/4096, f_lower=40)
Tvals = hp.sample_times 
start_Time, end_Time = np.searchsorted(Tvals, (-1, 0.01))
hp_cut = hp[start_Time : end_Time]
hc_cut = hc[start_Time : end_Time]
T_cut = Tvals[start_Time : end_Time]
pp.plot(T_cut, hp_cut)
pp.show()
quit()

sptilde, sctilde = waveform.get_fd_waveform(approximant="TaylorF2", mass1=6, mass2=6, delta_f=1.0/4, f_lower=40)

# FFT it to the time-domain
#tlen = int(1.0 / hp.delta_t / sptilde.delta_f)
#sptilde.resize(tlen/2 + 1)
#sp = types.TimeSeries(types.zeros(tlen), delta_t=hp.delta_t)
#fft.ifft(sptilde, sp)

#pp.plot(sp.sample_times, sp, label="TaylorF2 (IFFT)")
pp.plot(hp.sample_times, hp, label='IMRPhenomD')
pp.plot(hp.sample_times, hc, label='IMRPhenomD')

pp.ylabel('Strain')
pp.xlabel('Time (s)')
pp.legend()
pp.show()

