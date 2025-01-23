import numpy as np
import lalsimulation as lalsim
import lal
import matplotlib.pyplot as plt



#fix parameters
rate=8192.*2. #for time dt = 1./rate
#fLow = 8.2209#8.2988
#fRef = 8.50445#8.2988
dist = 300.
dist_SI=dist*lal.PC_SI*1.e6
inc=1.
s1x= 0.0
s1y= -0.0
s1z= 0.0
s2x= -0.0
s2y= 0.00
s2z= 0.0
phiRef=0.0  # if BHs are on x-axis initially
dT=1./rate
LMAX=4
ampO=-1#LMAX-2




modearray = lalsim.SimInspiralCreateModeArray()
for idxl in range(LMAX-1):
    l=idxl+2
    for idxm in range(2*l+1):
        m=l-idxm
        lalsim.SimInspiralModeArrayActivateMode(modearray, l, m);
lalParsNR=lal.CreateDict()
err_code =lalsim.SimInspiralWaveformParamsInsertModeArray(lalParsNR,modearray)
lalParsTay = lal.CreateDict()
err_code+=lalsim.SimInspiralWaveformParamsInsertPNAmplitudeOrder(lalParsTay,ampO)


def get_h22(m1, m2, fLow, fRef):
    m1_SI=m1*lal.MSUN_SI
    m2_SI=m2*lal.MSUN_SI
    hlmsSTT4=lalsim.SimInspiralChooseTDModes(phiRef, dT, m1_SI, m2_SI, s1x, s1y, s1z, s2x, s2y, s2z, fLow, fRef, dist_SI, lalParsTay, int(4), lalsim.IMRPhenomD)
    h22=lalsim.SphHarmTimeSeriesGetMode(hlmsSTT4,2,2)
    Tvals =  float(h22.epoch)+np.arange(len(h22.data.data))*h22.deltaT
    return Tvals, h22.real, h22.imag


print(get_h22(10, 10, 20, 20))


