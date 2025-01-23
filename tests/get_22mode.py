import numpy as np
import lalsimulation as lalsim
import lal
import matplotlib.pyplot as plt


q = 1.0/0.84
rate=8192.*2. #for time dt = 1./rate
fLow = 8.2209#8.2988
fRef = 8.50445#8.2988
dist = 300.
dist_SI=dist*lal.PC_SI*1.e6
inc=1.
Mtot=70.
m1=q/(1.+q)*Mtot
m2=1./(1.+q)*Mtot
m1_SI=m1*lal.MSUN_SI
m2_SI=m2*lal.MSUN_SI

s1x= 0.2320
s1y= -0.3250
s1z= 0.2440
s2x= -0.3580
s2y= 0.007
s2z= 0.2880

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

hlmsSTT4=lalsim.SimInspiralChooseTDModes(phiRef, dT, m1_SI, m2_SI, s1x, s1y, s1z, s2x, s2y, s2z, fLow, fRef, dist_SI, lalParsTay, int(4), lalsim.IMRPhenomD)

hlmSTT4_tmp=lalsim.SphHarmTimeSeriesGetMode(hlmsSTT4,2,2)
Tvals =  float(hlmSTT4_tmp.epoch)+np.arange(len(hlmSTT4_tmp.data.data))*hlmSTT4_tmp.deltaT
Time = np.arange(len(hlmSTT4_tmp.data.data))*hlmSTT4_tmp.deltaT
#np.savetxt("PNTime.txt", Tvals)
for l in range(2, LMAX+1):
    for m in range(-l, l+1):
        mode = lalsim.SphHarmTimeSeriesGetMode(hlmsSTT4,l,m).data.data
        #Tuniform, Zuniform = uniform_time(Tvals, mode)
        #Tclean, Zclean = clean_junk(Tuniform, Zuniform )
        #ReNew, ImNew = Zuniform.real , Zuniform.imag
        #ReNew, ImNew = Zclean.real , Zclean.imag

        np.savetxt('IMR{0}{1}'.format(l,m), np.stack((Tvals, mode.real,mode.imag), axis=-1))



