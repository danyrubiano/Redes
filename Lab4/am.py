# Import the plotting library
import numpy as np
from matplotlib import pyplot as plt
from scipy.io.wavfile import read,write
from scipy import fft, arange, ifft
import scipy.integrate as integrate
from scipy.signal import lfilter, firwin

rate,info=read("handel.wav")
dimension = info[0].size
if dimension==1:
    data = info
else:
    data = info[:,dimension-1]

large = len(data)

T = large/rate 

t1 = np.linspace(0,T,large)    			#linspace(start,stop,number)

t2 = np.linspace(0,T,200000*T)

data1 = np.interp(t2, t1, data)

t3 = np.linspace(0, 200000, 200000*T)


# Create the signals
carrier = np.cos(t3*np.pi);

am = carrier * data1 
"""
plt.plot(t1, data)
plt.show()

plt.plot(t3, carrier)
plt.show()

plt.plot(t2, am)
plt.show()
"""
plt.subplot(311)
plt.plot(t1[1000:2000], data[1000:2000])

plt.subplot(312)
plt.plot(t3[1000:2000], carrier[1000:2000])

plt.subplot(313)
plt.plot(t2[1000:2000], am[1000:2000])
plt.show()

##############################################################

""" Demodulacion AM, faltaria agregarle el canal de ruido a la señal original 
"""
am_demod = am * carrier

am_d = np.interp(t1, t2, am_demod)

plt.subplot(211)
NFFT = 1024     # the length of the windowing segments
Pxx, freqs, bins, im = plt.specgram(am_d, NFFT=NFFT, Fs=rate)

numtaps = 1001
nyq_rate = rate / 2
cutoff = 3500
fir = firwin(numtaps, cutoff/nyq_rate)
filtered_am = lfilter(fir, 1.0, am_d)

plt.subplot(212)
Pxx, freqs, bins, im = plt.specgram(filtered_am, NFFT=NFFT, Fs=rate)
plt.show()

plt.subplot(311)
plt.plot(t1, data)

plt.subplot(312)
plt.plot(t1, data)

plt.subplot(313)
plt.plot(t1, filtered_am)
plt.show()

write("audio_dem.wav", rate, filtered_am)


##############################################################

""" Modulacion FM, el profe dijo que era 4 veces mas 
"""
"""
t2_fm = np.linspace(0,T, 400000*T)

data_fm = np.interp(t2_fm, t1, data)

t3_fm = np.linspace(0, 400000, 400000*T)

A = 1
k = 0.15

# Create the signals
carrier_fm = np.sin(2*np.pi*t3_fm);

wct = rate * t2_fm

audio_integrate = integrate.cumtrapz(data_fm, t2_fm, initial=0)

fm = np.cos(np.pi*wct + audio_integrate*np.pi);

plt.subplot(311)
plt.plot(t1[1000:4000], data[1000:4000])

plt.subplot(312)
plt.plot(t3_fm[1000:4000], carrier_fm[1000:4000])

plt.subplot(313)
plt.plot(t2_fm[1000:4000], fm[1000:4000])
plt.show()

#################################################################

v=[0,0,1,1,0,1,1,0,0,1,0]
dim=100
Vx=[]
for i in range(1,11):
	f=np.ones(dim)
	x=i*v[i]
	Vx=np.concatenate((Vx,x))

plt.subplot(3,1,1)
plt.plot(Vx)
dim1=lon(Vx)
t=np.linspace(0,5,dim1)
f1=5
plt.subplot(3,1,2)
w1 = 2*np.pi*f1*t
y1=np.cos(w1)
plt.plot(t,y1)
plt.subplot(3,1,3)
mult=(Vx*y1)
plt.plot(t,mult)
"""