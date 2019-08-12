import numpy as np
from scipy import signal
from numpy import sin, linspace, pi
from scipy.io.wavfile import read,write
from scipy import fft, arange, ifft
from pylab import plot, show, title, xlabel, ylabel
from scipy.signal import bilinear, butter, lfilter
import matplotlib.pyplot as plt   
from scipy.signal import lfilter, firwin

def leerAudio(audio):
	rate,info=read(audio)
	dimension = info[0].size
	if dimension==1:
	    data = info
	else:
	    data = info[:,dimension-1]

	return data, rate


def plotTime(data, rate):
	large = len(data)
	T = large/rate 
	t = linspace(0,T,large)    			#linspace(start,stop,number)
	time1 = plot(t, data)
	title('Audio')
	xlabel('Tiempo [s]')
	ylabel('Amplitud [dB]')
	return time1, t


def plotFrecuece(data, rate):
	large = len(data)
	k = arange(large)
	T = large/rate
	frq = k/T
	Y1 = fft(data)
	Y2 = fft(data)/large
	frq1 = plot(frq,abs(Y1),'c')
	title('Gráficos de Frecuencia y Spectograma')
	ylabel('Amplitud de Frecuencia [dB]')
	xlabel('Frecuencia [Hz]')
	return Y1, frq, large, frq1


def plotSpecgram(data, rate):
	NFFT = 1024     # the length of the windowing segments
	Pxx, freqs, bins, im = plt.specgram(data, NFFT=NFFT, Fs=rate)
	#Pxx, freqs, bins, im = plt.specgram(data, Fs=rate)
	plt.show()


data, rate = leerAudio("beacon.wav")
plotSpecgram(data, rate)
#------------------------------------------------
# Create a FIR filter and apply it to signal.
#------------------------------------------------
# The Nyquist rate of the signal.
nyq_rate = rate / 2.
# The cutoff frequency of the filter: 6KHz
cutoff_hz = 1000
# Length of the filter (number of coefficients, i.e. the filter order + 1)
numtaps = 1001
# Use firwin to create a lowpass FIR filter
fir_coeff1 = firwin(numtaps, cutoff_hz/nyq_rate)

#----------------------------------------------------------
# now create the taps for a high pass filter.
# by multiplying tap coefficients by -1 and
# add 1 to the centre tap ( must be even order filter)

fir_coeff1 = [-1*a for a in fir_coeff1]
fir_coeff1[(numtaps+1)/2] = fir_coeff1[(numtaps+1)/2] + 1

# The cutoff frequency of the filter: 6KHz
cutoff_hz_2 = 3000

# Use firwin to create a lowpass FIR filter
fir_coeff2 = firwin(numtaps, cutoff_hz_2/nyq_rate)

fir_coeff2[(numtaps+1)/2] = fir_coeff2[(numtaps+1)/2] - 1

taps = [sum(pair) for pair in zip(fir_coeff1, fir_coeff2)]

filtered_x = lfilter(taps, 1.0, data)

plotFrecuece(filtered_x, rate)
show()

plotSpecgram(filtered_x, rate)

write("new.wav", rate, filtered_x)