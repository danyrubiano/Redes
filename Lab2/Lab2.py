import numpy as np
from scipy import signal
from numpy import sin, linspace, pi
from scipy.io.wavfile import read, write
from scipy import fft, arange, ifft
from pylab import plot, show, title, xlabel, ylabel
from scipy.signal import lfilter
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


def plotTime(data, rate, numtaps):
	large = len(data)
	T = large/rate 
	t = linspace(0,T,large)    			#linspace(start,stop,number)
	# The first N-1 samples are "corrupted" by the initial conditions
	warmup = numtaps - 1
	# The phase delay of the filtered signal
	delay = (warmup / 2) / rate
	# Plot the filtered signal, shifted to compensate for the phase delay
	time1 = plot(t-delay, data, 'r-')
	title('Audio en el Dominio del Tiempo')
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
	title('Audio en el Dominio de la Frecuencia')
	ylabel('Amplitud de Frecuencia [dB]')
	xlabel('Frecuencia [Hz]')
	return Y1, frq, large, frq1


def plotSpecgram(data, rate):
	NFFT = 1024     # the length of the windowing segments
	Pxx, freqs, bins, im = plt.specgram(data, NFFT=NFFT, Fs=rate)
	#Pxx, freqs, bins, im = plt.specgram(data, Fs=rate)
	plt.title('Spectograma de Frecuencias')
	plt.ylabel('Amplitud de Frecuencia [dB]')
	plt.xlabel('Tiempo [s]')
	plt.show()



def filt(data, rate):
    # The Nyquist rate of the signal.
    nyq_rate = rate / 2.
    # The cutoff frequency of the filter: 6KHz
    cutoff_low = 1850
    cutoff_high = 2050
    # Length of the filter (number of coefficients, i.e. the filter order + 1)
    numtaps = 1001
    # Use firwin to create a lowpass FIR filter
    fir = firwin(numtaps, cutoff_high/nyq_rate) #filtro paso bajo
    #fir = firwin(numtaps,[cutoff_low/nyq_rate, 0.99], pass_zero=False) #filtro paso alto
    #fir = firwin(numtaps,[cutoff_low/nyq_rate, cutoff_high/nyq_rate], pass_zero=False) # filtro paso banda
    # Use lfilter to filter the signal with the FIR filter
    filtered_signal = lfilter(fir, 1.0, data)
    plotFrecuece(filtered_signal, rate)
    show()
    plotTime(filtered_signal, rate, numtaps)
    show()
    plotSpecgram(filtered_signal, rate)
    write("audio_filtrado.wav", rate, filtered_signal)


#####################################################################################
#####################################################################################
## Main

data, rate = leerAudio("beacon.wav")
plotSpecgram(data, rate)

filt(data, rate)