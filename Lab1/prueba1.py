import numpy as np
from numpy import sin, linspace, pi
from scipy.io.wavfile import read,write
from scipy import fft, arange, ifft

import matplotlib.pyplot as plt

rate,info=read("beacon.wav")
print(rate) ## Frrecuencia de muestreo 
print(info)
dimension = info[0].size
print(dimension)					#data: datos del audio (arreglo de numpy)
if dimension==1:							#rate: frecuencia de muestreo
	data = info
	perfect = 1
else:
	data = info[:,dimension-1]
	perfect = 0

timp=len(data)/rate 
t=linspace(0,timp,len(data))    			#linspace(start,stop,number)
print(len(data))
print(timp)
#plt.title('Audio')
#plt.xlabel('Tiempo [s]')
#plt.ylabel('Amplitud [dB]')
#plt.plot(t, data)
#plt.show()



large = len(data)
print(large)
k = arange(large)
T = large/rate
frq = k/timp
Y = fft(data)

#plt.plot(frq,abs(Y),'c')
#plt.title('Gráfico de Frecuencia')
#plt.ylabel('Magnitud de Frecuencia [dB]')
#plt.xlabel('Frecuencia [Hz]')
#plt.show()

otro = ifft(Y)
#plt.title('Audio')
#plt.xlabel('Tiempo [s]')
#plt.ylabel('Amplitud [dB]')
#plt.plot(t, otro, 'b')
#plt.show()