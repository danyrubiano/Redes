import numpy as np
from numpy import sin, linspace, pi,fft
from scipy import fft, arange, ifft, signal
import matplotlib.pyplot as plt   
from random import randint
from math import floor

# instalar pycharm

T = 5
alpha = 0.90
Fs = alpha/T
num = T/Fs
f=1/T
t = 50
fs = 1/t

t1 = linspace(-t,t, (t/fs)+1)


x1 = np.sinc(t1)
##plt.plot(t1,x1, 'r')

x2 = (np.sinc(t1) * np.cos(alpha*np.pi*t1))/(1-(4*alpha*alpha*t1*t1)) #la función sinc tendrá periodo t = 2pi/w , w = pi, => t =2, por lo tanto hay que generar 0 y 1 cada t/2

"""
plt.plot(t1,x2, 'b')
plt.grid(True)
plt.xlim(-10,10)
plt.show()
"""


Y_k = abs(fft(x1))/len(x1)
Y_k2 = abs(fft(x2))/len(x2)

k = np.fft.fftfreq(len(x1),f)*fs
k2 = np.fft.fftfreq(len(x2),f)

"""
plt.plot(k,Y_k/max(Y_k), 'r')
plt.show()
plt.plot(k,Y_k2/max(Y_k2),'b')
plt.show()
"""






##tren de impulsos parte 2-1
#tren=np.array(range(10*T), dtype="int")
##t impar

contador=floor(T/2)

tren = np.zeros((len(t1)*1/(2*t))*10) #cantidad de puntos por periodo

for x in range(0,10):
	tren[x*len(t1)*1/(2*t)] = 2*randint(0,1)-1

"""
print(tren)
plt.plot(tren)
plt.grid(True)
plt.show()
"""
resultado=signal.convolve(x1,tren)
"""
plt.plot(resultado)
plt.grid(True)
plt.show()
"""


resultado2=np.convolve(x2,tren)

plt.grid(True)
plt.plot(resultado2)
plt.show()

"""
print(len(tren))
print(len(x1))
print(len(x2))
print(len(resultado))
print(len(resultado2))
"""
aux = np.zeros(300) #cantidad de puntos por periodo
i=0
for x in range(1200, 1500):
	aux[i] = resultado2[x]
	i = i + 1 

print(aux)


ojo=np.reshape(aux,(2*T,-1), order="F")

print(ojo)
plt.plot(ojo)
plt.show()

"""
# Number of data points
N = 2750
# Signal to noise ratio
SNR = 2

# Create some data with noise and a sinusoidal
# variation.
y = np.random.normal(0.0, 1/SNR, N) + 1.0
y += resultado2

plt.plot(y, 'b')
plt.show()

ojo=np.reshape(y,(2*T,len(y)/(2*T)), order="F")

print(ojo)
plt.plot(ojo)
plt.show()
"""







