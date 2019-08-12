import numpy as np
from numpy import sin, linspace, pi,fft
from scipy import fft, arange, ifft, signal
import matplotlib.pyplot as plt   
from random import randint
from math import floor

# instalar pycharm

T = 5
alpha = 0.22
Fs = alpha/T
num = T/Fs
f=1/T

t1 = linspace(-15,15, 101)
x1 = np.sinc(t1)
plt.plot(t1,x1, 'r')

x2 = (np.sinc(t1) * np.cos(alpha*np.pi*t1))/(1-(4*alpha*alpha*t1*t1))

plt.plot(t1,x2, 'b')
plt.show()


Y_k = abs(fft(x1))/len(x1)
Y_k2 = abs(fft(x2))/len(x2)

k = np.fft.fftfreq(len(x1),f)
k2 = np.fft.fftfreq(len(x2),f)


plt.plot(k,Y_k/max(Y_k), 'r')
plt.show()
plt.plot(k,Y_k2/max(Y_k2),'b')
plt.show()







##tren de impulsos parte 2-1
tren=np.array(range(10*T), dtype="int")
##t impar

contador=floor(T/2);
for x in range(0,10*T):
	
	
	if (contador==0):
		tren[x]=randint(0,1)
		contador=floor(T/2)*2
		if (tren[x]==0):
			tren[x]=-1

	else:
		tren[x]=0
		contador=contador-1

print(tren)
plt.plot(tren)
plt.show()

resultado=signal.convolve(x1,tren)

plt.plot(resultado)
plt.show()

resultado2=np.convolve(x2,tren)
plt.plot(resultado2)
plt.show()
print(len(tren))
print(len(x1))
print(len(x2))
print(len(resultado))
print(len(resultado2))


ojo=np.reshape(resultado,(T,len(resultado)/(T)))
print(ojo)
plt.plot(ojo,'b')
plt.show()




