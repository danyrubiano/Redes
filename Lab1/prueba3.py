import numpy as np
from numpy import sin, linspace, pi
from scipy.io.wavfile import read,write
from scipy import fft, arange, ifft

import matplotlib.pyplot as plt   

def mayor(lista):
    if lista ==[]:
        return("error")
    elif len(lista) == 1:
        return(lista)
    lista_nueva = 0
    index = 0
    i = 0
    while lista != []:
        primero = lista[0]
        if lista_nueva > primero:
            lista_nueva = lista_nueva
        else:
            lista_nueva =primero
            index = i
        lista = lista[1:]
        i = i+1
    return lista_nueva, index

rate,info=read("beacon.wav")
dimension = info[0].size
##print(dimension)					#data: datos del audio (arreglo de numpy)
if dimension==1:							#rate: frecuencia de muestreo
	data = info
	perfect = 1
else:
	data = info[:,dimension-1]
	perfect = 0


timp=len(data)/rate 
t=linspace(0,timp,len(data))

large = len(data)
#print(large)
k = arange(large)
T = large/rate
frq = k/T
Y = fft(data)/large

x= len(Y)

aux = Y

a, b =mayor(Y)

print(a)
print(b)
vmax = len(aux)
cota = vmax * 0.15

for i in range (0, vmax):
	if i < (b - cota):
		aux[i] = 0
	elif i > (b + cota):
		aux[i] = 0

plt.plot(frq,abs(aux),'c')
plt.title('Gráfico de Frecuencia Truncado al 15%')
plt.ylabel('Magnitud de Frecuencia [dB]')
plt.xlabel('Frecuencia [Hz]')
#plt.show()

temp = ifft(aux)
#plt.plot(t,temp, 'b')
#plt.title('Audio Procesado')
#plt.xlabel('Tiempo [s]')
#plt.ylabel('Amplitud [dB]')
#plt.show()


    



