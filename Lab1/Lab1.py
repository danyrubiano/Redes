import numpy as np
from numpy import sin, linspace, pi
from scipy.io.wavfile import read,write
from scipy import fft, arange, ifft
import matplotlib.pyplot as plt
from pylab import plot, show, title, xlabel, ylabel

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


def plotFrecueceNormalizada(data, rate, Y, large, frq):
	Y2 = Y/large
	frq2 = plot(frq,abs(Y2),'c')
	title('Gráficos de Frecuencia y Spectograma')
	ylabel('Amplitud de Frecuencia [dB]')
	xlabel('Frecuencia [Hz]')
	return frq2


def plotTimeAgain(data, rate, Y, t):
	otro = ifft(Y)
	time2 = plot(t, otro)
	title('Audio')
	xlabel('Tiempo [s]')
	ylabel('Amplitud [dB]')
	return time2

def plotTimeAgainNormalizado(data, rate, Y, t, large):
	otro = ifft(Y)/large
	time3 = plot(t, otro)
	title('Audio')
	xlabel('Tiempo [s]')
	ylabel('Amplitud [dB]')
	return time3

def buscarMayor(lista):
    if lista ==[]:
        return("error")
    elif len(lista) == 1:
        return(lista)
    mayor = 0
    index = 0
    i = 0
    while lista != []:
        primero = lista[0]
        if mayor > primero:
            mayor = mayor
        else:
            mayor =primero
            index = i
        lista = lista[1:]
        i = i+1
    return mayor, index


def truncar(data, rate, Y, indexMayor):
	aux = Y
	b = indexMayor
	vmax = len(aux)
	cota = vmax * 0.15

	for i in range(0, vmax):
		if i < (b - cota):
			aux[i] = 0
		elif i > (b + cota):
			aux[i] = 0

	return aux


def plotFrqTruncada(aux, frq):
	frq3 = plot(frq,abs(aux),'c')
	title('Gráfico de Frecuencia Truncado al 15%')
	ylabel('Magnitud de Frecuencia [dB]')
	xlabel('Frecuencia [Hz]')
	return frq3


def plotFrqTruncadaNormalizado(aux, frq, large):
	temp = aux/large
	frq4 = plot(frq,abs(temp),'c')
	title('Gráfico de Frecuencia Truncado al 15%')
	ylabel('Magnitud de Frecuencia [dB]')
	xlabel('Frecuencia [Hz]')
	return frq4


def plotTimeTruncado(aux, t):
	temp = ifft(aux)
	time4 = plot(t, temp, 'b')
	title('Audio Procesado')
	xlabel('Tiempo [s]')
	ylabel('Amplitud [dB]')
	return time4


def plotTimeTruncadoNormalizado(aux, t, large):
	temp = ifft(aux)/large
	time5 = plot(t, temp, 'b')
	title('Audio Procesado')
	xlabel('Tiempo [s]')
	ylabel('Amplitud [dB]')
	return time5



############################################################
############################################################
##Funcion Main

data, rate = leerAudio("beacon.wav")

time1, t = plotTime(data, rate)
show()

Y, frq, large, frq1 = plotFrecuece(data, rate)
show()

plotFrecueceNormalizada(data, rate, Y, large, frq)
show()

plotTimeAgain(data, rate, Y, t)
show()

plotTimeAgainNormalizado(data, rate, Y, t, large)
show()

mayor, indexMayor = buscarMayor(Y)

aux = truncar(data, rate, Y, indexMayor)

plotFrqTruncada(aux, frq)
show()

plotFrqTruncadaNormalizado(aux, frq, large)
show()

plotTimeTruncado(aux, t)
show()

plotTimeTruncadoNormalizado(aux, t, large)
show()

write("audio_truncado.wav", rate, abs(aux))

