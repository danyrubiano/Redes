import numpy as np
from numpy import sin, linspace, pi,fft
from scipy import fft, arange, ifft, signal
import matplotlib.pyplot as plt   
from random import randint
from math import floor



###valores por defecto
t1 = linspace(-16,16, 101)
sinc = np.sinc(t1)
alpha = 0.22##valor predeterminado
coseno_alzado = (np.sinc(t1) * np.cos(alpha*np.pi*t1))/(1-(4*alpha*alpha*t1*t1))


def tiempo_feecuencia():##funcion que expone los pulsos en el dominio de su tiempo y frecuencia
	

	##pulsos en el tiempo
	plt.subplot(2,1,1)
	plt.title('Funcion sinc en el tiempo')
	plt.xlabel('Tiempo(s)')
	plt.ylabel('Amplitud')
	plt.grid(True)
	plt.plot(t1,sinc, 'r')

	plt.subplot(2,1,2)
	plt.title('Funcion coseno alzado en el tiempo en el tiempo')
	plt.xlabel('Tiempo(s)')
	plt.ylabel('Amplitud')
	plt.grid(True)
	plt.plot(t1,coseno_alzado, 'b')
	plt.show()


	##pulsos en la frecuencia
	transformada_sinc = abs(fft(sinc))/len(sinc)
	transformada_coseno_alzado = abs(fft(coseno_alzado))/len(coseno_alzado)

	k = np.fft.fftfreq(len(sinc),1)
	k2 = np.fft.fftfreq(len(coseno_alzado),1)

	plt.subplot(2,1,1)
	plt.title('Funcion sinc en su frecuencia')
	plt.xlabel('Freq (Hz)')
	plt.ylabel('|Y(freq)|')
	plt.grid(True)
	plt.plot(k,transformada_sinc/max(transformada_sinc), 'r')

	plt.subplot(2,1,2)
	plt.title('Funcion coseno alzado en su frecuencia')
	plt.xlabel('Freq (Hz)')
	plt.ylabel('|Y(freq)|')
	plt.plot(k,transformada_coseno_alzado/max(transformada_coseno_alzado),'b')
	plt.grid(True)
	plt.show()


def comparar_rc(a,b,c):##funcion utilizada para exponer el pulso rc para 3 valores alpha entregados por el usuario
	alpha1=a
	alpha2=b
	alpha3=c
	rc1 = (np.sinc(t1) * np.cos(alpha1*np.pi*t1))/(1-(4*alpha1*alpha1*t1*t1))
	rc2 = (np.sinc(t1) * np.cos(alpha2*np.pi*t1))/(1-(4*alpha2*alpha2*t1*t1))
	rc3 = (np.sinc(t1) * np.cos(alpha3*np.pi*t1))/(1-(4*alpha3*alpha3*t1*t1))

	plt.subplot(3,1,1)
	plt.title('Funcion coseno alzado para alpha= %s' %(alpha1))
	plt.xlabel('Tiempo(s)')
	plt.ylabel('Amplitud')
	plt.grid(True)
	plt.plot(t1,rc1, 'r')

	plt.subplot(3,1,2)
	plt.title('Funcion coseno alzado para alpha= %s' %(alpha2))
	plt.xlabel('Tiempo(s)')
	plt.ylabel('Amplitud')
	plt.grid(True)
	plt.plot(t1,rc2, 'b')

	plt.subplot(3,1,3)
	plt.title('Funcion coseno alzado para alpha= %s' %(alpha3))
	plt.xlabel('Tiempo(s)')
	plt.ylabel('Amplitud')
	plt.grid(True)
	plt.plot(t1,rc3, 'g')
	plt.show()

	transformada_coseno_alzado1 = abs(fft(rc1))/len(rc1)
	transformada_coseno_alzado2 = abs(fft(rc2))/len(rc2)
	transformada_coseno_alzado3 = abs(fft(rc3))/len(rc3)
	k1 = np.fft.fftfreq(len(rc1),1)
	k2 = np.fft.fftfreq(len(rc2),1)
	k3 = np.fft.fftfreq(len(rc2),1)

	plt.subplot(3,1,1)
	plt.title('Funcion coseno alzado (frecuencia) para alpha= %s' %(alpha1))
	plt.xlabel('Freq (Hz)')
	plt.ylabel('|Y(freq)|')
	plt.grid(True)
	plt.plot(k1,transformada_coseno_alzado1/max(transformada_coseno_alzado1),'r')

	plt.subplot(3,1,2)
	plt.title('Funcion coseno alzado (frecuencia) para alpha= %s' %(alpha2))
	plt.xlabel('Freq (Hz)')
	plt.ylabel('|Y(freq)|')
	plt.grid(True)
	plt.plot(k2,transformada_coseno_alzado2/max(transformada_coseno_alzado2),'b')

	plt.subplot(3,1,3)
	plt.title('Funcion coseno alzado (frecuencia) para alpha= %s' %(alpha3))
	plt.xlabel('Freq (Hz)')
	plt.ylabel('|Y(freq)|')
	plt.grid(True)
	plt.plot(k3,transformada_coseno_alzado3/max(transformada_coseno_alzado3),'g')
	plt.show()




def bits_aleatorios():##funcion que expone el resultado de convolucionar los pulsos con un tren de 10 implusos aleatorios


	T=5###valor del periodo para el tren de impulsos
	tren=np.array(range(10*T), dtype="int")

	##tren para valores de periodo impar
	contador=floor(T/2);
	for x in range(0,10*T ):
	
		if (contador==0):
			tren[x]=randint(0,1)
			contador=floor(T/2)*2
			if (tren[x]==0):
				tren[x]=-1

		else:
			tren[x]=0
			contador=contador-1

	plt.title('Tren de impulsos a utilizar')
	plt.plot(tren)
	plt.show()

	resultado=np.convolve(sinc,tren,'same')##resultado de la convolucion del pulso sinc con el respectivo tren de impulsos
	resultado2=np.convolve(coseno_alzado,tren, 'same')##resultado de la convolucion del pulso coseno alzado con el respectivo tren de impulsos
	plt.subplot(2,1,1)
	plt.title('Funcion sinc luego de aplicar el tren de impulsos')
	plt.xlabel('Tiempo(s)')
	plt.ylabel('Amplitud')
	plt.grid(True)
	plt.plot(t1,resultado)

	plt.subplot(2,1,2)
	plt.title('Funcion coseno alzado luego de aplicar el tren de impulsos')
	plt.xlabel('Tiempo(s)')
	plt.ylabel('Amplitud')
	plt.grid(True)
	plt.plot(t1,resultado2)
	plt.show()

def eyediagram():##funcion utilizada para exponer los diagramas de ojos de los pulsos 

	alpha_ojo = 0.9999999999999##valor cercano a 1 entrega un diagrama mas limpio
	coseno_alzado2 = (np.sinc(t1) * np.cos(alpha_ojo*np.pi*t1))/(1-(4*alpha_ojo*alpha_ojo*t1*t1))
	T=5###valor del periodo para el tren de impulsos
	tren2=np.array(range(10000*T), dtype="int")

	##tren para valores de periodo impar
	contador=floor(T/2);
	for x in range(0,10000*T):
	
		if (contador==0):
			tren2[x]=randint(0,1)
			contador=floor(T/2)*2
			if (tren2[x]==0):
				tren2[x]=-1

		else:
			tren2[x]=0
			contador=contador-1


	resultado3=np.convolve(sinc,tren2,'same')##resultado de la convolucion del pulso sinc con el respectivo tren de impulsos
	resultado4=np.convolve(coseno_alzado2,tren2,'same')##resultado de la convolucion del pulso coseno alzado con el respectivo tren de impulsos

	##primero se exponen los diagramas de ojo resultantes solo de la convolucion

	ojo_sinc=np.reshape(resultado3,(2*T,len(resultado3)/(2*T)), order='F')
	ojo_coseno_alzado=np.reshape(resultado4,(2*T,len(resultado4)/(2*T)), order='F')
	
	plt.subplot(2,1,1)
	plt.title('Diagrama de ojo pulso Sinc')
	plt.xlabel('Tiempo(s)')
	plt.ylabel('Amplitud')
	plt.grid(True)	
	plt.plot(ojo_sinc,'b')

	plt.subplot(2,1,2)
	plt.title('Diagrama de ojo pulso coseno alzado')
	plt.xlabel('Tiempo(s)')
	plt.ylabel('Amplitud')
	plt.grid(True)	
	plt.plot(ojo_coseno_alzado,'b')

	plt.show()

	##ahora se expondran los diagramas de ojo con ruido awgn agregado
	# Numero de datos a utilizar
	N = 50000
	# razon señal ruido
	SNR = 20

	# Create some data with noise and a sinusoidal
	# variation.
	y = np.random.normal(0.0, 1.0/SNR, N) + 1.0
	
	ruido_sinc= y+resultado3
	ruido_rc=y+resultado4

	ojo_ruido_sinc=np.reshape(ruido_sinc,(2*T,len(ruido_sinc)/(2*T)), order='F')
	ojo_ruido_coseno_alzado=np.reshape(ruido_rc,(2*T,len(ruido_rc)/(2*T)), order='F')
	
	plt.subplot(2,1,1)
	plt.title('Diagrama de ojo pulso Sinc con ruido')
	plt.xlabel('Tiempo(s)')
	plt.ylabel('Amplitud')
	plt.grid(True)	
	plt.plot(ojo_ruido_sinc,'b')

	plt.subplot(2,1,2)
	plt.title('Diagrama de ojo pulso coseno alzado con ruido')
	plt.xlabel('Tiempo(s)')
	plt.ylabel('Amplitud')
	plt.grid(True)	
	plt.plot(ojo_ruido_coseno_alzado,'b')

	plt.show()




#######MENU#########
salida=0##variable a utilizar para mantener el while
while (salida==0):
	print('MENU')
	print('PARTE 1')
	print('1-Graficos en el dominio del tiempo y de su frecuencia')
	print('2-Grafico pulso rc para 3 valores de alpha')
	print('PARTE 2')
	print('3-Señal resultante del envio de 10 bits aleatorios')
	print('4-Diagramas de ojo de los pulsos Sinc y Coseno Alzado')
	print('5-Salir')

	opcion = input("Eleccion: ")
	opcion=int(opcion)

	if (opcion==1):
		tiempo_feecuencia()
		print('////////////////////////////////////////////////')
		print('////////////////////////////////////////////////')

	if (opcion==2):

		a1=input('valor para el primer valor alpha (0<alpha<1):')
		a1=float(a1)
		while(a1<=0 or a1>=1) :
			print('valor de alpha debe cumplir 0<alpha<1')
			a1=input('valor para el primer valor alpha (0<alpha<1):')
			a1=float(a1)
		
		a2=input('valor para el segundo valor alpha (0<alpha<1):')
		a2=float(a2)
		while(a2<=0 or a2>=1) :
			print('valor de alpha debe cumplir 0<alpha<1')
			a2=input('valor para el segundo valor alpha (0<alpha<1):')
			a2=float(a2)

		a3=input('valor para el tercer valor alpha (0<alpha<1):')
		a3=float(a3)
		while(a3<=0 or a3>=1) :
			print('valor de alpha debe cumplir 0<alpha<1')
			a3=input('valor para el tercer valor alpha (0<alpha<1):')
			a3=float(a3)
		
		comparar_rc(a1,a2,a3)
		print('////////////////////////////////////////////////')
		print('////////////////////////////////////////////////')

	if (opcion==3):
		bits_aleatorios()
		print('////////////////////////////////////////////////')
		print('////////////////////////////////////////////////')

	if (opcion==4):
		eyediagram()
		print('////////////////////////////////////////////////')
		print('////////////////////////////////////////////////')

	if (opcion==5):
		salida=1
		print('////////////////////////////////////////////////')
		print('////////////////////////////////////////////////')