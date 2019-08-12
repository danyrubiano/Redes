import numpy as np
from matplotlib import pyplot as plt
from scipy.io.wavfile import read,write
from scipy import fft, arange, ifft
import scipy.integrate as integrate
from scipy.signal import lfilter, firwin

rate,info=read("handel.wav")
dimension = info[0].size #es necesario saber si la señal de audio es de doble canal o no para poder trabajar en ella
if dimension==1:
     datos = info
else:
    datos = info[:,dimension-1]

def parte1():
##Debido a la naturaleza de la señal es necesario realizar una interpolacion para tener una mayor cantidad de puntos
	largo = len(datos)
	T = largo/rate 
	t1 = np.linspace(0,T,largo)    			
	t2 = np.linspace(0,T,200000*T)
	data1 = np.interp(t2, t1, datos)
	t3 = np.linspace(0, 200000, 200000*T)


# Creacion de la señal portadora para diferentes porcentajes

	carrier=np.cos(t3*2*np.pi);
	carrier15 = 1.15*np.cos(t3*2*np.pi);
	carrier100=7.2*np.cos(t3*2*np.pi);
	carrier125=9.584*np.cos(t3*2*np.pi);

#señales moduladas por amplitud en sus difentes porcentajes
	am = carrier * data1 
	am15=carrier15*data1
	am100=carrier100*data1
	am125=carrier125*data1


##para efectos graficos es necesario "cortar" la imagen para poder visualizarla de manera correcta
	plt.subplot(311)
	plt.subplots_adjust(hspace = 0.75)
	plt.title('Señal Original')
	plt.xlabel('Tiempo (s)')
	plt.ylabel('Amplitud (dB)')
	plt.grid(True)
	plt.plot(t1[1000:2000], datos[1000:2000])

	plt.subplot(312)
	plt.title('Señal Portadora AM')
	plt.xlabel('Tiempo (s)')
	plt.ylabel('Amplitud (dB)')
	plt.grid(True)
	plt.plot(t3[1000:2000], carrier[1000:2000])

	plt.subplot(313)
	plt.title('Señal Modulada por Amplitud')
	plt.xlabel('Tiempo (s)')
	plt.ylabel('Amplitud (dB)')
	plt.grid(True)
	plt.plot(t2[1000:2000], am[1000:2000])
	plt.show()



	plt.subplot(311)
	plt.subplots_adjust(hspace = 0.75)
	plt.title('Modulacion AM al 15%')
	plt.xlabel('Tiempo (s)')
	plt.ylabel('Amplitud (dB)')
	plt.grid(True)
	plt.plot(t2[1000:2000], am15[1000:2000])

	plt.subplot(312)
	plt.title('Modulacion AM al 100%')
	plt.xlabel('Tiempo (s)')
	plt.ylabel('Amplitud (dB)')
	plt.grid(True)
	plt.plot(t2[1000:2000], am100[1000:2000])

	plt.subplot(313)
	plt.title('Modulacion AM al 125%')
	plt.xlabel('Tiempo (s)')
	plt.ylabel('Amplitud (dB)')
	plt.grid(True)
	plt.plot(t2[1000:2000], am125[1000:2000])
	plt.show()

##############################################################



## Modulacion FM,  4 veces mas 

	t2_fm = np.linspace(0,T, 400000*T)
	data_fm = np.interp(t2_fm, t1, datos)
	t3_fm = np.linspace(0, 400000, 400000*T)


# Se crean las señales necesarias
	carrier_fm = np.sin(2*np.pi*t3_fm);
	wct = rate * t2_fm
	integral_audio = integrate.cumtrapz(data_fm, t2_fm, initial=0)##segun la teoria es necesaria la integral de la señal
	fm = np.cos(np.pi*wct + integral_audio*np.pi);

##señales fm a los porcentajes de interes
	fm15 = np.cos(np.pi*wct + 1.15*integral_audio*np.pi);
	fm100 = np.cos(np.pi*wct + 7.2*integral_audio*np.pi);
	fm125 = np.cos(np.pi*wct + 9.584*integral_audio*np.pi);



	plt.subplot(311)
	plt.subplots_adjust(hspace = 0.75)
	plt.title('Señal Original')
	plt.xlabel('Tiempo (s)')
	plt.ylabel('Amplitud (dB)')
	plt.grid(True)
	plt.plot(t1[1000:4000], datos[1000:4000])

	plt.subplot(312)
	plt.title('Señal Portadora FM')
	plt.xlabel('Tiempo (s)')
	plt.ylabel('Amplitud (dB)')
	plt.grid(True)
	plt.plot(t3_fm[1000:4000], carrier_fm[1000:4000])

	plt.subplot(313)
	plt.title('Señal Modulada por Frecuencia')
	plt.xlabel('Tiempo (s)')
	plt.ylabel('Amplitud (dB)')
	plt.grid(True)
	plt.plot(t2_fm[1000:4000], fm[1000:4000])
	plt.show()



	plt.subplot(311)
	plt.subplots_adjust(hspace = 0.75)
	plt.title('Modulacion FM al 15%')
	plt.xlabel('Tiempo (s)')
	plt.ylabel('Amplitud (dB)')
	plt.grid(True)
	plt.plot(t2[1000:4000], fm15[1000:4000])

	plt.subplot(312)
	plt.title('Modulacion FM al 100%')
	plt.xlabel('Tiempo (s)')
	plt.ylabel('Amplitud (dB)')
	plt.grid(True)
	plt.plot(t2[1000:4000], fm100[1000:4000])

	plt.subplot(313)
	plt.title('Modulacion FM al 125%')
	plt.xlabel('Tiempo (s)')
	plt.ylabel('Amplitud (dB)')
	plt.grid(True)
	plt.plot(t2[1000:4000], fm125[1000:4000])
	plt.show()

#################################################################
##A continuacion se exponen los espectrogramas de las diferentes modulaciones am, fm y sus respectivos porcentajes

##original,am y fm
	plt.subplot(311)
	plt.subplots_adjust(hspace = 0.75)
	plt.title('Espectrograma de la Señal Original')
	plt.xlabel('Tiempo (s)')
	plt.ylabel('Frecuencia (Hz)')
	plt.grid(True)
	plt.specgram(datos, NFFT=1024,Fs=rate)    

	plt.subplot(312)
	plt.title('Espectrograma de la Modulacion AM')
	plt.xlabel('Tiempo (s)')
	plt.ylabel('Frecuencia (Hz)')
	plt.grid(True)
	plt.specgram(am, NFFT=1024,Fs=rate)    

	plt.subplot(313)
	plt.title('Espectrograma de la Modulacion FM')
	plt.xlabel('Tiempo (s)')
	plt.ylabel('Frecuencia (Hz)')
	plt.grid(True)
	plt.specgram(fm, NFFT=1024,Fs=rate)    
	plt.show()


	##am a sus diferentes porcentajes
	plt.subplot(311)
	plt.subplots_adjust(hspace = 0.75)
	plt.title('Espectrograma de la Modulacion AM al 15%')
	plt.xlabel('Tiempo (s)')
	plt.ylabel('Frecuencia (Hz)')
	plt.grid(True)
	plt.specgram(am15, NFFT=1024,Fs=rate)    

	plt.subplot(312)
	plt.title('Espectrograma de la Modulacion AM al 100%')
	plt.xlabel('Tiempo (s)')
	plt.ylabel('Frecuencia (Hz)')
	plt.grid(True)
	plt.specgram(am100, NFFT=1024,Fs=rate)    

	plt.subplot(313)
	plt.title('Espectrograma de la Modulacion AM al 125%')
	plt.xlabel('Tiempo (s)')
	plt.ylabel('Frecuencia (Hz)')
	plt.grid(True)
	plt.specgram(am125, NFFT=1024,Fs=rate)    
	plt.show()


	##fm a sus diferentes porcentajes
	plt.subplot(311)
	plt.subplots_adjust(hspace = 0.75)
	plt.title('Espectrograma de la Modulacion FM al 15%')
	plt.xlabel('Tiempo (s)')
	plt.ylabel('Frecuencia (Hz)')
	plt.grid(True)
	plt.specgram(fm15, NFFT=1024,Fs=rate)    

	plt.subplot(312)
	plt.title('Espectrograma de la Modulacion FM al 100%')
	plt.xlabel('Tiempo (s)')
	plt.ylabel('Frecuencia (Hz)')
	plt.grid(True)
	plt.specgram(fm100, NFFT=1024,Fs=rate)    

	plt.subplot(313)
	plt.title('Espectrograma de la Modulacion FM al 125%')
	plt.xlabel('Tiempo (s)')
	plt.ylabel('Frecuencia (Hz)')
	plt.specgram(fm125, NFFT=1024,Fs=rate)    
	plt.grid(True)
	plt.show()


#### Demodulacion AM
	am_d = am * carrier
	tamano_modulada=len(am_d)
	transformada_modulada=fft(am_d)/tamano_modulada

	k = arange(tamano_modulada)
	T = tamano_modulada/200000
	frq = k/T

	for i in range(0,len(transformada_modulada)):
		if i<36900*T:
			transformada_modulada[i] = 0
		elif i>163000*T:
		    transformada_modulada[i] = 0


	inversa=fft(transformada_modulada)

	plt.plot(t2, inversa)
	plt.title("Señal AM Demodulada")
	plt.xlabel('Tiempo (s)')
	plt.ylabel('Amplitud (dB)')
	plt.grid(True)
	plt.show()

	am2 = np.interp(t1, t2, am_d)

	write("audio_dem.wav", rate, am2.astype(info.dtype))##se crea un audio para verificar que se mantiene el original sin mayores problemas


#XXXXXXXXXXXXXXXXXXXX Ruido Gaussiano XXXXXXXXXXXXXXXXXXXXXXXXXXXXX

def addNoise(N, SNR):
    # N Cantidad de datos
    # SNR razon señal ruido
    # Create some data with noise and a sinusoidal
    ruido = np.random.normal(0.0, 1.0/SNR, N)
    return ruido

def modular_ASK(x, bp):
	A1=1                    # Amplitud de la señal portadora para informacion 1
	A2=0                    # Amplitud de la señal portadora para informacion 0
	br=1/bp                 # frecuencia de bit
	f=br*10                 # frecuencia de carrier
	t2 = np.arange(bp/99, bp+bp/99, bp/99)                
	ss=len(t2)
	m=[]

	for i in range(0,len(x)):
	    if (x[i]==1):
	        y=A1*np.cos(2*np.pi*f*t2)
	    elif (x[i]==0):
	        y=A2*np.cos(2*np.pi*f*t2)
	    m = np.concatenate((m, y))
	return m, ss, f

def demodular_ASK(bp, ss, m, f):
	 ####DEmodulacion
	mn=[]
	n = ss

	while(n<=len(m)):
	  t = np.arange(bp/99, bp+bp/99, bp/99) 
	  y=np.cos(2*np.pi*f*t)                                        # señal portadora
	  mm=y*m[(n-(ss)):n]
	  t4 = np.arange(bp/99, bp+bp/99, bp/99) 
	  z=np.trapz(t4,mm)                                              # integracion
	  zz=round((2*z/bp))                                     
	  if(zz>0.75):                                 # nivel logica = (A1+A2)/2=0.75 
	    a=1
	  else:
	    a=0
	  mn = np.append(mn,a)
	  n += ss


	bit_demodulado=[]
	for n in range(0,len(mn)-1):
	    if mn[n]==1:
	       se=np.ones((100,), dtype=np.int)
	    elif mn[n]==0:
	        se=np.zeros((100,), dtype=np.int)
	    bit_demodulado = np.concatenate((bit_demodulado,se))

	return bit_demodulado, mn

def plotear_ASK(t1, bit, t3, m, t4, bit_demodulado, titulo):
	plt.subplot(3,1,1)
	plt.subplots_adjust(hspace = 0.75)
	plt.ylim(-0.2,1.2)
	plt.title("Señal Digital")
	plt.xlabel('Tiempo (s)')
	plt.ylabel('Información Bit')
	plt.plot(t1[0:8000],bit[0:8000])

	plt.subplot(3,1,2)
	plt.title(titulo)
	plt.xlabel('Tiempo (s)')
	plt.ylabel('Amplitud (dB)')
	plt.plot(t3[0:8000],m[0:8000])

	plt.subplot(3,1,3)
	plt.ylim(-0.2,1.2)
	plt.title("Señal Demodulada")
	plt.xlabel('Tiempo (s)')
	plt.ylabel('Información Bit')
	plt.plot(t4[0:8000],bit_demodulado[0:8000])
	plt.show()


def contar_errores(x, mn):
	cont_errores = 0

	for j in range(0, len(mn)-1):
	  if x[j] != mn[j]:
	    cont_errores += 1

	return cont_errores


##############################################################

def parte2():

	maximo = info.max() ##valor maximo de los datos
	cant_bits = len(bin(maximo)[2:])
	datos_bin = []
	for i in info:
	  binario = bin(i)[2:]
	  if(binario[0]== 'b'):
	    i = i*-1
	    binario = bin(i)[2:].zfill(cant_bits)
	    binario = '1' + binario
	  else:
	    binario = bin(i)[2:].zfill(cant_bits)
	    binario = '0' + binario
	  datos_bin.append(binario)
	cant_datos = int(len(datos_bin)/16) #cada elemento es de 16 bits
	data_acotado = datos_bin[0:cant_datos]


	info_digital = []

	for i in range(0,cant_datos):
	  for x in data_acotado[i]:
	    if x=='0':
	      info_digital.append(0)
	    elif x=='1':
	      info_digital.append(1)

	#print(len(info_digital))

	x=info_digital[0:10000]# Informacion binaria

	bp=0.000123112 # periodo por bit

	#### Ahora se reṕresenta la transmision de informacion binaria como una señal digital

	bit=[] 
	for n in range(0,len(x)):
	  if x[n]==1:
	    se=np.ones((100,), dtype=np.int)
	  elif x[n]==0:
	    se=np.zeros((100,), dtype=np.int)
	  bit = np.concatenate((bit,se))

	t1 = np.arange(bp/100,100*len(x)*(bp/100)+bp/100,bp/100)

	m, ss, f = modular_ASK(x, bp) # Modulador
	t3 = np.arange(bp/99, bp*len(x)+bp/99, bp/99)

	bit_demodulado, mn = demodular_ASK(bp, ss, m, f)
	t4 = np.arange(bp/100, 100*len(x)*(bp/100)+bp/100, bp/100)


	plotear_ASK(t1, bit, t3, m, t4, bit_demodulado, "Señal Modulada")

	list_SNR = []
	list_errores = []

	cont_errores = contar_errores(x, mn)
	list_SNR.append(0)
	list_errores.append(cont_errores/len(mn))


    #--------------------------------------------------------------------------#
	N = len(m)
	SNR = 50
	ruido = addNoise(N, SNR)

	m1 = m + ruido

	bit_demodulado, mn = demodular_ASK(bp, ss, m1, f)
	plotear_ASK(t1, bit, t3, m1, t4, bit_demodulado, "Señal Modulada con SNR 1/50")

	cont_errores = contar_errores(x, mn)
	list_SNR.append(1/SNR)
	list_errores.append(cont_errores/len(mn))

	#--------------------------------------------------------------------------#

	SNR = 10
	ruido = addNoise(N, SNR)

	m2 = m + ruido

	bit_demodulado, mn = demodular_ASK(bp, ss, m2, f)
	plotear_ASK(t1, bit, t3, m2, t4, bit_demodulado, "Señal Modulada con SNR 1/10")

	cont_errores = contar_errores(x, mn)
	list_SNR.append(1/SNR)
	list_errores.append(cont_errores/len(mn))	

	#--------------------------------------------------------------------------#

	SNR = 5
	ruido = addNoise(N, SNR)

	m3 = m + ruido

	bit_demodulado, mn = demodular_ASK(bp, ss, m3, f)
	plotear_ASK(t1, bit, t3, m3, t4, bit_demodulado, "Señal Modulada con SNR 1/5")

	cont_errores = contar_errores(x, mn)
	list_SNR.append(1/SNR)
	list_errores.append(cont_errores/len(mn))

	#--------------------------------------------------------------------------#

	SNR = 2
	ruido = addNoise(N, SNR)

	m4 = m + ruido

	bit_demodulado, mn = demodular_ASK(bp, ss, m4, f)
	plotear_ASK(t1, bit, t3, m4, t4, bit_demodulado, "Señal Modulada con SNR 1/2")

	cont_errores = contar_errores(x, mn)
	list_SNR.append(1/SNR)
	list_errores.append(cont_errores/len(mn))

	#--------------------------------------------------------------------------#

	SNR = 1
	ruido = addNoise(N, SNR)

	m5 = m + ruido

	bit_demodulado, mn = demodular_ASK(bp, ss, m5, f)
	plotear_ASK(t1, bit, t3, m5, t4, bit_demodulado, "Señal Modulada con SNR 1")

	cont_errores = contar_errores(x, mn)
	list_SNR.append(1/SNR)
	list_errores.append(cont_errores/len(mn))

	#--------------------------------------------------------------------------#
    
    
	#XXXXXXXXXXXXXXXXXXXXXX Contar Errores XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

	plt.plot(list_errores, list_SNR)
	plt.xlabel('Probabilidad de Error')
	plt.ylabel('SNR')
	plt.title("SNR vs Probabilidad de Error")
	plt.show()

	print("SNR:                  ",list_SNR)
	print("Probablidad de Error: ", list_errores)




#######MENU#########
salida=0##variable a utilizar para mantener el while
while (salida==0):
	print('MENU')
	print('1-Parte 1:Aplicaciones de modulacion AM,FM a diferentes porcentajes. Los respectivos espectros de frecuencia. Demodulador AM')
	print('2-Parte 2:Modulacion digital, demodular, simular un canal AWGN con 5 niveles de SNR y demodular esta señal con ruido ')
	print('3-Salir')

	opcion = input("Eleccion: ")
	opcion=int(opcion)

	if (opcion==1):
		parte1()
		print('////////////////////////////////////////////////')
		print('////////////////////////////////////////////////')

	if (opcion==2):

		parte2()
		print('////////////////////////////////////////////////')
		print('////////////////////////////////////////////////')

	if (opcion==3):
		salida=1
		print('////////////////////////////////////////////////')
		print('////////////////////////////////////////////////')
