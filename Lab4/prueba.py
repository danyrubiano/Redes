### sacado de https://www.mathworks.com/matlabcentral/fileexchange/44820-matlab-code-for-ask-modulation-and-demodulation/content/ASK_mod_demod_salim.m

# Import the plotting library
import numpy as np
from matplotlib import pyplot as plt
import scipy.integrate as integrate
from scipy.io.wavfile import read,write

rate, info = read("handel.wav")
print(len(info))

maximo = info.max()
cant_bits = len(bin(maximo)[2:])
datos = []
for i in info:
  binario = bin(i)[2:]
  if(binario[0]== 'b'):
    i = i*-1
    binario = bin(i)[2:].zfill(cant_bits)
    binario = '1' + binario
  else:
    binario = bin(i)[2:].zfill(cant_bits)
    binario = '0' + binario
  datos.append(binario)
cant_datos = int(len(datos)/16) #cada elemento es de 16 bits
data_acotado = datos[0:cant_datos]

#print(data_acotado)
#print(cant_datos)

info_digital = []

for i in range(0,cant_datos):
  for x in data_acotado[i]:
    if x=='0':
      info_digital.append(0)
    elif x=='1':
      info_digital.append(1)

print(len(info_digital))

x=info_digital[0:10000]                                    # Binary Information

##### no se que bit periodo ponerle, estaba pensando en ponerle 9seg/cantidad de bits, osea, 9/73104, porque cada bit va a pasar segun ese tiempo
bp=0.0001                                                 # bit period

#XX representation of transmitting binary information as digital signal XXX

bit=[] 
for n in range(0,len(x)):
  if x[n]==1:
    se=np.ones((100,), dtype=np.int)
  elif x[n]==0:
    se=np.zeros((100,), dtype=np.int)
  bit = np.concatenate((bit,se))

t1 = np.arange(bp/100,100*len(x)*(bp/100)+bp/100,bp/100)

plt.subplot(3,1,1)
plt.ylim(-0.2,1.2)
plt.plot(t1[0:8000],bit[0:8000])


#XXXXXXXXXXXXXXXXXXXXXXX Binary-ASK modulation XXXXXXXXXXXXXXXXXXXXXXXXXXX#
A1=1                    # Amplitude of carrier signal for information 1
A2=0                       # Amplitude of carrier signal for information 0
br=1/bp                                                         # bit rate
f=br*10                                                 # carrier frequency 
t2 = np.arange(bp/99, bp+bp/99, bp/99)                
ss=len(t2)
m=[]

for i in range(0,len(x)):
    if (x[i]==1):
        y=A1*np.cos(2*np.pi*f*t2)
    elif (x[i]==0):
        y=A2*np.cos(2*np.pi*f*t2)
    m = np.concatenate((m, y))


#XXXXXXXXXXXXXXXXXXXX Ruido Gaussiano XXXXXXXXXXXXXXXXXXXXXXXXXXXXX

def addNoise(N, SNR, si_no):
  # N Cantidad de datos
  #SNR razon señal ruido
  if si_no == 'true':
    # Create some data with noise and a sinusoidal
    out = 1
    ruido = np.random.normal(0.0, 1.0/SNR, N)
  elif si_no == 'false':
    out = 0
  return out, ruido

#XXXXXXXXXXXXXXXXXX Añadiendo ruido a la señal modulada
N = len(m)
out, ruido = addNoise(N, 8, 'true')

if out == 1:
  m = m + ruido

t3 = np.arange(bp/99, bp*len(x)+bp/99, bp/99)

plt.subplot(3,1,2)
plt.plot(t3[0:8000],m[0:8000])


#XXXXXXXXXXXXXXXXXXXX Binary ASK demodulation XXXXXXXXXXXXXXXXXXXXXXXXXXXXX

mn=[]
n = ss

while(n<=len(m)):
  t = np.arange(bp/99, bp+bp/99, bp/99) 
  y=np.cos(2*np.pi*f*t)                                        # carrier siignal 
  mm=y*m[(n-(ss)):n]
  t4 = np.arange(bp/99, bp+bp/99, bp/99) 
  z=np.trapz(t4,mm)                                              # intregation 
  zz=round((2*z/bp))                                     
  if(zz>0.75):                                 # logic level = (A1+A2)/2=0.75 
    a=1
  else:
    a=0
  mn = np.append(mn,a)
  n += ss



#XXXXX Representation of binary information as digital signal which achived 
#after ASK demodulation XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
bit_demodulado=[]
for n in range(0,len(mn)-1):
    if mn[n]==1:
       se=np.ones((100,), dtype=np.int)
    elif mn[n]==0:
        se=np.zeros((100,), dtype=np.int)
    bit_demodulado = np.concatenate((bit_demodulado,se))

t4 = np.arange(bp/100, 100*len(x)*(bp/100)+bp/100, bp/100)

plt.subplot(3,1,3)
plt.ylim(-0.2,1.2)
plt.plot(t4[0:8000],bit_demodulado[0:8000])
plt.show()

t3 = np.arange(bp/99, bp*len(x)+bp/99, bp/99)

plt.plot(t3[0:8000],m[0:8000])
plt.show()


#XXXXXXXXXXXXXXXXXXXXXX Contar Errores XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
cont_errores = 0

#print(len(x))
#print(len(mn))

for j in range(0, len(mn)-1):
  if x[j] != mn[j]:
    cont_errores += 1

print(cont_errores)



