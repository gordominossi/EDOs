#EP2 - MAP3122 - Métodos Numéricos e Aplicações

########### Exercício 2.1 ##############
##### Resolução por Euler Explícito #####

# Nome: Gabriel Moreira Minossi  | NUSP:  9349346
# Nome: Vinicius Bueno de Moraes | NUSP: 10256432

import numpy as np #Import de Bibliotecas
import math
import matplotlib.pyplot as plt

print()
print("MAP3122 - EP2")
print("Exercício 2.1 - Resolução por Euler Explícito")
print()

h = 10/6000 #Declaração de Variáveis
n = 6000
x = 1.5
y = 1.5
ts = []
coelhos = []
raposas = []

j=0
for i in range(n+1):
    ts.append(j*h)
    j = j+1

for i in ts: #Método Euler Explícito
    if i == 0:
        coelhos.append(x)
        raposas.append(y)
    else:
        x = x + h * ((2/3) * x - (4/3) * x * y)
        y = y + h * (x * y - y)
        coelhos.append(x)
        raposas.append(y)

plt.figure(figsize=(12, 6)) #Plot do Gráfico
plt.suptitle("2.1 - Resolução por Euler Explícito")
plt.subplot(1, 2, 1)
plt.grid()
plt.plot(coelhos, raposas, "-b")
plt.title("Retrato de Fase (Coelhos, Raposas)")
plt.ylabel("Raposas")
plt.xlabel("Coelhos")
plt.subplot(1, 2, 2)
plt.grid()
plt.plot(ts, coelhos, "-b", label= "Coelhos")
plt.plot(ts, raposas, "-r", label = "Raposas")
plt.title("Evolução das populações com o tempo \n para h ≅ %s | n = %s" %(round(h, 5), n))
plt.legend(loc="upper left")
plt.ylabel("População")
plt.xlabel("t")
plt.show()

print("O programa foi executado com sucesso para n = 6000 e os gráficos exibidos")
print()
