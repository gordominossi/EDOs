# EP2 - MAP3122 - Métodos Numéricos e Aplicações

########### Exercício 2.2 ##############
##### Resolução por Euler Implícito #####

# Nome: Gabriel Moreira Minossi  | NUSP:  9349346
# Nome: Vinicius Bueno de Moraes | NUSP: 10256432

import numpy as np  # Import de Bibliotecas
import math
import matplotlib.pyplot as plt

print()
print("MAP3122 - EP2")
print("Exercício 2.2 - Resolução por Euler Implícito")
print()

h = 10/6000 #Declaração de Variáveis
n = 6000
x = 1.5
y = 1.5
ts = []
coelhos = []
raposas = []

for k in range(n+1): #Define o tempo em n intervalos
    ts.append(k*h)


def J(u): #Define o cálculo do Jacobiano
    x = u[0]
    y = u[1]
    return np.array([[1 - h * (2 - 4 * y) / 3, h * x * (4 / 3)],
                     [- h * y,                 1 - h * (x - 1)]])


def F(u): #Define a função f
    x = u[0]
    y = u[1]
    return np.array([(2 * x - 4 * x * y) / 3, x * y - y])


def G(u1, u): #Define a Função G
    return u1 - h * F(u1) - u

u1 = [x, y] #Cálculo do Método de Euler Implícito
for t in ts:
    if t == 0:
        coelhos.append(x)
        raposas.append(y)
    else:
        u = u1
        for iteration in range(7):
            u1 = u1 - np.dot(np.linalg.inv(J(u)), G(u1, u))
        x = u1[0]
        y = u1[1]
        coelhos.append(x)
        raposas.append(y)

plt.figure(figsize=(12, 6))  # Plot do Gráfico
plt.suptitle("2.2 - Resolução por Euler Implícito")
plt.subplot(1, 2, 1)
plt.grid()
plt.plot(coelhos, raposas, "-b")
plt.title("Retrato de Fase (Coelhos, Raposas)")
plt.ylabel("Raposas")
plt.xlabel("Coelhos")
plt.subplot(1, 2, 2)
plt.grid()
plt.plot(ts, coelhos, "-b", label="Coelhos")
plt.plot(ts, raposas, "-r", label="Raposas")
plt.title("Evolução das populações com o tempo \n para h ≅ %s | n = %s" %
          (round(h, 5), n))
plt.legend(loc="upper left")
plt.ylabel("População")
plt.xlabel("t")
plt.show()

print("O programa foi executado com sucesso para n = 6000 e os gráficos exibidos")
print()
