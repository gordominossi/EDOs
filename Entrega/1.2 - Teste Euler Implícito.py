# EP2 - MAP3122 - Métodos Numéricos e Aplicações

########### Exercício 1.2 ##############
####### Teste Euler Implícito ##########

# Nome: Gabriel Moreira Minossi  | NUSP:  9349346
# Nome: Vinicius Bueno de Moraes | NUSP: 10256432

import numpy as np
import math
import matplotlib.pyplot as plt

print()
print("MAP3122 - EP2")
print("Exercício 1.2 - Teste de Euler Implícito")
print()

n = 5000
h = (3 - 1.1) / n
x = -8.79

ts = []
for k in range(n + 1):
    ts.append(1.1 + k * h)


def F(t, x):
    return 2 * t + (x - t ** 2) ** 2


def G(t1, x1, x):
    return x1 - h * F(t1, x1) - x


def J(t, x):
    return 1 - 2 * h * (x - t ** 2)


xFinalEulerImplícito = [x]
x1 = x
for t in ts[1:]:
    x = x1
    for iteration in range(7):
        x1 = x1 - J(t, x1) ** -1 * G(t, x1, x)
    xFinalEulerImplícito.append(x1)


xFinalExplicito = []
# Cálculo da Solução Explicita
for t in ts:
    xFinalExplicito.append(t ** 2 + 1 / (1 - t))

Erro = []
# Cálculo do Erro
for k in range(n + 1):
    Erro.append(abs(xFinalExplicito[k] - xFinalEulerImplícito[k]))


# Plot do Gráfico
plt.figure(figsize=(14,5))
plt.suptitle("1.2 - Comparação entre a Solução Explícita e o Método de Euler Implícito")

plt.subplot(1,3,1)
plt.title("Solução Explícita")
plt.plot(ts, xFinalExplicito, "-b")
plt.ylabel("Resultado")
plt.xlabel("t")
plt.grid()

plt.subplot(1,3,2)
plt.title("Método - Euler Implícito")
plt.plot(ts, xFinalEulerImplícito, "-b")
plt.ylabel("Resutado")
plt.xlabel("t")
plt.grid()

plt.subplot(1,3,3)
plt.title("Erro entre as Soluções ao Lado")
plt.plot(ts, Erro, "-r")
plt.ylabel("Erro")
plt.xlabel("t")
plt.grid()

plt.show()

print("O programa foi executado com sucesso para n = 5000 e os gráficos exibidos")
print()