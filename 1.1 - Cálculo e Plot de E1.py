# EP2 - MAP3122 - Métodos Numéricos e Aplicações

################ Exercício 1.1 ##################
############ Cálculo e Plot de E1 ###############

# Nome: Gabriel Moreira Minossi  | NUSP:  9349346
# Nome: Vinicius Bueno de Moraes | NUSP: 10256432

import numpy as np  # Import de Bibliotecas
import math
import matplotlib.pyplot as plt

print()
print("MAP3122 - EP2")
print("Exercício 1.1 - Cálculo e Plot de E1")
print()

# Declaração da Matriz A
A = [[-2, -1, -1, -2],
     [+1, -2, +2, -1],
     [-1, -2, -2, -1],
     [+2, -1, +1, -2]]
# Declaração da condição inicial
x = [+1, +1, +1, -1]

n = int(input("Digite o valor de n: "))
# Cálculo de h (passo)
h = 2/n
xFinalRK4 = []
xFinalRK4.append(x)

ts = []
for k in range(n):
    ts.append(k * h)

# Cálculo da solução pelo método RK4
for t in ts:
    k1 = np.dot(A, x)
    k2 = np.dot(A, x + h * k1 / 2)
    k3 = np.dot(A, x + h * k2 / 2)
    k4 = np.dot(A, x + h * k3)
    x = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    xFinalRK4.append(x)

xFinalExplicito = []
xCopia = []
xIntermed = []
# Cálculo da Solução Explicita
for t in ts:
    xIntermed.append(math.exp(-t)*math.sin(t)+math.exp(-3*t)*math.cos(3*t))
    xIntermed.append(math.exp(-t)*math.cos(t)+math.exp(-3*t)*math.sin(3*t))
    xIntermed.append(-math.exp(-t)*math.sin(t)+math.exp(-3*t)*math.cos(3*t))
    xIntermed.append(-math.exp(-t)*math.cos(t)+math.exp(-3*t)*math.sin(3*t))
    xCopia = xIntermed.copy()
    xFinalExplicito.append(xCopia)
    xIntermed.clear()

Erro = []
# Cálculo do Erro
for k in range(n):
    E = 0
    for j in range(4):
        ECalc = abs(xFinalExplicito[k][j] - xFinalRK4[k][j])
        E = max(ECalc, E)
    Erro.append(E)


plt.title("Gráfico do Erro em função de t para passo igual = %s | n = %s" % (h, n))
plt.ylabel("Erro")
plt.xlabel("t")
plt.grid()
plt.plot(ts, Erro)
plt.show()

print()
print("O programa foi executado com sucesso para n = %s e o gráfico de E1 exibido" % n)
print("Para outro n execute o programa novamente")
print()
