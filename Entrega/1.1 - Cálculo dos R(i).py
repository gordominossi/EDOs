#EP2 - MAP3122 - Métodos Numéricos e Aplicações

############ Exercício 1.1 ###############
########## Cálculo dos R(i) ##############

# Nome: Gabriel Moreira Minossi  | NUSP:  9349346
# Nome: Vinicius Bueno de Moraes | NUSP: 10256432

import numpy as np #Import de Bibliotecas
import math
import matplotlib.pyplot as plt

print()
print("MAP3122 - EP2")
print("Exercício 1.1 - Cálculo dos R(i)")

A = [[-2, -1, -1, -2], [1, -2, 2, -1], [-1, -2, -2, -1], [2, -1, 1, -2]] #Declaração da Matriz A
ns = [20, 40, 80, 160, 320, 640] #Valores usados para R
rs = []

for i in ns: #Cálculo dos Resultados
    n = i
    h = 2/n #Cálculo de h (passo)
    x = [1,1,1,-1]
    xFinalRK4=[x]
    
    for i in range(n): #Cálculo da solução pelo método RK4
        k1 = h * np.dot(A, x)
        k2 = h * np.dot(A, x+(k1/2))
        k3 = h * np.dot(A, x+(k2/2))
        k4 = h * np.dot(A, x+k3)
        k = (k1+2*k2+2*k3+k4)/6
        x = x + k
        xFinalRK4.append(x)
    k = 0

    ts = []
    j = 0
    for i in range(n):
        ts.append(j*h)
        j = j + 1
    j = 0

    xFinalExplicito = []
    xCopia = []
    xIntermed = []

    for t in ts: #Cálculo da Solução Explicita
        xIntermed.append(math.exp(-t)*math.sin(t)+math.exp(-3*t)*math.cos(3*t))
        xIntermed.append(math.exp(-t)*math.cos(t)+math.exp(-3*t)*math.sin(3*t))
        xIntermed.append(-math.exp(-t)*math.sin(t)+math.exp(-3*t)*math.cos(3*t))
        xIntermed.append(-math.exp(-t)*math.cos(t)+math.exp(-3*t)*math.sin(3*t))
        xCopia = xIntermed.copy()
        xFinalExplicito.append(xCopia)
        xIntermed.clear()

    Erro = []

    while k < n: #Cálculo do Erro
        j = 0
        E = 0
        while j < 4:  
            ECalc = abs(xFinalExplicito[k][j] - xFinalRK4[k][j])
            if ECalc > E:
                E = ECalc
            else: 
                E = E
            j = j + 1
        Erro.append(E)
        k = k + 1
    j = 0
    k = 0

    rs.append(max(Erro))

print()
while k < 5: #Saída e Resultado do código, com os Ris
    R = rs[k]/rs[k+1]
    print ("O valor de R%s é: %s" % (k+1, R))
    k = k + 1

print() #Final
print("O programa foi executado com sucesso e os valores de R(i) exibidos")
print()