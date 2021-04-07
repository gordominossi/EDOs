# EP2 - MAP3122 - Métodos Numéricos e Aplicações

########### Exercício 2.2 ##############
##### Resolução por Euler Implícito #####

# Nome: Gabriel Moreira Minossi  | NUSP:  9349346
# Nome: Vinicius Bueno de Moraes | NUSP: 10256432

import numpy as np  # Import de Bibliotecas
import math
import matplotlib.pyplot as plt

print()
print("MAP3122 - EP3")
print("Exercício 2.3 - Comparação entre os Métodos de Euler Impícito e Euler Explícito")
print()

#Declaração de Variáveis
ns = [250, 500, 1000, 2000, 4000]

for n in ns:
    
    h = 10/n
    ts = []

    j=0
    for i in range(n+1): #Define discretização
        ts.append(j*h)
        j = j+1
    j = 0

    x = 1.5
    y = 1.5
    coelhosExplicito = []
    raposasExplicito = []
    coelhosImplicito = []
    raposasImplicito = []
    ErroX = []
    ErroY = []

    ################## Resolução por Euler Implícito ########################

    for i in ts: #Método Euler Explícito
        if i == 0:
            coelhosExplicito.append(x)
            raposasExplicito.append(y)
        else:
            x = x + h * ((2/3) * x - (4/3) * x * y)
            y = y + h * (x * y - y)
            coelhosExplicito.append(x)
            raposasExplicito.append(y)

    ################## Resolução por Euler Implícito ########################

    x = 1.5
    y = 1.5        

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
            coelhosImplicito.append(x)
            raposasImplicito.append(y)
        else:
            u = u1
            for iteration in range(7):
                u1 = u1 - np.dot(np.linalg.inv(J(u)), G(u1, u))
            x = u1[0]
            y = u1[1]
            coelhosImplicito.append(x)
            raposasImplicito.append(y)

    i = 0
    for e in range(len(ts)):
        ErroX.append(coelhosImplicito [i] - coelhosExplicito[i])
        ErroY.append(raposasImplicito [i] - raposasExplicito[i])
        i = i + 1

    ##################### Plot do Gráfico ############################

    plt.figure(figsize=(12, 6))
    plt.suptitle("2.3 - Erro - Divergência da Solução por Euler Implícito e Explícito | n = %s" % n)
    plt.grid()
    plt.plot(ts, ErroX, "-b", label="Ex")
    plt.plot(ts, ErroY, "-r", label="Ey")
    plt.legend(loc="upper left")
    plt.ylabel("Erro (Euler Implícito - Euler Explícito)")
    plt.xlabel("t")
    plt.show()

print("O programa foi executado com sucesso e os gráficos para o Erro entre os Métodos exibidos")
print()
