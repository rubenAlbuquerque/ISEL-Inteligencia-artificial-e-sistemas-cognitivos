import numpy as np
import matplotlib.pyplot as plt
# from estado import *


class Estado:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __repr__(self):
        return f'({self.x}, {self.y})'

    def __str__(self) -> str:
        return f'({self.x}, {self.y})'

    def __eq__(self, other) -> bool:
        return self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))


class Wavefront:
    def __init__(self, nomeArquivo: str, mostrarGrafico: bool = True):
        self.mundo, self.s, self.alvo = self.carregarMundo(nomeArquivo)
        self.mostrarGrafico = mostrarGrafico
        self.valorMundo = self.propagarValor([self.alvo])
        self.path = []

    def carregarMundo(self, nomeArquivo: str):
        with open(nomeArquivo, "r") as arquivo:
            lines = arquivo.readlines()
            mundo: list[list[int]] = np.zeros((len(lines), len(lines[0].replace('\n', ''))), dtype=int)

            estadoInicial = Estado(0, 0)
            alvo = Estado(0, 0)

            m = 0
            for x in lines:
                n = 0
                for y in x[:-1]:
                    if y.__eq__('O'):
                        mundo[m][n] = -1
                    elif y.__eq__('A'):
                        mundo[m][n] = 2
                        alvo = Estado(n, m)
                    elif y.__eq__('>'):
                        # mundo[m][n] = 1
                        estadoInicial = Estado(n, m)
                    n += 1
                m += 1

            # print(mundo)
            # print(estadoInicial, 'Start')
            # print(alvo, 'End')
            return mundo, estadoInicial, alvo

    def propagarValor(self, objetivos, gain: int = 10):
        V = {}
        frenteDeOnda = []
        gama = len(self.mundo)/(len(self.mundo)+1)  # Gama proporcional oa mundo
        for o in objetivos:
            V[o] = gain
            frenteDeOnda.append(o)
        while len(frenteDeOnda) > 0:  # Enquanto houver estados na frente de onda
            s = frenteDeOnda.pop(0)
            for a in self.adjacentes(s):
                v = V[s] * gama  # Atenua o valor
                if v > V.get(a, -1):  # Caso haja um valor pior no estado adjacente, substitui e adiciona à frente de onda
                    V[a] = v
                    frenteDeOnda.append(a)
        return V

    def adjacentes(self, s: Estado):
        # Estados adjacentes apenas na vertical e horizontal evitando obstáculos
        adjacentes: list[Estado] = []
        if s.x > 0:
            e = Estado(s.x - 1, s.y)
            if self.mundo[e.y][e.x] != -1:
                adjacentes.append(e)
        if s.x < len(self.mundo[0]) - 1:
            e = Estado(s.x + 1, s.y)
            if self.mundo[e.y][e.x] != -1:
                adjacentes.append(e)
        if s.y > 0:
            e = Estado(s.x, s.y - 1)
            if self.mundo[e.y][e.x] != -1:
                adjacentes.append(e)
        if s.y < len(self.mundo) - 1:
            e = Estado(s.x, s.y + 1)
            if self.mundo[e.y][e.x] != -1:
                adjacentes.append(e)
        return adjacentes

    def getPath(self, s: Estado):
        # Retorna os estados no caminho do estado inicial até o alvo
        self.path = []
        while s != self.alvo:
            self.path.append(s)
            sn = max(self.adjacentes(s), key=lambda s: self.valorMundo[s])
            s = sn
        return self.path

    def showPath(self, path = []):
        mundo = [[x for x in y] for y in self.mundo]
        for v in self.valorMundo:
            mundo[v.y][v.x] = self.valorMundo[v]

        for s in path:
            mundo[s.y][s.x] = -5

        plt.imshow(mundo)
        plt.show()


if __name__ == '__main__':
    w = Wavefront("amb2.txt")  # Carrega o mundo, colocando o agente de volta ao início
    path = w.getPath(w.s)
    w.showPath(path)
