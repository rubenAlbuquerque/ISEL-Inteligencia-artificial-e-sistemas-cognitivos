import random as rnd
from SearchAlgorithms import SearchAlgorithms
import matplotlib.pyplot as plt
import numpy as np

class TravellingSalesman:
    def __init__(self, cities, initial_temp=100, alpha=0.99):
        self.cities = cities
        self.initial_temp = initial_temp
        self.alpha = alpha
        
    def initialState(self):
        return rnd.sample(self.cities, len(self.cities))
    
    def randomNeighbor(self, state):
        neighbor = state.copy()
        i = rnd.randint(0, len(neighbor) - 1)
        j = rnd.randint(0, len(neighbor) - 1)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        return neighbor
    
    def bestNeighbor(self, state):
        best = state
        bestValue = self.stateValue(best)        
        for neighbor in self.neighbors(state):
            value = self.stateValue(neighbor)
            if value > bestValue:
                best = neighbor
                bestValue = value
        return best
    
    def neighbors(self, state):
        neighbors = []
        for i in range(len(state) - 1):
            for j in range(i + 1, len(state)):
                neighbor = state.copy()
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighbors.append(neighbor)
        return neighbors

    def stateValue(self, state):
        distance = 0
        for i in range(len(state) - 1):
            distance += self.distance(state[i], state[i + 1])
        return -distance

    def distance(self, city1, city2):
        return ((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2) ** 0.5
    
    def printState(self, dicionario):
        fig = plt.figure(figsize=(9, 9))
        fig.subplots_adjust(hspace=0.3, wspace=0.2, top=0.9, bottom=0.1, left=0.1, right=0.9)
        for i, sub in  enumerate(dicionario.items()):
            i+=1
            stateinicial = sub[1][0]
            plt.subplot(2, 2, i)
            distance = self.stateValue(sub[1]) * -1
            plt.title(f"{sub[0]} \nDistance: {distance:.1f}")
            plt.xlim(-5, 110)
            plt.ylim(0, 110)
            # plt.grid(True)
            sub[1].append(stateinicial)
            states = sub[1]
            if i == 1:
                plt.plot(*zip(*states), 'o')
            else:
                plt.plot(*zip(*states), 'o-')
            
            for i in range(len(states)-1):
                plt.annotate(f'  {i+1}', (states[i][0], states[i][1]))
        
        plt.show()




if __name__ == '__main__':
    print_results = True
    n = 24
    w = 100
    # criar uma lista de coordenadas random para o problema do caixeiro viajante
    cities = [(rnd.randint(0, w), rnd.randint(0, w)) for i in range(n)]
    
    # criar o problema do caixeiro viajante
    problem = TravellingSalesman(cities)

    # guardar as coordenadas das cidades num dicionario para fazer plot da evolucao no fim
    print_cities = {"Cidades": cities.copy()}
    
    # utilizar o algoritmo de procura para resolver o problema do caixeiro viajante - hill climbing 
    result = SearchAlgorithms.stochasticHillClimbing(problem)
    print_cities["Stochastic Hill Climbing"] = result

    # utilizar o algoritmo de procura para resolver o problema do caixeiro viajante - hill climbing random restart
    result = SearchAlgorithms.hillClimbingWithRandomRestart(problem)
    print_cities["hill Climbing With Random Restart"] = result

    # utilizar o algoritmo de procura para resolver o problema do caixeiro viajante - simulated Annealing
    result = SearchAlgorithms.simulatedAnnealing(problem, SearchAlgorithms.schedule )
    print_cities["Simulated Annealing"] = result

    # print dos resultados do dicionario print_cities num plot
    if print_results:
        problem.printState(print_cities)




