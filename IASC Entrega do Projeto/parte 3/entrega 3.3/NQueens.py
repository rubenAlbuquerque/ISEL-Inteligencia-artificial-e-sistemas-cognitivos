import random as rnd
from SearchAlgorithms import SearchAlgorithms
import matplotlib.pyplot as plt
import numpy as np

class NQueens:
    # class para resolver o problema das N rainhas
    def __init__(self, state, printState: bool = True):
        self.state = state #[rnd.randint(1, N) for i in range(N)]

    def initialState(self):
        return self.state
    
    def randomNeighbor(self, state: list):
        randomState = state.copy()

        # Choose random index
        index = rnd.randint(0, len(state)-1)
        pos = randomState[index]

        # Move queen to a random position
        newPos = rnd.randint(1, len(state))
        while newPos == pos:
            newPos = rnd.randint(1, len(state))

        randomState[index] = newPos

        # def randomNeighbor(self, state):
        #     # devolve um vizinho aleatorio do estado atual
        #     neighbor = state.copy()
        #     neighbor[rnd.randint(0, self.N - 1)] = rnd.randint(1, self.N)
        #     return neighbor
        return randomState
    
    def stateValue(self, state):
        # devolve o valor do estado atual
        # o valor do estado e o numero de rainhas que se atacam
        # value = 0
        # for i in range(self.N):
        #     for j in range(i + 1, self.N):
        #         if state[i] == state[j] or abs(state[i] - state[j]) == abs(i - j):
        #             value += 1
        # return value~
    # def stateValue(self, state: list):
        # number of collisions
        value = 0
        for i in range(len(state) - 1):
            for j in range(i+1, len(state)):
                if state[j] == state[i] or abs(state[j] - state[i]) == j - i:
                    value += 1

        value *= -1
        return value  # returns negative value so that the best is closer to 0

    def bestNeighbor(self, state):
        # devolve o melhor vizinho do estado atual
        # bestNeighbor = state.copy()
        # bestValue = self.stateValue(state)
        # for i in range(self.N):
        #     for j in range(1, self.N + 1):
        #         neighbor = state.copy()
        #         neighbor[i] = j
        #         value = self.stateValue(neighbor)
        #         if value < bestValue:
        #             bestNeighbor = neighbor.copy()
        #             bestValue = value
    
        # return bestNeighbor
        bestState = state.copy()

        # Guarantee that returns a different state
        bestValue = -len(state)**2

        # Randomize order to find the best neighbor
        indexes = [i for i in range(len(state))]
        rnd.shuffle(indexes)

        for index in indexes:
            newState = state.copy()
            for pos in range(1, len(state)+1):
                if pos != state[index]:
                    # Move queen
                    newState[index] = pos

                    # If the new state is better than the current best
                    newValue = self.stateValue(newState)
                    if newValue > bestValue:
                        bestState = newState.copy()
                        bestValue = newValue

        return bestState

    def printState(self, dicionario: dict, msg: str = ''):
        
        fig = plt.figure(figsize=(8, 8))
        fig.subplots_adjust(hspace=0.3, wspace=0.2, top=0.9, bottom=0.1, left=0.1, right=0.9)

        for i, sub in  enumerate(dicionario.items()):
            i+=1
            plt.subplot(2, 2, i)
            nAttacks = self.stateValue(sub[1]) * -1
            plt.title(f"{sub[0]} \nAttacks: {nAttacks:.0f}")
            states = sub[1]

            # Build matrix to plot
            board = [[1 if j+1 == states[i] else 0 for j in range(len(states))] for i in range(len(states))]
            plt.imshow(board)
        plt.show()
    


if __name__ == '__main__':
    print_results = True
    n = 20

    # criar uma lista de rainhas random para o problema do N-rainhas 
    state = [rnd.randint(1, n) for i in range(n)]

    # criar um problema das N rainhas
    problem = NQueens(state)
    print_results = {"Initial board": state.copy()}

    # utilizar o algoritmo de procura para resolver o problema das N rainhas -stochasticHillClimbing
    result = SearchAlgorithms.stochasticHillClimbing(problem)
    print_results["Stochastic Hill Climbing"] = result

    # utilizar o algoritmo de procura para resolver o problema das N rainhas - hillClimbing random restart
    result = SearchAlgorithms.hillClimbingWithRandomRestart(problem)
    print_results["hill Climbing With Random Restart"] = result


    # utilizar o algoritmo de procura para resolver o problema das N rainhas -simulatedAnnealing
    result = SearchAlgorithms.simulatedAnnealing(problem, SearchAlgorithms.schedule)
    print_results["Simulated Annealing"] = result

    # print dos resultados do dicionario print_cities num plot
    if print_results:
        problem.printState(print_results)




