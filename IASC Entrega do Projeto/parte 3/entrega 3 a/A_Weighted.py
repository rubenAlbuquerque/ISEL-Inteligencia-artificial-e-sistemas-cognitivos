import numpy as np
import matplotlib.pyplot as plt


class AStar:
    def __init__(self, world, start, end, file, heuristic="manhattan"):
        self.world = world
        self.start = start
        self.end = end
        self.file = file
        self.width, self.height = world.shape

        if heuristic == "manhattan": 
            self.heuristic = self.manhattan_distance
        elif heuristic == "euclidean":
            self.heuristic = self.euclidean_distance
    
    # carregar o world de um ficheiro para uma matriz
    def load_world(self):
        print(self.file)
        with open(self.file, "r") as arquivo:
            lines = arquivo.readlines()
            mundo: list[list[int]] = np.zeros((len(lines), len(lines[0].replace('\n', ''))), dtype=int)
            m = 0
            for x in lines:
                n = 0
                for y in x[:-1]:
                    if y.__eq__('O'):
                        mundo[m][n] = -1
                    elif y.__eq__('A'):
                        mundo[m][n] = 2
                    elif y.__eq__('>'):
                        mundo[m][n] = 1
                    n += 1
                m += 1
        return mundo
    
    def manhattan_distance(self, a, b):
        # Use the Manhattan distance as heuristic
        x1, y1 = a
        x2, y2 = b
        return abs(x1 - x2) + abs(y1 - y2)
    
    def euclidean_distance(self, a, b):
        # Use the Euclidean distance as heuristic
        x1, y1 = a
        x2, y2 = b
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    
    def neighbors(self, cell):
        # Possible next states and their costs
        x, y = cell
        states = []
        if x > 0 and self.world[x-1][y] != "O":
            states.append((x-1, y))
        if x < self.width-1 and self.world[x+1][y] != "O":
            states.append((x+1, y))
        if y > 0 and self.world[x][y-1] != "O":
            states.append((x, y-1))
        if y < self.height-1 and self.world[x][y+1] != "O":
            states.append((x, y+1))
        return states

    def solve(self):
        open_set = set([self.start])
        closed_set = set()
        came_from = {}
        g_score = {self.start: 0}
        f_score = {self.start: self.heuristic(self.start, self.end)}

        while open_set:
            current = None
            current_f_score = None

            # Get the cell with the lowest f_score
            for cell in open_set:
                if current is None or f_score[cell] < current_f_score:
                    current = cell
                    current_f_score = f_score[cell]

            # If we have reached the end, reconstruct the path
            if current == self.end:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]

            open_set.remove(current)
            closed_set.add(current)

            for neighbor in self.neighbors(current):
                if neighbor in closed_set:
                    continue
                tentative_g_score = g_score[current] + 1
                if neighbor not in open_set:
                    open_set.add(neighbor)
                elif tentative_g_score >= g_score[neighbor]:
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, self.end)
        return None

    def print_path(self, path):
        w = self.load_world()
        plt.figure(figsize=(8, 8))
        plt.title("AStar")
        # plt.grid(True)
        plt.xticks(np.arange(0, self.width, 1))
        plt.yticks(np.arange(0, self.height, 1))
        plt.imshow(w)
        x, y = zip(*path)
        plt.plot(y, x, '-o', color='red')
        plt.show()



if __name__ == "__main__":

    # Read the .txt file
    filename = "amb1.txt"
    with open(filename, "r") as file:
        data = file.read()

    # Create a 2D array with the data from the file
    world = np.array([list(line) for line in data.split("\n") if line])
    start = np.where(world == ">")
    end = np.where(world == "A")


    agent = AStar(world, (start[0][0], start[1][0]), (end[0][0], end[1][0]), file=filename)

    # Run the algorithm and get the path
    path = agent.solve()

    # Print the path
    # print(path)

    if path:
        agent.print_path(path)
    else:
        print("No path found.")
