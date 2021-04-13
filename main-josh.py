import random
import time

from queue import PriorityQueue, Queue
from typing import List, Tuple


#This dictionary holds the terrain values for each type
terrainDict = {
    1: 0.1, #flat
    2: 0.3, #hilly
    3: 0.7, #forested
    4: 0.9  #maze of caves
}



#This function generates our landscape
def generate_environment(agent):
    landscape = [[0]*agent.dim]*agent.dim
    
    # Randomly assign the terrains
    for i in range(agent.dim):
        for j in range(agent.dim):
            landscape[i][j] = terrainDict[random.randint(1, 4)]
    return landscape

#The initial choice of cell at the beginning of the agent's path
def pick_cell(agent):
    
    i = random.randint(0, agent.dim - 1)
    j = random.randint(0, agent.dim - 1)
    
    return i, j

class Agent:

    def __init__(self, d):

        self.dim: int = d

        self.environment: List[List[float]] = generate_environment(self)

        self.beliefStates: List[List[float]] = [[1 / (self.dim * self.dim) for _ in range(self.dim)] for _ in
                                                    range(self.dim)]
                                                    
        self.probQueue: PriorityQueue[Tuple[float, Tuple[int, int]]] = PriorityQueue()
        for i in range(d):
            for j in range(d):
                self.probQueue.put((-1*self.beliefStates[i][j], (i,j)))

        self.target = pick_cell(self)

        return

#This function calculates the Manhattan distanc between two given cells
def calcDist(current, previous):
    
    #formula for Manhattan distance
    distance = abs(current[0] - previous[0]) + abs(current[1] - previous[1])
    return distance


def updateAgent(agent, observed_cell, probability):
    newQueue = PriorityQueue()
    total = 0
    while not agent.probQueue.empty():
        current = agent.probQueue.get()
        false_negative_rate = agent.environment[observed_cell[0]][observed_cell[1]]
        prior_belief = abs(agent.beliefStates[current[1][0]][current[1][1]])

        # Take the absolute value of the previously observed cell
        probability = abs(probability)

        # Update belief states
        if current[1] == observed_cell:
            # Probability that the target is in the cell given observations and failure
            agent.beliefStates[current[1][0]][current[1][1]] = false_negative_rate * probability / (1 - probability + false_negative_rate)
        else:
            # Probability that the target is in the cell given observations
            agent.beliefStates[current[1][0]][current[1][1]] = prior_belief / (1 - probability + false_negative_rate)
        total += agent.beliefStates[current[1][0]][current[1][1]]

    # Update probQueue
    for i in range(agent.dim):
        for j in range(agent.dim):
            agent.beliefStates[i][j] = agent.beliefStates[i][j] *  (1/total)
            priority = -1 * (agent.beliefStates[i][j])
            newQueue.put((priority, (i, j)))
    agent.probQueue = newQueue
    return

def updateAgentImproved(agent, observed_cell, probability):
    newQueue = PriorityQueue()
    total = 0
    while not agent.probQueue.empty():
        current = agent.probQueue.get()
        false_negative_rate = agent.environment[observed_cell[0]][observed_cell[1]]
        prior_belief = abs(agent.beliefStates[current[1][0]][current[1][1]])

        # Take the absolute value of the previously observed cell
        probability = abs(probability)

        # Update belief states
        if current[1] == observed_cell:
            # Probability that the target is in the cell given observations and failure
            agent.beliefStates[current[1][0]][current[1][1]] = false_negative_rate * probability / (1 - probability + false_negative_rate)
        else:
            # Probability that the target is in the cell given observations
            agent.beliefStates[current[1][0]][current[1][1]] = prior_belief / (1 - probability + false_negative_rate)
        total += agent.beliefStates[current[1][0]][current[1][1]]

    # Update probQueue
    for i in range(agent.dim):
        for j in range(agent.dim):
            agent.beliefStates[i][j] = agent.beliefStates[i][j] * (1/total)
            priority = (calcDist(observed_cell, (i, j)) + 1) / agent.beliefStates[i][j]
            newQueue.put((priority, (i, j)))
    agent.probQueue = newQueue
    return

#Checks the cell for target
def check_for_target(agent, current):

    if current == agent.target:
        false_negative_rate = random.uniform(0, 1)

        # Check for false negative
        if false_negative_rate > agent.environment[current[0]][current[1]]:
            return True

    return False

#Returns average terrain value of a given cell's neighbors
def avg_neighbors(agent, i, j):

    neighbor_count = 0
    total = 0

    # Add up the landscape scores of its neighbors
    for x in range(-1, 2):

        for y in range(-1, 2):

            if x != 0 or y != 0:

                if 0 <= i + x < agent.dim and 0 <= j + y < agent.dim:

                    total += agent.environment[i + x][j + y]

                    #Increment neighbor count
                    neighbor_count += 1

    # Get the average score of terrain values
    neighbor_avg = total / neighbor_count
    return neighbor_avg

#Picks the optimal starting cell based on flatness and result of avg_neighbors
def improved_start(agent):
    
    temp_min = 1
    start_cell = None

    #Iterate through environment and select flat cell, calculate average neighbor terrain value accordingly
    for i in range(agent.dim):
        for j in range(agent.dim):

            if agent.environment[i][j] == 0.1:

                #Calculate average terrain neighbor value
                average = avg_neighbors(agent, i, j)

                #Update min terrain neighbor value
                if average < temp_min:
                    temp_min = average
                    start_cell = (i, j)

    #return optimal starting cell
    return start_cell

def BaseAgent1(agent):
    target_found = False
    previous = None

    cells_searched = 1
    cells_moved = 0
    current = pick_cell(agent)
    target_found = check_for_target(agent, current)

    updateAgent(agent, current, -1 / agent.dim ** 2)
    previous = (0, current)
    while not target_found:
        # Takes the top cell and put it back
        current = agent.probQueue.get()
        agent.probQueue.put(current)
        
        # Determines new curr cell after handling ties
        
        # Create a list to track cells removed from the queue in this step and add initial cell to the List
        temp = [current]

        while not agent.probQueue.empty():
            # Remove new cells from the queue until we find one that doesn't match the current probability
            popped = agent.probQueue.get()
            if popped[0] != current[0]:
                # If probability didn't match, put cell back in queue
                agent.probQueue.put(popped)
                break
            else:
                # If probability did match, add cell to list of temp
                temp.append(popped)

        # Initialize a minimum distance variable and find the cell with minimum distance from the center
        minDistance = agent.dim * 2
        minDistance_cell = None
        for cell in temp:
            if calcDist(cell[1], previous[1]) < minDistance:
                minDistance = calcDist(cell[1], previous[1])
                minDistance_cell = cell[1]

        # Put all cells back except for minDistance_cell
        chosen_cell = None
        for cell in temp:
            if cell[1] == minDistance_cell:
                chosen_cell = cell
            else:
                agent.probQueue.put(cell)

        current = chosen_cell
        agent.probQueue.put(current)
        # Calculate number of actions moving to curr cell would take
        cells_moved += calcDist(current[1], previous[1])
        cells_searched += 1
        target_found = check_for_target(agent, current[1])
        updateAgent(agent, current[1], agent.beliefStates[current[1][0]][current[1][1]])
        previous = current
    return cells_searched, cells_moved, cells_searched + cells_moved
    
def BaseAgent2(agent):
    # Assigns initial probability values based on likelihood target is seen
    newQueue = PriorityQueue()
    total = 0

    for i in range(agent.dim):
        for j in range(agent.dim):
            agent.beliefStates[i][j] *= (1 - agent.environment[i][j])
            total += agent.beliefStates[i][j]

    # Normalize probabilities
    scaling_factor = 1 / total
    for i in range(agent.dim):
        for j in range(agent.dim):
            agent.beliefStates[i][j] = agent.beliefStates[i][j] * scaling_factor
            newQueue.put((-1 * agent.beliefStates[i][j], (i, j)))
    agent.probQueue = newQueue

    target_found = False
    previous = None

    cells_searched = 1
    cells_moved = 0

    # Takes the top cell and put it back
    current = agent.probQueue.get()
    agent.probQueue.put(current)

    target_found = check_for_target(agent, current[1])
    updateAgent(agent, current[1], agent.beliefStates[current[1][0]][current[1][1]])

    # Update prev cell
    previous = current
    while not target_found:
        # Takes the top cell and put it back
        current = agent.probQueue.get()
        agent.probQueue.put(current)
        # Determines new curr cell after handling ties
        
        # Create a list to track cells removed from the queue in this step and add initial cell to the List
        temp = [current]

        while not agent.probQueue.empty():
            # Remove new cells from the queue until we find one that doesn't match the current probability
            popped = agent.probQueue.get()
            if popped[0] != current[0]:
                # If probability didn't match, put cell back in queue
                agent.probQueue.put(popped)
                break
            else:
                # If probability did match, add cell to list of temp
                temp.append(popped)

        # Initialize a minimum distance variable and find the cell with minimum distance from the center
        minDistance = agent.dim * 2
        minDistance_cell = None
        for cell in temp:
            if calcDist(cell[1], previous[1]) < minDistance:
                minDistance = calcDist(cell[1], previous[1])
                minDistance_cell = cell[1]

        # Put all cells back except for minDistance_cell
        chosen_cell = None
        for cell in temp:
            if cell[1] == minDistance_cell:
                chosen_cell = cell
            else:
                agent.probQueue.put(cell)

        current = chosen_cell
        agent.probQueue.put(current)
        # Calculate number of actions moving to curr cell would take
        cells_moved += calcDist(current[1], previous[1])
        cells_searched += 1
        target_found = check_for_target(agent, current[1])
        updateAgent(agent, current[1], agent.beliefStates[current[1][0]][current[1][1]])
        previous = current
    return cells_searched, cells_moved, cells_searched + cells_moved

def Rule1(agent):
    target_found = False
    previous = None

    cells_searched = 1
    cells_moved = 0
    current = pick_cell(agent)
    target_found = check_for_target(agent, current)

    updateAgent(agent, current, -1 / agent.dim ** 2)
    previous = (0, current)    
    while not target_found:
        # Remove top cell
        current = agent.probQueue.get()
        agent.probQueue.put(current)

        cells_searched += 1
        target_found = check_for_target(agent, current[1])
        updateAgent(agent, current[1], current[0])
    return cells_searched, cells_moved, cells_searched + cells_moved
 
def Rule2(agent):
    # Assigns initial probability values based on likelihood target is seen
    newQueue = PriorityQueue()
    total = 0
    for i in range(agent.dim):
        for j in range(agent.dim):
            agent.beliefStates[i][j] *= (1 - agent.environment[i][j])
            total += agent.beliefStates[i][j]
    # Normalize probabilities
    scaling_factor = 1 / total
    for i in range(agent.dim):
        for j in range(agent.dim):
            agent.beliefStates[i][j] = agent.beliefStates[i][j] * scaling_factor
            newQueue.put((-1 * agent.beliefStates[i][j], (i, j)))
    agent.probQueue = newQueue

    target_found = False
    previous = None
    cells_searched = 1
    cells_moved = 0
    # Takes the top cell and put it back
    current = agent.probQueue.get()
    agent.probQueue.put(current)

    target_found = check_for_target(agent, current[1])
    updateAgent(agent, current[1], agent.beliefStates[current[1][0]][current[1][1]])

    # Update prev cell
    previous = current


    while not target_found:
        # Remove top cell
        current = agent.probQueue.get()
        agent.probQueue.put(current)

        cells_searched += 1
        target_found = check_for_target(agent, current[1])
        updateAgent(agent, current[1], current[0])

def ImprovedAgent(agent):
    target_found = False
    previous = None

    cells_searched = 1
    cells_moved = 0
    current = improved_start(agent)

    # There was no flat terrain in the board, but this would only happen with a 1x1 or a 2x2
    if current == None:
        current = pick_cell(agent)

    target_found = check_for_target(agent, current)
    updateAgentImproved(agent, current, -1 / agent.dim ** 2)
    previous = (0, current)

    while not target_found:
        # Takes the top cell and put it back
        current = agent.probQueue.get()
        agent.probQueue.put(current)

        # Determines new curr cell after handling ties

        # Create a list to track cells removed from the queue in this step and add initial cell to the List
        temp = [current]

        while not agent.probQueue.empty():
            # Remove new cells from the queue until we find one that doesn't match the current probability
            popped = agent.probQueue.get()
            if popped[0] != current[0]:
                # If probability didn't match, put cell back in queue
                agent.probQueue.put(popped)
                break
            else:
                # If probability did match, add cell to list of temp
                temp.append(popped)

        # Initialize a minimum distance variable and find the cell with minimum distance from the center
        minDistance = agent.dim * 2
        minDistance_cell = None
        for cell in temp:
            if calcDist(cell[1], previous[1]) < minDistance:
                minDistance = calcDist(cell[1], previous[1])
                minDistance_cell = cell[1]

        # Put all cells back except for minDistance_cell
        chosen_cell = None
        for cell in temp:
            if cell[1] == minDistance_cell:
                chosen_cell = cell
            else:
                agent.probQueue.put(cell)

        current = chosen_cell
        agent.probQueue.put(current)

        # Calculate number of actions moving to curr cell would take
        cells_moved += calcDist(current[1], previous[1])
        cells_searched += 1
        target_found = check_for_target(agent, current[1])
        updateAgent(agent, current[1], agent.beliefStates[current[1][0]][current[1][1]])
        previous = current
    return cells_searched, cells_moved, cells_searched + cells_moved

def main():
    iterations = 10
    results = [0]*5

    for i in range(iterations):
        landscape = Agent(10)
        print("Target Cell: ", landscape.target)
        print("Beginning search")
        start = time.time()
        cells_searched, cells_moved, agent_actions = ImprovedAgent(landscape)
        results[0] += cells_searched
        results[1] += cells_moved
        results[2] += agent_actions
        results[3] += time.time() - start
        results[4] += 1

    for i in range(0, 5):
        results[i] /= iterations
    print(results)

if __name__ == '__main__':
    main()
