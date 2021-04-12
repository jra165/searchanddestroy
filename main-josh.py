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

#This function calculates the Manhattan distanc between two given cells
def calcDist(current, previous):
    
    #formula for Manhattan distance
    distance = abs(current[0] - previous[0]) + abs(current[1] - previous[1])
    return distance

#This function generates our landscape
def generate_environment(agent):
    landscape = [[0]*agent.dim]*agent.dim
    
    # Randomly assign the terrains
    for i in range(agent.dim):
        for j in range(agent.dim):
            landscape[i][j] = terrainDict[random.randint(1, 4)]
    return landscape

#The initial choice of cell at the beginning of the agent's path
def pick_random_cell(agent):
    
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

        self.target = pick_random_cell(self)

        return

def change_target(agent):
        prev_target = agent.target
        successful_move = False
        new_target = None

        while not successful_move:
            move_flag = random.randint(0, 3)
            # Go up
            if move_flag == 0:
                new_target = (prev_target[0], prev_target[1]-1)

            # Go down
            elif move_flag == 1:
                new_target = (prev_target[0], prev_target[1]+1)

            # Go left
            elif move_flag == 2:
                new_target = (prev_target[0]-1, prev_target[1])

            # Go right
            elif move_flag == 3:
                new_target = (prev_target[0]+1, prev_target[1] + 1)

            if new_target[0] >= 0 and new_target[1] < agent.dim:
                successful_move = True

        agent.target = new_target
        return

def _updateprobQueue(agent, observed_cell: Tuple[int, int], observed_prior_probability: float, version: str, part2_flag: bool) -> None:
    """
    Update probability  values in the probability_priorities queue
    """

    # Loop through queue of cells and updated probability values
    updated_queue = PriorityQueue()
    total_probability_sum = 0
    while not agent.probQueue.empty():
        curr_cell = agent.probQueue.get()
        if curr_cell[1] == observed_cell:
            agent.beliefStates[curr_cell[1][0]][curr_cell[1][1]] = _calculate_new_probability(agent, curr_cell,
                                                                                                    observed_cell,
                                                                                                    observed_prior_probability,
                                                                                                    "observed")
        else:
            # print("Else condition")
            agent.beliefStates[curr_cell[1][0]][curr_cell[1][1]] = _calculate_new_probability(agent, curr_cell,
                                                                                                    observed_cell,
                                                                                                    observed_prior_probability,
                                                                                                    "other")
        # Add newly calculated probability value to counter
        total_probability_sum += agent.beliefStates[curr_cell[1][0]][curr_cell[1][1]]

    # Recreate the probability_queue with updated values
    scaling_factor = 1 / total_probability_sum
    for i in range(agent.dim):
        for j in range(agent.dim):
            if version == "rule_1" or version == "rule_2" or version == "agent_1" or version == "agent_2":
                agent.beliefStates[i][j] = agent.beliefStates[i][j] * scaling_factor

                if not part2_flag:
                    updated_queue.put((-1 * agent.beliefStates[i][j], (i, j)))
                else:
                    priority = -1 * (agent.beliefStates[i][j])
                    updated_queue.put((priority, (i, j)))

            # NOTE the version == "improved" if condition should rly go here but agent_3 is kind of wack
            elif version == "agent_3" or version == "improved":
                agent.beliefStates[i][j] = agent.beliefStates[i][j] * scaling_factor
                if not part2_flag:
                    updated_queue.put(((calcDist(observed_cell, (i, j)) + 1) / agent.beliefStates[i][j], (i, j)))
                else:
                    priority = (calcDist(observed_cell, (i, j)) + 1) / agent.beliefStates[i][j]
                    updated_queue.put((priority, (i, j)))
    agent.probQueue = updated_queue

#Calculates the new probability
def _calculate_new_probability(agent, curr_cell: Tuple[float, Tuple[int, int]], observed_cell: Tuple[int, int],
                                   observed_prior: float, flag: str):
        false_negative_rate = agent.environment[observed_cell[0]][observed_cell[1]]
        prior_belief = abs(agent.beliefStates[curr_cell[1][0]][curr_cell[1][1]])
        observed_prior = abs(observed_prior)

        # print('FNR: ', false_negative_rate)
        # print('Prior belief: ', prior_belief)
        # print('Observed prior: ', observed_prior)

        if flag == "observed":
            new_probability = false_negative_rate * observed_prior / (
                    1 - observed_prior + false_negative_rate)
        else:
            new_probability = prior_belief / (1 - observed_prior + false_negative_rate)

        return new_probability

#Checks the cell for target
def check_for_target(agent, current):

    if current == agent.target:
        false_negative_rate = random.uniform(0, 1)

        # Check for false negative
        if false_negative_rate > agent.environment[current[0]][current[1]]:
            return True

    return False

#JOSHUA RETURN TO THIS ONE LATER
def _handle_ties(agent, curr_cell: Tuple[float, Tuple[int, int]], prev_cell: Tuple[float, Tuple[int, int]]) -> Tuple[float, Tuple[int, int]]:
    # Create a list to track cells removed from the queue in this step and add initial cell to the List
    popped_cells: List[Tuple[float, Tuple[int, int]]] = [curr_cell]
    loop = True

    while loop and not agent.probQueue.empty():
        # Remove new cells from the queue until we find one that doesn't match the current probability
        popped = agent.probQueue.get()
        if popped[0] != curr_cell[0]:
            loop = False
            # If probability didn't match, put cell back in queue
            agent.probQueue.put(popped)
        else:
            # If probability did match, add cell to list of popped_cells
            popped_cells.append(popped)

    # Initialize a minimum distance variable and find the cell with minimum distance from the center
    minDistance = agent.dim * 2
    minDistance_cell = None
    for cell in popped_cells:
        if calcDist(cell[1], prev_cell[1]) < minDistance:
            minDistance = calcDist(cell[1], prev_cell[1])
            minDistance_cell = cell[1]

    # Put all cells back except for minDistance_cell
    chosen_cell = None
    for cell in popped_cells:
        if cell[1] == minDistance_cell:
            chosen_cell = cell
        else:
            agent.probQueue.put(cell)

    return chosen_cell

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

def begin_search(agent, version: str, part2_flag=None) -> Tuple[int, int, int]:

    if version == "rule_2" or version == "agent_2" or version == "agent_3":
        # Assigns initial probability values based on likelihood target is seen
        updated_queue = PriorityQueue()
        total_probability_sum = 0

        for i in range(agent.dim):
            for j in range(agent.dim):
                agent.beliefStates[i][j] *= (1 - agent.environment[i][j])
                total_probability_sum += agent.beliefStates[i][j]

        # Normalize probabilities
        scaling_factor = 1 / total_probability_sum
        for i in range(agent.dim):
            for j in range(agent.dim):
                agent.beliefStates[i][j] = agent.beliefStates[i][j] * scaling_factor
                updated_queue.put((-1 * agent.beliefStates[i][j], (i, j)))
        agent.probQueue = updated_queue

    target_found = False
    prev_cell = None

    search_count = 1
    move_count = 0

    # Pick first cell, random if for rule 1 or agent 1
    if version == "rule_1" or version == "agent_1":
        curr_cell = _pick_random_cell()
        target_found = check_for_target(agent, curr_cell)

        # Find whether we're 5 distance if target not found for part 2
        if not target_found and part2_flag:
            _update_part_2(agent, curr_cell)

        _updateprobQueue(agent, curr_cell, -1 / agent.dim ** 2, version, part2_flag)
        prev_cell = (0, curr_cell)

    # Pick optimal flat terrain cell as the initial random cell
    elif version == "improved":
        curr_cell = improved_start(agent)

        # There was no flat terrain in the board, but this would only happen with a 1x1 or a 2x2
        if curr_cell == None:
            curr_cell = _pick_random_cell()

        # Find whether we're 5 distance if target not found for part 2
        if not target_found and part2_flag:
            _update_part_2(agent, curr_cell)

        target_found = check_for_target(agent, curr_cell)
        _updateprobQueue(agent, curr_cell, -1 / self.dim ** 2, version, part2_flag)
        prev_cell = (0, curr_cell)

    else:
        # Takes the top cell and put it back
        curr_cell = agent.probQueue.get()
        agent.probQueue.put(curr_cell)

        # Find whether we're 5 distance if target not found for part 2
        if not target_found and part2_flag:
            _update_part_2(agent, curr_cell[1])

        target_found = check_for_target(agent, curr_cell[1])
        _updateprobQueue(agent, curr_cell[1], agent.beliefStates[curr_cell[1][0]][curr_cell[1][1]], version, part2_flag)

        # Update prev cell
        prev_cell = curr_cell

    while not target_found:
        if version == "rule_1" or version == "rule_2":
            # Remove top cell
            curr_cell = agent.probQueue.get()
            agent.probQueue.put(curr_cell)

            # Find whether we're 5 distance if target not found for part 2
            if not target_found and part2_flag:
                _update_part_2(agent, curr_cell[1])

            search_count += 1
            target_found = check_for_target(agent, curr_cell[1])
            _updateprobQueue(agent, curr_cell[1], curr_cell[0], version, part2_flag)

        elif version == "agent_1" or version == "agent_2" or version == "agent_3" or version == "improved":

            # Takes the top cell and put it back
            curr_cell = agent.probQueue.get()
            agent.probQueue.put(curr_cell)

            # Determines new curr cell after handling ties
            curr_cell = _handle_ties(agent, curr_cell, prev_cell)
            agent.probQueue.put(curr_cell)

            # Find whether we're 5 distance if target not found for part 2
            if not target_found and part2_flag:
                _update_part_2(agent, curr_cell[1])

            # Calculate number of actions moving to curr cell would take
            move_count += calcDist(curr_cell[1], prev_cell[1])
            search_count += 1
            target_found = check_for_target(agent, curr_cell[1])
            _updateprobQueue(agent, curr_cell[1], agent.beliefStates[curr_cell[1][0]][curr_cell[1][1]], version, part2_flag)
            prev_cell = curr_cell

    print("Target found")
    print("Cell", curr_cell)
    print("Searches: ", search_count)
    print("Moves: ", move_count)
    print("Actions: ", search_count + move_count)
    return search_count, move_count, search_count + move_count


def main():
    iterations = 5
    results = [0.0, 0.0, 0.0, 0.0]

    for _ in range(iterations):
        print("hello world")
        landscape = Agent(10)
        print("Target Cell: ", landscape.target)
        print("Beginning search")
        start = time.time()
        search_count, move_count, action_count = begin_search(landscape, "agent_3", False)
        results[3] += time.time() - start
        results[0] += search_count
        results[1] += move_count
        results[2] += action_count
        print("goodbye world")

    for i in range(0, 4):
        results[i] /= iterations
    print(results)

if __name__ == '__main__':
    main()
