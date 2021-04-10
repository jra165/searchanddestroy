#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 21:25:14 2021

@author: Joshua Atienza, Janet Zhang
@net-id: jra165, jrz42
@project: search-and-destroy
"""

import random
import time

from queue import PriorityQueue, Queue
from typing import List, Tuple


terrainDict = {
    1: 0.1,
    2: 0.3,
    3: 0.7,
    4: 0.9
}

def createEnv(initial, dimension):
    env = [[initial]*dimension]*dimension
    for x in range(dimension):
        for y in range(dimension):
            env[x][y] = terrainDict[random.randint(1, 4)]
    return env

env = createEnv(0, 10)


class Agent:
    def __init__(self, dimension):
        self.dim = dimension
        self.beliefState = createEnv(1/(dimension**2), dimension)
        self.probabilities = PriorityQueue()
        for i in range(dimension):
            for j in range(dimension):
                self.probabilities.put(-self.beliefState[i][j], (i,j))
        self.target = [random.randint(0, dimension-1), random.randint(0,dimension-1)]
        self.observations = createEnv(0, dimension)
        self.memory = Queue(10)
        return

SimpleAgent = Agent(10)
for i in range(10):
    for j in range(10):
        SimpleAgent.probabilities.put((-1*SimpleAgent.beliefState[i][j], (i,j)))

def probabilitiesUpdate(Agent, tile, oldProb, part1):
    newQueue = PriorityQueue()
    probSum = 0
    while not Agent.probabilities.empty():
        curr = Agent.probabilities.get()
        if curr[1] == tile:
            Agent.beliefState[curr[1][0]][curr[1][1]] = Agent.probabilitiesCalculate(curr, tile, oldProb, "observed")
        else: 
            Agent.beliefState[curr[1][0]][cur[1][1]] = Agent.probabilitiesCalculate(curr, tile, oldProb, "other")
        probSum += Agent.beliefState[curr[1][0]][curr[1][1]]
    factor = 1/probSum
    for i in range(Agent.dimension):
        for j in range(Agent.dimension):
            Agent.beliefState[i][j] = Agent.beliefState[i][j] * factor
            if part1: 
                newQueue.put(((calcDistance(tile, (i,j)+1))/Agent.beliefState[i][j], (i,j)))
            else:
                newQueue.put((0-(Agent.beliefState[i][j]), (i,j)))

def probabilitiesUpdateImproved(env, Agent, tile, oldProb, part1):
    newQueue = PriorityQueue()
    probSum = 0
    while not Agent.probabilities.empty():
        curr = Agent.probabilities.get()
        if curr[1] == tile:
            Agent.beliefState[curr[1][0]][curr[1][1]] = calcProbs(env, Agent, curr, tile, oldProb, True)
        else: 
            Agent.beliefState[curr[1][0]][cur[1][1]] = calcProbs(env, Agent, curr, tile, oldProb, False)
        probSum += Agent.beliefState[curr[1][0]][curr[1][1]]
    factor = 1/probSum
    for i in range(Agent.dim):
        for j in range(Agent.dim):
            Agent.beliefState[i][j] = Agent.beliefState[i][j] * factor
            if part1: 
                newQueue.put(((calcDist(tile, (i,j)+1))/Agent.beliefState[i][j], (i,j)))
            else:
                newQueue.put((calcDist(tile,(i,j))+1/Agent.beliefState[i][j]+Agent.observations[i][j], (i,j)))

def calcProbs(env, agent, curr, tile, searched, old):
    falseNeg = env[tile[0]][tile[1]]
    oldBelief = abs(agent.beliefState[curr[1][0]][curr[1][1]])
    searched = abs(searched)
    if old:
        newBelief = falsNeg * searched / (1- searched + falseNeg)
    else:
        newBelief = oldBelief / (1- searched + falseNeg)
    return newBelief

def searchTile(env, Agent, tile):
    if curr == Agent.target:
        falseNeg = random.uniform(0,1)
        if falseNeg > env[curr[9]][curr[1]]:
            return True
    return False

def updatePart2(env, Agent, tile):
    if Agent.memory.qsize() == 10:
        tile, isClose = Agent.memory.get()
        factor = 1
        if not isClose:
            factor = -1
        factor *= -1
        for i in range(-5, 6):
            for j in range(-5,6):
                if 0 <= tile[0] + i < Agent.dim and 0<= tile[i] + j < Agent.dim and calcDist(tile, (tile[0]+i, tile[1]+j)) <= 5:
                    Agent.observations[tile[0]+i][tile[1]+j] += factor*0.25

    else:
        distfromTarget = calcDist(curr, Agent.target)
        factor = 1
        if distfromTarget > 5:
            factor = -1
        for i in range(-5, 6):
            for j in range(-5,6):
                if 0 <= tile[0] + i < Agent.dim and 0<= tile[i] + j < Agent.dim and calcDist(tile, (tile[0]+i, tile[1]+j)) <= 5:
                    Agent.observations[tile[0]+i][tile[1]+j] += factor*0.25

def checkTile(env, Agent, curr):
    if curr == Agent.target:
        falseNeg = random.uniform(0,1)
        if falseNeg > env[curr[0]][curr[1]]:
            return True
    return False



def startSearch(env, Agent, param, part2):
    tileFound = False
    tilePrev = None
    tilesSearched = 1
    tilesTravelled = 0

    # Assigns initial probability values based on likelihood target is seen
    if param == "Rule2" or param == "Agent2" or param == "Agent3":
        newQueue = PriorityQueue()
        probSum = 0
        for i in range(Agent.dim):
            for j in range(Agent.dim):
                Agent.beliefState[i][j] *= (1 - env[i][j])
                probSum += Agent.beliefState[i][j]
        Agent.probabilities = newQueue
    if param == "Rule1" or param == "Agent1":
        a = random.randint(0, Agent.dim - 1)
        b = random.randint(0, Agent.dim - 1)
        curr = [a, b]
        tileFound = checkTile(env, Agent, curr)

        # Find whether we're 5 distance if target not found for part 2
        if not tileFound and part2:
            updatePart2(env, Agent, curr)
        probabilitiesUpdate(Agent, curr, -1/(Agent.dim**2), not part2)
        tilePrev = (0, curr)
    
    # Pick optimal flat terrain cell as the initial random cell
    elif version == "improved":
        #pick improved
        minScore = 1
        optimal = None
        for i in range(Agent.dim):
            for j in range(Agent.dim):
                if env[i][j] == 0.1:
                    #Calc landscape score
                    neighbors = 0
                    neighborsSum = 0
                    for m in range(-1,2):
                        for n in range(-1,2):
                            if m!=0 or n!= 0:
                                if 0<=i+m < Agent.dim and 0 <= j+n <Agent.dim:
                                    neighborsSum += env[a + k][b+l]
                                    neighbors += 1
                    neighborsAvg = neighborsSum/neighbors
                    if neighborsAvg < minScore:
                        minScore = neighborsAvg
                        optimal = (i,j)
        
        # There was no flat terrain in the board, but this would only happen with a 1x1 or a 2x2
        if optimal != None:
            curr = optimal

        # Find whether we're 5 distance if target not found for part 2
        if not tileFound and part2:
            updatePart2(env, Agent, curr)
        
        tileFound = checkTile(env, Agent, curr)
        probabilitiesUpdate(Agent, curr, -1/Agent.dim**2, not part2)
        tilePrev = (0, curr)

    else:
        # Takes the top cell and put it back
        curr = Agent.probabilities.get()
        Agent.probabilites.put(curr)

        # Find whether we're 5 distance if target not found for part 2
        if not tileFound and part2:
            updatePart2(env, Agent, curr[1])
            #Idt used
        
        tileFound = checkTile(env, Agent, curr[1])
        probabilitiesUpdate(Agent, curr, Agent.beliefState[curr[1][0]][curr[1][1]], not part2)
        
        # Update prev cell
        tilePrev = curr

    while not tileFound:
        if param == "Rule1" or param == "Rule2":
            curr = Agent.probabilities.get()
            Agent.probabilities.put(curr)
            if not tileFound and part2:
                updatePart2(env, Agent, curr[1])
            tilesSearched += 1
            tileFound = checkTile(env, Agent, curr[1])
            probabitiesUpdate(Agent, curr[1], curr[0], not part2)
        elif param == "Agent1" or param == "Agent2" or param == "Agent3" or param == "improved":
            curr = Agent.probabilities.get()
            Agent.probabilities.put(curr)
            popped = [curr]
            while True and not Agent.probabilities.empty():
                temp = probablities.get()
                if temp[0] != curr[0]:
                    Agent.probabilities.put(temp)
                    break
                else:
                    popped.append(temp)
            minDist = Agent.dim * 2
            minDistTile = None
            for tile in popped:
                if calcDist(tile[1], tilePrev[1]) < minDist:
                    minDist = calcDist(tile[1], tilePrev[1])
                    minDistTile = tile[1]
            chosen = None
            for tile in popped:
                if tile[1] == minDistTile:
                    chosen = tile
                else:
                    Agent.probabilities.put(tile)

            curr = chosen
            Agent.probabilities.put(curr)
            if not tileFound and part2:
                updatePart2(curr[1])
            tilesTravelled += calcDist(curr[1], prevTile[1])

    

startSearch(env, SimpleAgent, "Agent1", False)

class Landscape:
    def __init__(self, d):

        self._d: int = d
        self._landscape: List[List[float]] = self.generate_landscape()
        self._belief_states: List[List[float]] = [[1 / (self._d * self._d) for _ in range(self._d)] for _ in
                                                  range(self._d)]
        self._probability_queue: PriorityQueue[Tuple[float, Tuple[int, int]]] = self._initialize_probability_queue()
        self.target = [random.randint(0, self._d - 1), random.randint(0, self._d - 1)]
        self.proximity_history: List[List[float]] = [[0 for _ in range(self._d)] for _ in
                                                  range(self._d)]
        self.most_recent_searches = Queue(10)
        

    def generate_landscape(self) -> List[List[float]]:
        """
        Generate initial landscape. The value of each cell corresponds to the probability of a false negative for that cell.
        0.1: flat
        0.3: hilly
        0.7: forested
        0.9: caves
        :return: the generated landscape
        """

        landscape = [[0.0 for _ in range(self._d)] for _ in range(self._d)]

        # Randomly assign the terrains
        for i in range(self._d):
            for j in range(self._d):
                terrain_probability = random.uniform(0, 1)

                # Assign cell to flat
                if terrain_probability < 0.2:
                    landscape[i][j] = 0.1

                # Assign cell to hilly
                elif terrain_probability < 0.5:
                    landscape[i][j] = 0.3

                # Assign cell to forested
                elif terrain_probability < 0.8:
                    landscape[i][j] = 0.7

                # Assign cell to caves
                else:
                    landscape[i][j] = 0.9

        return landscape

    def print_landscape(self) -> None:
        """
        Print landscape.
        """
        for row in range(self._d):
            print(self._landscape[row])
        print()

    def print_belief_states(self) -> None:
        """
        Print belief states.
        """
        for row in range(self._d):
            print(self._belief_states[row])
        print()

    def _initialize_probability_queue(self) -> PriorityQueue:
        probability_queue = PriorityQueue()
        for i in range(self._d):
            for j in range(self._d):
                probability_queue.put((-1 * self._belief_states[i][j], (i, j)))

        return probability_queue

    def _update_probability_queue(self, observed_cell: Tuple[int, int], observed_prior_probability: float,
                                  version: str, part2_flag: bool) -> None:
        """
        Update probability  values in the probability_priorities queue
        """

        # Loop through queue of cells and updated probability values
        updated_queue = PriorityQueue()
        total_probability_sum = 0
        while not self._probability_queue.empty():
            curr_cell = self._probability_queue.get()
            if curr_cell[1] == observed_cell:
                self._belief_states[curr_cell[1][0]][curr_cell[1][1]] = self._calculate_new_probability(curr_cell,
                                                                                                        observed_cell,
                                                                                                        observed_prior_probability,
                                                                                                        "observed")
            else:
                # print("Else condition")
                self._belief_states[curr_cell[1][0]][curr_cell[1][1]] = self._calculate_new_probability(curr_cell,
                                                                                                        observed_cell,
                                                                                                        observed_prior_probability,
                                                                                                        "other")
            # Add newly calculated probability value to counter
            total_probability_sum += self._belief_states[curr_cell[1][0]][curr_cell[1][1]]

        # Recreate the probability_queue with updated values
        scaling_factor = 1 / total_probability_sum
        for i in range(self._d):
            for j in range(self._d):
                if version == "rule_1" or version == "rule_2" or version == "agent_1" or version == "agent_2":
                    self._belief_states[i][j] = self._belief_states[i][j] * scaling_factor
                    if not part2_flag:
                        updated_queue.put((-1 * self._belief_states[i][j], (i, j)))
                    else:
                        priority = -1 * (self._belief_states[i][j] + self.proximity_history[i][j])
                        updated_queue.put((priority, (i, j)))

                # NOTE the version == "improved" if condition should rly go here but agent_3 is kind of wack
                elif version == "agent_3" or version == "improved":
                    self._belief_states[i][j] = self._belief_states[i][j] * scaling_factor
                    if not part2_flag:
                        updated_queue.put(((self._calculate_distance(observed_cell, (i, j)) + 1) / self._belief_states[i][j], (i, j)))
                    else:
                        priority = (self._calculate_distance(observed_cell, (i, j)) + 1) / self._belief_states[i][j] + self.proximity_history[i][j]
                        updated_queue.put((priority, (i, j)))
        self._probability_queue = updated_queue

    def _calculate_new_probability(self, curr_cell: Tuple[float, Tuple[int, int]], observed_cell: Tuple[int, int],
                                   observed_prior: float, flag: str):
        false_negative_probability = self._landscape[observed_cell[0]][observed_cell[1]]
        prior_belief = abs(self._belief_states[curr_cell[1][0]][curr_cell[1][1]])
        observed_prior = abs(observed_prior)

        # print('FNR: ', false_negative_probability)
        # print('Prior belief: ', prior_belief)
        # print('Observed prior: ', observed_prior)

        if flag == "observed":
            new_probability = false_negative_probability * observed_prior / (
                    1 - observed_prior + false_negative_probability)
        else:
            new_probability = prior_belief / (1 - observed_prior + false_negative_probability)

        return new_probability

    def _search_cell(self, curr_cell: Tuple[int, int]) -> bool:
        if curr_cell == self.target:
            false_negative_probability = random.uniform(0, 1)
            # Check for false negative
            if false_negative_probability > self._landscape[curr_cell[0]][curr_cell[1]]:
                return True
        return False

    def _update_proximity_history(self, within_5_from_target: bool, cur_cell: Tuple[int, int], dequeue: bool):
        factor = 1
        if not within_5_from_target:
            factor = -1

        # When dequeueing, want to negate the factor previously added / subtracted originally
        if dequeue:
            factor *= -1

        for i in range(-5, 6):
            for j in range(-5, 6):
                if 0 <= cur_cell[0] + i < self._d and 0 <= cur_cell[1] + j < self._d:
                    # Increment or decrement the proximity history priority adjustment factor
                    if self._calculate_distance(cur_cell, (cur_cell[0]+i, cur_cell[1]+j)) <= 5:
                        self.proximity_history[cur_cell[0]+i][cur_cell[1]+j] += factor * 0.25

    def _update_part_2(self, curr_cell: Tuple[int, int]) -> None:
        # Find whether target is 5 from current cell
        distance_from_target = self._calculate_distance(curr_cell, self.target)
        within_5_from_target = False
        if distance_from_target <= 5:
            within_5_from_target = True
        print("Within 5 from target?", within_5_from_target)

        # Update proximity history
        self._update_proximity_history(within_5_from_target, curr_cell, False)

        # Update most recently searched
        if self.most_recent_searches.qsize() == 10:
            expired_cell, expired_cell_within_5_from_target = self.most_recent_searches.get()
            self._update_proximity_history(expired_cell_within_5_from_target, expired_cell, True)
        self.most_recent_searches.put((curr_cell, within_5_from_target))

        # Move target
        targetOld = self.target
        targetNew = NULL
        while True:
            rInt = random.randint(0, 3)
            if rInt == 0:
                targetNew = (targetOld[0], targetOld[1]-1)
            elif rInt == 1:
                targetNew = (targetOld[0], targetOld[1]+1)
            if rInt == 2:
                targetNew = (targetOld[0]-1, targetOld[1])
            elif rInt == 3:
                targetNew = (targetOld[0]+1, targetOld[1]+1)
            if targetNew[0] >=0 and targetNew[1] < self._d:
                break
        self.target = targetNew


    def begin_search(self, version: str, part2_flag=None) -> Tuple[int, int, int]:

        if version == "rule_2" or version == "agent_2" or version == "agent_3":
            # Assigns initial probability values based on likelihood target is seen
            self._update_belief_states_rule_2(version)

        print("Initial belief states")
        self.print_belief_states()

        target_found = False
        prev_cell = None

        search_count = 1
        move_count = 0

        # Pick first cell, random if for rule 1 or agent 1
        if version == "rule_1" or version == "agent_1":
            curr_cell = [random.randint(0, self._d - 1), random.randint(0, self._d - 1)]
            # print("Current cell", curr_cell)
            # print("Target cell", self.target)
            # print(search_count)
            # print(move_count)
            target_found = self._search_cell(curr_cell)

            # Find whether we're 5 distance if target not found for part 2
            if not target_found and part2_flag:
                self._update_part_2(curr_cell)

            self._update_probability_queue(curr_cell, -1 / self._d ** 2, version, part2_flag)
            prev_cell = (0, curr_cell)

        # Pick optimal flat terrain cell as the initial random cell
        elif version == "improved":
            curr_cell = self._pick_improved_initial_cell()

            # There was no flat terrain in the board, but this would only happen with a 1x1 or a 2x2
            if curr_cell == None:
                curr_cell = self._pick_random_cell()

            # Find whether we're 5 distance if target not found for part 2
            if not target_found and part2_flag:
                self._update_part_2(curr_cell)

            # print("Current cell", curr_cell)
            # print("Target cell", self.target)
            # print(search_count)
            # print(move_count)
            target_found = self._search_cell(curr_cell)
            self._update_probability_queue(curr_cell, -1 / self._d ** 2, version, part2_flag)
            prev_cell = (0, curr_cell)

        else:
            # Takes the top cell and put it back
            curr_cell = self._probability_queue.get()
            self._probability_queue.put(curr_cell)

            # Find whether we're 5 distance if target not found for part 2
            if not target_found and part2_flag:
                self._update_part_2(curr_cell[1])

            print("Current cell", curr_cell[1])
            print("Target cell", self.target)
            print(search_count)
            print(move_count)
            target_found = self._search_cell(curr_cell[1])
            self._update_probability_queue(curr_cell[1], self._belief_states[curr_cell[1][0]][curr_cell[1][1]], version, part2_flag)
            self.print_belief_states()

            # Update prev cell
            prev_cell = curr_cell

        self.print_belief_states()

        while not target_found:
            if version == "rule_1" or version == "rule_2":
                # Remove top cell
                curr_cell = self._probability_queue.get()
                self._probability_queue.put(curr_cell)

                # Find whether we're 5 distance if target not found for part 2
                if not target_found and part2_flag:
                    self._update_part_2(curr_cell[1])

                # print("Current cell", curr_cell)
                # print("Target cell", self.target)
                search_count += 1
                # print(search_count)
                target_found = self._search_cell(curr_cell[1])
                self._update_probability_queue(curr_cell[1], curr_cell[0], version, part2_flag)
                self.print_belief_states()

            elif version == "agent_1" or version == "agent_2" or version == "agent_3" or version == "improved":

                # Takes the top cell and put it back
                curr_cell = self._probability_queue.get()
                self._probability_queue.put(curr_cell)

                # Determines new curr cell after handling ties
                curr_cell = self._handle_ties(curr_cell, prev_cell)
                self._probability_queue.put(curr_cell)

                # Find whether we're 5 distance if target not found for part 2
                if not target_found and part2_flag:
                    self._update_part_2(curr_cell[1])

                # Calculate number of actions moving to curr cell would take
                move_count += self._calculate_distance(curr_cell[1], prev_cell[1])

                print("Current cell", curr_cell)
                print("Target cell", self.target)
                search_count += 1
                print(search_count)
                print(move_count)
                target_found = self._search_cell(curr_cell[1])
                self._update_probability_queue(curr_cell[1], self._belief_states[curr_cell[1][0]][curr_cell[1][1]], version, part2_flag)
                self.print_belief_states()

                # Update prev cell
                prev_cell = curr_cell

        print("Target found")
        print("Cell", curr_cell)
        print("Searches: ", search_count)
        print("Moves: ", move_count)
        print("Actions: ", search_count + move_count)
        return search_count, move_count, search_count + move_count
        # self.print_belief_states()

    def _handle_ties(self, curr_cell: Tuple[float, Tuple[int, int]], prev_cell: Tuple[float, Tuple[int, int]]) -> Tuple[
        float, Tuple[int, int]]:
        # Create a list to track cells removed from the queue in this step and add initial cell to the List
        popped_cells: List[Tuple[float, Tuple[int, int]]] = [curr_cell]
        loop = True

        while loop and not self._probability_queue.empty():
            # Remove new cells from the queue until we find one that doesn't match the current probability
            popped = self._probability_queue.get()
            if popped[0] != curr_cell[0]:
                loop = False
                # If probability didn't match, put cell back in queue
                self._probability_queue.put(popped)
            else:
                # If probability did match, add cell to list of popped_cells
                popped_cells.append(popped)

        # Initialize a minimum distance variable and find the cell with minimum distance from the center
        min_distance = self._d * 2
        min_distance_cell = None
        for cell in popped_cells:
            if self._calculate_distance(cell[1], prev_cell[1]) < min_distance:
                min_distance = self._calculate_distance(cell[1], prev_cell[1])
                min_distance_cell = cell[1]

        # Put all cells back except for min_distance_cell
        chosen_cell = None
        for cell in popped_cells:
            if cell[1] == min_distance_cell:
                chosen_cell = cell
            else:
                self._probability_queue.put(cell)

        return chosen_cell

    def _calculate_distance(self, curr_cell: Tuple[int, int], prev_cell: Tuple[int, int]) -> int:
        """
        Calculate manhattan distance between cells
        :param curr_cell:
        :param prev_cell:
        :return: manhattan distance
        """
        distance = abs(curr_cell[0] - prev_cell[0]) + abs(curr_cell[1] - prev_cell[1])
        return distance

    def _update_belief_states_rule_2(self, version: str):

        updated_queue = PriorityQueue()
        total_probability_sum = 0

        for i in range(self._d):
            for j in range(self._d):
                self._belief_states[i][j] *= (1 - self._landscape[i][j])
                total_probability_sum += self._belief_states[i][j]

        # Normalize probabilities
        self._probability_queue = self._normalize_belief_states_rule_2(total_probability_sum, updated_queue)

    def _normalize_belief_states_rule_2(self, total_probability_sum: int, updated_queue: PriorityQueue):
        """
        Scale the probability values so the total probability is 1
        :param total_probability_sum:
        :param updated_queue:
        :return:
        """

        scaling_factor = 1 / total_probability_sum
        print("scaling: ", scaling_factor)
        for i in range(self._d):
            for j in range(self._d):
                self._belief_states[i][j] = self._belief_states[i][j] * scaling_factor
                updated_queue.put((-1 * self._belief_states[i][j], (i, j)))

        return updated_queue

    def _pick_random_cell(self) -> Tuple[int, int]:
        """
        Pick a random cell in the board.
        """
        i = random.randint(0, self._d - 1)
        j = random.randint(0, self._d - 1)
        return i, j

    def _pick_improved_initial_cell(self) -> Tuple[int, int]:
        """
        Pick a flat cell that is surrounded by the flattest land
        :return: Flat cell that is surrounded by the flattest land
        """
        min_landscape_score = 1
        optimal_cell = None

        for i in range(self._d):
            for j in range(self._d):

                # Only consider cells with flat terrain
                if self._landscape[i][j] == 0.1:
                    cell_landscape_score = self._calculate_landscape_score(i, j)
                    if cell_landscape_score < min_landscape_score:
                        min_landscape_score = cell_landscape_score
                        optimal_cell = (i, j)

        return optimal_cell

    def _calculate_landscape_score(self, i: int, j: int) -> float:
        """
        Return the average terrain value of a cell's neighbors
        :param i: x coordinate of cell
        :param j: y coordinate of cell
        :return: average terrain value of the cell's neighbors
        """
        neighbors = 0
        neighbor_sum = 0

        # Add up the landscape scores of its neighbors
        for x in range(-1, 2):
            for y in range(-1, 2):
                if x != 0 or y != 0:
                    if 0 <= i + x < self._d and 0 <= j + y < self._d:
                        neighbor_sum += self._landscape[i + x][j + y]
                        neighbors += 1

        # Get the average score
        neighbor_average = neighbor_sum / neighbors

        return neighbor_average

    # def _get_possible_neighbors(self, i: int, j: int) -> int:
    #     """
    #     Calculate max possible neighbors based on whether cell is corner, edge, or neither.
    #     Args:
    #         i: x dimension index of cell
    #         j: y dimension index of cell
    #
    #     Returns:
    #         int: possible number of neighbors
    #     """
    #     if i == 0 or i == self._d - 1:
    #         if j == 0 or j == self._d - 1:
    #             # Corner
    #             possible_neighbors = 3
    #         else:
    #             # Edge
    #             possible_neighbors = 5
    #
    #     elif j == 0 or j == self._d - 1:
    #         # Edge
    #         possible_neighbors = 5
    #     else:
    #         # Central cell
    #         possible_neighbors = 8
    #
    #     return possible_neighbors


def main():
    iterations = 5
    results = [0.0, 0.0, 0.0, 0.0]

    for _ in range(iterations):
        print("hello world")
        landscape = Landscape(10)
        print("Target Cell: ", landscape.target)
        landscape.print_landscape()
        landscape.print_belief_states()
        print("Beginning search")
        start = time.time()
        search_count, move_count, action_count = landscape.begin_search("agent_3", False)
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
