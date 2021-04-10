import random
import time

from queue import PriorityQueue, Queue
from typing import List, Tuple


def _calculate_distance(curr_cell, prev_cell):
    distance = abs(curr_cell[0] - prev_cell[0]) + abs(curr_cell[1] - prev_cell[1])
    return distance


terrainDict = {
    1: 0.1,
    2: 0.3,
    3: 0.7,
    4: 0.9
}

def generate_landscape(agent):
    landscape = [[0]*agent._d]*agent._d
    # Randomly assign the terrains
    for i in range(agent._d):
        for j in range(agent._d):
            landscape[i][j] = terrainDict[random.randint(1, 4)]
    return landscape

def pick_random_cell(agent):
    i = random.randint(0, agent._d - 1)
    j = random.randint(0, agent._d - 1)
    return i, j

class Agent:
    def __init__(self, d):
        self._d: int = d
        self._landscape: List[List[float]] = generate_landscape(self)
        self._belief_states: List[List[float]] = [[1 / (self._d * self._d) for _ in range(self._d)] for _ in
                                                    range(self._d)]
        self._probability_queue: PriorityQueue[Tuple[float, Tuple[int, int]]] = PriorityQueue()
        for i in range(d):
            for j in range(d):
                self._probability_queue.put((-1*self._belief_states[i][j], (i,j)))
        self.target = pick_random_cell(self)
        self.proximity_history: List[List[float]] = [[0 for _ in range(self._d)] for _ in
                                                    range(self._d)]
        self.most_recent_searches = Queue(10)
        return

def _change_target(agent) -> None:
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

            if new_target[0] >= 0 and new_target[1] < agent._d:
                successful_move = True

        agent.target = new_target

def _update_probability_queue(agent, observed_cell: Tuple[int, int], observed_prior_probability: float, version: str, part2_flag: bool) -> None:
    """
    Update probability  values in the probability_priorities queue
    """

    # Loop through queue of cells and updated probability values
    updated_queue = PriorityQueue()
    total_probability_sum = 0
    while not agent._probability_queue.empty():
        curr_cell = agent._probability_queue.get()
        if curr_cell[1] == observed_cell:
            agent._belief_states[curr_cell[1][0]][curr_cell[1][1]] = _calculate_new_probability(agent, curr_cell,
                                                                                                    observed_cell,
                                                                                                    observed_prior_probability,
                                                                                                    "observed")
        else:
            # print("Else condition")
            agent._belief_states[curr_cell[1][0]][curr_cell[1][1]] = _calculate_new_probability(agent, curr_cell,
                                                                                                    observed_cell,
                                                                                                    observed_prior_probability,
                                                                                                    "other")
        # Add newly calculated probability value to counter
        total_probability_sum += agent._belief_states[curr_cell[1][0]][curr_cell[1][1]]

    # Recreate the probability_queue with updated values
    scaling_factor = 1 / total_probability_sum
    for i in range(agent._d):
        for j in range(agent._d):
            if version == "rule_1" or version == "rule_2" or version == "agent_1" or version == "agent_2":
                agent._belief_states[i][j] = agent._belief_states[i][j] * scaling_factor

                if not part2_flag:
                    updated_queue.put((-1 * agent._belief_states[i][j], (i, j)))
                else:
                    priority = -1 * (agent._belief_states[i][j] + agent.proximity_history[i][j])
                    updated_queue.put((priority, (i, j)))

            # NOTE the version == "improved" if condition should rly go here but agent_3 is kind of wack
            elif version == "agent_3" or version == "improved":
                agent._belief_states[i][j] = agent._belief_states[i][j] * scaling_factor
                if not part2_flag:
                    updated_queue.put(((_calculate_distance(observed_cell, (i, j)) + 1) / agent._belief_states[i][j], (i, j)))
                else:
                    priority = (_calculate_distance(observed_cell, (i, j)) + 1) / agent._belief_states[i][j] + agent.proximity_history[i][j]
                    updated_queue.put((priority, (i, j)))
    agent._probability_queue = updated_queue

def _calculate_new_probability(agent, curr_cell: Tuple[float, Tuple[int, int]], observed_cell: Tuple[int, int],
                                   observed_prior: float, flag: str):
        false_negative_probability = agent._landscape[observed_cell[0]][observed_cell[1]]
        prior_belief = abs(agent._belief_states[curr_cell[1][0]][curr_cell[1][1]])
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

def _search_cell(agent, curr_cell: Tuple[int, int]) -> bool:
    if curr_cell == agent.target:
        false_negative_probability = random.uniform(0, 1)
        # Check for false negative
        if false_negative_probability > agent._landscape[curr_cell[0]][curr_cell[1]]:
            return True
    return False

def _update_proximity_history(agent, within_5_from_target: bool, cur_cell: Tuple[int, int], dequeue: bool):
    factor = 1
    if not within_5_from_target:
        factor = -1

    # When dequeueing, want to negate the factor previously added / subtracted originally
    if dequeue:
        factor *= -1

    for i in range(-5, 6):
        for j in range(-5, 6):
            if 0 <= cur_cell[0] + i < agent._d and 0 <= cur_cell[1] + j < agent._d:
                # Increment or decrement the proximity history priority adjustment factor
                if _calculate_distance(cur_cell, (cur_cell[0]+i, cur_cell[1]+j)) <= 5:
                    agent.proximity_history[cur_cell[0]+i][cur_cell[1]+j] += factor * 0.25

def _update_part_2(agent, curr_cell: Tuple[int, int]) -> None:
        # Find whether target is 5 from current cell
        distance_from_target = _calculate_distance(curr_cell, agent.target)
        within_5_from_target = False
        if distance_from_target <= 5:
            within_5_from_target = True
        print("Within 5 from target?", within_5_from_target)

        # Update proximity history
        _update_proximity_history(agent, within_5_from_target, curr_cell, False)

        # Update most recently searched
        if agent.most_recent_searches.qsize() == 10:
            expired_cell, expired_cell_within_5_from_target = agent.most_recent_searches.get()
            _update_proximity_history(agent, expired_cell_within_5_from_target, expired_cell, True)
        agent.most_recent_searches.put((curr_cell, within_5_from_target))

        # Move target
        _change_target(agent)

def _handle_ties(agent, curr_cell: Tuple[float, Tuple[int, int]], prev_cell: Tuple[float, Tuple[int, int]]) -> Tuple[float, Tuple[int, int]]:
    # Create a list to track cells removed from the queue in this step and add initial cell to the List
    popped_cells: List[Tuple[float, Tuple[int, int]]] = [curr_cell]
    loop = True

    while loop and not agent._probability_queue.empty():
        # Remove new cells from the queue until we find one that doesn't match the current probability
        popped = agent._probability_queue.get()
        if popped[0] != curr_cell[0]:
            loop = False
            # If probability didn't match, put cell back in queue
            agent._probability_queue.put(popped)
        else:
            # If probability did match, add cell to list of popped_cells
            popped_cells.append(popped)

    # Initialize a minimum distance variable and find the cell with minimum distance from the center
    min_distance = agent._d * 2
    min_distance_cell = None
    for cell in popped_cells:
        if _calculate_distance(cell[1], prev_cell[1]) < min_distance:
            min_distance = _calculate_distance(cell[1], prev_cell[1])
            min_distance_cell = cell[1]

    # Put all cells back except for min_distance_cell
    chosen_cell = None
    for cell in popped_cells:
        if cell[1] == min_distance_cell:
            chosen_cell = cell
        else:
            agent._probability_queue.put(cell)

    return chosen_cell

def _calculate_landscape_score(agent, i: int, j: int) -> float:
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
                if 0 <= i + x < agent._d and 0 <= j + y < agent._d:
                    neighbor_sum += agent._landscape[i + x][j + y]
                    neighbors += 1

    # Get the average score
    neighbor_average = neighbor_sum / neighbors
    return neighbor_average

def _pick_improved_initial_cell(agent) -> Tuple[int, int]:
    """
    Pick a flat cell that is surrounded by the flattest land
    :return: Flat cell that is surrounded by the flattest land
    """
    min_landscape_score = 1
    optimal_cell = None

    for i in range(agent._d):
        for j in range(agent._d):

            # Only consider cells with flat terrain
            if agent._landscape[i][j] == 0.1:
                cell_landscape_score = _calculate_landscape_score(agent, i, j)
                if cell_landscape_score < min_landscape_score:
                    min_landscape_score = cell_landscape_score
                    optimal_cell = (i, j)

    return optimal_cell

def begin_search(agent, version: str, part2_flag=None) -> Tuple[int, int, int]:

    if version == "rule_2" or version == "agent_2" or version == "agent_3":
        # Assigns initial probability values based on likelihood target is seen
        updated_queue = PriorityQueue()
        total_probability_sum = 0

        for i in range(agent._d):
            for j in range(agent._d):
                agent._belief_states[i][j] *= (1 - agent._landscape[i][j])
                total_probability_sum += agent._belief_states[i][j]

        # Normalize probabilities
        scaling_factor = 1 / total_probability_sum
        for i in range(agent._d):
            for j in range(agent._d):
                agent._belief_states[i][j] = agent._belief_states[i][j] * scaling_factor
                updated_queue.put((-1 * agent._belief_states[i][j], (i, j)))
        agent._probability_queue = updated_queue

    target_found = False
    prev_cell = None

    search_count = 1
    move_count = 0

    # Pick first cell, random if for rule 1 or agent 1
    if version == "rule_1" or version == "agent_1":
        curr_cell = _pick_random_cell()
        target_found = _search_cell(agent, curr_cell)

        # Find whether we're 5 distance if target not found for part 2
        if not target_found and part2_flag:
            _update_part_2(agent, curr_cell)

        _update_probability_queue(agent, curr_cell, -1 / agent._d ** 2, version, part2_flag)
        prev_cell = (0, curr_cell)

    # Pick optimal flat terrain cell as the initial random cell
    elif version == "improved":
        curr_cell = _pick_improved_initial_cell(agent)

        # There was no flat terrain in the board, but this would only happen with a 1x1 or a 2x2
        if curr_cell == None:
            curr_cell = _pick_random_cell()

        # Find whether we're 5 distance if target not found for part 2
        if not target_found and part2_flag:
            _update_part_2(agent, curr_cell)

        target_found = _search_cell(agent, curr_cell)
        _update_probability_queue(agent, curr_cell, -1 / self._d ** 2, version, part2_flag)
        prev_cell = (0, curr_cell)

    else:
        # Takes the top cell and put it back
        curr_cell = agent._probability_queue.get()
        agent._probability_queue.put(curr_cell)

        # Find whether we're 5 distance if target not found for part 2
        if not target_found and part2_flag:
            _update_part_2(agent, curr_cell[1])

        target_found = _search_cell(agent, curr_cell[1])
        _update_probability_queue(agent, curr_cell[1], agent._belief_states[curr_cell[1][0]][curr_cell[1][1]], version, part2_flag)

        # Update prev cell
        prev_cell = curr_cell

    while not target_found:
        if version == "rule_1" or version == "rule_2":
            # Remove top cell
            curr_cell = agent._probability_queue.get()
            agent._probability_queue.put(curr_cell)

            # Find whether we're 5 distance if target not found for part 2
            if not target_found and part2_flag:
                _update_part_2(agent, curr_cell[1])

            search_count += 1
            target_found = _search_cell(agent, curr_cell[1])
            _update_probability_queue(agent, curr_cell[1], curr_cell[0], version, part2_flag)

        elif version == "agent_1" or version == "agent_2" or version == "agent_3" or version == "improved":

            # Takes the top cell and put it back
            curr_cell = agent._probability_queue.get()
            agent._probability_queue.put(curr_cell)

            # Determines new curr cell after handling ties
            curr_cell = _handle_ties(agent, curr_cell, prev_cell)
            agent._probability_queue.put(curr_cell)

            # Find whether we're 5 distance if target not found for part 2
            if not target_found and part2_flag:
                _update_part_2(agent, curr_cell[1])

            # Calculate number of actions moving to curr cell would take
            move_count += _calculate_distance(curr_cell[1], prev_cell[1])
            search_count += 1
            target_found = _search_cell(agent, curr_cell[1])
            _update_probability_queue(agent, curr_cell[1], agent._belief_states[curr_cell[1][0]][curr_cell[1][1]], version, part2_flag)
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
