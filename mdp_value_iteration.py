import copy


# Class that defines an MDP for a grid world, and the different properties in it
class GridWorldMDP:

    # definition of all the elements in the MDP and grid world
    def __init__(self, rows, cols, walls, terminal_states, reward, transition, discount_factor, epsilon):
        self.num_rows = rows
        self.num_columns = cols
        self.wall_location = walls
        self.terminal_states = terminal_states
        self.reward_non_terminal = reward
        self.transition_prob = transition
        self.discount_rate = discount_factor
        self.epsilon = epsilon
        self.actions = ['N', 'E', 'S', 'W']
        self.states = self.create_states()
        self.transition_model = self.create_transition_model(self.states, self.actions)
        self.reward_model = self.create_reward_model(self.states)

    # Method to perform value iteration in the current MDP
    def value_iteration(self):
        iteration = 0
        utility_dic = {}  # dictionary to hold the maximum utility of each state
        moves_dic = {}  # dictionary to hold the best possible move of each state

        for state in self.states:
            if state in self.wall_location:  # state is a wall
                utility_dic[state] = "--------------"
                moves_dic[state] = "-"
            elif state in self.terminal_states:  # state is a terminal state
                utility_dic[state] = 0
                moves_dic[state] = "T"
            else:
                utility_dic[state] = 0
                moves_dic[state] = " "

        print("################ VALUE ITERATION ###########################")

        while True:
            delta = 0
            old_utility_dic = copy.deepcopy(utility_dic)  # keep copy of the old utilities

            print("Iteration: " + str(iteration))

            self.print_grid_world_utility(utility_dic)

            for state in self.states:
                if state not in self.wall_location:

                    best_action = max_a(utility_dic, self.transition_model[state])  # get the best action, with utility
                    moves_dic[state] = best_action[0]  # save best move
                    utility_dic[state] = (self.reward_model[state] + (self.discount_rate * best_action[1]))

                    if abs(old_utility_dic[state] - utility_dic[state]) > delta:
                        delta = abs(old_utility_dic[state] - utility_dic[state])

            if delta <= self.epsilon * (1 - self.discount_rate) / self.discount_rate:  # check to keep looping
                break

            iteration += 1

        print("Final policy")
        self.print_grid_world_policy(moves_dic)

    # Prints the grid world in terms of the utility of each state
    def print_grid_world_utility(self, utility_dic):
        for i in range(self.num_rows, 0, -1):
            for j in range(1, self.num_columns + 1):
                print(str(utility_dic[State(j, i)]) + " ", end='')

            print("\n")

    # Prints the grid world in terms of the move of each state
    def print_grid_world_policy(self, moves_dic):
        for i in range(self.num_rows, 0, -1):
            for j in range(1, self.num_columns + 1):
                state = State(j, i)
                if state in self.terminal_states:  # Terminal state
                    print("T ", end='')
                else:
                    print(str(moves_dic[state]) + " ", end='')

            print("\n")

    # Method that calculates a move of a state based on its current position. If there is a wall or move
    # is not possible, it returns the same state
    def move(self, state, action):
        if action == 'N' and state.y < self.num_rows and State(state.x, state.y + 1) not in self.wall_location:
            return State(state.x, state.y + 1)

        if action == 'E' and state.x < self.num_columns and State(state.x + 1, state.y) not in self.wall_location:
            return State(state.x + 1, state.y)

        if action == 'S' and state.y > 1 and State(state.x, state.y - 1) not in self.wall_location:
            return State(state.x, state.y - 1)

        if action == 'W' and state.x > 1 and State(state.x - 1, state.y) not in self.wall_location:
            return State(state.x - 1, state.y)

        return state

    # Method to create the states in the grid world, for the MDP
    def create_states(self):
        states = []
        for i in range(1, self.num_rows + 1):
            for j in range(1, self.num_columns + 1):
                states.append(State(j, i))
        return states

    # Method to create the reward model (dictionary) for each state
    def create_reward_model(self, states):
        reward_model = {}
        # define reward function
        for state in states:
            if state in self.terminal_states:
                reward_model[state] = self.terminal_states[state]
            else:
                reward_model[state] = self.reward_non_terminal
        return reward_model

    # Method to create the transition model. It gives a dictionary of each state, with all their possible moves
    # and their probabilities
    def create_transition_model(self, states, actions):
        transition_model = {}   # dictionary to hold all the possible moves for each state
        for state in states:

            transition_model[state] = {}  # dictionary for a particular state

            for action in actions:

                move_left = actions[(actions.index(action) - 1) % 4]  # moving to the left of current action
                move_right = actions[(actions.index(action) + 1) % 4]  # moving to the right of current action

                # no movements from terminal state
                if state in self.terminal_states:
                    transition_model[state][action] = [(0.0, self.move(state, action)),
                                                       (0.0, self.move(state, move_left)),
                                                       (0.0, self.move(state, move_right))]
                else:
                    transition_model[state][action] = [(0.8, self.move(state, action)),
                                                       (0.1, self.move(state, move_left)),
                                                       (0.1, self.move(state, move_right))]
        return transition_model


# Function that performs the Bellman equation and determines the best state (move) based on utility
def max_a(utility_dict, transition_actions):
    actions = []  # holds the possible values for each action of a single state

    for key, value in transition_actions.items():  # for each possible move N, E, S, W
        temp = 0
        for i in range(len(value)):
            prob, next_state = value[i]
            temp += prob * float(utility_dict[next_state])

        actions.append((key, temp))

    return max(actions, key=lambda x: x[1])


# Class that defines a state in the grid world
class State:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return "x= " + str(self.x) + " y= " + str(self.y)

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


def parse_file(input_filename):
    rows = 0
    cols = 0
    walls = []
    terminal_states = {}
    non_terminal_rewards = 0
    transition_probs = 0
    discount_rate = 0
    epsilon = 0

    with open(input_filename, "r") as file:
        for line in file:
            if ':' in line:
                key, value = line.strip().split(':')
                if key.strip() == 'size':   # checking for size
                    rows = int(value.strip().split(" ")[1])
                    cols = int(value.strip().split(" ")[0])

                elif key.strip() == 'walls':    # checking for location of walls
                    walls_indexes = value.split(",")
                    for wall in walls_indexes:  # adding each wall to a state
                        walls.append(State(int(wall.strip().split(" ")[0]), int(wall.strip().split(" ")[1])))

                elif key.strip() == 'terminal_states':
                    terminal_state_rewards = value.split(",")
                    for terminal_state in terminal_state_rewards:
                        row, col, reward = terminal_state.strip().split(" ")
                        terminal_states[State(int(row), int(col))] = float(reward)

                elif key.strip() == 'reward':
                    non_terminal_rewards = float(value.strip())

                elif key.strip() == 'transition_probabilities':
                    transition_probs = value.strip().split(" ")

                elif key.strip() == 'discount_rate':
                    discount_rate = float(value.strip())

                elif key.strip() == 'epsilon':
                    epsilon = float(value.strip())

    print(rows, cols, walls, terminal_states, non_terminal_rewards, transition_probs, discount_rate, epsilon)

    return GridWorldMDP(rows, cols, walls, terminal_states, non_terminal_rewards, transition_probs, discount_rate,
                        epsilon)


if __name__ == "__main__":
    # type in filename of input. The file should be formatted as the sample input file, so the program
    # can parse it correctly
    grid_world_mdp = parse_file("mdp_input.txt")
    grid_world_mdp.value_iteration()