class Square:
    def __init__(self, t, reward=None, name=None, utility=0):
        self.__type = t
        self.__reward = reward
        self.__name = name
        self.__utility = utility
        self.__neighbors = {}
        self.__policy = None
        self.__q = None

    def get_type(self):
        return self.__type

    def get_reward(self):
        return self.__reward

    def set_reward(self, reward):
        self.__reward = reward

    def get_name(self):
        return self.__name

    def get_utility(self):
        return self.__utility

    def update_utility(self, utility):
        self.__utility = utility

    def update_neighbors(self, lst):
        self.__neighbors.update(lst)

    def get_neighbors(self):
        return self.__neighbors

    @staticmethod
    def __is_valid(state):
        if state is None:
            return False
        elif state.get_type() == "B":
            return False
        else:
            return True

    @staticmethod
    def __argmax(lst):
        return lst.index(max(lst))

    def best_policy(self, p):
        utils = []
        direction = ["N", "E", "S", "W"]
        neighbor = self.__neighbors

        for i in range(len(direction)):
            val = 0
            if self.__is_valid(neighbor[direction[i - 1]]):
                val += p["L"] * neighbor[direction[i - 1]].get_utility()
            else:
                val += p["L"] * self.get_utility()

            if self.__is_valid(neighbor[direction[i]]):
                val += p["U"] * neighbor[direction[i]].get_utility()
            else:
                val += p["U"] * self.get_utility()

            if self.__is_valid(neighbor[direction[(i + 1) % len(direction)]]):
                val += p["R"] * neighbor[direction[(i + 1) % len(direction)]].get_utility()
            else:
                val += p["R"] * self.get_utility()

            utils.append(val)

        index = self.__argmax(utils)

        self.update_policy(direction[int(index)])

        return utils

    def get_policy(self):
        return self.__policy

    def update_policy(self, policy):
        self.__policy = policy

    def policy_evaluation(self, p, direction):
        directions = ["N", "E", "S", "W"]
        neighbor = self.__neighbors
        i = directions.index(direction)

        val = 0
        if self.__is_valid(neighbor[directions[i - 1]]):
            val += p["L"] * neighbor[directions[i - 1]].get_utility()
        else:
            val += p["L"] * self.get_utility()

        if self.__is_valid(neighbor[directions[i]]):
            val += p["U"] * neighbor[directions[i]].get_utility()
        else:
            val += p["U"] * self.get_utility()

        if self.__is_valid(neighbor[directions[(i + 1) % len(directions)]]):
            val += p["R"] * neighbor[directions[(i + 1) % len(directions)]].get_utility()
        else:
            val += p["R"] * self.get_utility()

        return val

    def get_q_values(self):
        return self.__q

    def get_q_value(self, i):
        return self.__q[i]

    def set_q_values(self, q):
        self.__q = q

    def update_q_value(self, val, i):
        self.__q[i] = val

    def get_next(self, action):
        directions = ["N", "E", "S", "W"]
        neighbor = self.__neighbors
        i = action
        next_states = []

        for j in range(-1, 2):
            if self.__is_valid(neighbor[directions[(i + j) % len(directions)]]):
                next_states.append(neighbor[directions[(i + j) % len(directions)]])
            else:
                next_states.append(self)

        return next_states


class Grid:
    def __init__(self, width, height):
        self.__width = width
        self.__height = height
        self.__mtx = []
        self.__policy = []

    def init_world(self, terminals, blocks=None):
        count = 0
        for y in range(self.__height):
            one = []
            for x in range(self.__width):
                if self.__is_terminal(y, x, terminals):
                    t = self.__get_terminal(y, x, terminals)
                    s = Square("T", t["reward"], utility=t["reward"])
                    one.append(s)
                elif self.__is_blocked(y, x, blocks):
                    s = Square("B")
                    one.append(s)
                else:
                    s = Square("N", name="s{0}".format(count))
                    one.append(s)
                    count += 1
            self.__mtx.append(one)

        self.__make_neighbors()

    def __make_neighbors(self):
        for y in range(self.__height):
            for x in range(self.__width):
                s = self.__mtx[y][x]
                if s.get_type() == "N":
                    if y - 1 >= 0:
                        a = self.__mtx[y - 1][x]
                        s.update_neighbors({"N": a})
                    else:
                        s.update_neighbors({"N": None})

                    if y + 1 < self.__height:
                        a = self.__mtx[y + 1][x]
                        s.update_neighbors({"S": a})
                    else:
                        s.update_neighbors({"S": None})

                    if x - 1 >= 0:
                        a = self.__mtx[y][x - 1]
                        s.update_neighbors({"W": a})
                    else:
                        s.update_neighbors({"W": None})

                    if x + 1 < self.__width:
                        a = self.__mtx[y][x + 1]
                        s.update_neighbors({"E": a})
                    else:
                        s.update_neighbors({"E": None})

    @staticmethod
    def __is_terminal(y, x, terminals):
        for terminal in terminals:
            if (terminal["x"] == x) and (terminal["y"] == y):
                return True
        return False

    @staticmethod
    def __get_terminal(y, x, terminals):
        for terminal in terminals:
            if (terminal["x"] == x) and (terminal["y"] == y):
                return terminal

    @staticmethod
    def __is_blocked(y, x, blocks):
        for block in blocks:
            if (block["x"] == x) and (block["y"] == y):
                return True
        return False

    def get_utilities(self):
        util = []
        for states in self.__mtx:
            for state in states:
                if state.get_type() == "N":
                    util.append(state.get_utility())
        return util

    def reset_utilities(self):
        for states in self.__mtx:
            for state in states:
                if state.get_type() == "N":
                    state.update_utility(0)

    def reset_policies(self):
        for states in self.__mtx:
            for state in states:
                if state.get_type() == "N":
                    state.update_policy(None)

    def reset_q_values(self):
        for states in self.__mtx:
            for state in states:
                if state.get_type() == "N":
                    state.set_q_values([0] * 4)
                elif state.get_type() == "T":
                    state.set_q_values([state.get_utility()] * 4)

    def get_q_matrix(self):
        matrix = []
        for states in self.__mtx:
            for state in states:
                if state.get_type() == "N":
                    matrix.append(state.get_q_values())

        return matrix

    def set_reward(self, reward):
        for states in self.__mtx:
            for state in states:
                if state.get_type() == "N":
                    state.set_reward(reward)

    def get_policies(self):
        policy = []

        for items in self.__mtx:
            for item in items:
                if item.get_type() == "N":
                    policy.append(item.get_policy())

        return policy

    def get_states(self):
        return self.__mtx

    def get_state(self, name):
        for states in self.__mtx:
            for state in states:
                if state.get_name() == name:
                    return state

    def print_q_values(self):
        for i in range(4):
            print("q{0:<4}".format(i), end=" ")
            for states in self.__mtx:
                for state in states:
                    if state.get_type() == "N":
                        print("{0:<8.3f}".format(state.get_q_value(i)), end=" ")
            print()

    def print_optimal_policy(self):
        for states in self.__mtx:
            print("{0:4}".format(" "), end=" ")
            for state in states:
                if state.get_type() == "N":
                    print(state.get_policy(), end=" ")
                else:
                    print(state.get_type(), end=" ")
            print()
