import numpy as np
import matplotlib.pyplot as plt


def value_iteration(grid, reward, discount, probability, threshold=0.0000000000000000000001):
    grid.reset_utilities()
    grid.reset_policies()
    grid.set_reward(reward)
    p = {"U": probability, "L": (1 - probability) / 2, "R": (1 - probability) / 2}

    states_arr = grid.get_states()
    print("Value iteration: reward: {0}, discount: {1}, probability {2}\n".format(reward, discount, probability))
    print("{0:^4}".format(" "), end=" ")

    for states in states_arr:
        for state in states:
            if state.get_type() == "N":
                print("{0:^8}".format(state.get_name()), end=" ")

    print()

    count = 0
    while 1:
        print("it{0:<4}".format(count), end=" ")
        old_utils = grid.get_utilities()
        for states in states_arr:
            for state in states:
                if state.get_type() == "N":
                    util = state.get_reward() + discount * max(state.best_policy(p))
                    state.update_utility(util)
                    print("{0:<8.3f}".format(util), end=" ")
        new_utils = grid.get_utilities()

        print()

        count += 1
        if is_converged_vi(old_utils, new_utils, threshold):
            print("\nOptimal policy:")
            print("{\"N\": North, \"E\": East, \"S\": South, \"W\": West, \"T\": Terminal, \"B\": Blocked}\n")
            grid.print_optimal_policy()
            break

    print()


def policy_iteration(grid, reward, discount, probability, threshold=0.0000000000000000000001):
    grid.reset_utilities()
    grid.reset_policies()
    grid.set_reward(reward)
    p = {"U": probability, "L": (1 - probability) / 2, "R": (1 - probability) / 2}

    states_arr = grid.get_states()

    print("Policy iteration: reward: {0}, discount: {1}, probability {2}\n".format(reward, discount, probability))
    print("{0:^4}".format(" "), end=" ")

    for states in states_arr:
        for state in states:
            if state.get_type() == "N":
                print("{0:^8}".format(state.get_name()), end=" ")

    print()

    count = 0

    old_policy = grid.get_policies()

    fill_random(old_policy)

    while 1:
        print("it{0:<4}".format(count), end=" ")
        while 1:
            index = 0
            old_utils = grid.get_utilities()
            for states in states_arr:
                for state in states:
                    if state.get_type() == "N":
                        util = state.get_reward() + discount * state.policy_evaluation(p, old_policy[index])
                        state.update_utility(util)
                        index += 1
            new_utils = grid.get_utilities()

            if is_converged_vi(old_utils, new_utils, threshold):
                break

        for states in states_arr:
            for state in states:
                if state.get_type() == "N":
                    state.best_policy(p)
                    print("{0:<8.3f}".format(state.get_utility()), end=" ")

        new_policy = grid.get_policies()

        print()
        count += 1
        if is_converged_pi(old_policy, new_policy):
            print("\nOptimal policy:")
            print("{\"N\": North, \"E\": East, \"S\": South, \"W\": West, \"T\": Terminal, \"B\": Blocked}\n")
            grid.print_optimal_policy()
            print()
            break
        else:
            old_policy = new_policy


def q_function(grid, initial_state, reward, discount, alpha, epsilon, probability, N):
    grid.reset_utilities()
    grid.reset_policies()
    grid.reset_q_values()
    grid.set_reward(reward)
    p = {"U": probability, "L": (1 - probability) / 2, "R": (1 - probability) / 2}
    directions = ["N", "E", "S", "W"]

    print(
        "Q-learning: initial state: {0}, reward: {1}, discount: {2}, alpha: {3}, epsilon: {4},  probability {5}\n".format(
            initial_state, reward, discount, alpha, epsilon, probability))
    print("{0:^4}".format(" "), end=" ")

    states_arr = grid.get_states()

    for states in states_arr:
        for state in states:
            if state.get_type() == "N":
                print("{0:^8}".format(state.get_name()), end=" ")

    print()

    for i in range(N):
        state = grid.get_state(initial_state)

        while 1:

            if np.random.random() < epsilon:
                action = np.random.randint(4)
            else:
                action = np.argmax(state.get_q_values())

            q_val = state.get_q_value(action)
            next_states = state.get_next(action)

            action_next = []
            q_val_next = []

            for j in range(len(next_states)):
                action_next.append(np.argmax(next_states[j].get_q_values()))
                q_val_next.append(next_states[j].get_q_value(action_next[j]))

            q_val = q_val + alpha * (state.get_reward() + (
                discount * (p['U'] * q_val_next[1] + p['L'] * q_val_next[0] + p['R'] * q_val_next[2]) - q_val))

            state.update_q_value(q_val, action)
            q_vals = state.get_q_values()
            state.update_policy(directions[q_vals.index(max(q_vals))])

            if np.random.random() < p['U']:
                next_state = next_states[1]
            else:
                if np.random.random() < p['L']:
                    next_state = next_states[0]
                else:
                    next_state = next_states[2]

            if next_state.get_type() == "T":
                break
            else:
                state = next_state

        '''
        if (i % 100) == 0:
            mtx = grid.get_q_matrix()
            f = np.array(mtx)
            plt.plot(f)
           # plt.show()
        '''

    grid.print_q_values()
    print("\nOptimal policy")
    print("{\"N\": North, \"E\": East, \"S\": South, \"W\": West, \"T\": Terminal, \"B\": Blocked}\n")
    grid.print_optimal_policy()


def is_converged_vi(old, new, th):
    for i in range(len(old)):
        o = old[i]
        n = new[i]
        if abs(n - o) > th:
            return False
    return True


def is_converged_pi(old, new):
    for i in range(len(old)):
        if old[i] != new[i]:
            return False
    return True


def fill_random(arr):
    direction = ["N", "E", "S", "W"]

    for i in range(len(arr)):
        val = np.random.randint(4)
        arr[i] = direction[val]
