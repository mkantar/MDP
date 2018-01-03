from world import Grid
import numpy as np
import mdp


def main():
    g = Grid(4, 4)

    terminals = [{"x": 3, "y": 0, "reward": 1}, {"x": 1, "y": 3, "reward": 1}, {"x": 2, "y": 3, "reward": -10},
                 {"x": 3, "y": 3, "reward": 10}]
    blocks = [{"x": 1, "y": 1}]

    g.init_world(terminals, blocks)

    np.random.seed(62)

    mdp.value_iteration(g, -0.02, 0.8, 0.8)

    mdp.policy_iteration(g, -0.02, 0.8, 0.8)

    mdp.q_function(g, "s6", -0.02, 0.8, 0.1, 0.1, 0.8, 1000000)


if __name__ == '__main__':
    main()
