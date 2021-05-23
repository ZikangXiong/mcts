import numpy as np
from mcts.backups import monte_carlo
from mcts.default_policies import immediate_reward
from mcts.graph import StateNode
from mcts.mcts import get_next_node, get_best_trace
from mcts.tree_policies import UCB1
from mcts.utils import rand_max


class MazeAction(object):
    def __init__(self, move):
        self.move = np.asarray(move)

    def __eq__(self, other):
        return all(self.move == other.move)

    def __hash__(self):
        return int(10 * self.move[0] + self.move[1])


class MazeState(object):
    def __init__(self, pos):
        self.pos = np.asarray(pos)
        self.actions = [MazeAction([1, 0]),
                        MazeAction([0, 1]),
                        MazeAction([-1, 0]),
                        MazeAction([0, -1])]

    def perform(self, action):
        pos = self.pos + action.move
        pos = np.clip(pos, 0, 2)
        return MazeState(pos)

    def reward(self, parent, action):
        if all(self.pos == np.array([2, 2])):
            return 10
        else:
            return -1

    def is_terminal(self):
        return False

    def __eq__(self, other):
        return all(self.pos == other.pos)

    def __hash__(self):
        return int(10 * self.pos[0] + self.pos[1])


if __name__ == '__main__':
    tree_policy = UCB1(c=1.41)
    default_policy = immediate_reward
    backup = monte_carlo
    root = StateNode(parent=None, state=MazeState([0, 0]))

    for _ in range(1500):
        node = get_next_node(root, tree_policy)
        node.reward = default_policy(node)
        backup(node)

    trace = get_best_trace(root, tree_policy)
    states = [node.state.pos for node in trace]
    print(states)
