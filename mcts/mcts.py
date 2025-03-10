from __future__ import print_function

import random
from . import utils


class MCTS(object):
    """
    The central MCTS class, which performs the tree search. It gets a
    tree policy, a default policy, and a backup strategy.
    See e.g. Browne et al. (2012) for a survey on monte carlo tree search
    """

    def __init__(self, tree_policy, default_policy, backup):
        self.tree_policy = tree_policy
        self.default_policy = default_policy
        self.backup = backup

    def __call__(self, root, n=1500):
        """
        Run the monte carlo tree search.

        :param root: The StateNode
        :param n: The number of roll-outs to be performed
        :return:
        """
        if root.parent is not None:
            raise ValueError("Root's parent must be None.")

        for _ in range(n):
            node = get_next_node(root, self.tree_policy)
            node.reward = self.default_policy(node)
            self.backup(node)

        return utils.rand_max(root.children.values(), key=lambda x: x.q).action


def expand(state_node):
    action = random.choice(state_node.untried_actions)
    return state_node.children[action].sample_state()


def best_child(state_node, tree_policy):
    best_action_node = utils.rand_max(state_node.children.values(),
                                      key=tree_policy)
    return best_action_node.sample_state()


def next_layer_node(state_node, tree_policy):
    if state_node.untried_actions:
        return expand(state_node)
    else:
        state_node = best_child(state_node, tree_policy)
    return state_node


def get_next_node(state_node, tree_policy):
    while not state_node.state.is_terminal():
        if state_node.untried_actions:
            return expand(state_node)
        else:
            state_node = best_child(state_node, tree_policy)
    return state_node


def get_best_trace(state_node, tree_policy):
    trace = [state_node]
    while not state_node.state.is_terminal():
        if state_node.untried_actions:
            trace.append(expand(state_node))
            break
        else:
            state_node = best_child(state_node, tree_policy)
            trace.append(state_node)
    return trace
