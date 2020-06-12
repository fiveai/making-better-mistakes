import torch
from nltk import Tree
from typing import List
from collections import defaultdict
from better_mistakes.trees import get_label


class SoftmaxCascade(torch.nn.Module):
    def __init__(self, hierarchy: Tree, classes: List[str]):
        """
        Initialise the cascade with a given hierarchy.

        Args:
            hierarchy: The hierarchy used to define the loss.
            classes: A list of classes defining the order of all nodes.
        """
        super(SoftmaxCascade, self).__init__()

        # get the position of all leaf classes
        positions_leaves = hierarchy.treepositions("leaves")
        num_classes = len(positions_leaves)

        # the tree positions of all the edges
        positions_edges = {get_label(hierarchy[p]): p for p in hierarchy.treepositions()}
        positions_edges = [positions_edges[c] for c in classes]  # we use classes in the given order
        num_edges = len(positions_edges)

        # we require one class == one node in the tree (including root)
        assert num_edges == len(classes), "Number of classes doesnt match: {} != {}".format(num_classes, len(classes))

        # check that the labels match
        for i, position in enumerate(positions_edges):
            assert get_label(hierarchy[position]) == classes[i], "Labels do not match!"

        # map from position tuples to leaf/edge indices
        index_map_edges = {position: i for i, position in enumerate(positions_edges)}

        # for each edge get the indices of its children
        index_map_children = defaultdict(list)
        for position in positions_edges:
            if len(position):  # skip the root
                index_map_children[position[:-1]].append(index_map_edges[position])
            else:
                # add the root (normalises to one) with a dummy key
                index_map_children["ROOT"].append(index_map_edges[()])

        num_softmaxes = len(index_map_children)

        # number of softmaxes is the number of non-leaf nodes (+ the one for the root)
        assert num_softmaxes == num_edges - num_classes + 1, "Number of softmaxes doesnt match: {} != {}".format(num_softmaxes, num_edges - num_classes)

        # store each softmax as a binary mask
        self.softmax_masks = torch.nn.Parameter(torch.zeros([num_softmaxes, num_edges], dtype=torch.bool), requires_grad=False)
        for i, children in enumerate(index_map_children.values()):
            for j in children:
                self.softmax_masks[i, j] = True

        # edge indices corresponding to the path from each index to the root
        edges_from_leaf = [[index_map_edges[position[:i]] for i in range(len(position), -1, -1)] for position in positions_edges]

        # store list of conditionals contributing to a final prob as one hot
        self.path_onehot = torch.nn.Parameter(torch.zeros([num_edges, num_edges]), requires_grad=False)

        for position, edges in zip(positions_edges, edges_from_leaf):
            leaf_idx = index_map_edges[position]
            self.path_onehot[leaf_idx, leaf_idx] = 1
            for j in edges:
                self.path_onehot[leaf_idx, j] = 1

    def normalise(self, inputs, norm_fn=torch.softmax):
        """
        Normalise according to the cascade of softmaxes.

        Args:
            inputs: A vector of shape [batch_norm, num_edges] to normalise.
            norm_fn: The function used to perform normalisation.
        """
        out = inputs * 0
        for mask in self.softmax_masks:
            out[:, mask] = norm_fn(inputs[:, mask], dim=1)
        return out

    def cross_entropy(self, inputs, target, weights):
        """
        Compute the cross entropy.

        Args:
            inputs: Unnormalised inputs (logits).
            target: The index of the ground truth class.
            weights: The weights ordered as the blabla.
        """
        batch_size = inputs.size()[0]
        normalised_inputs = self.normalise(inputs, norm_fn=torch.log_softmax)
        onehot = self.path_onehot[target]
        return -torch.sum(normalised_inputs * onehot * weights) / batch_size

    def final_probabilities(self, inputs, norm_fn=torch.exp):
        """
        Get final probabilities as a product of conditionals.

        Args:
            inputs: A vector of shape [batch_norm, num_edges].
            norm_fn: The normalisation function, set to torch.exp tp get
                probabilities and to lambda x:x to get log probabilities.
        """
        log_prob = self.normalise(inputs, norm_fn=torch.log_softmax)
        return norm_fn(torch.mm(log_prob, torch.transpose(self.path_onehot, 0, 1)))
