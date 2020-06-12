import numpy as np
import torch
from typing import List
from nltk.tree import Tree

from better_mistakes.trees import get_label
from better_mistakes.data.softmax_cascade import SoftmaxCascade


class HierarchicalLLLoss(torch.nn.Module):
    """
    Hierachical log likelihood loss.

    The weights must be implemented as a nltk.tree object and each node must
    be a float which corresponds to the weight associated with the edge going
    from that node to its parent. The value at the origin is not used and the
    shapre of the weight tree must be the same as the associated hierarchy.

    The input is a flat probability vector in which each entry corresponds to
    a leaf node of the tree. We use alphabetical ordering on the leaf nodes
    labels, which corresponds to the 'normal' imagenet ordering.

    Args:
        hierarchy: The hierarchy used to define the loss.
        classes: A list of classes defining the order of the leaf nodes.
        weights: The weights as a tree of similar shape as hierarchy.
    """

    def __init__(self, hierarchy: Tree, classes: List[str], weights: Tree):
        super(HierarchicalLLLoss, self).__init__()

        assert hierarchy.treepositions() == weights.treepositions()

        # the tree positions of all the leaves
        positions_leaves = {get_label(hierarchy[p]): p for p in hierarchy.treepositions("leaves")}
        num_classes = len(positions_leaves)

        # we use classes in the given order
        positions_leaves = [positions_leaves[c] for c in classes]

        # the tree positions of all the edges (we use the bottom node position)
        positions_edges = hierarchy.treepositions()[1:]  # the first one is the origin

        # map from position tuples to leaf/edge indices
        index_map_leaves = {positions_leaves[i]: i for i in range(len(positions_leaves))}
        index_map_edges = {positions_edges[i]: i for i in range(len(positions_edges))}

        # edge indices corresponding to the path from each index to the root
        edges_from_leaf = [[index_map_edges[position[:i]] for i in range(len(position), 0, -1)] for position in positions_leaves]

        # get max size for the number of edges to the root
        num_edges = max([len(p) for p in edges_from_leaf])

        # helper that returns all leaf positions from another position wrt to the original position
        def get_leaf_positions(position):
            node = hierarchy[position]
            if isinstance(node, Tree):
                return node.treepositions("leaves")
            else:
                return [()]

        # indices of all leaf nodes for each edge index
        leaf_indices = [[index_map_leaves[position + leaf] for leaf in get_leaf_positions(position)] for position in positions_edges]

        # save all relevant information as pytorch tensors for computing the loss on the gpu
        self.onehot_den = torch.nn.Parameter(torch.zeros([num_classes, num_classes, num_edges]), requires_grad=False)
        self.onehot_num = torch.nn.Parameter(torch.zeros([num_classes, num_classes, num_edges]), requires_grad=False)
        self.weights = torch.nn.Parameter(torch.zeros([num_classes, num_edges]), requires_grad=False)

        # one hot encoding of the numerators and denominators and store weights
        for i in range(num_classes):
            for j, k in enumerate(edges_from_leaf[i]):
                self.onehot_num[i, leaf_indices[k], j] = 1.0
                self.weights[i, j] = get_label(weights[positions_edges[k]])
            for j, k in enumerate(edges_from_leaf[i][1:]):
                self.onehot_den[i, leaf_indices[k], j] = 1.0
            self.onehot_den[i, :, j + 1] = 1.0  # the last denominator is the sum of all leaves

    def forward(self, inputs, target):
        """
        Foward pass, computing the loss.

        Args:
            inputs: Class _probabilities_ ordered as the input hierarchy.
            target: The index of the ground truth class.
        """
        # add a sweet dimension to inputs
        inputs = torch.unsqueeze(inputs, 1)
        # sum of probabilities for numerators
        num = torch.squeeze(torch.bmm(inputs, self.onehot_num[target]))
        # sum of probabilities for denominators
        den = torch.squeeze(torch.bmm(inputs, self.onehot_den[target]))
        # compute the neg logs for non zero numerators and store in there
        idx = num != 0
        num[idx] = -torch.log(num[idx] / den[idx])
        # weighted sum of all logs for each path (we flip because it is numerically more stable)
        num = torch.sum(torch.flip(self.weights[target] * num, dims=[1]), dim=1)
        # return sum of losses / batch size
        return torch.mean(num)


class HierarchicalCrossEntropyLoss(HierarchicalLLLoss):
    """
    Combines softmax with HierachicalNLLLoss. Note that the softmax is flat.
    """

    def __init__(self, hierarchy: Tree, classes: List[str], weights: Tree):
        super(HierarchicalCrossEntropyLoss, self).__init__(hierarchy, classes, weights)

    def forward(self, inputs, index):
        return super(HierarchicalCrossEntropyLoss, self).forward(torch.nn.functional.softmax(inputs, 1), index)


class CosineLoss(torch.nn.Module):
    """
    Cosine Distance loss.
    """

    def __init__(self, embedding_layer):
        super(CosineLoss, self).__init__()
        self._embeddings = embedding_layer

    def forward(self, inputs, target):
        inputs = torch.nn.functional.normalize(inputs, p=2, dim=1)
        emb_target = self._embeddings(target)
        return 1 - torch.nn.functional.cosine_similarity(inputs, emb_target).mean()


class CosinePlusXentLoss(torch.nn.Module):
    """
    Cosine Distance + Cross-entropy loss.
    """

    def __init__(self, embedding_layer, xent_weight=0.1):
        super(CosinePlusXentLoss, self).__init__()
        self._embeddings = embedding_layer
        self.xent_weight = xent_weight

    def forward(self, inputs, target):
        inputs_cosine = torch.nn.functional.normalize(inputs, p=2, dim=1)
        emb_target = self._embeddings(target)
        loss_cosine = 1 - torch.nn.functional.cosine_similarity(inputs_cosine, emb_target).mean()
        loss_xent = torch.nn.functional.cross_entropy(inputs, target).mean()
        return loss_cosine + self.xent_weight * loss_xent


class RankingLoss(torch.nn.Module):
    """
    Ranking Loss implementation used for DeViSe
    """

    def __init__(self, embedding_layer, batch_size, single_random_negative, margin=0.1):
        super(RankingLoss, self).__init__()
        self._embeddings = embedding_layer
        self._vocab_len = embedding_layer.weight.size()[0]
        self._margin = margin
        self._single_random_negative = single_random_negative
        self._batch_size = batch_size

    def forward(self, inputs, target):
        inputs = torch.nn.functional.normalize(inputs, p=2, dim=1)
        dot_product = torch.mm(inputs, self._embeddings.weight.t())
        true_embeddings = self._embeddings(target)
        negate_item = torch.sum(inputs * true_embeddings, dim=1, keepdim=True)
        # use a single random negative sample that violates the margin (as discussed in the Devise paper after the definition of the loss)
        if self._single_random_negative:
            dot_product_pruned = torch.zeros_like(negate_item)
            mask_margin_violating = self._margin + dot_product - negate_item > 0
            for i in range(self._batch_size):
                mask = mask_margin_violating[i, :] != 0
                num_valid = torch.sum(mask).item()
                margin_violating_samples_i = (mask_margin_violating[i, :] != 0).nonzero().squeeze()
                if num_valid > 1:
                    rnd_id = np.random.choice(num_valid, 1)
                    rnd_val = dot_product[i, margin_violating_samples_i[rnd_id]]
                else:
                    rnd_val = dot_product[i, margin_violating_samples_i]
                dot_product_pruned[i, 0] = rnd_val

            dot_product = dot_product_pruned

        full_rank_mat = self._margin + dot_product - negate_item
        relu_mat = torch.nn.ReLU()(full_rank_mat)
        summed_mat = torch.sum(relu_mat, dim=1)
        return summed_mat.mean()


class YOLOLoss(torch.nn.Module):
    """
    Loss implemented in YOLO-v2.

    The hierarchy must be implemented as a nltk.tree object and the weight tree
    is a tree of the same shape as the hierarchy whose labels node must be
    floats which corresponds to the weight associated with the cross entropy
    at the node. Values at leaf nodes are ignored.

    There must be one input probability per node exept the origin. The
    probabilities at each node represent the conditional probability of this
    node give its parent. We use nltk internal ordering for the
    leaf nodes and their indices are obtained using Tree.treepositions().
    """

    def __init__(self, hierarchy: Tree, classes: List[str], weights: Tree):
        """
        Initialise the loss with a given hierarchy.

        Args:
            hierarchy: The hierarchy used to define the loss.
            classes: A list of classes defining the order of all nodes.
            weights: The weights as a tree of similar shapre as hierarchy.
        """
        super(YOLOLoss, self).__init__()

        assert hierarchy.treepositions() == weights.treepositions()

        self.cascade = SoftmaxCascade(hierarchy, classes)

        weights_dict = {get_label(hierarchy[p]): get_label(weights[p]) for p in weights.treepositions()}
        self.weights = torch.nn.Parameter(torch.unsqueeze(torch.tensor([weights_dict[c] for c in classes], dtype=torch.float32), 0), requires_grad=False)

    def forward(self, inputs, target):
        """
        Foward pass, computing the loss.

        Args:
            inputs: Class _logits_ ordered as the input hierarchy.
            target: The index of the ground truth class.
        """
        return self.cascade.cross_entropy(inputs, target, self.weights)
