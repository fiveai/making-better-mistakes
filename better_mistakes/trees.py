import pickle
import lzma
import os
import numpy as np
from math import exp, fsum
from nltk.tree import Tree
from copy import deepcopy


class DistanceDict(dict):
    """
    Small helper class implementing a symmetrical dictionary to hold distance data.
    """

    def __init__(self, distances):
        self.distances = {tuple(sorted(t)): v for t, v in distances.items()}

    def __getitem__(self, i):
        if i[0] == i[1]:
            return 0
        else:
            return self.distances[(i[0], i[1]) if i[0] < i[1] else (i[1], i[0])]

    def __setitem__(self, i):
        raise NotImplementedError()


def get_label(node):
    if isinstance(node, Tree):
        return node.label()
    else:
        return node


def load_hierarchy(dataset, data_dir):
    """
    Load the hierarchy corresponding to a given dataset.

    Args:
        dataset: The dataset name for which the hierarchy should be loaded.
        data_dir: The directory where the hierarchy files are stored.

    Returns:
        A nltk tree whose labels corresponds to wordnet wnids.
    """
    if dataset in ["tiered-imagenet-84", "tiered-imagenet-224"]:
        fname = os.path.join(data_dir, "tiered_imagenet_tree.pkl")
    elif dataset in ["ilsvrc12", "imagenet"]:
        fname = os.path.join(data_dir, "imagenet_tree.pkl")
    elif dataset in ["inaturalist19-84", "inaturalist19-224"]:
        fname = os.path.join(data_dir, "inaturalist19_tree.pkl")
    else:
        raise ValueError("Unknown dataset {}".format(dataset))

    with open(fname, "rb") as f:
        return pickle.load(f)


def load_distances(dataset, dist_type, data_dir):
    """
    Load the distances corresponding to a given hierarchy.

    Args:
        dataset: The dataset name for which the hierarchy should be loaded.
        dist_type: The distance type, one of ['jc', 'ilsvrc'].
        data_dir: The directory where the hierarchy files are stored.
        shuffle_distances: Create random hierarchy maintaining the same weights

    Returns:
        A dictionary of the form {(wnid1, wnid2): distance} for all precomputed
        distances for the hierarchy.
    """
    assert dist_type in ["ilsvrc", "jc"]

    if dataset in ["tiered-imagenet-224", "tiered-imagenet-84"]:
        dataset = "tiered-imagenet"
    elif dataset in ["ilsvrc12", "imagenet"]:
        dataset = "imagenet"
    elif dataset in ["inaturalist19-224", "inaturalist19-84"]:
        dataset = "inaturalist19"

    with lzma.open(os.path.join(data_dir, "{}_{}_distances.pkl.xz".format(dataset, dist_type).replace("-", "_")), "rb") as f:
        return DistanceDict(pickle.load(f))


def get_uniform_weighting(hierarchy: Tree, value):
    """
    Construct unit weighting tree from hierarchy.

    Args:
        hierarchy: The hierarchy to use to generate the weights.
        value: The value to fill the tree with.

    Returns:
        Weights as a nltk.Tree whose labels are the weights associated with the
        parent edge.
    """
    weights = deepcopy(hierarchy)
    for p in weights.treepositions():
        node = weights[p]
        if isinstance(node, Tree):
            node.set_label(value)
        else:
            weights[p] = value
    return weights


def get_exponential_weighting(hierarchy: Tree, value, normalize=True):
    """
    Construct exponentially decreasing weighting, where each edge is weighted
    according to its distance from the root as exp(-value*dist).

    Args:
        hierarchy: The hierarchy to use to generate the weights.
        value: The decay value.
        normalize: If True ensures that the sum of all weights sums
            to one.

    Returns:
        Weights as a nltk.Tree whose labels are the weights associated with the
        parent edge.
    """
    weights = deepcopy(hierarchy)
    all_weights = []
    for p in weights.treepositions():
        node = weights[p]
        weight = exp(-value * len(p))
        all_weights.append(weight)
        if isinstance(node, Tree):
            node.set_label(weight)
        else:
            weights[p] = weight
    total = fsum(all_weights)  # stable sum
    if normalize:
        for p in weights.treepositions():
            node = weights[p]
            if isinstance(node, Tree):
                node.set_label(node.label() / total)
            else:
                weights[p] /= total
    return weights


def get_weighting(hierarchy: Tree, weighting="uniform", **kwargs):
    """
    Get different weightings of edges in a tree.

    Args:
        hierarchy: The tree to generate the weighting for.
        weighting: The type of weighting, one of 'uniform', 'exponential'.
        **kwards: Keyword arguments passed to the weighting function.
    """
    if weighting == "uniform":
        return get_uniform_weighting(hierarchy, **kwargs)
    elif weighting == "exponential":
        return get_exponential_weighting(hierarchy, **kwargs)
    else:
        raise NotImplementedError("Weighting {} is not implemented".format(weighting))


def get_classes(hierarchy: Tree, output_all_nodes=False):
    """
    Return all classes associated with a hierarchy. The classes are sorted in
    alphabetical order using their label, putting all leaf nodes first and the
    non-leaf nodes afterwards.

    Args:
        hierarhcy: The hierarchy to use.
        all_nodes: Set to true if the non-leaf nodes (excepted the origin) must
            also be included.

    Return:
        A pair (classes, positions) of the array of all classes (sorted) and the
        associated tree positions.
    """

    def get_classes_from_positions(positions):
        classes = [get_label(hierarchy[p]) for p in positions]
        class_order = np.argsort(classes)  # we output classes in alphabetical order
        positions = [positions[i] for i in class_order]
        classes = [classes[i] for i in class_order]
        return classes, positions

    positions = hierarchy.treepositions("leaves")
    classes, positions = get_classes_from_positions(positions)

    if output_all_nodes:
        positions_nl = [p for p in hierarchy.treepositions() if p not in positions]
        classes_nl, positions_nl = get_classes_from_positions(positions_nl)
        classes += classes_nl
        positions += positions_nl

    return classes, positions
