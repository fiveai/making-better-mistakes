import os.path
import configargparse
import numpy as np
import pickle
import csv
from distutils.util import strtobool as boolean
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from copy import deepcopy
from tqdm import tqdm

from nltk import Tree
import networkx as nx

from hierarchies.data.metadata import load_ilsvrc_metadata
from hierarchies.wordnet.synsets import wnid_to_synset, synset_to_wnid
from hierarchies.wordnet.distances import ilsvrc_dist, wup_sim
from hierarchies.util.folders import make_expm_folder

from operator import itemgetter
from tree_format import format_tree

# We change slightly the hierarchy such that traffic lights, traffic signs, and bubble
# are connected to physical object instead of abstraction. This is clearly a mistake in
# the original hierarchy. The problems where:
#   * Traffic sign and traffic light are connected to the abstract sign, use signboard instead.
#   * Bubble is connected to the abstract idea of a sphere, use sphere artifact instead.
EDGE_REPLACE = {
    ('n06793231', 'n06794110'): ['n04217882', 'n06794110'],
    ('n06874019', 'n06874185'): ['n04217882', 'n06874185'],
    ('n13899200', 'n13899404'): ['n04274530', 'n13899404']
}


parser = configargparse.ArgParser(description='Compute the hierarchical tree for ilsvrc as a nltk tree and saves it using pickle.')
parser.add('--dataset', type=str, default='imagenet', choices=['imagenet', 'tiered_imagenet'])
parser.add('--wordnet_edges', type=str, metavar='PATH', default='../data/wordnet_edges.txt')
parser.add('--wordnet_words', type=str, metavar='PATH', default='../data/wordnet_words.txt')
parser.add('--out_dir', type=str, metavar='PATH', default='../data/')
parser.add('--data_dir', type=str, metavar='PATH', default='../data/')
parser.add('-interactive', action='store_true')

opts = parser.parse_args()

# setup path choices
if opts.dataset == 'tiered_imagenet':
    path_choices = [1, 1, 0, 1, 2, 1, 1, 1, 1, 2, 0, 1, 1, 1]
elif opts.dataset == 'imagenet':
    path_choices = [0, 0, 1, 1, 1, 0, 1, 2, 1, 1, 1, 1, 0, 2, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 1, 1, 0]

# get all the wnids for the dataset
with open(os.path.join(opts.data_dir, opts.dataset + '_classes.txt')) as f:
    wnids = [ line.strip('\n') for line in f.readlines()]

# open the words file for nice output
with open(opts.wordnet_words) as f:
    reader = csv.reader(f, delimiter='\t')
    words = {row[0]: row[1].split(',')[0].lower() for row in reader}

# open the edge file in order to build the tree (parent, child)
with open(opts.wordnet_edges) as f:
    reader = csv.reader(f, delimiter=' ')
    edges = []
    for row in reader:
        if tuple(row) in EDGE_REPLACE:
            print('Replacing edge', [words[r] for r in row], end=' ')
            row = list(EDGE_REPLACE[tuple(row)])
            print('by', [words[r] for r in row])
        edges.append(row)

# that's the number of classes/leaf nodes we have
num_classes = len(wnids)
print('Building tree for', num_classes, 'classes')

# get the root
# root = wnid_to_synset(wnids[0])
# for w1 in tqdm(wnids):
#     root = root.lowest_common_hypernyms(wnid_to_synset(w1))[0]
# print('Using', root, '/', synset_to_wnid(root), 'as root')
# root = synset_to_wnid(root)

# because we modify slightly the hierarchy, we need to set the root explicitely here
root = 'n00001930'  # phyical entity

# build full dependency graph using networkx
graph = nx.DiGraph()
for parent, child in edges:
    graph.add_edge(child, parent)

# get edges that are on the longest path from the leaf classes to the root, resolving alternatives as we go
paths = {}
for node in wnids:
    all_paths = list(nx.all_simple_paths(graph, node, root))
    max_len = max([len(p) for p in all_paths])
    max_paths = [p for p in all_paths if len(p) == max_len]
    if(len(max_paths)>1):
        print('Going from', words[node], 'to', words[root], 'has multiple possible paths:')
        for i, path in enumerate(max_paths):
            print('  {}:'.format(i), ', '.join([words[n] for n in path]))
        if opts.interactive:
            print('Choose wisely:', end=' ')
            i = int(input())
        else:
            i = path_choices.pop(0)
        print('Picked', i)
        paths[node] = max_paths[i][1:]

    else:
        paths[node] = max_paths[0][1:]

# select only those edges that are on the shortest path from the ilsvrc classes to root
# paths = {node: nx.shortest_path(graph, node, root)[1:] for node in wnids}

def get_edges(paths):
    edges = set()
    for node, path in paths.items():
        path = [node] + path
        for i in range(1, len(path)):
            edges.add((path[i], path[i-1]))  # still keep the confusing parent/child ordering
    return edges

# get all edges that are still there
edges = get_edges(paths)

# Check that we do indeed have a tree
children = [e[1] for e in edges]
assert len(children) == len(set(children))  # check that all children have only one parent

# prune all non-leaf nodes that have only one child
num_children = Counter([e[0] for e in edges])
paths = {node: [n for n in path if num_children[n]>1 or n==root] for node, path in paths.items()}
edges = get_edges(paths)

# check that we dont have nodes with single child
parents = [e[0] for e in edges]
assert len([i for i, c in Counter(parents).items() if c==1 and i!=root]) == 0

# build the actualy tree using dicts
tree = defaultdict(dict)
for parent, child in edges:
    tree[parent][child] = tree[child]

# build nltk tree recursively
def convert(label, children):
    if children:
        return Tree(label, [convert(k, children[k]) for k in sorted(children.keys())])
    else:
        return label
nltk_tree = convert(root, tree[root])

# check that we do indeed have the right number of leaf classes
assert len(nltk_tree.leaves()) == num_classes

# pickle nltk tree
with open(os.path.join(opts.out_dir, opts.dataset + '_tree.pkl'), 'wb') as f:
    pickle.dump(nltk_tree, f)

# also dumps a list of all paths
with open(os.path.join(opts.out_dir, opts.dataset + '_paths.txt'), 'w') as f:
    writer = csv.writer(f, delimiter=' ')
    for i in range(len(wnids)):
        writer.writerow([wnids[i]] + [n for n in paths[wnids[i]]])

# and another (somewhat) human readable file with the tree (using tree_format lib)
def convert_for_printing(label, children):
    if children:
        return (words[label], [convert_for_printing(k, children[k]) for k in sorted(children.keys())])
    else:
        return (words[label], [])
tree_for_printing = convert_for_printing(root, tree[root])

fmt = format_tree(tree_for_printing, format_node=itemgetter(0), get_children=itemgetter(1))

with open(os.path.join(opts.out_dir, opts.dataset + '_hierarchy.txt'), 'w') as f:
    f.write(fmt)