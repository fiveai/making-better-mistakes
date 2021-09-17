import lzma
import os
import json
import argparse
import numpy as np
import pickle
from collections import defaultdict, Counter
from tqdm import tqdm

from nltk import Tree
import networkx as nx

from tree_format import format_tree
from operator import itemgetter

TAXONOMY = [
    'kingdom',
    'phylum',
    'class',
    'order',
    'family',
    'genus', 
    'name'
]

parser = argparse.ArgumentParser(description='Compute the hierarchical tree for inat as a nltk tree and saves it using pickle.')
parser.add_argument('--input', type=str, metavar='PATH', required=True)  # train2019.json
parser.add_argument('--out_dir', type=str, metavar='PATH', default='../data/')
parser.add_argument('--data_dir', type=str, metavar='PATH', default='../data/')
opts = parser.parse_args()

# get all the paths (sorted by id)
categories = json.load(open(opts.input))['categories']

def get_taxonomy_str(category, idx):
    # we do this such that the classes can be sorted alphabetically
    # and the leaf nodes are first.
    if TAXONOMY[idx] == 'name':
        return 'nat{0:04d}'.format(category['id'])
    else:
        return 'node_' + category[TAXONOMY[idx]]

# all species
species = [get_taxonomy_str(category, 6) for category in categories]
num_classes = len(species)
assert num_classes == len(categories)
print('Building tree for', num_classes, 'classes')

# get all the edges
root = 'root'
edges = []
for category in categories:
    edges.append((root, get_taxonomy_str(category, 0)))
    edges.extend([
        (get_taxonomy_str(category, i-1), get_taxonomy_str(category, i)) for i in range(1, len(TAXONOMY))
    ])
edges = list(set(edges))

# build full dependency graph using networkx
graph = nx.DiGraph()
for parent, child in edges:
    graph.add_edge(child, parent)

# get edges that are on the longest path from the leaf classes to the root, resolving alternatives as we go
paths = {}
for node in species:
    all_paths = list(nx.all_simple_paths(graph, node, root))
    max_len = max([len(p) for p in all_paths])
    max_paths = [p for p in all_paths if len(p) == max_len]
    if(len(max_paths)>1):
        print('Going from', node, 'to', root, 'has multiple possible paths:')
        for i, path in enumerate(max_paths):
            print('  {}:'.format(i), ', '.join(path))
        print('Meeeeh')
    else:
        paths[node] = max_paths[0][1:]

def get_edges(paths):
    edges = set()
    for node, path in paths.items():
        path = [node] + path
        for i in range(1, len(path)):
            edges.add((path[i], path[i-1]))  # still keep the confusing parent/child ordering
    return edges

# Check that we do indeed have a tree
children = [e[1] for e in edges]
assert len(children) == len(set(children))  # check that all children have only one parent

# prune all non-leaf nodes that have only one child (actually we don't here)
# num_children = Counter([e[0] for e in edges])
# paths = {node: [n for n in path if num_children[n]>1 or n==root] for node, path in paths.items()}
# edges = get_edges(paths)

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
leaves = [l[2:] for l in nltk_tree.leaves()]
oops = [c for c in leaves if c not in species]
oops2 = [c for c in species if c not in leaves]
assert len(nltk_tree.leaves()) == num_classes

# pickle nltk tree
with open(os.path.join(opts.out_dir, 'inaturalist19_tree.pkl'), 'wb') as f:
    pickle.dump(nltk_tree, f)

# and another (somewhat) human readable file with the tree (using tree_format lib)
def convert_for_printing(label, children):
    if children:
        return (label, [convert_for_printing(k, children[k]) for k in sorted(children.keys())])
    else:
        return (label, [])
tree_for_printing = convert_for_printing(root, tree[root])

# dump in a human readable format
fmt = format_tree(tree_for_printing, format_node=itemgetter(0), get_children=itemgetter(1))
with open(os.path.join(opts.out_dir, 'inaturalist19_hierarchy.txt'), 'w') as f:
    f.write(fmt)