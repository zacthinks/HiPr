# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 12:15:48 2024

@author: lin_s
"""

import pandas as pd
import math
import pickle
from collections import deque


class DFNode:
    '''A node object to be used with DFTree that primarily contains a DataFrame.'''

    def __init__(self, df, label, node_id, parent=None, order=0, attribs={}):
        self.df = df
        self.label = label
        self.node_id = node_id
        if isinstance(parent, DFNode):
            self.parent = parent
            self.tree = parent.tree
        elif isinstance(parent, DFTree):
            self.parent = None
            self.tree = parent
        self.order = order
        self.attribs = attribs
        self.collapsed = False

    def delete(self, save_children=True):
        if save_children and self.has_children():
            for child in self.children:
                child.parent = self.parent
        del self.tree.node_dict[self.node_id]

    def clear_children(self):
        for child in self.children:
            child.delete(False)

    def __repr__(self):
        return f'{self.label} (n={self.n})'

    @property
    def n(self):
        return len(self.df)

    @property
    def parent_id(self):
        return self.parent.node_id

    @parent_id.setter
    def parent_id(self, val):
        self.parent = self.tree[val]

    @property
    def children(self):
        children = [node for node in self.tree.nodes if node.parent is self]
        children.sort(key=lambda x: (x.order, -x.n))
        return children

    @property
    def descendants(self):
        descendants = []
        queue = self.children
        while len(queue) > 0:
            node = queue.pop(0)
            descendants.append(node)
            queue = node.children + queue
        return descendants

    def has_children(self):
        return len(self.children) > 0

    def is_root(self):
        return self.parent is None

    def collapse(self):
        if self.is_root():
            raise Exception("The root node can't be collapsed. (I mean it could be but for technical reasons it can't.) But why would you want to do that anyway?")
        if self.has_children():
            self.collapsed = True
        else:
            print('Node has no children.')

    def expand(self):
        self.collapsed = False

    def subtree_str(self, show_id=False):
        return self.tree.to_string(show_id, self.node_id)

    def print_subtree(self, show_id=False):
        print(self.subtree_str(show_id))

    def move_to_end(self):
        self.order = max([child.order for child in self.parent.children]) + 1

    def add_child(self, df, label, attribs={}, add_to_end=False):
        self.tree.add_node(df, label, self.node_id, attribs, add_to_end)

    def split(self, df_split_func):
        '''Splits a node based on a some function that takes its DataFrame and returns a list of triples (label, DataFrame, attribute dictionary).'''
        if len(self.children) > 0:
            raise Exception('Can only split nodes that do not already have children')
        lbl_df_attribds = df_split_func(self.df)
        for lbl, df, attribd in lbl_df_attribds:
            self.add_child(df, lbl, attribd)

    def split_by_group(self, column_names):
        '''Splits a node based on grouping the values of a particular column in its DataFrame.'''
        def splitter(df):
            lbl_dfs = list(df.groupby(column_names))
            attribds = []
            for lbl, _ in lbl_dfs:
                attribd = {}
                if not isinstance(column_names, list):
                    cols = [column_names]
                    lbl = [lbl]
                else:
                    cols = column_names
                for i, col in enumerate(cols):
                    attribd[col] = lbl[i]
                attribds.append(attribd)
            return [(lbl, df, attribd) for ((lbl, df), attribd) in zip(lbl_dfs, attribds)]
        self.split(splitter)

    def split_by_bins(self, column_name, bins):
        '''Splits a node based on a particular column in its DataFrame and a number of bin values, like in a historgram.'''
        def splitter(df):
            l = []
            for i, b in enumerate(bins):
                if i == 0:
                    subset = df[df[column_name] < b]
                    if len(subset) > 0:
                        l.append((f'{column_name} < {b}', subset, {'by': column_name, 'range': (None, b)}))
                else:
                    subset = df[(df[column_name] >= bins[i - 1]) & (df[column_name] < b)]
                    if len(subset) > 0:
                        l.append((f'{bins[i-1]} <= {column_name} < {b}', subset, {'by': column_name, 'range': (bins[i - 1], b)}))
            subset = df[df[column_name] >= bins[-1]]
            if len(subset) > 0:
                l.append((f'{bins[-1]} <= {column_name}', subset, {'by': column_name, 'range': (bins[-1], None)}))
            return l
        self.split(splitter)

    def relabel_children_func(self, new_label_func):
        '''Renames the children of the node (assuming they have certain similarities in structure) using a function that takes each child node and outputs a string (or any object that can be coerced into a string) and an attribute dictionary to be used to update the node's attributes'.'''
        for child in self.children:
            new_label, new_attribs = new_label_func(child)
            child.label = new_label
            child.attribs.update(new_attribs)

    def example(self, columns=None, n=1):
        sample = self.df.sample(n)
        if columns:
            sample = sample[columns]
        return sample


class DFTree:
    '''A dictionary of DFNodes where each node contains a separate DataFrame. Includes methods to visualize the tree and split and combine nodes.'''
    root_id = 0

    def __init__(self, df, label='ROOT', attribs={}):
        self.node_dict = {self.root_id: DFNode(df, label, self.root_id, self, attribs=attribs)}

    @property
    def nodes(self):
        return list(self.node_dict.values())

    def node_count(self):
        return len(self.nodes)

    def __getitem__(self, key):
        if isinstance(key, list):
            return [self.node_dict[k] for k in key]
        else:
            return self.node_dict[key]

    def __setitem__(self, key, val):
        self.node_dict[key] = val

    def __delitem__(self, key):
        self[key].delete()

    def add_node(self, df, label, parent_id, attribs={}, add_to_end=False):
        new_id = max(self.node_dict) + 1
        self[new_id] = DFNode(df, label, new_id, self[parent_id], 0, attribs)
        if add_to_end:
            self[new_id].move_to_end()
        return self[new_id]

    def combine_nodes(self, nodes,
                      combined_label_func=lambda df, ids, labels: '[OTHER]',
                      combined_attribs_func=lambda df, ids, attribds: {},
                      move_to_end=True,
                      delete_combined=True):
        '''combines specified sibling nodes (merges their DataFrames). Custom functions to generate the combined label and attribute dictionary can be provided.'''
        if isinstance(nodes, list) and len(nodes) > 1:
            nodes = [node if isinstance(node, DFNode) else self[node] for node in nodes]
        else:
            raise Exception('Must provide a list of at least 2 nodes or node ids to merge.')

        parent_id = {node.parent_id for node in nodes}
        if len(parent_id) != 1:
            raise Exception(f'Nodes to combine must be under the same parent. Parent ids provided: {parent_id}')
        parent_id = parent_id.pop()

        new_df = pd.concat([node.df for node in nodes])
        node_ids = [node.node_id for node in nodes]
        old_labels = [node.label for node in nodes]
        old_attribds = [node.attribs for node in nodes]
        new_node = self.add_node(new_df,
                                 combined_label_func(new_df, node_ids, old_labels),
                                 parent_id,
                                 combined_attribs_func(new_df, node_ids, old_attribds),
                                 add_to_end=move_to_end)
        if delete_combined:
            for node_id in node_ids:
                del self.node_dict[node_id]
        else:
            for node in nodes:
                node.parent = new_node
            new_node.collapse()

    def collapse_nodes(self, nodes,
                       combined_label_func=lambda df, ids, labels: '[OTHER]',
                       combined_attribs_func=lambda df, ids, attribds: {},
                       move_to_end=True,
                       delete_combined=False):
        self.combine_nodes(nodes, combined_label_func, combined_attribs_func, move_to_end, delete_combined)

    def combine_nodes_func(self, node_criteria_func,
                           combined_label_func=lambda df, ids, labels: '[OTHER]',
                           combined_attribs_func=lambda df, ids, attribds: {},
                           node_id=None, levels=-1, move_to_end=True, delete_combined=True):
        node_id = node_id if node_id is not None else self.root_id
        if levels == 0:
            return

        to_check = deque([(self[node_id], levels)])

        while len(to_check) > 0:
            to_merge = []
            current_node, level = to_check.popleft()
            if level == 0:
                continue
            for child in current_node.children:
                if not child.has_children():
                    if node_criteria_func(child):
                        to_merge.append(child)
                else:
                    to_check.append((child, level - 1))
            if len(to_merge) > 1:
                self.combine_nodes(to_merge, combined_label_func, combined_attribs_func, move_to_end=move_to_end, delete_combined=delete_combined)

    def collpase_nodes_func(self, node_criteria_func,
                            combined_label_func=lambda df, ids, labels: '[OTHER]',
                            combined_attribs_func=lambda df, ids, attribds: {},
                            node_id=None, levels=-1, move_to_end=True, delete_combined=False):
        self.combine_nodes_func(node_criteria_func, combined_label_func, combined_attribs_func, node_id, levels, move_to_end, delete_combined)

    def combine_small_n(self,
                        max_n,
                        combined_label_func=None,
                        combined_attribs_func=lambda df, ids, attribds: {},
                        starting_node=None, levels=-1, move_to_end=True, delete_combined=True):
        starting_node = starting_node if starting_node is not None else self.root_id
        if combined_label_func is None:
            def combined_label_func(df, ids, labels):
                return f'[{len(ids)} combined nodes with n<={max_n}]'

        self.combine_nodes_func(lambda node: node.n <= max_n, combined_label_func, combined_attribs_func, starting_node, levels, move_to_end, delete_combined)

    def collapse_small_n(self,
                         max_n,
                         combined_label_func=None,
                         combined_attribs_func=lambda df, ids, attribds: {},
                         starting_node=None, levels=-1, move_to_end=True, delete_combined=False):
        self.combine_small_n(max_n, combined_label_func, combined_attribs_func, starting_node, levels, move_to_end, delete_combined)

    def filter(self, func):
        return_list = []
        for node in self.nodes:
            if func(node):
                return_list.append(node)

        return return_list

    def to_string(self, show_id=False, node_id=None):
        node_id = node_id if node_id is not None else self.root_id
        total_n = self[node_id].n
        total_nodes = 1
        digits = 1 if max(self.node_dict) == 0 else math.floor(math.log(max(self.node_dict), 10)) + 1
        return_str = f'{self[node_id].label} (n={total_n:,})'
        if show_id:
            return_str = f'{node_id:0{digits}d}: ' + return_str

        queue = [(child, '├───') for child in self[node_id].children]
        if len(queue) > 0:
            queue[-1] = queue[-1][0], '└───'

        while len(queue) > 0:
            node, leading = queue.pop(0)
            parent_n = node.parent.n
            newline = f"{leading} {node.label} (n={node.n:,}, {node.n/parent_n*100:.2f}% of parent, {node.n/total_n*100:.{digits}f}% of root)"
            if show_id:
                newline = f'{node.node_id:0{digits}d}: ' + newline
            return_str += '\n' + newline
            total_nodes += 1

            if node.has_children():
                child_leading = leading[:-4]
                child_leading += '│   ' if leading[-4] == '├' else '    '
                if node.collapsed:
                    descendants_count = len(node.descendants)
                    total_nodes += descendants_count
                    return_str += f"\n{' '*(digits+2) if show_id else ''}{child_leading}└─── [{descendants_count} collapsed node{'s' if descendants_count != 1 else ''}]"
                else:
                    to_add = [(child, child_leading + '├───') for child in node.children]
                    to_add[-1] = to_add[-1][0], child_leading + '└───'
                    queue = to_add + queue

        return f"DFTree with {total_nodes} node{'s' if self.node_count() != 1 else ''}\n\n" + return_str

    def to_pickle(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def __str__(self):
        return self.to_string(True)

    def __repr__(self):
        return self.to_string()


def read_pickle(filename):
    dft = None
    with open(filename, 'rb') as f:
        dft = pickle.load(f)
    return dft


def iris_test():
    iris = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
    iris_tree = DFTree(iris, 'Flowers')
    iris_tree[0].split_by_group('species')
    iris_tree[2].split_by_bins('petal_width', [0, 1, 1.5, 2])
    iris_tree[5].split_by_bins('sepal_width', [3])
    iris_tree[4].split_by_group('sepal_width')
    iris_tree.combine_nodes([18, 19])
    iris_tree.combine_small_n(5)
    iris_tree[3].split_by_group('petal_width')
    iris_tree[3].collapse()
    print(iris_tree)
    return iris_tree
