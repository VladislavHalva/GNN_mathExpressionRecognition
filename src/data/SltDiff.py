# ###
# Mathematical expression recognition tool.
# Written as a part of masters thesis at VUT FIT Brno, 2022

# Author: Vladislav Halva
# Login: xhalva04
# ###

from src.data.SltParser import SltParser

class SltDiff:
    """
    Computes difference of Symbol Layout Trees of two formulas.
    """
    def __init__(self, tokenizer, a, a_eindex, a_etype, a_erel, b, b_eindex, b_etype, b_erel):
        """
        :param tokenizer: trained tokenizer
        :param a: first graph node symbol logits
        :param a_eindex: first graph edge index
        :param a_etype: first graph edge types
        :param a_erel: first graph edge relations
        :param b: second graph node symbol logits
        :param b_eindex: second graph edge index
        :param b_etype: second graph edge types
        :param b_erel: second graph edge relations
        """
        # load first SLT, decode tokens and clean SLT
        self.a_eindex = a_eindex
        self.a_tk = SltParser.decode_node_tokens(tokenizer, a)
        self.a_tk, self.a_eindex_pc, self.a_eindex_bb, self.a_erel_pc, self.a_root = \
            SltParser.clean_slt(self.a_tk, a_erel, a_eindex, a_etype, training_time=False)
        # load second SLT, decode tokens and clean SLT
        self.b_eindex = b_eindex
        self.b_tk = SltParser.decode_node_tokens(tokenizer, b)
        self.b_tk, self.b_eindex_pc, self.b_eindex_bb, self.b_erel_pc, self.b_root = \
            SltParser.clean_slt(self.b_tk, b_erel, b_eindex, b_etype, training_time=False)
        # init results dictionary
        self.result = {}

    def eval(self):
        """
        Evaluates SLT difference.
        """
        self.result['a_nodes'] = len(self.a_tk)
        self.result['b_nodes'] = len(self.b_tk)
        self.result['nodes_count_match'] = self.result['a_nodes'] == self.result['b_nodes']
        self.result['a_edges'] = self.a_eindex_pc.shape[0]
        self.result['b_edges'] = self.b_eindex_pc.shape[0]
        self.result['edges_count_match'] = self.result['a_edges'] == self.result['b_edges']

        self.traverse_trees()

    def traverse_trees(self):
        """
        Computes global statistics of graphs difference.
        """
        self.result['a_nodes_missing'] = 0
        self.result['b_nodes_missing'] = 0
        self.result['node_class_errors'] = 0
        self.result['edge_class_errors'] = 0
        self.traverse_subtree(self.a_root, self.b_root)

        self.result['structure_match'] = \
            self.result['a_nodes_missing'] == self.result['b_nodes_missing'] == 0
        self.result['exact_match'] = \
            self.result['structure_match'] and \
            self.result['node_class_errors'] == self.result['edge_class_errors'] == 0
        self.result['exact_match_1'] = \
            self.result['structure_match'] and \
            (self.result['node_class_errors'] + self.result['edge_class_errors']) <= 1
        self.result['exact_match_2'] = \
            self.result['structure_match'] and \
            (self.result['node_class_errors'] + self.result['edge_class_errors']) <= 2
        self.result['exact_match_3'] = \
            self.result['structure_match'] and \
            (self.result['node_class_errors'] + self.result['edge_class_errors']) <= 3

    def traverse_subtree(self, a_root, b_root):
        """
        DFS traversal of SLT trees. Computes local difference statistics.
        Traverses both graph simultaneously and compares them.
        :param a_root: first graph current root
        :param b_root: second graph current root
        """
        # get currect root token if exists
        a_token = self.a_tk[a_root] if a_root is not None else None
        b_token = self.b_tk[b_root] if b_root is not None else None
        # compare current roots
        if a_root is None:
            self.result['a_nodes_missing'] += 1
        if b_root is None:
            self.result['b_nodes_missing'] += 1
        if a_root is not None and b_root is not None and self.a_tk[a_root] != self.b_tk[b_root]:
            self.result['node_class_errors'] += 1
        # get children
        a_children = None
        b_children = None
        if a_root is not None:
            a_children = SltParser.get_children(a_root, self.a_eindex_pc, self.a_eindex_bb)
        if b_root is not None:
            b_children = SltParser.get_children(b_root, self.b_eindex_pc, self.b_eindex_bb)
        # if neither of trees has children at this point of traversal -> return
        if a_children is None and b_children is None:
            return
        # pad children with Nones to match children's count of both trees
        if a_children is not None and b_children is not None:
            max_children = max(len(a_children), len(b_children))
            a_children += [{'id': None}] * (max_children - len(a_children))
            b_children += [{'id': None}] * (max_children - len(b_children))
        elif a_children is None:
            a_children = [{'id': None}] * len(b_children)
        elif b_children is None:
            b_children = [{'id': None}] * len(a_children)
        # check edge relations
        for a_child, b_child in zip(a_children, b_children):
            if a_child['id'] is not None and b_child['id'] is not None:
                a_erel = self.a_erel_pc[a_child['e_id']]
                b_erel = self.b_erel_pc[b_child['e_id']]
                if a_erel != b_erel:
                    self.result['edge_class_errors'] += 1
        # explore children subtrees
        for child_i in range(len(a_children)):
            self.traverse_subtree(a_children[child_i]['id'], b_children[child_i]['id'])

    def get_result(self):
        """
        :return: SLT graphs comparison results.
        """
        return self.result
