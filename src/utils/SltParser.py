import numpy as np
import torch

from src.definitions.SltEdgeTypes import SltEdgeTypes
from src.definitions.SrtEdgeTypes import SrtEdgeTypes
from src.definitions.exceptions.SltStructureError import SltStructureError


class SltParser:
    @staticmethod
    def get_root(tokens, pc_edge_index):
        tokens_ids = np.array(list(range(len(tokens))))
        targets = np.array(list(set([edge[1] for edge in pc_edge_index])))
        tokens_ids = np.delete(tokens_ids, targets)

        if tokens_ids.shape[0] == 0 or tokens_ids.shape[0] > 1:
            return None
        return tokens_ids[0]

    @staticmethod
    def get_children(root_id, pc_edge_index, bb_edge_index):
        children = [{'id': edge[1], 'e_id': idx} for idx, edge in enumerate(pc_edge_index) if edge[0] == root_id]
        if len(children) <= 1:
            return children
        else:
            children_ids = [ch['id'] for ch in children]
            bb_src_node_ids = bb_edge_index[:, 0]
            bb_tgt_node_ids = bb_edge_index[:, 1]

            bb_src_node_ids = [node for node in bb_src_node_ids if node in children_ids]
            bb_tgt_node_ids = [node for node in bb_tgt_node_ids if node in children_ids]

            first_node = [node_id for node_id in bb_src_node_ids if node_id not in bb_tgt_node_ids]
            children_order = first_node
            for _ in range(len(children_ids) - 1):
                current_node = children_order[-1]
                next_child = [edge[1] for edge in bb_edge_index if edge[0] == current_node]
                if len(next_child) == 0:
                    raise SltStructureError()
                else:
                    children_order.append(next_child[0])

            children_ordered = []
            for child_id in children_order:
                children_ordered.append(next(child for child in children if child['id'] == child_id))

        return children_ordered

    @staticmethod
    def parse_slt_subtree(root_id, x, pc_edge_index, bb_edge_index, pc_edge_rel):
        output = []
        root_symbol = x[root_id]
        # identify child nodes
        children = SltParser.get_children(root_id, pc_edge_index, bb_edge_index)
        # recursion - process child node subtrees
        children_subtrees = [SltParser.parse_slt_subtree(child['id'], x, pc_edge_index, bb_edge_index, pc_edge_rel) for child in children]

        # separate subtrees based on connection type
        above = [subtree for i, subtree in enumerate(children_subtrees) if
                 pc_edge_rel[children[i]['e_id']] == SrtEdgeTypes.ABOVE]
        below = [subtree for i, subtree in enumerate(children_subtrees) if
                 pc_edge_rel[children[i]['e_id']] == SrtEdgeTypes.BELOW]
        right = [subtree for i, subtree in enumerate(children_subtrees) if
                 pc_edge_rel[children[i]['e_id']] == SrtEdgeTypes.RIGHT or
                 pc_edge_rel[children[i]['e_id']] == SrtEdgeTypes.TO_ENDNODE]
        inside = [subtree for i, subtree in enumerate(children_subtrees) if
                  pc_edge_rel[children[i]['e_id']] == SrtEdgeTypes.INSIDE]
        superscript = [subtree for i, subtree in enumerate(children_subtrees) if
                       pc_edge_rel[children[i]['e_id']] == SrtEdgeTypes.SUPERSCRIPT]
        subscript = [subtree for i, subtree in enumerate(children_subtrees) if
                     pc_edge_rel[children[i]['e_id']] == SrtEdgeTypes.SUBSCRIPT]

        # append root symbol
        output.append(root_symbol)

        # symbol specific formatting
        if root_symbol == r"\frac":
            output.append('{')
            for a in above:
                output.extend(a)
            output.extend(['}', '{'])
            for b in below:
                output.extend(b)
            output.append('}')
            # already used --> empty lists
            above = []
            below = []
        elif root_symbol == r"\sqrt":
            output.append('{')
            for i in inside:
                output.extend(i)
            output.append('}')
            # already used --> empty list
            inside = []
            pass

        # common rules for super/subscripts and right positions
        if len(subscript) > 0:
            output.extend(['_', '{'])
            for s in subscript:
                output.extend(s)
            for b in below:
                output.extend(b)
            output.append('}')
        if len(superscript) > 0:
            output.extend(['^', '{'])
            for s in superscript:
                output.extend(s)
            for a in above:
                output.extend(a)
            output.append('}')
        if len(right) > 0:
            for r in right:
                output.extend(r)
        return output

    @staticmethod
    def remove_nodes(x, x_remove_ids, edge_index1, edge_index2, edge_relations1):
        x = np.delete(x, x_remove_ids)
        edge_index1_r_ids = []
        edge_index2_r_ids = []
        for i, edge in enumerate(edge_index1):
            if edge[0] in x_remove_ids or edge[1] in x_remove_ids:
                edge_index1_r_ids.append(i)
        for i, edge in enumerate(edge_index2):
            if edge[0] in x_remove_ids or edge[1] in x_remove_ids:
                edge_index2_r_ids.append(i)
        edge_index1 = np.delete(edge_index1, edge_index1_r_ids, axis=0)
        edge_index2 = np.delete(edge_index2, edge_index2_r_ids, axis=0)
        edge_relations1 = np.delete(edge_relations1, edge_index1_r_ids)
        return x, edge_index1, edge_index2, edge_relations1

    @staticmethod
    def slt_to_latex_predictions(tokenizer, x, edge_relations, edge_index, edge_type):
        # get symbols
        x = x.numpy()
        tokens = [tokenizer.decode([token_id]) for token_id in x]
        return SltParser.slt_to_latex(tokenizer, tokens, edge_relations, edge_index, edge_type)

    @staticmethod
    def slt_to_latex(tokenizer, tokens, edge_relations, edge_index, edge_type):
        # keep only parent-child edges
        edge_pc_indices = ((edge_type == SltEdgeTypes.PARENT_CHILD).nonzero(as_tuple=True)[0])
        edge_bb_indices = ((edge_type == SltEdgeTypes.LEFTBROTHER_RIGHTBROTHER).nonzero(as_tuple=True)[0])
        pc_edge_index = edge_index.t()[edge_pc_indices]
        pc_edge_relations = edge_relations[edge_pc_indices]
        # and save brother edges for ordering purposes
        bb_edge_index = edge_index.t()[edge_bb_indices]

        # TODO might remove node that is not endnode, if edge type incorrect
        if False:
            # remove "to-endnode" edges
            pc_edge_no_endnode = ((pc_edge_relations != SrtEdgeTypes.TO_ENDNODE).nonzero(as_tuple=True)[0])
            pc_edge_index = pc_edge_index[pc_edge_no_endnode]
            pc_edge_relations = pc_edge_relations[pc_edge_no_endnode]

            pc_edge_index = pc_edge_index.numpy()
            pc_edge_relations = pc_edge_relations.numpy()
            bb_edge_index = bb_edge_index.numpy()

            # remove standalone nodes - end leaf nodes
            if pc_edge_index.shape[0] > 0:
                # remove only if there are any parent-child edges
                # otherwise the whole graph is single node and wanna keep it
                src_nodes_ids = [edge[0] for edge in pc_edge_index]
                tgt_nodes_ids = [edge[1] for edge in pc_edge_index]
                src_nodes_ids.extend(tgt_nodes_ids)
                connected_node_ids = src_nodes_ids
                connected_node_ids = list(set(connected_node_ids))
                standalone_node_ids = [i for i, _ in enumerate(tokens) if i not in connected_node_ids]
                tokens, pc_edge_index, bb_edge_index, pc_edge_relations = \
                    SltParser.remove_nodes(tokens, standalone_node_ids, pc_edge_index, bb_edge_index, pc_edge_relations)

        pc_edge_index = pc_edge_index.numpy()
        pc_edge_relations = pc_edge_relations.numpy()
        bb_edge_index = bb_edge_index.numpy()

        # remove end leaf nodes
        end_node_ids = [i for i, token in enumerate(tokens) if not token]
        tokens, pc_edge_index, bb_edge_index, pc_edge_relations = \
            SltParser.remove_nodes(tokens, end_node_ids, pc_edge_index, bb_edge_index, pc_edge_relations)

        if pc_edge_index.shape[0] == 0:
            if tokens.shape[0] == 0:
                # if there is no node left, return empty string
                return ""
            else:
                # if no edges, but single node --> the only node in graph is root
                root_id = 0
        else:
            root_id = SltParser.get_root(tokens, pc_edge_index)
            if root_id is None:
                return ""

        latex = SltParser.parse_slt_subtree(root_id, tokens, pc_edge_index, bb_edge_index, pc_edge_relations)
        return ' '.join(latex)
