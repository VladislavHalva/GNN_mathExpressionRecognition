import networkx as nx
import torch
from matplotlib import pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from src.definitions.SltEdgeTypes import SltEdgeTypes
from src.definitions.SrtEdgeTypes import SrtEdgeTypes


class SltParser:
    @staticmethod
    def get_root(edge_index):
        sources = [edge[0] for edge in edge_index]
        targets = [edge[1] for edge in edge_index]
        for source in sources:
            if source in targets:
                del source
        if len(sources) == 0 or len(sources) > 1:
            return None
        return sources[0]

    @staticmethod
    def remove_selfloops(edge_index):
        return [edge for edge in edge_index if edge[0] != edge[1]]

    @staticmethod
    def remove_standalone_nodes(tokens, edge_index, edge_rel):
        sources = [edge[0] for edge in edge_index]
        targets = [edge[1] for edge in edge_index]
        for i, token in enumerate(tokens):
            if token not in sources and token not in targets:
                del token
        return tokens

    @staticmethod
    def get_children(root_id, edge_index):
        return [{'id': edge[1], 'e_id': idx} for idx, edge in enumerate(edge_index) if edge[0] == root_id]

    @staticmethod
    def parse_slt_subtree(root_id, x, edge_index, edge_rel):
        output = []
        root_symbol = x[root_id]
        # identify child nodes
        children = SltParser.get_children(root_id, edge_index)
        # recursion - process child node subtrees
        children_subtrees = [SltParser.parse_slt_subtree(child['id'], x, edge_index, edge_rel) for child in children]

        # separate subtrees based on connection type
        above = [subtree for i, subtree in enumerate(children_subtrees) if edge_rel[children[i]['e_id']] == SrtEdgeTypes.ABOVE]
        below = [subtree for i, subtree in enumerate(children_subtrees) if edge_rel[children[i]['e_id']] == SrtEdgeTypes.BELOW]
        right = [subtree for i, subtree in enumerate(children_subtrees) if edge_rel[children[i]['e_id']] == SrtEdgeTypes.RIGHT]
        inside = [subtree for i, subtree in enumerate(children_subtrees) if edge_rel[children[i]['e_id']] == SrtEdgeTypes.INSIDE]
        superscript = [subtree for i, subtree in enumerate(children_subtrees) if edge_rel[children[i]['e_id']] == SrtEdgeTypes.SUPERSCRIPT]
        subscript = [subtree for i, subtree in enumerate(children_subtrees) if edge_rel[children[i]['e_id']] == SrtEdgeTypes.SUB]

        # append root symbol
        output.append(root_symbol)

        # symbol specific formatting
        if root_symbol == r"\frac":
            output.append('{')
            output.extend(above)
            output.extend(['}', '{'])
            output.extend(below)
            output.append('}')
            # already used --> empty lists
            above = []
            below = []
        elif root_symbol == r"\sqrt":
            output.append('{')
            output.extend(inside)
            output.append('}')
            # already used --> empty list
            inside = []
            pass

        # common rules for super/subscripts and right positions
        if len(superscript) > 0:
            output.extend(['^', '{'])
            output.extend(superscript)
            output.extend(above)
            output.append('}')
        if len(subscript) > 0:
            output.extend(['_', '{'])
            output.extend(superscript)
            output.extend(below)
            output.append('}')
        if len(right) > 0:
            output.extend(right)

        return output


    @staticmethod
    def slt_to_latex(tokenizer, x_pred, edge_rel_pred, edge_index, edge_type):
        # get symbols
        node_preds = torch.exp(x_pred)
        max, max_id = node_preds.max(dim=1)
        token_ids = max_id.tolist()
        tokens = [tokenizer.decode([token_id]) for token_id in token_ids]

        # get node relations
        edge_preds = torch.exp(edge_rel_pred)
        max, max_id = edge_preds.max(dim=1)
        edge_relations = max_id

        # keep only parent-child edges
        edge_pc_indices = ((edge_type == SltEdgeTypes.PARENT_CHILD).nonzero(as_tuple=True)[0])
        edge_index = edge_index.t()[edge_pc_indices]
        edge_relations = edge_relations[edge_pc_indices]

        # remove "to-endnode" edges
        edge_no_endnode = ((edge_relations != SrtEdgeTypes.TO_ENDNODE).nonzero(as_tuple=True)[0])
        edge_index = edge_index[edge_no_endnode]
        edge_relations = edge_relations[edge_no_endnode]

        edge_index = edge_index.tolist()
        edge_relations = edge_relations.tolist()

        # remove selfloops
        edge_index = SltParser.remove_selfloops(edge_index)

        # remove standalone nodes
        tokens = SltParser.remove_standalone_nodes(tokens, edge_index, edge_relations)

        root_id = SltParser.get_root(edge_index)
        if not root_id:
            # TODO error
            return ""

        latex = SltParser.parse_slt_subtree(root_id, tokens, edge_index, edge_relations)
        return latex.join(' ')


