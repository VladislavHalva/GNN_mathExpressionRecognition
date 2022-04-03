import torch

from src.definitions.SltEdgeTypes import SltEdgeTypes
from src.definitions.SrtEdgeTypes import SrtEdgeTypes


class SltParser:
    @staticmethod
    def get_root(edge_index):
        sources = list(set([edge[0] for edge in edge_index]))
        targets = list(set([edge[1] for edge in edge_index]))
        keep = [True for _ in range(len(sources))]
        for i, source in enumerate(sources):
            if source in targets:
                keep[i] = False
        roots = [source for i, source in enumerate(sources) if keep[i]]

        if len(roots) == 0 or len(roots) > 1:
            return None
        return roots[0]

    @staticmethod
    def remove_selfloops(edge_index, edge_rel):
        keep = [True for _ in range(len(edge_index))]
        for i, edge in enumerate(edge_index):
            if edge[0] == edge[1]:
                keep[i] = False

        edge_index = [edge for i, edge in enumerate(edge_index) if keep[i]]
        edge_rel = [r for i, r in enumerate(edge_rel) if keep[i]]
        return edge_index, edge_rel

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
        above = [subtree for i, subtree in enumerate(children_subtrees) if
                 edge_rel[children[i]['e_id']] == SrtEdgeTypes.ABOVE]
        below = [subtree for i, subtree in enumerate(children_subtrees) if
                 edge_rel[children[i]['e_id']] == SrtEdgeTypes.BELOW]
        right = [subtree for i, subtree in enumerate(children_subtrees) if
                 edge_rel[children[i]['e_id']] == SrtEdgeTypes.RIGHT]
        inside = [subtree for i, subtree in enumerate(children_subtrees) if
                  edge_rel[children[i]['e_id']] == SrtEdgeTypes.INSIDE]
        superscript = [subtree for i, subtree in enumerate(children_subtrees) if
                       edge_rel[children[i]['e_id']] == SrtEdgeTypes.SUPERSCRIPT]
        subscript = [subtree for i, subtree in enumerate(children_subtrees) if
                     edge_rel[children[i]['e_id']] == SrtEdgeTypes.SUBSCRIPT]

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
    def slt_to_latex_predictions(tokenizer, x_pred, edge_rel_pred, edge_index, edge_type):
        # get symbols
        node_preds = torch.exp(x_pred)
        _, max_id = node_preds.max(dim=1)
        token_ids = max_id.tolist()
        tokens = [tokenizer.decode([token_id]) for token_id in token_ids]

        # get node relations
        edge_preds = torch.exp(edge_rel_pred)
        _, max_id = edge_preds.max(dim=1)
        edge_relations = max_id

        return SltParser.slt_to_latex(tokenizer, tokens, edge_relations, edge_index, edge_type)

    @staticmethod
    def slt_to_latex(tokenizer, tokens, edge_relations, edge_index, edge_type):
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

        # remove self loops
        # edge_index, edge_relations = SltParser.remove_selfloops(edge_index, edge_relations)

        # remove standalone nodes - end leaf nodes
        if len(edge_index) > 0:
            max_src = max([edge[0] for edge in edge_index])
            max_tgt = max([edge[1] for edge in edge_index])
            max_id = max(max_src, max_tgt)
            tokens = tokens[:(max_id + 1)]

        # print(tokens)
        # print([[i, SrtEdgeTypes.to_string(rel)] for i,rel in enumerate(edge_relations)])

        root_id = SltParser.get_root(edge_index)
        if root_id is None:
            return ""

        latex = SltParser.parse_slt_subtree(root_id, tokens, edge_index, edge_relations)
        return ' '.join(latex)
