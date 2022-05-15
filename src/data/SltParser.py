import numpy as np

from src.definitions.SltEdgeTypes import SltEdgeTypes
from src.definitions.SrtEdgeTypes import SrtEdgeTypes
from src.definitions.exceptions.SltStructureError import SltStructureError


class SltParser:
    @staticmethod
    def get_root(tokens, pc_edge_index):
        # THERE HAS TO BE ONLY SINGLE TREE (one root in graph)
        tokens_ids = np.array(list(range(len(tokens))))
        targets = np.array(list(set([edge[1] for edge in pc_edge_index if edge[1] < len(tokens)])))
        tokens_ids = np.delete(tokens_ids, targets)

        if tokens_ids.shape[0] != 1:
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
                # TODO řádek nad hází chybu
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
        children_subtrees = [SltParser.parse_slt_subtree(child['id'], x, pc_edge_index, bb_edge_index, pc_edge_rel) for
                             child in children]

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

        # TODO add above and below

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
    def identify_reachable_nodes(tokens, pc_edge_index, bb_edge_index, root_id):
        token_ids = list(range(len(tokens)))

        # BFS traversal to identify reachable nodes (from root)
        visited = []
        queue = []
        reachable = []
        queue.append(root_id)
        reachable.append(root_id)
        while queue:
            current_root = queue.pop(0)
            if current_root not in visited:
                visited.append(current_root)
                current_root_children = SltParser.get_children(current_root, pc_edge_index, bb_edge_index)
                children_ids = [child['id'] for child in current_root_children]
                queue.extend(children_ids)
                reachable.extend(children_ids)

        reachable = list(set(reachable))
        return reachable

    @staticmethod
    def remove_unconnected_edges(node_ids, pc_edge_index, pc_edge_relations, bb_edge_index):
        pc_remove_ids = []
        bb_remove_ids = []
        for i, edge in enumerate(pc_edge_index):
            if edge[0] not in node_ids or edge[1] not in node_ids:
                pc_remove_ids.append(i)
        for i, edge in enumerate(bb_edge_index):
            if edge[0] not in node_ids or edge[1] not in node_ids:
                bb_remove_ids.append(i)

        pc_edge_index = np.delete(pc_edge_index, pc_remove_ids, axis=0)
        pc_edge_relations = np.delete(pc_edge_relations, pc_remove_ids)
        bb_edge_index = np.delete(bb_edge_index, bb_remove_ids, axis=0)
        return pc_edge_index, pc_edge_relations, bb_edge_index

    @staticmethod
    def remove_nodes(x, x_remove_ids, edge_index1, edge_index2, edge_relations1):
        # determine how many positions will each of the x-elems move to the left in array
        x_remove_ids = np.asarray(x_remove_ids, dtype=np.uint)
        x_shift_positions = np.zeros((len(x)))
        for i, x_shift_positions_i in enumerate(x_shift_positions):
            x_shift_positions[i] = (x_remove_ids < i).sum()
        x_shift_positions = np.asarray(x_shift_positions, dtype=np.uint)
        # remove nodes
        x = np.delete(x, x_remove_ids)
        # remove edges that refer from or to a removed node
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
        # remove edge relations belonging to removed edges
        edge_relations1 = np.delete(edge_relations1, edge_index1_r_ids)
        # shift node indices in edge_index arrays so that they match the x-elems indices after removal
        edge_index1 = SltParser.shift_indices_edge(edge_index1, x_shift_positions)
        edge_index2 = SltParser.shift_indices_edge(edge_index2, x_shift_positions)
        return x, edge_index1, edge_index2, edge_relations1

    @staticmethod
    def shift_indices_edge(edge_index, x_shift_positions):
        for i in range(edge_index.shape[0]):
            for j in range(edge_index.shape[1]):
                edge_index[i][j] = edge_index[i][j] - x_shift_positions[edge_index[i][j]]
        return edge_index

    @staticmethod
    def clean_slt(tokens, edge_relations, edge_index, edge_type, training_time=False):
        # keep only parent-child edges
        edge_pc_indices = ((edge_type == SltEdgeTypes.PARENT_CHILD).nonzero(as_tuple=True)[0])
        edge_bb_indices = ((edge_type == SltEdgeTypes.LEFTBROTHER_RIGHTBROTHER).nonzero(as_tuple=True)[0])
        pc_edge_index = edge_index.t()[edge_pc_indices]
        pc_edge_relations = edge_relations[edge_pc_indices]
        # and save brother edges for ordering purposes
        bb_edge_index = edge_index.t()[edge_bb_indices]

        pc_edge_index = pc_edge_index.numpy()
        pc_edge_relations = pc_edge_relations.numpy()
        bb_edge_index = bb_edge_index.numpy()

        # identify root - needs to be done before removing any nodes
        if pc_edge_index.shape[0] == 0:
            if tokens.shape[0] == 0:
                # if there is no node at all --> return empty string
                root_id = None
            else:
                # if no edges, but single node --> the only node in graph is root
                root_id = 0
        else:
            root_id = SltParser.get_root(tokens, pc_edge_index)

        # remove end leaf nodes
        # for training time's sake change nodes classified as [EOS]
        # that have subtrees (somewhere in the middle of the tree) to empty string
        # -> will not prune their whole subtree
        src_nodes_ids = [edge[0] for edge in pc_edge_index]
        end_node_ids = []
        for i, token in enumerate(tokens):
            if token == '[EOS]':
                if i in src_nodes_ids and training_time:
                    tokens[i] = ''
                else:
                    end_node_ids.append(i)

        tokens, pc_edge_index, bb_edge_index, pc_edge_relations = \
            SltParser.remove_nodes(tokens, end_node_ids, pc_edge_index, bb_edge_index, pc_edge_relations)

        # remove nodes not reachable from root
        # Note: matters in training time only - otherwise only end leaf nodes will be removed
        # (tree level generation always ends with EOS)
        reachable_node_ids = SltParser.identify_reachable_nodes(tokens, pc_edge_index, bb_edge_index, root_id)
        unreachable_nodes_ids = [token_id for token_id, _ in enumerate(tokens) if token_id not in reachable_node_ids]
        tokens, pc_edge_index, bb_edge_index, pc_edge_relations = \
            SltParser.remove_nodes(tokens, unreachable_nodes_ids, pc_edge_index, bb_edge_index, pc_edge_relations)

        # remove edges implying non-existing nodes (were within the unreachable subtrees)
        pc_edge_index, pc_edge_relations, bb_edge_index = \
            SltParser.remove_unconnected_edges([i for i in range(len(tokens))], pc_edge_index, pc_edge_relations,
                                               bb_edge_index)

        if len(tokens) == 0:
            root_id = None

        return tokens, pc_edge_index, bb_edge_index, pc_edge_relations, root_id

    @staticmethod
    def decode_node_tokens(tokenizer, x):
        x = x.numpy()
        tokens = [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in x]
        tokens = np.asarray(tokens)
        return tokens

    @staticmethod
    def slt_to_latex(tokenizer, x, edge_relations, edge_index, edge_type):
        tokens = SltParser.decode_node_tokens(tokenizer, x)

        tokens, pc_edge_index, bb_edge_index, pc_edge_relations, root_id = \
            SltParser.clean_slt(tokens, edge_relations, edge_index, edge_type)

        # g = nx.Graph()
        # for edge in pc_edge_index:
        #     g.add_edge(edge[0], edge[1])
        #
        # labeldict = {}
        # for i, x_i in enumerate(tokens):
        #     labeldict[i] = str(i) + tokens[i]
        # g = nx.relabel_nodes(g, labeldict)
        #
        # pos = graphviz_layout(g, prog="dot")
        # nx.draw(g, pos, with_labels=True)
        # plt.show()

        if root_id is None:
            return "", []

        latex = SltParser.parse_slt_subtree(root_id, tokens, pc_edge_index, bb_edge_index, pc_edge_relations)
        return ' '.join(latex), latex
