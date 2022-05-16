# ###
# Mathematical expression recognition tool.
# Written as a part of masters thesis at VUT FIT Brno, 2022

# Author: Vladislav Halva
# Login: xhalva04
# ###

from torch_geometric.data import Data


class GMathData(Data):
    """
        Dataset item wrapper for dataloader.
    """
    def __init__(
            self,
            x=None, edge_index=None, edge_attr=None,
            gt=None, gt_ml=None, tgt_y=None, tgt_edge_index=None,
            tgt_edge_type=None, tgt_edge_relation=None,
            comp_symbols=None, filename=None
    ):
        """
        :param x: source graph node features
        :param edge_index: source graph edge index
        :param edge_attr: source graph edge features
        :param gt: tokenized latex sequence ground-truth created from latex string
        :param gt_ml: tokenized latex sequence ground-truth created form MathML representation
        :param tgt_y: output graph groudtruth node symbols
        :param tgt_edge_index: output graph groundtruth edge index
        :param tgt_edge_type: output graph groundtruth edge types given by SltEdgeTypes
        :param tgt_edge_relation: output graph groundtruth edge relations given by SrtEdgeTypes
        :param comp_symbols: source graph groundtruth node symbols
        :param filename: filename
        """
        super().__init__()
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.gt = gt
        self.gt_ml = gt_ml
        self.tgt_y = tgt_y
        self.tgt_edge_index = tgt_edge_index
        self.tgt_edge_type = tgt_edge_type
        self.tgt_edge_relation = tgt_edge_relation
        self.comp_symbols = comp_symbols
        self.filename = filename

    def __inc__(self, key, value, *args, **kwargs):
        """
        :param key: class attribute name
        :return: values increment for batch
        """
        if key == 'edge_index':
            return self.x.size(0)
        if key == 'tgt_edge_index':
            return self.tgt_y.size(0)
        if key == 'comp_symbols':
            return self.tgt_y.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        """
        :param key: class attribute name
        :return: dimension in which item shall be concatenated for batch
        """
        if 'index' in key or 'face' in key:
            return 1
        else:
            return 0
