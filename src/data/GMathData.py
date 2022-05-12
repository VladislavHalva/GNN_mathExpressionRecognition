from torch_geometric.data import Data


class G2GData(Data):
    def __init__(
            self,
            x=None, edge_index=None, edge_attr=None,
            gt=None, gt_ml=None, tgt_y=None, tgt_edge_index=None,
            tgt_edge_type=None, tgt_edge_relation=None,
            comp_symbols=None, filename=None
    ):
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
        if key == 'edge_index':
            return self.x.size(0)
        if key == 'tgt_edge_index':
            return self.tgt_y.size(0)
        if key == 'comp_symbols':
            return self.tgt_y.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if 'index' in key or 'face' in key:
            return 1
        else:
            return 0
