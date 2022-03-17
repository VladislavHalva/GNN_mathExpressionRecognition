import torch
from torch import nn
from torch.nn import Parameter
from torch.nn.utils.rnn import pack_padded_sequence
from torch_geometric.nn import MessagePassing, Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_sparse import SparseTensor, fill_diag, matmul, mul
import torch.nn.functional as F

from src.definitions.SltEdgeTypes import SltEdgeTypes


class GCNDecLayer(MessagePassing):
    def __init__(self, device, hidden_size, embed_size, is_first=False):
        super(GCNDecLayer, self).__init__(node_dim=0, aggr='add')
        # node_dim = axis along which propagation is done
        # aggr = aggregation function (add = SUM)

        self.device = device
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.is_first = is_first

        self.lin_src_x_att = Linear(embed_size, embed_size, bias=True, weight_initializer='glorot')
        self.lin_src_x_context = Linear(embed_size, hidden_size, bias=True, weight_initializer='glorot')

        self.lin_gg = Linear(embed_size, hidden_size, bias=True, weight_initializer='glorot')
        self.lin_pc = Linear(embed_size, hidden_size, bias=True, weight_initializer='glorot')
        self.lin_bb = Linear(embed_size, hidden_size, bias=True, weight_initializer='glorot')
        self.lin_cc = Linear(embed_size, hidden_size, bias=True, weight_initializer='glorot')

        self.lin_h = Linear(hidden_size, embed_size, bias=True, weight_initializer='glorot')
        self.lin_z = Linear(hidden_size, embed_size, bias=True, weight_initializer='glorot')

        self.bias = Parameter(torch.Tensor(embed_size))
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src_x_att.reset_parameters()
        self.lin_src_x_context.reset_parameters()
        self.lin_gg.reset_parameters()
        self.lin_pc.reset_parameters()
        self.lin_bb.reset_parameters()
        self.lin_cc.reset_parameters()
        self.lin_h.reset_parameters()
        self.lin_z.reset_parameters()
        zeros(self.bias)

    def forward(self, src_x, x, edge_index, edge_type, restrict_update_to=None):
        # linear transformation of nodes given by different possible edge types
        gg_h = self.lin_gg(x)
        pc_h = self.lin_pc(x)
        bb_h = self.lin_bb(x)
        cc_h = self.lin_cc(x)

        # differ edges based on their type - grandparent-grandchild, parent-child, brothers
        gg_edges_mask = (edge_type == SltEdgeTypes.GRANDPARENT_GRANDCHILD).to(torch.long).unsqueeze(1)
        pc_edges_mask = (edge_type == SltEdgeTypes.PARENT_CHILD).to(torch.long).unsqueeze(1)
        bb_edges_mask = (edge_type == SltEdgeTypes.LEFTBROTHER_RIGHTBROTHER).to(torch.long).unsqueeze(1)
        cc_edges_mask = (edge_type == SltEdgeTypes.CURRENT_CURRENT).to(torch.long).unsqueeze(1)

        out = self.propagate(
            edge_index=edge_index, x=x, src_x=src_x,
            gg_h=gg_h, pc_h=pc_h, bb_h=bb_h, cc_h=cc_h,
            gg_edges_mask=gg_edges_mask, pc_edges_mask=pc_edges_mask,
            bb_edges_mask=bb_edges_mask, cc_edges_mask=cc_edges_mask,
            size=None)
        out += self.bias

        if restrict_update_to:
            # during eval time update features only for currently generated node
            # mask new features that shall be preserved
            update_mask = torch.zeros(x.size(0), dtype=torch.long).to(self.device)
            update_mask[restrict_update_to] = 1
            update_mask = update_mask.unsqueeze(1)
            out = out * update_mask
            # mask old features that shall be preserved
            preserve_mask = torch.abs(update_mask - 1)
            x = x * preserve_mask
            # combine old and new
            out = out + x

        return out

    def message(self, x_j, gg_h_j, pc_h_j, bb_h_j, cc_h_j, gg_edges_mask, pc_edges_mask, bb_edges_mask, cc_edges_mask):
        # get node features lin. transformation based on the connection type to current node
        gg_h_j_masked = gg_h_j * gg_edges_mask
        pc_h_j_masked = pc_h_j * pc_edges_mask
        bb_h_j_masked = bb_h_j * bb_edges_mask
        if self.is_first:
            # if first layer - current node features set to zero - to learn
            cc_h_j_masked = torch.zeros(cc_h_j.size()).to(self.device) * cc_edges_mask
        else:
            cc_h_j_masked = cc_h_j * cc_edges_mask

        # merge results for all edge types
        h_j = torch.sum(torch.stack([gg_h_j_masked, pc_h_j_masked, bb_h_j_masked, cc_h_j_masked]), dim=0)

        # identify left brother and parent nodes for each node
        x_prev_bro_j = x_j * bb_edges_mask
        x_prev_pa_j = x_j * pc_edges_mask

        # pad to stack
        if h_j.size(1) > x_prev_bro_j.size(1):
            x_prev_bro_j = F.pad(x_prev_bro_j, (0,h_j.size(1) - x_prev_bro_j.size(1)), mode='constant', value=0.0)
            x_prev_pa_j = F.pad(x_prev_pa_j, (0,h_j.size(1) - x_prev_pa_j.size(1)), mode='constant', value=0.0)
        else:
            h_j = F.pad(h_j, (0, x_prev_bro_j.size(1) - h_j.size(1)), mode='constant', value=0.0)

        bro_pa_h_j = torch.stack([x_prev_bro_j, x_prev_pa_j, h_j], dim=0)
        bro_pa_h_j = bro_pa_h_j.permute(1, 0, 2)
        return bro_pa_h_j

    def update(self, bro_par_h, src_x):
        # linear tr. of source graph for attention and context vector
        src_x_att = self.lin_src_x_att(src_x)
        src_x_context = self.lin_src_x_context(src_x)
        # recover feature vectors of brother, parent and new of current from message fun.
        bro_par_h = bro_par_h.permute(1, 0, 2)
        x_prev_bro, x_prev_pa, h = torch.unbind(bro_par_h, dim=0)
        # remove padding added due to stacking
        x_prev_bro = x_prev_bro[:, 0:self.embed_size]
        x_prev_pa = x_prev_pa[:, 0:self.embed_size]
        h = h[:, 0:self.hidden_size]
        # GLU activation on h - gcn feature vector
        h_in_and_condition = torch.cat([h, h], dim=1)
        h = F.glu(h_in_and_condition, dim=1)
        # compute context vector
        q = self.lin_h(h) + x_prev_bro + x_prev_pa
        alpha = q @ src_x_att.t().contiguous()
        alpha = F.softmax(alpha, dim=0)
        c = alpha @ src_x_context
        # combine context vector and feature vector acquired from graph convolution
        z = h + c
        z = self.lin_z(z)
        return z
