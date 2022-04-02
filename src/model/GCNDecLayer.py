import torch
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing, Linear
from torch_geometric.nn.inits import zeros
import torch.nn.functional as F

from src.definitions.SltEdgeTypes import SltEdgeTypes


class GCNDecLayer(MessagePassing):
    def __init__(self, device, f_size, in_size, out_size, is_first=False):
        """
        :param device: torch device - cpu/gpu
        :param f_size: input graph features size
        :param in_size: processed (output) graph features size on input
        :param out_size: processed (output) graph features size on output
        :param is_first: whether this layer is first in sequence (required during training time only)
        """
        super(GCNDecLayer, self).__init__(node_dim=0, aggr='add')
        # node_dim = axis along which propagation is done
        # aggr = aggregation function (add = SUM)

        self.device = device
        self.f_size = f_size
        self.in_size = in_size
        self.out_size = out_size
        self.is_first = is_first

        self.lin_f_att = Linear(f_size, in_size, bias=True, weight_initializer='glorot')
        self.lin_f_context = Linear(f_size, out_size, bias=True, weight_initializer='glorot')

        self.lin_gg = Linear(in_size, out_size, bias=True, weight_initializer='glorot')
        self.lin_pc = Linear(in_size, out_size, bias=True, weight_initializer='glorot')
        self.lin_bb = Linear(in_size, out_size, bias=True, weight_initializer='glorot')
        self.lin_cc = Linear(in_size, out_size, bias=True, weight_initializer='glorot')

        self.lin_h = Linear(out_size, in_size, bias=True, weight_initializer='glorot')
        self.lin_z = Linear(out_size, out_size, bias=True, weight_initializer='glorot')

        self.bias = Parameter(torch.Tensor(out_size))
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_f_att.reset_parameters()
        self.lin_f_context.reset_parameters()
        self.lin_gg.reset_parameters()
        self.lin_pc.reset_parameters()
        self.lin_bb.reset_parameters()
        self.lin_cc.reset_parameters()
        self.lin_h.reset_parameters()
        self.lin_z.reset_parameters()
        zeros(self.bias)

    def forward(self, f, x, edge_index, edge_type, restrict_update_to=None):
        # linear transformation of nodes given by different possible edge types
        gg_h = self.lin_gg(x)
        pc_h = self.lin_pc(x)
        bb_h = self.lin_bb(x)
        cc_h = self.lin_cc(x)

        # differ edges based on their type - grandparent-grandchild, parent-child, brothers, self-loops
        gg_edges_mask = (edge_type == SltEdgeTypes.GRANDPARENT_GRANDCHILD).to(torch.long).unsqueeze(1)
        pc_edges_mask = (edge_type == SltEdgeTypes.PARENT_CHILD).to(torch.long).unsqueeze(1)
        bb_edges_mask = (edge_type == SltEdgeTypes.LEFTBROTHER_RIGHTBROTHER).to(torch.long).unsqueeze(1)
        cc_edges_mask = (edge_type == SltEdgeTypes.CURRENT_CURRENT).to(torch.long).unsqueeze(1)

        # propagate messages in neigh. and get attn to src graph - message method
        out = self.propagate(
            edge_index=edge_index, x=x, f=f,
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
            # if first layer - current node features set to zero - emulate eval time processing
            cc_h_j_masked = torch.zeros(cc_h_j.size()).to(self.device)
        else:
            cc_h_j_masked = cc_h_j * cc_edges_mask

        # merge results for all edge types
        h_j = torch.sum(torch.stack([gg_h_j_masked, pc_h_j_masked, bb_h_j_masked, cc_h_j_masked]), dim=0)

        # get brother and parent node features for each node for attention in update (old values)
        x_bro_j = x_j * bb_edges_mask
        x_pa_j = x_j * pc_edges_mask

        # pad and stack hidden message, and parent+brother features for node in (packing)
        if h_j.size(1) > x_bro_j.size(1):
            x_bro_j = F.pad(x_bro_j, (0, h_j.size(1) - x_bro_j.size(1)), mode='constant', value=0.0)
            x_pa_j = F.pad(x_pa_j, (0, h_j.size(1) - x_pa_j.size(1)), mode='constant', value=0.0)
        else:
            h_j = F.pad(h_j, (0, x_bro_j.size(1) - h_j.size(1)), mode='constant', value=0.0)

        # pack the values a permute, so that the will be correctly distributed among nodes
        packed_msg_j = torch.stack([x_bro_j, x_pa_j, h_j], dim=0)
        # [3, num_edges, features] -> [num_edges, 3, features]
        packed_msg_j = packed_msg_j.permute(1, 0, 2)
        return packed_msg_j

    def update(self, packed_msg, f):
        # linear tr. of source graph for attention and context vector
        f_att = self.lin_f_att(f)
        f_context = self.lin_f_context(f)
        # recover feature vectors of brother, parent and new of current from message fun.
        # [num_nodes, 3, features] -> [3, num_nodes, features]
        packed_msg = packed_msg.permute(1, 0, 2)
        # get slices == unpack
        x_bro, x_pa, h = torch.unbind(packed_msg, dim=0)
        # remove padding added due to stacking
        x_bro = x_bro[:, 0:self.in_size]
        x_pa = x_pa[:, 0:self.in_size]
        h = h[:, 0:self.out_size]
        # GLU activation on h - gcn feature vector
        h_in_and_condition = torch.cat([h, h], dim=1)
        h = F.glu(h_in_and_condition, dim=1)
        # compute context vector
        q = self.lin_h(h) + x_bro + x_pa
        alpha = q @ f_att.t().contiguous()
        alpha = F.softmax(alpha, dim=0)
        c = alpha @ f_context
        # combine context vector and feature vector acquired from graph convolution
        z = h + c
        z = self.lin_z(z)
        return z
