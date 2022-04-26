import glob
import os
import re
import shutil
from itertools import zip_longest

import numpy as np
import torch
import torch.nn.functional as F

from src.definitions.SltEdgeTypes import SltEdgeTypes
from src.utils.SltParser import SltParser


def cpy_simple_train_gt():
    orig_images_root = 'assets/crohme/train/img'
    orig_inkmls_root = 'assets/crohme/train/inkml'
    tgt_images_root = 'assets/crohme/train_simple/img'
    tgt_inkmls_root = 'assets/crohme/train_simple/inkml'
    for subdir, _, files in os.walk(tgt_images_root):
        for file in files:
            image_file = file
            file_name = '.'.join(file.split('.')[:-1])
            orig_image_path = glob.glob(orig_images_root + "/**/" + image_file, recursive=True)
            if len(orig_image_path) == 1:
                orig_image_path = orig_image_path[0]
                orig_image_relative_from_root = os.path.relpath(orig_image_path, orig_images_root)
                orig_inkml_path = os.path.join(orig_inkmls_root, orig_image_relative_from_root)
                orig_inkml_path = '.'.join(orig_inkml_path.split('.')[:-1])
                orig_inkml_path += '.inkml'
                if os.path.exists(orig_inkml_path):
                    tgt_inkml_path = os.path.join(tgt_inkmls_root, file_name + '.inkml')
                    shutil.copyfile(orig_inkml_path, tgt_inkml_path)
                else:
                    print('inkml does not exists')
            else:
                print('not found or multiple with same name: ' + image_file)


def create_attn_gt(data_batch, end_node_token_id):
    comp_symbols = data_batch.comp_symbols
    # create empty attention matrix
    attn_gt = np.zeros((comp_symbols.shape[0], data_batch.tgt_y.shape[0]))
    # set subgraph attention for all symbols as softmax along all corresponding compoennts
    for cs_i in range(len(comp_symbols)):
        attn_gt[cs_i][comp_symbols[cs_i]] = 1
    attn_gt = np.transpose(attn_gt)
    attn_gt_rows_sum = attn_gt.sum(axis=1)
    attn_gt_rows_sum[attn_gt_rows_sum<1] = 1
    attn_gt = attn_gt / attn_gt_rows_sum[:, np.newaxis]
    attn_gt = torch.from_numpy(attn_gt)
    # get indices of components belonging to each of batch items
    batch_indices = torch.unique(data_batch.x_batch)
    batch_masks = []
    for batch_index in batch_indices:
        batch_mask = (data_batch.x_batch == batch_index).nonzero(as_tuple=False)
        batch_masks.append(batch_mask)
    # set attention for end leaf nodes as uniform among all components belonging to the same batch item
    # --> whole formula
    for i, y_i in enumerate(data_batch.tgt_y):
        if y_i == end_node_token_id:
            batch_index = data_batch.tgt_y_batch[i]
            attn_gt[i][batch_masks[batch_index]] = 1 / batch_masks[batch_index].shape[0]

    data_batch.attn_gt = attn_gt
    return data_batch


def calc_and_print_acc(data, tokenizer, during_training=False):
    result = {}

    y_pred = torch.argmax(data.y_score, dim=1)
    y_edge_rel_pred = torch.argmax(data.y_edge_rel_score, dim=1)

    latex = SltParser.slt_to_latex_predictions(tokenizer, y_pred, y_edge_rel_pred, data.y_edge_index, data.y_edge_type)
    # latex = SltParser.slt_to_latex_predictions(tokenizer, out.tgt_y.squeeze(1), out.tgt_edge_relation, out.tgt_edge_index, out.tgt_edge_type)

    gt_ml = tokenizer.decode(data.gt_ml.tolist())
    gt_ml = re.sub(' +', ' ', gt_ml)

    print('GT: ' + tokenizer.decode(data.tgt_y.tolist()))
    print('PR: ' + tokenizer.decode(y_pred.tolist()))
    print('GT: ' + gt_ml)
    print('PR: ' + latex)
    print("\n")

    if during_training:
        target_tokens = data.tgt_y
        predicted_tokens = y_pred
        tokens_count = target_tokens.shape[0]
        correct_tokens_count = torch.sum((target_tokens == predicted_tokens))
        result['tokens_count'] = tokens_count
        result['correct_tokens_count'] = correct_tokens_count

        tgt_edge_pc_indices = ((data.tgt_edge_type == SltEdgeTypes.PARENT_CHILD).nonzero(as_tuple=True)[0])
        tgt_pc_edge_relation = data.tgt_edge_relation[tgt_edge_pc_indices]
        out_pc_edge_relation = data.y_edge_rel_score[tgt_edge_pc_indices]
        out_pc_edge_relation = out_pc_edge_relation.argmax(dim=-1)
        edges_count = tgt_pc_edge_relation.shape[0]
        correct_edges_count = torch.sum((tgt_pc_edge_relation == out_pc_edge_relation))
        result['edges_count'] = edges_count
        result['correct_edges_count'] = correct_edges_count

    target_string = gt_ml
    predicted_string = latex
    symbols_count = 0
    correct_symbols_count = 0
    for gt, pred in zip_longest(target_string, predicted_string, fillvalue=None):
        symbols_count += 1
        if gt is not None and pred is not None and gt == pred:
            correct_symbols_count += 1
    result['symbols_count'] = symbols_count
    result['correct_symbols_count'] = correct_symbols_count

    return result