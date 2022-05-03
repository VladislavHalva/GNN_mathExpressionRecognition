import os.path
import re
from datetime import date
from itertools import zip_longest
from rapidfuzz.distance import Levenshtein
import numpy as np
import torch

from src.data.GPairData import GPairData
from src.utils.SltDiff import SltDiff
from src.utils.SltParser import SltParser


def create_attn_gt(data_batch, end_node_token_id):
    comp_symbols = data_batch.comp_symbols
    # create empty attention matrix
    attn_gt = np.zeros((comp_symbols.shape[0], data_batch.tgt_y.shape[0]))
    # set subgraph attention for all symbols as softmax along all corresponding components
    for cs_i in range(len(comp_symbols)):
        attn_gt[cs_i][comp_symbols[cs_i]] = 1
    attn_gt = np.transpose(attn_gt)
    attn_gt_rows_sum = attn_gt.sum(axis=1)
    attn_gt_rows_sum[attn_gt_rows_sum < 1] = 1
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


def split_databatch(databatch):
    db = databatch
    batch_ids = torch.unique(db.x_batch, sorted=True)
    data_elems = []
    for batch_id in batch_ids:
        # create masks for the items for which the dataloader follows batch == batch-item reference is available
        x_b_ids = (db.x_batch == batch_id).nonzero(as_tuple=True)[0]
        y_b_ids = (db.y_batch == batch_id).nonzero(as_tuple=True)[0]
        tgt_y_b_ids = (db.tgt_y_batch == batch_id).nonzero(as_tuple=True)[0]
        gt_b_ids = (db.gt_batch == batch_id).nonzero(as_tuple=True)[0]
        gt_ml_b_ids = (db.gt_ml_batch == batch_id).nonzero(as_tuple=True)[0]
        if db.filename is not None:
            filename_b_id = (db.filename_batch == batch_id).nonzero(as_tuple=True)[0]
            filename_b = db.filename[filename_b_id]
        else:
            filename_b = None

        # get latex GTs for current batch item
        gt_b = db.gt[gt_b_ids]
        gt_ml_b = db.gt_ml[gt_ml_b_ids]

        # get source graph elements for current batch item
        x_b = db.x[x_b_ids]
        comp_symbols_b = db.comp_symbols[x_b_ids]
        x_score_b = db.x_score[x_b_ids]
        edge_index_b_ids = [i for i, src in enumerate(db.edge_index[0]) if
                            src in x_b_ids and db.edge_index[1][i] in x_b_ids]
        edge_index_b = db.edge_index.t()[edge_index_b_ids].t()
        edge_index_b[0] = (edge_index_b[0].view(-1, 1) == x_b_ids).int().argmax(dim=1)
        edge_index_b[1] = (edge_index_b[1].view(-1, 1) == x_b_ids).int().argmax(dim=1)
        edge_attr_b = db.edge_attr[edge_index_b_ids]

        # get target graph elements for current batch item
        tgt_y_b = db.tgt_y[tgt_y_b_ids]
        attn_gt_b = db.attn_gt[tgt_y_b_ids]
        tgt_edge_index_b_ids = [i for i, src in enumerate(db.tgt_edge_index[0]) if
                                src in tgt_y_b_ids and db.tgt_edge_index[1][i] in tgt_y_b_ids]
        tgt_edge_index_b = db.tgt_edge_index.t()[tgt_edge_index_b_ids].t()
        tgt_edge_index_b[0] = (tgt_edge_index_b[0].view(-1, 1) == tgt_y_b_ids).int().argmax(dim=1)
        tgt_edge_index_b[1] = (tgt_edge_index_b[1].view(-1, 1) == tgt_y_b_ids).int().argmax(dim=1)
        tgt_edge_type_b = db.tgt_edge_type[tgt_edge_index_b_ids]
        tgt_edge_relation_b = db.tgt_edge_relation[tgt_edge_index_b_ids]

        # get output graph elements for current batch item
        y_b = db.y[y_b_ids]
        y_score_b = db.y_score[y_b_ids]
        y_edge_index_b_ids = [i for i, src in enumerate(db.y_edge_index[0]) if
                              src in y_b_ids and db.y_edge_index[1][i] in y_b_ids]
        y_edge_index_b = db.y_edge_index.t()[y_edge_index_b_ids].t()
        y_edge_index_b[0] = (y_edge_index_b[0].view(-1, 1) == y_b_ids).int().argmax(dim=1)
        y_edge_index_b[1] = (y_edge_index_b[1].view(-1, 1) == y_b_ids).int().argmax(dim=1)
        y_edge_type_b = db.y_edge_type[y_edge_index_b_ids]
        y_edge_rel_score_b = db.y_edge_rel_score[y_edge_index_b_ids]

        # create new data-element with data for current batch item only
        data_b = GPairData(
            x=x_b, edge_index=edge_index_b, edge_attr=edge_attr_b,
            gt=gt_b, gt_ml=gt_ml_b, tgt_y=tgt_y_b, tgt_edge_index=tgt_edge_index_b,
            tgt_edge_type=tgt_edge_type_b, tgt_edge_relation=tgt_edge_relation_b,
            comp_symbols=comp_symbols_b, filename=filename_b
        )
        data_b.x_score = x_score_b
        data_b.attn_gt = attn_gt_b
        data_b.y = y_b
        data_b.y_score = y_score_b
        data_b.y_edge_index = y_edge_index_b
        data_b.y_edge_type = y_edge_type_b
        data_b.y_edge_rel_score = y_edge_rel_score_b
        data_elems.append(data_b)

    return data_elems


def compute_single_item_stats(data, tokenizer):
    stats = {}

    # get token and edge relations predictions
    y_pred = torch.argmax(data.y_score, dim=1)
    y_edge_rel_pred = torch.argmax(data.y_edge_rel_score, dim=1)

    stats['gt_node_symbols'] = tokenizer.decode(data.tgt_y.tolist())
    stats['pred_node_symbols'] = tokenizer.decode(y_pred.tolist())

    # parse latex string from Symbol Layout Tree
    latex, latex_symbols = SltParser.slt_to_latex(
        tokenizer, y_pred, y_edge_rel_pred,
        data.y_edge_index, data.y_edge_type)
    stats['latex_pred'] = latex

    # decode ground-truth latex string - created from MathML GT
    gt_ml = tokenizer.decode(data.gt_ml.tolist())
    gt_ml = re.sub(' +', ' ', gt_ml)
    gt_ml_symbols = [tokenizer.decode([symbol_id]) for symbol_id in data.gt_ml.tolist() if
                     tokenizer.decode([symbol_id]).strip() != '']
    stats['latex_gt'] = gt_ml

    # calculate edit-distance
    # on latex GT created from MathML --> Latex GT in files often differs
    stats['edit_distance_str'] = Levenshtein.distance(gt_ml, latex)
    stats['edit_distance_seq'] = Levenshtein.distance(gt_ml_symbols, latex_symbols)

    # calculate difference of Symbol Layout Trees
    slt_diff = SltDiff(
        tokenizer,
        y_pred, data.y_edge_index, data.y_edge_type, y_edge_rel_pred,
        data.tgt_y, data.tgt_edge_index, data.tgt_edge_type, data.tgt_edge_relation
    )
    slt_diff.eval()
    stats['slt_diff'] = slt_diff.get_result()

    # compute vague count of correct symbols in string sequences
    target_string = [tokenizer.decode([char], skip_special_tokens=True) for char in data.gt_ml.tolist()]
    target_string = [char for char in target_string if char != '' and char != ' ']
    predicted_string = latex_symbols
    symbols_count = 0
    correct_symbols_count = 0
    for gt, pred in zip_longest(target_string, predicted_string, fillvalue=None):
        symbols_count += 1
        if gt is not None and pred is not None and gt == pred:
            correct_symbols_count += 1
    stats['seq_symbols_count'] = symbols_count
    stats['seq_correct_symbols_count'] = correct_symbols_count

    return stats


def mathml_unicode_to_latex_label(label, skip_curly_brackets=False):
    if label in ['alpha', 'beta', 'sin', 'cos', 'tan', 'rightarrow', 'sum', 'int', 'pi',
                 'leq', 'lim', 'geq', 'infty', 'prime', 'times', 'pm', 'log']:
        return '\\' + label

    if label in ['}', '{'] and not skip_curly_brackets:
        return '\\' + label

    transcriptions = {
        '÷': '\\div',
        '×': '\\times',
        '±': '\\pm',
        '∑': '\\sum',
        'π': '\\pi',
        '∫': '\\int',
        'θ': '\\theta',
        '∞': '\\infty',
        '…': '\\ldots',
        'β': '\\beta',
        '→': '\\rightarrow',
        '≤': '\\leq',
        '≥': '\\geq',
        '<': '\\lt',
        '>': '\\gt',
        'σ': '\\sigma',
        'ϕ': '\\phi',
        '′': '\\prime',
        'Γ': '\\gamma',
        'γ': '\\gamma',
        'μ': '\\mu',
        'λ': '\\lambda',
        'Δ': '\\Delta',
        '∃': '\\exists',
        '∀': '\\forall',
        '∈': '\\in',
        '∂': '\\partial',
        '≠': '\\neq',
        'α': '\\alpha',
        '−': '-'
    }
    if label in transcriptions.keys():
        return transcriptions[label]

    return label


def create_latex_result_file(directory, filename, latex, author):
    if not os.path.exists(directory):
        return False

    today = date.today()
    today = today.strftime("%B %d %Y")
    file_content = f"---\ntitle: {filename}\nauthor: {author}\ndate: {today}\n---\n\n$${latex}$$"
    try:
        with open(os.path.join(directory, filename+'.txt'), 'w') as fd:
            fd.write(file_content)
    except Exception as e:
        return False

    return True

