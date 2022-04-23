import glob
import os
import shutil

import numpy as np
import torch


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
    # for i, y_i in enumerate(data_batch.tgt_y.squeeze(1)):
    #     if y_i == end_node_token_id:
    #         batch_index = data_batch.tgt_y_batch[i]
    #         attn_gt[i][batch_masks[batch_index]] = 1 / batch_masks[batch_index].shape[0]

    data_batch.attn_gt = attn_gt
    return data_batch
