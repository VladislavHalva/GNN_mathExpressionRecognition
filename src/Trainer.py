import logging
import os.path
from datetime import datetime
from math import ceil

import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from src.data.GMathDataset import CrohmeDataset
from src.data.LatexVocab import LatexVocab
from src.definitions.SltEdgeTypes import SltEdgeTypes
from src.definitions.exceptions.ModelParamsError import ModelParamsError
from src.model.Model import Model
from src.utils.loss import calculate_loss
from src.utils.utils import create_attn_gt, split_databatch, compute_single_item_stats, create_latex_result_file


class Trainer:
    def __init__(
        self,
        model_name,
        tokenizer_path,
        vocab_path=None,
        load_vocab=False,
        inkml_folder_vocab=None,
        load_model=None,
        writer=None,
        temp_path=None
    ):
        # define metaparameters
        self.components_shape = (32, 32)
        self.edge_features = 10
        self.edge_h_size = 64
        self.enc_in_size = 256
        self.enc_h_size = 256
        self.enc_out_size = 128
        self.dec_in_size = 256
        self.dec_h_size = 256
        self.emb_size = 256

        self.enc_vgg_dropout_p = 0.0
        self.enc_gat_dropout_p = 0.0
        self.dec_emb_dropout_p = 0.0
        self.dec_att_dropout_p = 0.0

        self.substitute_terms = False

        # use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Device: {self.device}")

        # load or create tokenizer
        if load_vocab and os.path.exists(tokenizer_path):
            self.tokenizer = LatexVocab.load_tokenizer(tokenizer_path)
            logging.info(f"Tokenizer loaded from: {tokenizer_path}")
        elif inkml_folder_vocab is not None and vocab_path is not None and os.path.exists(vocab_path) and os.path.exists(inkml_folder_vocab):
            include_latex_gt = not self.substitute_terms
            LatexVocab.generate_formulas_file_from_inkmls(inkml_folder_vocab, vocab_path, substitute_terms=self.substitute_terms, latex_gt=include_latex_gt, mathml_gt=True)
            self.tokenizer = LatexVocab.create_tokenizer(vocab_path, min_freq=2)
            LatexVocab.save_tokenizer(self.tokenizer, tokenizer_path)
            logging.info(f"Tokenizer created as: {tokenizer_path}")
        else:
            raise ModelParamsError('vocabulary could not be initialized')
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.end_node_token_id = self.tokenizer.encode("[EOS]", add_special_tokens=False).ids[0]
        logging.info(f"Vocab size: {self.vocab_size}")

        # init model
        now = datetime.now()
        self.model_name = model_name + '_' + now.strftime("%y-%m-%d_%H-%M-%S")

        self.model = Model(
            self.device, self.edge_features, self.edge_h_size,
            self.enc_in_size, self.enc_h_size, self.enc_out_size, self.dec_in_size, self.dec_h_size, self.emb_size,
            self.vocab_size, self.end_node_token_id, self.tokenizer,
            self.enc_vgg_dropout_p, self.enc_gat_dropout_p, self.dec_emb_dropout_p, self.dec_att_dropout_p)
        self.model.double()
        if load_model is not None and os.path.exists(load_model):
            self.model.load_state_dict(torch.load(load_model, map_location=self.device))
            logging.info(f"Model loaded: {load_model}")
        self.model.to(self.device)
        # wandb.watch(self.model)

        # init summary writer
        if writer is not None and os.path.exists(writer):
            self.writer = SummaryWriter(os.path.join(writer, self.model_name))
        else:
            self.writer = False

        if temp_path is not None and os.path.exists(temp_path):
            self.temp_path = temp_path
        else:
            self.temp_path = temp_path

        self.eval_during_training = False
        self.eval_train_settings = None

    def unset_eval_during_training(self):
        self.eval_during_training = False
        self.eval_train_settings = None

    def set_eval_during_training(self, images_root, inkmls_root, batch_size, print_stats,
                                 print_item_level_stats, each_nth_epoch=1):
        if os.path.exists(images_root) and os.path.exists(inkmls_root) and each_nth_epoch > 0:
            self.eval_during_training = True
            self.eval_train_settings = {
                'images_root': images_root,
                'inkmls_root': inkmls_root,
                'batch_size': batch_size,
                'print_stats': print_stats,
                'print_item_level_stats': print_item_level_stats,
                'each_nth_epoch': each_nth_epoch
            }

    def train(self, images_root, inkmls_root, epochs, batch_size=1, save_model_dir=None, save_checkpoint_each_nth_epoch=0):
        logging.info("\nTraining...")

        optimizer = optim.Adam(self.model.parameters(), lr=0.0003)

        trainset = CrohmeDataset(images_root, inkmls_root, self.tokenizer, self.components_shape, self.temp_path, self.substitute_terms, img_transform=True)
        trainloader = DataLoader(trainset, batch_size, True, follow_batch=['x', 'tgt_y'])

        self.model.train()

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_stats = {
                'symbol_count': 0,
                'correct_symbol_count': 0,
                'edge_count': 0,
                'correct_edge_count': 0,
                'src_node_count': 0,
                'correct_src_node_count': 0,
                'src_edge_count': 0,
                'correct_src_edge_count': 0,
                'attn_relevant_items': 0,
                'attn_block1_abs_diff': 0,
                'attn_block2_abs_diff': 0,
                'attn_block3_abs_diff': 0
            }
            logging.info(f"\nEPOCH: {epoch}")

            for i, data_batch in enumerate(tqdm(trainloader)):
                data_batch = create_attn_gt(data_batch, self.end_node_token_id)
                data_batch = data_batch.to(self.device)

                optimizer.zero_grad()
                out = self.model(data_batch)

                loss = calculate_loss(out, self.end_node_token_id, self.device, self.writer, writer_idx=epoch * len(trainloader) + i)
                loss.backward()

                # gradient clipping
                nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)

                optimizer.step()

                batch_stats = self.evaluate_training(out.detach())
                epoch_stats['symbol_count'] += batch_stats['symbol_count']
                epoch_stats['correct_symbol_count'] += batch_stats['correct_symbol_count']
                epoch_stats['edge_count'] += batch_stats['edge_count']
                epoch_stats['correct_edge_count'] += batch_stats['correct_edge_count']
                epoch_stats['src_node_count'] += batch_stats['src_node_count']
                epoch_stats['correct_src_node_count'] += batch_stats['correct_src_node_count']
                epoch_stats['src_edge_count'] += batch_stats['src_edge_count']
                epoch_stats['correct_src_edge_count'] += batch_stats['correct_src_edge_count']
                epoch_stats['attn_relevant_items'] += batch_stats['attn_relevant_items']
                epoch_stats['attn_block1_abs_diff'] += batch_stats['attn_block1_abs_diff']
                epoch_stats['attn_block2_abs_diff'] += batch_stats['attn_block2_abs_diff']
                epoch_stats['attn_block3_abs_diff'] += batch_stats['attn_block3_abs_diff']
                if self.writer:
                    self.writer.add_scalar('ItemAttnMeanAbsDiff_blk1/train', batch_stats['attn_block1_mean_abs_diff'], epoch * len(trainloader) + i)
                    self.writer.add_scalar('ItemAttnMeanAbsDiff_blk2/train', batch_stats['attn_block2_mean_abs_diff'], epoch * len(trainloader) + i)
                    self.writer.add_scalar('ItemAttnMeanAbsDiff_blk3/train', batch_stats['attn_block3_mean_abs_diff'], epoch * len(trainloader) + i)

                epoch_loss += loss.item()
                if self.writer:
                    self.writer.add_scalar('ItemLoss/train', loss.item(), epoch * len(trainloader) + i)

            logging.info(f" epoch loss total: {epoch_loss}")
            logging.info(f" epoch loss avg:   {epoch_loss/len(trainloader)}")

            if self.writer:
                self.writer.add_scalar('EpochLossTotal/train', epoch_loss, epoch)
                self.writer.add_scalar('EpochLossAvg/train', epoch_loss / len(trainloader), epoch)

            if save_checkpoint_each_nth_epoch != 0 and epoch % save_checkpoint_each_nth_epoch == save_checkpoint_each_nth_epoch - 1:
                save_model_check_name = self.model_name + '_' + str(epoch) + '.pth'
                torch.save(self.model.state_dict(), os.path.join(save_model_dir, save_model_check_name))

            epoch_stats['symbol_acc'] = epoch_stats['correct_symbol_count'] / epoch_stats['symbol_count'] if epoch_stats['symbol_count'] > 0 else 0
            epoch_stats['edge_acc'] = epoch_stats['correct_edge_count'] / epoch_stats['edge_count'] if epoch_stats['edge_count'] > 0 else 0
            epoch_stats['src_symbol_acc'] = epoch_stats['correct_src_node_count'] / epoch_stats['src_node_count'] if epoch_stats['src_node_count'] > 0 else 0
            epoch_stats['src_edge_acc'] = epoch_stats['correct_src_edge_count'] / epoch_stats['src_edge_count'] if epoch_stats['src_edge_count'] > 0 else 0
            epoch_stats['attn_block1_mean_abs_diff'] = epoch_stats['attn_block1_abs_diff'] / epoch_stats['attn_relevant_items'] if epoch_stats['attn_relevant_items'] > 0 else 0
            epoch_stats['attn_block2_mean_abs_diff'] = epoch_stats['attn_block2_abs_diff'] / epoch_stats['attn_relevant_items'] if epoch_stats['attn_relevant_items'] > 0 else 0
            epoch_stats['attn_block3_mean_abs_diff'] = epoch_stats['attn_block3_abs_diff'] / epoch_stats['attn_relevant_items'] if epoch_stats['attn_relevant_items'] > 0 else 0

            if self.writer:
                self.writer.add_scalar('SetSymAcc/train', epoch_stats['symbol_acc'], epoch)
                self.writer.add_scalar('SetEdgeAcc/train', epoch_stats['edge_acc'], epoch)
                self.writer.add_scalar('SetSrcSymAcc/train', epoch_stats['src_symbol_acc'], epoch)
                self.writer.add_scalar('SetSrcEdgeAcc/train', epoch_stats['src_edge_acc'], epoch)
                self.writer.add_scalar('SetAttnMeanAbsDiff_blk1/train', epoch_stats['attn_block1_mean_abs_diff'], epoch)
                self.writer.add_scalar('SetAttnMeanAbsDiff_blk2/train', epoch_stats['attn_block2_mean_abs_diff'], epoch)
                self.writer.add_scalar('SetAttnMeanAbsDiff_blk3/train', epoch_stats['attn_block3_mean_abs_diff'], epoch)

            logging.info(f" symbol class acc: {epoch_stats['symbol_acc'] * 100:.3f}%")
            logging.info(f" edge class acc:   {epoch_stats['edge_acc'] * 100:.3f}%")


            if self.eval_during_training and epoch % self.eval_train_settings['each_nth_epoch'] == self.eval_train_settings['each_nth_epoch'] - 1:
                self.evaluate(
                    self.eval_train_settings['images_root'],
                    self.eval_train_settings['inkmls_root'],
                    self.eval_train_settings['batch_size'],
                    self.writer, epoch,
                    self.eval_train_settings['print_stats'],
                    self.eval_train_settings['print_item_level_stats']
                )

        if save_model_dir is not None and os.path.exists(save_model_dir):
            save_model_final_name = self.model_name + '_final.pth'
            torch.save(self.model.state_dict(), os.path.join(save_model_dir, save_model_final_name))
            logging.info(f"Model saved as: {os.path.join(save_model_dir, save_model_final_name)}")

    def evaluate_training(self, data):
        stats = {}

        # evaluate nodes predictions = symbols
        y_pred = torch.argmax(data.y_score, dim=1)
        target_symbols = data.tgt_y
        predicted_symbols = y_pred
        stats['symbol_count'] = target_symbols.shape[0]
        stats['correct_symbol_count'] = torch.sum((target_symbols == predicted_symbols))

        # evaluate edges predictions
        tgt_edge_pc_indices = ((data.tgt_edge_type == SltEdgeTypes.PARENT_CHILD).nonzero(as_tuple=True)[0])
        tgt_pc_edge_relation = data.tgt_edge_relation[tgt_edge_pc_indices]
        out_pc_edge_relation = data.y_edge_rel_score[tgt_edge_pc_indices]
        out_pc_edge_relation = out_pc_edge_relation.argmax(dim=-1)
        stats['edge_count'] = tgt_pc_edge_relation.shape[0]
        stats['correct_edge_count'] = torch.sum((tgt_pc_edge_relation == out_pc_edge_relation))

        # evaluate src symbol prediction
        x_pred = torch.argmax(data.x_score, dim=1)
        x_gt_node = data.attn_gt.argmax(dim=0)
        x_gt = data.tgt_y[x_gt_node]
        stats['src_node_count'] = x_gt.shape[0]
        stats['correct_src_node_count'] = torch.sum(x_gt == x_pred)

        # evaluate src edge prediction
        x_edge_pred = torch.argmax(data.edge_type_score, dim=1)
        x_edge_gt = data.edge_type
        stats['src_edge_count'] = x_edge_gt.shape[0]
        stats['correct_src_edge_count'] = torch.sum(x_edge_pred == x_edge_gt)

        # evaluate attention accuracy
        alpha_batch_mask = (data.y_batch.unsqueeze(1) - data.x_batch.unsqueeze(0) != 0).long()
        no_end_node_indices = (data.tgt_y != self.end_node_token_id)
        no_end_node_mask = no_end_node_indices.unsqueeze(1).repeat(1, data.x.shape[0])
        relevant_attn_mask = torch.logical_and(alpha_batch_mask, no_end_node_mask)
        relevant_items_count = torch.sum(relevant_attn_mask.long())

        attn_gt = data.attn_gt
        block1_attn = data.gcn1_alpha * relevant_attn_mask
        block2_attn = data.gcn2_alpha * relevant_attn_mask
        block3_attn = data.gcn3_alpha * relevant_attn_mask
        block1_abs_diff = torch.abs(attn_gt - block1_attn).sum()
        block2_abs_diff = torch.abs(attn_gt - block2_attn).sum()
        block3_abs_diff = torch.abs(attn_gt - block3_attn).sum()
        block1_mean_abs_diff = block1_abs_diff / relevant_items_count
        block2_mean_abs_diff = block2_abs_diff / relevant_items_count
        block3_mean_abs_diff = block3_abs_diff / relevant_items_count
        stats['attn_relevant_items'] = relevant_items_count
        stats['attn_block1_abs_diff'] = block1_abs_diff
        stats['attn_block2_abs_diff'] = block2_abs_diff
        stats['attn_block3_abs_diff'] = block3_abs_diff
        stats['attn_block1_mean_abs_diff'] = block1_mean_abs_diff
        stats['attn_block2_mean_abs_diff'] = block2_mean_abs_diff
        stats['attn_block3_mean_abs_diff'] = block3_mean_abs_diff

        return stats

    def evaluate(self, images_root, inkmls_root, batch_size=1, writer=False, epoch=None, print_stats=True,
                 print_item_level_stats=False, store_results_dir=None, results_author=''):
        logging.info("\nEvaluation...")
        self.model.eval()

        if store_results_dir is None or not os.path.exists(store_results_dir):
            store_results_dir = None

        # load data
        testset = CrohmeDataset(images_root, inkmls_root, self.tokenizer, self.components_shape, self.temp_path, self.substitute_terms)
        testloader = DataLoader(testset, batch_size, False, follow_batch=['x', 'tgt_y', 'gt', 'gt_ml', 'filename'])

        # init statistics
        stats = {
            'exact_match': 0,
            'exact_match_1': 0,
            'exact_match_2': 0,
            'exact_match_pct': 0,
            'exact_match_1_pct': 0,
            'exact_match_2_pct': 0,
            'structure_match': 0,
            'structure_match_pct': 0,
            'edit_distances_str': [],
            'edit_distances_seq': [],
            'edit_distance_str_avg': 0,
            'edit_distance_seq_avg': 0
        }

        with torch.no_grad():
            for i, data_batch in enumerate(tqdm(testloader)):
                data_batch = create_attn_gt(data_batch, self.end_node_token_id)
                data_batch = data_batch.to(self.device)

                x = data_batch.x

                out = self.model(data_batch)
                if self.device == torch.device('cuda'):
                    out = out.cpu()

                x_gt_node = out.attn_gt.argmax(dim=0)
                x_gt = out.tgt_y[x_gt_node]
                comp_pred = torch.argmax(out.comp_class, dim=1)
                # print(x_gt)
                # print(comp_pred)

                plot_attn = False
                if plot_attn:
                    for i, y in enumerate(data_batch.tgt_y):
                        token = self.tokenizer.decode([y.item()], skip_special_tokens=False)
                        print(token)
                        y_attn_gt = data_batch.attn_gt[i]
                        y_attn_gcn1 = out.gcn1_alpha[i]
                        x_j = x.clone().squeeze(1)
                        x_a1 = x.clone().squeeze(1)
                        for i, x_i in enumerate(x_j):
                            x_j[i] = x_i * y_attn_gt[i]
                            x_j[i] /= torch.max(x_j[i])
                        for i, x_i in enumerate(x_a1):
                            x_a1[i] = x_i * y_attn_gcn1[i]
                            x_a1[i] /= torch.max(x_a1[i])
                        cols = 4
                        rows = ceil(y_attn_gt.shape[0] / cols)
                        row = 0
                        col = 0
                        _, axs = plt.subplots(rows, cols, figsize=(32, 32))
                        axs = axs.flatten()
                        for x_i, ax in zip(x_j, axs):
                            ax.imshow(x_i)
                        plt.show()
                        _, axs = plt.subplots(rows, cols, figsize=(32, 32))
                        axs = axs.flatten()
                        for x_i, ax in zip(x_a1, axs):
                            ax.imshow(x_i)
                        plt.show()

                # split result batch to separate data elements
                out_elems = split_databatch(out)
                for out_elem in out_elems:
                    item_stats = compute_single_item_stats(out_elem, self.tokenizer)
                    stats['edit_distances_str'].append(item_stats['edit_distance_str'])
                    stats['edit_distances_seq'].append(item_stats['edit_distance_seq'])
                    stats['structure_match'] += 1 if item_stats['slt_diff']['structure_match'] else 0
                    stats['exact_match'] += 1 if item_stats['slt_diff']['exact_match'] else 0
                    stats['exact_match_1'] += 1 if item_stats['slt_diff']['exact_match_1'] else 0
                    stats['exact_match_2'] += 1 if item_stats['slt_diff']['exact_match_2'] else 0

                    if print_item_level_stats:
                        seq_symbol_precision = 0
                        if item_stats['seq_symbols_count'] > 0:
                            seq_symbol_precision = item_stats['seq_correct_symbols_count'] / item_stats['seq_symbols_count']
                        logging.info(f"  gt symbols:        {item_stats['gt_node_symbols']}")
                        logging.info(f"  pr symbols:        {item_stats['pred_node_symbols']}")
                        logging.info(f"  pr symbols sp:     {item_stats['pred_node_symbols_with_special']}")
                        logging.info(f"  gt latex:          {item_stats['latex_gt']}")
                        logging.info(f"  pr latex:          {item_stats['latex_pred']}")
                        logging.info(f"  e-distance str:    {item_stats['edit_distance_str']}")
                        logging.info(f"  e-distance seq:    {item_stats['edit_distance_seq']}")
                        logging.info(f"  seq-sym-pr:        {seq_symbol_precision*100:.5f}%")
                        logging.info(f"  SLT struct-match:  {item_stats['slt_diff']['structure_match']}")
                        logging.info(f"  SLT exact-match:   {item_stats['slt_diff']['exact_match']}")
                        logging.info(f"  SLT exact-match-1: {item_stats['slt_diff']['exact_match_1']}")
                        logging.info(f"  SLT exact-match-2: {item_stats['slt_diff']['exact_match_2']}")
                        logging.info(f"  SLT sym-cls-err:   {item_stats['slt_diff']['node_class_errors']}")
                        logging.info(f"  SLT edge-cls-err:  {item_stats['slt_diff']['edge_class_errors']}")

                    if store_results_dir is not None:
                        create_latex_result_file(store_results_dir, out_elem.filename, item_stats['latex_pred'], results_author)

                    if writer and epoch is not None:
                        self.writer.add_scalar('ItemEditDistStr/eval', item_stats['edit_distance_str'], epoch + len(testloader) + i)
                        self.writer.add_scalar('ItemEditDistSeq/eval', item_stats['edit_distance_seq'], epoch + len(testloader) + i)

        stats['exact_match_pct'] = stats['exact_match'] / len(testset) if len(testset) > 0 else 0
        stats['exact_match_1_pct'] = stats['exact_match_1'] / len(testset) if len(testset) > 0 else 0
        stats['exact_match_2_pct'] = stats['exact_match_2'] / len(testset) if len(testset) > 0 else 0
        stats['structure_match_pct'] = stats['structure_match'] / len(testset) if len(testset) > 0 else 0
        stats['edit_distances_str_avg'] = np.asarray(stats['edit_distances_str']).mean()
        stats['edit_distances_seq_avg'] = np.asarray(stats['edit_distances_seq']).mean()

        if print_stats:
            logging.info(f" exact-match:    {stats['exact_match_pct']*100:.3f}% = {stats['exact_match']}")
            logging.info(f" exact-match -1: {stats['exact_match_1_pct']*100:.3f}% = {stats['exact_match_1']}")
            logging.info(f" exact-match -2: {stats['exact_match_2_pct']*100:.3f}% = {stats['exact_match_2']}")
            logging.info(f" struct-match:   {stats['structure_match_pct']*100:.3f}% = {stats['structure_match']}")
            logging.info(f" e-dist str avg: {stats['edit_distances_str_avg']:.3f}")
            logging.info(f" e-dist seq avg: {stats['edit_distances_seq_avg']:.3f}")

        if writer and epoch is not None:
            self.writer.add_scalar('SetExactMatch/eval', stats['exact_match_pct'], epoch)
            self.writer.add_scalar('SetExactMatch-1/eval', stats['exact_match_1_pct'], epoch)
            self.writer.add_scalar('SetExactMatch-2/eval', stats['exact_match_2_pct'], epoch)
            self.writer.add_scalar('SetStructMatch/eval', stats['structure_match_pct'], epoch)
            self.writer.add_scalar('SetEditDistStrAvg/eval', stats['edit_distances_str_avg'], epoch)
            self.writer.add_scalar('SetEditDistSeqAvg/eval', stats['edit_distances_seq_avg'], epoch)

        self.model.train()
        return stats
