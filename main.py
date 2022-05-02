import logging
import os.path
import re
import sys
import timeit
from itertools import chain

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import optim, nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime

from src.data.CrohmeDataset import CrohmeDataset
from src.data.LatexVocab import LatexVocab
from src.definitions.SltEdgeTypes import SltEdgeTypes
from src.definitions.SrtEdgeTypes import SrtEdgeTypes
from src.model.Model import Model
from src.utils.SltParser import SltParser
from src.utils.loss import loss_termination
from src.utils.utils import cpy_simple_train_gt, create_attn_gt, calc_and_print_acc, split_databatch

def eval_training_batch(data):
    result = {}
    y_pred = torch.argmax(data.y_score, dim=1)
    y_edge_rel_pred = torch.argmax(data.y_edge_rel_score, dim=1)

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

    tokens_acc = correct_tokens_count / tokens_count if tokens_count > 0 else 0
    print(f"tok acc: {tokens_acc:.5f} = {correct_tokens_count} / {tokens_count}")

    edges_acc = correct_edges_count / edges_count if edges_count > 0 else 0
    print(f"edg acc: {edges_acc:.5f} = {correct_edges_count} / {edges_count}")

    return result


def evaluate_model(model, images_root, inkmls_root, tokenizer, components_shape):
    logging.info("Evaluation...")
    testset = CrohmeDataset(images_root, inkmls_root, tokenizer, components_shape)
    testloader = DataLoader(testset, 5, False, follow_batch=['x', 'tgt_y', 'gt', 'gt_ml'])
    model.eval()

    symbols_count = 0
    correct_symbols_count = 0
    exact_match = 0
    exact_match_1 = 0
    exact_match_2 = 0
    edit_distances = []
    with torch.no_grad():
        for i, data_batch in enumerate(testloader):
            data_batch = create_attn_gt(data_batch, end_node_token_id)
            data_batch = data_batch.to(device)

            out = model(data_batch)
            if device == torch.device('cuda'):
                out = out.cpu()

            out_elems = split_databatch(out)
            for out_elem in out_elems:
                acc = calc_and_print_acc(out_elem, tokenizer)
                symbols_count += acc['symbols_count']
                correct_symbols_count += acc['correct_symbols_count']
                edit_distances.append(acc['edit_distance'])
                exact_match += 1 if acc['slt_diff']['exact_match'] else 0
                exact_match_1 += 1 if acc['slt_diff']['exact_match_1'] else 0
                exact_match_2 += 1 if acc['slt_diff']['exact_match_2'] else 0

    symbols_acc = correct_symbols_count / symbols_count if symbols_count > 0 else 0
    print(f"sym acc: {symbols_acc:.5f} = {correct_symbols_count} / {symbols_count}")
    print(f"e-match: {exact_match}, e-match-1: {exact_match_1}, e-match-2: {exact_match_2}")
    print(f"e-match: {exact_match/len(testset)}, e-match-1: {exact_match_1/len(testset)}, e-match-2: {exact_match_2/len(testset)}")
    print(f"avg edit distance: {np.asarray(edit_distances).mean()}")

    model.train()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    logging.getLogger('matplotlib.font_manager').disabled = True

    load_vocab = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 150
    batch_size = 4
    components_shape = (32, 32)
    edge_features = 19
    edge_h_size = 128
    enc_in_size = 400
    enc_h_size = 256
    enc_out_size = 256
    dec_h_size = 256
    emb_size = 256

    load_model = False
    load_model_path = "checkpoints/"
    load_model_name = "MER_19_400_256_tiny_22-05-02_02-39-34_final.pth"

    train = True
    evaluate = True
    save_run = True
    print_train_info = True

    # to build vocabulary
    dist_inkmls_root = 'assets/crohme/train/inkml'
    # for training
    train_images_root = 'assets/crohme/simple/img/'
    train_inkmls_root = 'assets/crohme/simple/inkml/'
    # for test
    test_images_root = 'assets/crohme/simple/img/'
    test_inkmls_root = 'assets/crohme/simple/inkml/'

    # folder where data item representations will be stored
    tmp_path = 'temp'

    if load_vocab:
        tokenizer = LatexVocab.load_tokenizer('assets/tokenizer.json')
    else:
        LatexVocab.generate_formulas_file_from_inkmls(dist_inkmls_root, 'assets/vocab.txt', latex_gt=True,
                                                      mathml_gt=True)
        tokenizer = LatexVocab.create_tokenizer('assets/vocab.txt', min_freq=2)
        LatexVocab.save_tokenizer(tokenizer, 'assets/tokenizer.json')

    vocab = tokenizer.get_vocab()
    vocab_size = tokenizer.get_vocab_size()
    end_node_token_id = tokenizer.encode("[EOS]", add_special_tokens=False).ids[0]

    logging.info(f"Vocab size: {vocab_size}")
    logging.info(f"Device: {device}")

    model = Model(
        device,
        components_shape, edge_features, edge_h_size,
        enc_in_size, enc_h_size, enc_out_size, dec_h_size, emb_size,
        vocab_size, end_node_token_id, tokenizer)
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
    model.float()
    if load_model:
        model.load_state_dict(torch.load(os.path.join(load_model_path, load_model_name), map_location=device))
        logging.info(f"Model loaded: {load_model_name}")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    loss_f = nn.CrossEntropyLoss()

    now = datetime.datetime.now()
    model_name = 'MER_' + '19_400_256_tiny' + '_' + now.strftime("%y-%m-%d_%H-%M-%S")

    if train:
        logging.info("Training...")
        trainset = CrohmeDataset(train_images_root, train_inkmls_root, tokenizer, components_shape)

        trainloader = DataLoader(trainset, batch_size, False, follow_batch=['x', 'tgt_y'])
        if save_run:
            writer = SummaryWriter('runs/' + model_name)

        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            running_loss = 0
            if print_train_info:
                print("EPOCH: " + str(epoch))

            for i, data_batch in tqdm(enumerate(trainloader)):
                data_batch = create_attn_gt(data_batch, end_node_token_id)
                data_batch = data_batch.to(device)

                optimizer.zero_grad()
                out = model(data_batch)

                # calculate loss as cross-entropy on output graph node predictions
                loss_out_node = loss_f(out.y_score, out.tgt_y)

                # calculate additional loss penalizing classification non-end nodes as end nodes
                loss_end_nodes = loss_termination(out.y_score, out.tgt_y, end_node_token_id)

                x_gt_node = out.attn_gt.argmax(dim=0)
                x_gt = out.tgt_y[x_gt_node]
                loss_enc_nodes = F.cross_entropy(out.x_score, x_gt)

                # calculate loss for attention to source graph - average
                gcn_alpha_avg = torch.cat((out.gcn1_alpha.unsqueeze(0), out.gcn2_alpha.unsqueeze(0), out.gcn3_alpha.unsqueeze(0)), dim=0)
                gcn_alpha_avg = torch.mean(gcn_alpha_avg, dim=0)
                gcn_alpha_avg = F.softmax(gcn_alpha_avg, dim=1)
                loss_gcn_alpha_avg = F.kl_div(
                    gcn_alpha_avg.type(torch.double),
                    out.attn_gt.type(torch.double),
                    reduction='batchmean', log_target=False).type(torch.float)

                if epoch % 50 == 49:
                    print(out.attn_gt.argmax(dim=1))
                    print(out.gcn1_alpha.argmax(dim=1))
                    print(out.gcn2_alpha.argmax(dim=1))
                    print(out.gcn3_alpha.argmax(dim=1))
                    print(gcn_alpha_avg.argmax(dim=1))
                    print("\n")

                # calculate loss as cross-entropy on output graph SRT edge type predictions
                tgt_edge_pc_indices = ((out.tgt_edge_type == SltEdgeTypes.PARENT_CHILD).nonzero(as_tuple=True)[0])
                tgt_pc_edge_relation = out.tgt_edge_relation[tgt_edge_pc_indices]
                out_pc_edge_relation = out.y_edge_rel_score[tgt_edge_pc_indices]
                loss_out_edge = loss_f(out_pc_edge_relation, tgt_pc_edge_relation)

                loss = \
                    loss_out_node + \
                    loss_out_edge + \
                    0.3 * loss_gcn_alpha_avg + \
                    0.5 * loss_end_nodes + \
                    0.5 * loss_enc_nodes

                loss.backward()

                epoch_loss += loss.item()
                running_loss += loss.item()

                # eval_res = eval_training_batch(out.detach())

                if save_run:
                    writer.add_scalar('Loss/train', loss.item(), epoch * len(trainloader) + i)
                if i % 100 == 0 and i != 0:
                    if print_train_info:
                        print("running_loss: " + str(running_loss / 100))
                    if save_run:
                        writer.add_scalar('RunningLoss/train', (running_loss / 100), epoch * len(trainloader) + i)
                    running_loss = 0

                optimizer.step()

            if print_train_info:
                print(epoch_loss / len(trainset))
            if save_run:
                writer.add_scalar('EpochLoss/train', epoch_loss / len(trainset), epoch)
                if epoch % 30 == 29:
                    pass
                    # torch.save(model.state_dict(), 'checkpoints/' + model_name + '_epoch' + str(epoch) + '.pth')

            if epoch % 20 == 19:
                evaluate_model(model, test_images_root, test_inkmls_root, tokenizer, components_shape)

        if save_run:
            torch.save(model.state_dict(), 'checkpoints/' + model_name + '_final' + '.pth')
            logging.info("Model final state saved")

    if evaluate:
        evaluate_model(model, test_images_root, test_inkmls_root, tokenizer, components_shape)
