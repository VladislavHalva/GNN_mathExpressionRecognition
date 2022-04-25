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
from src.model.Model import Model
from src.utils.SltParser import SltParser
from src.utils.loss import loss_termination
from src.utils.utils import cpy_simple_train_gt, create_attn_gt


def evaluate_model(model, images_root, inkmls_root, tokenizer, components_shape):
    logging.info("Evaluation...")
    testset = CrohmeDataset(images_root, inkmls_root, tokenizer, components_shape)
    testloader = DataLoader(testset, 1, False, follow_batch=['x', 'tgt_y'])
    model.eval()
    with torch.no_grad():
        for i, data_batch in enumerate(testloader):
            data_batch = create_attn_gt(data_batch, end_node_token_id)
            data_batch = data_batch.to(device)

            out = model(data_batch)
            if device == torch.device('cuda'):
                out = out.cpu()

            # print(out.attn_gt.argmax(dim=-1))
            # print(out.gcn1_alpha.argmax(dim=-1))
            # print(out.gcn2_alpha.argmax(dim=-1))
            # print(out.gcn3_alpha.argmax(dim=-1))

            y_pred = F.softmax(out.y_score, dim=1)
            y_pred = torch.argmax(y_pred, dim=1)
            y_edge_rel_pred = F.softmax(out.y_edge_rel_score, dim=1)
            y_edge_rel_pred = torch.argmax(y_edge_rel_pred, dim=1)

            latex = SltParser.slt_to_latex_predictions(tokenizer, y_pred, y_edge_rel_pred, out.y_edge_index,
                                                       out.y_edge_type)
            # latex = SltParser.slt_to_latex_predictions(tokenizer, out.tgt_y.squeeze(1), out.tgt_edge_relation, out.tgt_edge_index, out.tgt_edge_type)

            # print(y_pred)
            print('GT: ' + tokenizer.decode(out.tgt_y.squeeze(1).tolist()))
            print('PR: ' + tokenizer.decode(y_pred.tolist()))
            gt_ml = tokenizer.decode(out.gt_ml.tolist())
            gt_ml = re.sub(' +', ' ', gt_ml)
            print('GT: ' + gt_ml)
            print('PR: ' + latex)
            # print('nodes count: ' + str(out.y_score.shape[0]))
            # print('edges count: ' + str(out.y_edge_rel_score.shape[0]))
            print("\n")
    model.train()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    logging.getLogger('matplotlib.font_manager').disabled = True

    load_vocab = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 400
    batch_size = 4
    components_shape = (32, 32)
    edge_features = 19
    enc_in_size = 400
    enc_h_size = 256
    enc_out_size = 256
    dec_h_size = 256
    emb_size = 256

    load_model = False
    load_model_path = "checkpoints/"
    load_model_name = "MER_enc_train_19_256_400_256_simple_22-04-25_01-05-49_final.pth"

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
        components_shape, edge_features,
        enc_in_size, enc_h_size, enc_out_size, dec_h_size, emb_size,
        vocab_size, end_node_token_id)
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
    model.float()
    if load_model:
        model.load_state_dict(torch.load(os.path.join(load_model_path, load_model_name), map_location=device))
        logging.info(f"Model loaded: {load_model_name}")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    loss_f = nn.CrossEntropyLoss()

    now = datetime.datetime.now()
    model_name = 'MER_enc_train_' + '19_400_256_simple' + '_' + now.strftime("%y-%m-%d_%H-%M-%S")

    if train:
        logging.info("Training...")
        trainset = CrohmeDataset(train_images_root, train_inkmls_root, tokenizer, components_shape)

        trainloader = DataLoader(trainset, batch_size, True, follow_batch=['x', 'tgt_y'])
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
                loss_out_node = loss_f(out.y_score, out.tgt_y.squeeze(1))

                # calculate loss as cross-entropy on decoder embeddings
                loss_embeds = loss_f(out.embeds, out.tgt_y.squeeze(1))

                # calculate additional loss penalizing classification non-end nodes as end nodes
                loss_end_nodes = loss_termination(out.y_score, out.tgt_y.squeeze(1), end_node_token_id)

                x_gt_node = out.attn_gt.argmax(dim=0)
                x_gt = out.tgt_y[x_gt_node].squeeze(1)
                loss_enc_nodes = F.cross_entropy(out.x_score, x_gt)

                # calculate loss for attention to source graph
                loss_gcn1_alpha = F.mse_loss(out.gcn1_alpha.type(torch.double), out.attn_gt.type(torch.double)).type(torch.float)
                loss_gcn2_alpha = F.mse_loss(out.gcn2_alpha.type(torch.double), out.attn_gt.type(torch.double)).type(torch.float)
                loss_gcn3_alpha = F.mse_loss(out.gcn3_alpha.type(torch.double), out.attn_gt.type(torch.double)).type(torch.float)

                # calculate loss as cross-entropy on output graph SRT edge type predictions
                tgt_edge_pc_indices = ((out.tgt_edge_type == SltEdgeTypes.PARENT_CHILD).nonzero(as_tuple=True)[0])
                tgt_pc_edge_relation = out.tgt_edge_relation[tgt_edge_pc_indices]
                out_pc_edge_relation = out.y_edge_rel_score[tgt_edge_pc_indices]
                loss_out_edge = loss_f(out_pc_edge_relation, tgt_pc_edge_relation)

                # print(f"node: {loss_out_node}, edge: {loss_out_edge}, gcn1: {loss_gcn1_alpha}, gcn2: {loss_gcn2_alpha}, gcn3: {loss_gcn3_alpha}, end: {loss_end_nodes}")

                loss = loss_enc_nodes + loss_out_node + loss_embeds + loss_out_edge + 0.99 * loss_gcn1_alpha + 0.99 * loss_gcn2_alpha + 0.99 * loss_gcn3_alpha + 0.5 * loss_end_nodes
                # loss = loss_enc_nodes

                # print(f"encoder: {loss_enc_nodes}, node: {loss_out_node}, gcn1: {loss_gcn1_alpha}, gcn2: {loss_gcn2_alpha}, gcn3: {loss_gcn3_alpha}, edge: {loss_out_edge}")

                loss.backward()

                epoch_loss += loss.item()
                running_loss += loss.item()

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
                if epoch % 50 == 49:
                    torch.save(model.state_dict(), 'checkpoints/' + model_name + '_epoch' + str(epoch) + '.pth')

            if epoch % 20 == 19:
                evaluate_model(model, test_images_root, test_inkmls_root, tokenizer, components_shape)

        if save_run:
            torch.save(model.state_dict(), 'checkpoints/' + model_name + '_final' + '.pth')
            logging.info("Model final state saved")

    if evaluate:
        evaluate_model(model, test_images_root, test_inkmls_root, tokenizer, components_shape)
