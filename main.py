import logging
import os.path
from itertools import chain

import networkx as nx
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


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    logging.getLogger('matplotlib.font_manager').disabled = True

    load_vocab = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 1
    batch_size = 2
    components_shape = (32, 32)
    edge_features = 19
    enc_in_size = 256
    enc_h_size = 400
    enc_out_size = 400
    dec_h_size = 256
    emb_size = 256

    load_model = False
    load_model_path = "checkpoints/"
    load_model_name = "MER_3L_19_256_400_256_simple_22-03-24_20-57-21_final.pth"

    train = True
    train_sufficient_loss = 0.05
    eval = False
    save_run = False
    print_train_info = False

    # to build vocabulary
    dist_inkmls_root = 'assets/crohme/train/inkml'
    # for training
    train_images_root = 'assets/crohme/simple/img/'
    train_inkmls_root = 'assets/crohme/simple/inkml/'
    # for test
    test_images_root = 'assets/crohme/simple/img/'
    test_inkmls_root = 'assets/crohme/simple/inkml/'

    if load_vocab:
        tokenizer = LatexVocab.load_tokenizer('assets/tokenizer.json')
    else:
        LatexVocab.generate_formulas_file_from_inkmls(dist_inkmls_root, 'assets/vocab.txt')
        tokenizer = LatexVocab.create_tokenizer('assets/vocab.txt', min_freq=1)
        LatexVocab.save_tokenizer(tokenizer, 'assets/tokenizer.json')

    vocab = tokenizer.get_vocab()
    vocab_size = tokenizer.get_vocab_size()
    end_node_token_id = tokenizer.encode("[EOS]", add_special_tokens=False).ids[0]

    model = Model(
        device,
        components_shape, edge_features,
        enc_in_size, enc_h_size, enc_out_size, dec_h_size, emb_size,
        vocab_size, end_node_token_id)
    model.float()
    if load_model:
        model.load_state_dict(torch.load(os.path.join(load_model_path, load_model_name)))
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    now = datetime.datetime.now()
    model_name = 'MER_3L_'+'19_256_256_simple'+'_'+now.strftime("%y-%m-%d_%H-%M-%S")

    if train:
        trainset = CrohmeDataset(train_images_root, train_inkmls_root, tokenizer, components_shape)
        trainloader = DataLoader(trainset, batch_size, False, follow_batch=['x', 'tgt_x'])
        if save_run:
            writer = SummaryWriter('runs/' + model_name)

        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            running_loss = 0
            if print_train_info:
                print("EPOCH: " + str(epoch))

            for i, data_batch in tqdm(enumerate(trainloader)):
                data_batch = data_batch.to(device)

                optimizer.zero_grad()
                out = model(data_batch)

                exit()

                # calculate loss as cross-entropy on output graph node predictions
                loss_out_node = F.nll_loss(out.out_x_pred, out.tgt_x.squeeze(1))

                # calculate loss as cross-entropy on output graph SRT edge type predictions
                tgt_edge_pc_indices = ((out.tgt_edge_type == SltEdgeTypes.PARENT_CHILD).nonzero(as_tuple=True)[0])
                tgt_pc_edge_relation = out.tgt_edge_relation[tgt_edge_pc_indices]
                out_pc_edge_relation = out.out_edge_pred[tgt_edge_pc_indices]
                loss_out_edge = F.nll_loss(out_pc_edge_relation, tgt_pc_edge_relation)

                loss = loss_out_node + loss_out_edge
                loss.backward()

                epoch_loss += loss.item()
                running_loss += loss.item()

                if save_run:
                    writer.add_scalar('Loss/train', loss.item(), epoch*len(trainloader) + i)
                if i % 100 == 0 and i != 0:
                    if print_train_info:
                        print("running_loss: " + str(running_loss / 100))
                    if save_run:
                        writer.add_scalar('RunningLoss/train', (running_loss / 100), epoch*len(trainloader) + i)
                    running_loss = 0

                optimizer.step()

            if save_run:
                writer.add_scalar('EpochLoss/train', epoch_loss / len(trainset), epoch)
            if print_train_info:
                print(epoch_loss / len(trainset))
            if save_run and False:
                torch.save(model.state_dict(), 'checkpoints/' + model_name + '_epoch' + str(epoch) + '.pth')

            if epoch_loss / len(trainset) < train_sufficient_loss:
                print("LOSS LOW ENOUGH")
                break

        torch.save(model.state_dict(), 'checkpoints/' + model_name + '_final' + '.pth')

    if eval:
        testset = CrohmeDataset(test_images_root, test_inkmls_root, tokenizer, components_shape)
        testloader = DataLoader(testset, 1, False, follow_batch=['x', 'tgt_x'])

        model.train()
        with torch.no_grad():
            for i, data_batch in enumerate(testloader):
                data_batch = data_batch.to(device)
                out = model(data_batch)
                break

                latex = SltParser.slt_to_latex_predictions(tokenizer, out.out_x_pred, out.out_edge_pred, out.tgt_edge_index, out.tgt_edge_type)
                print('GT: ' + tokenizer.decode(out.gt.tolist()))
                print('PR: ' + latex)
                print('nodes count: ' + str(out.out_x_pred.shape[0]))
                print('edges count: ' + str(out.out_edge_pred.shape[0]))
                print("\n")

                # pc_edge_mask = (out.tgt_edge_type == SltEdgeTypes.PARENT_CHILD).to(torch.long).unsqueeze(1)
                # pc_edge_index = out.tgt_edge_index.t() * pc_edge_mask
                # pc_edge_index = pc_edge_index.t()
                #
                # data = Data(x=out.out_x, edge_index=pc_edge_index)
                # G = to_networkx(data)
                # pos = graphviz_layout(G, prog="dot")
                # nx.draw(G, pos)
                # plt.show()

