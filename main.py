import logging
import os.path

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
from networkx.drawing.nx_pydot import graphviz_layout

from src.data.CrohmeDataset import CrohmeDataset
from src.data.LatexVocab import LatexVocab
from src.definitions.SltEdgeTypes import SltEdgeTypes
from src.model.Model import Model

import datetime

from test import test

if __name__ == '__main__':
    test()
    exit()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    logging.getLogger('matplotlib.font_manager').disabled = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 1
    batch_size = 1
    components_shape = (32, 32)
    input_edge_size = 19
    input_feature_size = 256
    hidden_size = 256
    embed_size = 256

    dist_inkmls_root = 'assets/crohme/train/inkml'

    images_root = 'assets/crohme/dev/img/'
    inkmls_root = 'assets/crohme/dev/inkml/'
    lgs_root = 'assets/crohme/dev/lg/'

    # LatexVocab.generate_formulas_file_from_inkmls(dist_inkmls_root, 'assets/vocab.txt')
    # tokenizer = LatexVocab.create_tokenizer('assets/vocab.txt', min_freq=1)
    # LatexVocab.save_tokenizer(tokenizer, 'assets/tokenizer.json')
    tokenizer = LatexVocab.load_tokenizer('assets/tokenizer.json')
    vocab = tokenizer.get_vocab()
    vocab_size = tokenizer.get_vocab_size()
    end_node_token_id = tokenizer.encode("[EOS]", add_special_tokens=False).ids[0]

    dataset = CrohmeDataset(images_root, inkmls_root, lgs_root, tokenizer, components_shape)
    trainloader = DataLoader(dataset, batch_size, False, follow_batch=['x', 'tgt_x'])

    train = True
    eval = False

    load_model = True
    load_model_path = "checkpoints/"
    load_model_name = "MER_19_256_256_train_22-03-17_16-53-52_epoch9.pth"

    model = Model(device, components_shape, input_edge_size, input_feature_size, hidden_size, embed_size, vocab_size, end_node_token_id)
    model.float()
    if load_model:
        model.load_state_dict(torch.load(os.path.join(load_model_path, load_model_name)))
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    now = datetime.datetime.now()
    model_name = 'MER_'+'19_256_256_train'+'_'+now.strftime("%y-%m-%d_%H-%M-%S")
    # writer = SummaryWriter('runs/' + model_name)

    if train:
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            running_loss = 0
            print("EPOCH: " + str(epoch))
            for i, data_batch in tqdm(enumerate(trainloader)):
                data_batch = data_batch.to(device)

                continue

                optimizer.zero_grad()
                out = model(data_batch)

                preds = torch.exp(out.out_x_pred)
                max, max_id = preds.max(dim=1)
                # print(max)
                # print(max_id)
                # print(tokenizer.decode(out.y.tolist()))
                # print(tokenizer.decode(max_id.tolist()))

                # calculate loss as cross-entropy on output graph node predictions
                loss_out_node = F.nll_loss(out.out_x_pred, out.tgt_x.squeeze(1))

                # calculate loss as cross-entropy on output graph SRT edge type predictions
                tgt_edge_pc_indices = ((out.tgt_edge_type == SltEdgeTypes.PARENT_CHILD).nonzero(as_tuple=True)[0])
                tgt_pc_edge_relation = out.tgt_edge_relation[tgt_edge_pc_indices]
                out_pc_edge_relation = out.out_edge_pred[tgt_edge_pc_indices]
                loss_out_edge = F.nll_loss(out_pc_edge_relation, tgt_pc_edge_relation)

                loss = loss_out_node + loss_out_edge
                loss.backward()

                epoch_loss += batch_size * loss.item()
                running_loss += loss.item()

                # writer.add_scalar('Loss/train', loss.item(), epoch*dataset.__len__()/len(dataset) + i)
                if i % 100 == 0:
                    print("running_loss: " + str(running_loss / 100))
                    # writer.add_scalar('RunningLoss/train', (running_loss / 100), epoch*dataset.__len__()/len(dataset) + i)
                    running_loss = 0

                if i % 1000 == 0 and i != 0:
                    pass
                    # torch.save(model.state_dict(), 'checkpoints/' + model_name + '_epoch' + str(epoch) + '.pth')

                optimizer.step()

            # writer.add_scalar('EpochLoss/train', epoch_loss / len(dataset), epoch + 1)
            print(epoch_loss)

        # torch.save(model.state_dict(), 'checkpoints/' + model_name + '_epoch' + str(epoch) + '.pth')

    if eval:
        model.eval()
        with torch.no_grad():
            for i, data_batch in enumerate(trainloader):
                data_batch = data_batch.to(device)
                out = model(data_batch)

                preds = torch.exp(out.out_x_pred)
                max, max_id = preds.max(dim=1)
                print(tokenizer.decode(out.y.tolist()))
                print(tokenizer.decode(max_id.tolist()))

                pc_edge_mask = (out.tgt_edge_type == SltEdgeTypes.PARENT_CHILD).to(torch.long).unsqueeze(1)
                pc_edge_index = out.tgt_edge_index.t() * pc_edge_mask
                pc_edge_index = pc_edge_index.t()

                data = Data(x=out.out_x, edge_index=pc_edge_index)
                G = to_networkx(data)
                pos = graphviz_layout(G, prog="dot")
                nx.draw(G, pos)
                plt.show()

