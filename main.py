import logging
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

from src.data.CrohmeDataset import CrohmeDataset
from src.data.LatexVocab import LatexVocab
from src.definitions.SltEdgeTypes import SltEdgeTypes
from src.model.Model import Model

import datetime

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
    logging.getLogger('matplotlib.font_manager').disabled = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    components_shape = (32, 32)
    input_edge_size = 19
    input_feature_size = 256
    hidden_size = 128
    embed_size = 256

    img_path = 'assets/crohme/train/img/train_2014/7_em_59.png'
    inkml_path = 'assets/crohme/train/inkml/train_2014/7_em_59.inkml'
    lg_path = 'assets/crohme/train/lg/train_2014/7_em_59.lg'

    dist_inkmls_root = 'assets/crohme/train/inkml'

    images_root = 'assets/crohme/train/img/'
    inkmls_root = 'assets/crohme/train/inkml/'
    lgs_root = 'assets/crohme/train/lg/'

    LatexVocab.generate_formulas_file_from_inkmls(dist_inkmls_root, 'assets/vocab.txt')
    # tokenizer = LatexVocab.create_tokenizer('assets/vocab.txt', min_freq=1)
    # LatexVocab.save_tokenizer(tokenizer, 'assets/tokenizer.json')
    tokenizer = LatexVocab.load_tokenizer('assets/tokenizer.json')
    vocab = tokenizer.get_vocab()
    vocab_size = tokenizer.get_vocab_size()
    end_node_token_id = tokenizer.encode("[EOS]", add_special_tokens=False).ids[0]

    dataset = CrohmeDataset(images_root, inkmls_root, lgs_root, tokenizer, components_shape)
    trainloader = DataLoader(dataset, 2, True, follow_batch=['x', 'tgt_x'])

    model = Model(components_shape, input_edge_size, input_feature_size, hidden_size, embed_size, vocab_size, end_node_token_id)
    model.float()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    now = datetime.datetime.now()
    model_name = 'MER_'+'train'+'_'+now.strftime("%y-%m-%d_%H-%M-%S")
    writer = SummaryWriter('runs/' + model_name)

    model.train()
    for epoch in range(1):
        print("EPOCH: " + str(epoch))
        for i, data_batch in tqdm(enumerate(trainloader)):
            data_batch = data_batch.to(device)

            optimizer.zero_grad()
            out = model(data_batch)

            # calculate loss as cross-entropy on output graph node predictions
            loss_out_node = F.nll_loss(out.out_x_pred, out.tgt_x.squeeze(1))

            # calculate loss as cross-entropy on output graph SRT edge type predictions
            tgt_edge_pc_indices = ((out.tgt_edge_type == SltEdgeTypes.PARENT_CHILD).nonzero(as_tuple=True)[0])
            tgt_pc_edge_relation = out.tgt_edge_relation[tgt_edge_pc_indices]
            out_pc_edge_relation = out.out_edge_pred[tgt_edge_pc_indices]
            loss_out_edge = F.nll_loss(out_pc_edge_relation, tgt_pc_edge_relation)

            loss = loss_out_node + loss_out_edge
            loss.backward()

            writer.add_scalar('Loss/train', loss.item(), epoch*dataset.__len__() + i)
            if i % 100 == 0:
                print(loss.item())

            optimizer.step()

    torch.save(model.state_dict(), 'checkpoints/' + model_name + '.pth')

    model.eval()
    with torch.no_grad():
        for i, data_batch in enumerate(trainloader):
            data_batch = data_batch.to(device)
            out = model(data_batch)
            data = Data(x=out.out_x, edge_index=out.tgt_edge_index)
            G = to_networkx(data)
            nx.draw(G)
            plt.show()

