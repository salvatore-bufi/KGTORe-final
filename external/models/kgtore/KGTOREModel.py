from abc import ABC
from torch_geometric.nn import LGConv
import torch
import torch_geometric
import numpy as np
import random
from torch_sparse import matmul
from torch_scatter import scatter_add


class KGTOREModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 edges_lr,
                 embedding_size,
                 kg_size,
                 l_w,
                 l_ind,
                 ind_edges,
                 n_layers,
                 edge_index,
                 random_seed,
                 name="KGTORE",
                 **kwargs
                 ):
        super().__init__()

        # set seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.edges_lr = edges_lr
        self.l_w = l_w
        self.l_ind = l_ind
        self.n_layers = n_layers
        self.weight_size_list = [self.embedding_size] * (self.n_layers + 1)
        self.alpha = torch.tensor([1 / (k + 1) for k in range(len(self.weight_size_list))])
        self.adj = edge_index
        self.kg_size = kg_size

        self.Gu = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_users, self.embedding_size))).to(self.device),
            requires_grad=True)
        # Gi == pesi delle features fake cndivise che costruiscono item
        self.Gi = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_items, self.kg_size))).to(self.device),
            requires_grad=True)
        # features matrix (for edges)
        self.F = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.kg_size, self.embedding_size))).to(self.device))

        propagation_network_list = []

        for layer in range(self.n_layers):
            propagation_network_list.append((LGConv(), 'x, edge_index -> x'))

        self.propagation_network = torch_geometric.nn.Sequential('x, edge_index', propagation_network_list)
        self.propagation_network.to(self.device)
        self.softplus = torch.nn.Softplus()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def propagate_embeddings(self, evaluate=False):
        weight = torch.sigmoid(self.Gi)
        Gi = torch.matmul(weight, self.F)
        ego_embeddings = torch.cat((self.Gu.to(self.device), Gi.to(self.device)), 0)
        all_embeddings = [ego_embeddings]

        for layer in range(0, self.n_layers):
            if evaluate:
                self.propagation_network.eval()
                with torch.no_grad():
                    all_embeddings += [list(
                        self.propagation_network.children()
                    )[layer](all_embeddings[layer].to(self.device), self.adj.to(self.device))]
            else:
                all_embeddings += [list(
                    self.propagation_network.children()
                )[layer](all_embeddings[layer].to(self.device), self.adj.to(self.device))]

        if evaluate:
            self.propagation_network.train()

        all_embeddings = sum([all_embeddings[k] * self.alpha[k] for k in range(len(all_embeddings))])
        gu, _ = torch.split(all_embeddings, [self.num_users, self.num_items], 0)

        return gu.to(self.device), Gi.to(self.device)

    def forward(self, inputs, **kwargs):
        gu, gi = inputs
        gamma_u = torch.squeeze(gu)
        gamma_i = torch.squeeze(gi)
        xui = torch.sum(gamma_u * gamma_i, -1)
        return xui

    def predict(self, gu, gi, **kwargs):
        return torch.matmul(gu, torch.transpose(gi, 0, 1))

    def train_step(self, batch):

        gu, gi = self.propagate_embeddings()
        user, pos, neg = batch
        xu_pos = self.forward(inputs=(gu[user[:, 0]], gi[pos[:, 0]]))
        xu_neg = self.forward(inputs=(gu[user[:, 0]], gi[neg[:, 0]]))
        difference = torch.clamp(xu_pos - xu_neg, -80.0, 1e8)
        bpr_loss = torch.sum(self.softplus(-difference))
        reg_loss = self.l_w * (torch.norm(self.Gu, 2) +
                               torch.norm(self.F, 2))
        loss = bpr_loss + reg_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
