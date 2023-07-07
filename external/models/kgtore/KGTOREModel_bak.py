from abc import ABC
from .EdgeLayer import LGConv
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
                 num_interactions,
                 learning_rate,
                 edges_lr,
                 embedding_size,
                 l_w,
                 l_ind,
                 ind_edges,
                 n_layers,
                 edge_index,
                 edge_features,
                 item_features,
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

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_users = num_users
        self.num_items = num_items
        self.num_interactions = num_interactions
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.edges_lr = edges_lr
        self.l_w = l_w
        self.l_ind = l_ind
        self.n_layers = n_layers
        self.weight_size_list = [self.embedding_size] * (self.n_layers + 1)
        self.alpha = torch.tensor([1 / (k + 1) for k in range(len(self.weight_size_list))])
        self.edge_index = torch.tensor(edge_index, dtype=torch.int64, device=self.device)
        self.edge_features = edge_features.to(self.device)
        self.item_features = item_features.to(self.device)

        _, self.cols = self.edge_index.clone()
        self.items = self.cols[:self.num_interactions]
        self.items -= self.num_users

        row = self.edge_index[0]
        deg = scatter_add(torch.ones((self.edge_index.size(1),), device=self.device), row, dim=0,
                          dim_size=(self.edge_index.max() + 1))
        deg_inv = deg.pow_(-1)
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        self.edge_attr_weight = deg_inv[row]
        self.edge_attr_weight[num_interactions:] = 1.

        self.user_inter = self.edge_index[0, 0:int(self.edge_index.size(1) / 2)]
        self.item_inter = self.edge_index[0, int(self.edge_index.size(1) / 2):] - self.num_users

        # ADDITIVE OPTIONS
        # self.a = torch.nn.Parameter(
        #     torch.abs(torch.nn.init.xavier_normal_(torch.empty((int(self.edge_index.size(1)/2), 1)))).to(self.device),
        #     requires_grad=True)
        # self.b = torch.nn.Parameter(
        #     torch.abs(torch.nn.init.xavier_normal_(torch.empty((int(self.edge_index.size(1)/2), 1)))).to(self.device),
        #     requires_grad=True)
        self.a = torch.nn.Parameter(
            torch.abs(torch.nn.init.xavier_normal_(torch.empty(self.num_items, 1))).to(self.device), requires_grad=True)
        self.b = torch.nn.Parameter(
            torch.abs(torch.nn.init.xavier_normal_(torch.empty(self.num_users, 1))).to(self.device), requires_grad=True)

        self.Gu = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_users, self.embedding_size))).to(self.device),
            requires_grad=True)

        self.Gi = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_items, self.embedding_size))).to(self.device),
            requires_grad=True)

        # features matrix (for edges)
        self.feature_dim = edge_features.size(1)
        self.F = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.feature_dim, self.embedding_size))).to(self.device))

        propagation_network_list = []
        for layer in range(self.n_layers):
            propagation_network_list.append((LGConv(), 'x, edge_index -> x'))
        self.propagation_network = torch_geometric.nn.Sequential('x, edge_index', propagation_network_list).to(
            self.device)

        self.softplus = torch.nn.Softplus()

        self.optimizer = torch.optim.Adam([self.Gu, self.Gi, self.a, self.b], lr=self.learning_rate)
        self.edges_optimizer = torch.optim.Adam([self.F], lr=self.edges_lr)

        self.ind_edges = ind_edges
        self.n_selected_edges = int(self.num_interactions * self.ind_edges)
        self.l_ind = self.l_ind / self.ind_edges

        self.edge_path = dict()
        self.edge_len = dict()
        for e in range(self.edge_features.size(0)):
            self.edge_path[e] = self.edge_features[e].storage._col
            self.edge_len[e] = len(self.edge_path[e])

    def propagate_embeddings(self, evaluate=False):

        # a = torch.sigmoid(self.a[self.item_inter])  # force values in [0 - 1]
        # b = torch.sigmoid(self.b[self.user_inter])
        a = torch.clamp(self.a[self.item_inter], min=0, max=1)
        b = torch.clamp(self.b[self.user_inter], min=0, max=1)
        b_minus = 1 - a
        a_minus = 1 - b

        edge_embeddings_u_i_0 = matmul(self.edge_features, self.F)
        edge_embeddings_u_i = edge_embeddings_u_i_0 * b

        edge_embeddings_i_u_0 = matmul(self.item_features, self.F)[self.items]
        edge_embeddings_i_u = edge_embeddings_i_u_0 * a

        ego_embeddings = torch.cat((self.Gu, self.Gi), 0).to(self.device)
        all_embeddings = [ego_embeddings]
        edge_embeddings = torch.cat([edge_embeddings_u_i, edge_embeddings_i_u], dim=0).to(self.device)

        for layer in range(0, self.n_layers):
            if evaluate:
                self.propagation_network.eval()
                with torch.no_grad():
                    all_embeddings += [list(
                        self.propagation_network.children())[layer](
                        all_embeddings[layer], self.edge_index, edge_embeddings, self.edge_attr_weight, a_minus,
                        b_minus)
                    ]
            else:
                all_embeddings += [list(
                    self.propagation_network.children())[layer](
                    all_embeddings[layer], self.edge_index, edge_embeddings, self.edge_attr_weight, a_minus, b_minus)
                ]

        if evaluate:
            self.propagation_network.train()

        all_embeddings = sum([all_embeddings[k] * self.alpha[k] for k in range(len(all_embeddings))])
        gu, gi = torch.split(all_embeddings, [self.num_users, self.num_items], 0)

        return gu.to(self.device), gi.to(self.device)

    def forward(self, inputs, **kwargs):
        gu, gi = inputs
        gamma_u = torch.squeeze(gu)
        gamma_i = torch.squeeze(gi)
        xui = torch.sum(gamma_u * gamma_i, 1)
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
        # reg_loss = self.l_w * (1 / 2) * (self.Gu[user[:, 0]].norm(2).pow(2) +
        #                                  self.Gi[pos[:, 0]].norm(2).pow(2) +
        #                                  self.Gi[neg[:, 0]].norm(2).pow(2) +
        #                                  self.a.norm(2).pow(2) +
        #                                  self.b.norm(2).pow(2)) / float(batch[0].shape[0])
        reg_loss = self.l_w * (self.Gu[user[:, 0]].norm(2) +
                               self.Gi[pos[:, 0]].norm(2) +
                               self.Gi[neg[:, 0]].norm(2))

        features_reg_loss = self.l_w * torch.norm(self.F, 2)

        # independence loss over the features within the same path
        if self.l_ind > 0:
            selected_edges = random.sample(list(range(self.num_interactions)), self.n_selected_edges)
            ind_loss = [torch.abs(torch.corrcoef(self.F[self.edge_path[e]])).sum() - self.edge_len[e] for e in
                        selected_edges]
            ind_loss = sum(ind_loss) / self.n_selected_edges * self.l_ind

        loss = bpr_loss + reg_loss
        self.optimizer.zero_grad()
        self.edges_optimizer.zero_grad()

        loss.backward()
        loss.backward(retain_graph=True)
        if self.l_ind > 0:
            ind_loss.backward(retain_graph=True)
        features_reg_loss.backward()

        self.optimizer.step()
        self.edges_optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
