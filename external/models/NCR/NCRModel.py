from abc import ABC
import torch
from .LogicModules import LogicNot, LogicOr
import numpy as np
import random


class NCRModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users: int,
                 num_items: int,
                 learning_rate: float,
                 embedding_size: int,
                 l_w: float,
                 random_seed: int,
                 history: dict,
                 logical_layers: int = 2,
                 name="NCR",
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
        self.embed_k = embedding_size
        self.learning_rate = learning_rate
        self.l_w = l_w
        self.logical_layers = logical_layers
        self.history = history

        # user-item embedding
        self.Gu = torch.nn.Embedding(self.num_users, self.embed_k)
        torch.nn.init.xavier_uniform_(self.Gu.weight)
        self.Gu.to(self.device)
        self.Gi = torch.nn.Embedding(self.num_items, self.embed_k)
        torch.nn.init.xavier_uniform_(self.Gi.weight)
        self.Gi.to(self.device)

        # ------------- Logic
        # anchor true vector - randomly initialized (does not change during training)
        self.true = torch.nn.Parameter(
            torch.from_numpy(np.random.uniform(0, 0.1, size=self.embed_k).astype(np.float32)), requires_grad=False).to(
            self.device)
        self.logic_not = LogicNot(self.embed_k, self.embed_k, self.logical_layers)
        self.logic_or = LogicOr(self.embed_k * 2, self.embed_k, self.logical_layers)

        # Logic not
        # self.logic_not_modules = [j for i in range(self.logical_layers - 1) for j
        #                           in [torch.nn.Linear(self.embed_k, self.embed_k), torch.nn.ReLU()]]
        # self.logic_not_modules.append(torch.nn.Linear(self.embed_k, self.embed_k))
        # self.logic_not = torch.nn.Sequential(*self.logic_not_modules)

        # Logic and
        # self.logic_and_modules = [j for i in range(self.logical_layers - 1) for j
        #                           in [torch.nn.Linear(self.embed_k, self.embed_k), torch.nn.ReLU()]]
        # self.logic_and_modules.append(torch.nn.Linear(self.embed_k, self.embed_k))
        # self.logic_and = torch.nn.Sequential(*self.logic_and_modules)

        # Logic or

        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, inputs, **kwargs):
        users, items = inputs
        gamma_u = torch.squeeze(self.Gu.weight[users]).to(self.device)
        gamma_i = torch.squeeze(self.Gi.weight[items]).to(self.device)

        # expand user vector to prepare for concatenating with history item vectors
        gamma_u = gamma_u.view(gamma_u.size(0), 1, gamma_u.size(1))
        # ci serve un [u, [items_id vector] ]
        # [torch.tensor(list(t[i].keys())) for i in t.keys()]   --t = i_train_dict --per ogni utente tensore lista item
        # [(len(torch.tensor(list(t[i].keys()))) ,torch.tensor(list(t[i].keys()))) for i in t.keys()]




        xui = torch.sum(gamma_u * gamma_i, 1)

        return xui, gamma_u, gamma_i

    def predict(self, start, stop, **kwargs):
        return torch.matmul(self.Gu.weight[start:stop].to(self.device),
                            torch.transpose(self.Gi.weight.to(self.device), 0, 1))

    def train_step(self, batch):
        user, pos, neg = batch
        xu_pos, gamma_u, gamma_i_pos = self.forward(inputs=(user[:, 0], pos[:, 0]))
        xu_neg, _, gamma_i_neg = self.forward(inputs=(user[:, 0], neg[:, 0]))

        loss = -torch.mean(torch.nn.functional.logsigmoid(xu_pos - xu_neg))
        reg_loss = self.l_w * (1 / 2) * (gamma_u.norm(2).pow(2) +
                                         gamma_i_pos.norm(2).pow(2) +
                                         gamma_i_neg.norm(2).pow(2)) / user.shape[0]
        loss += reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
