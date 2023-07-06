import torch
from torch import Tensor


class LogicNot(torch.nn.Module):
    def __init__(self, input_dim: int = 64, output_dim: int = 64, n_layers: int = 2, bias=False):
        super().__init__()
        self.logic_not_modules = [j for i in range(n_layers - 1) for j
                                  in [torch.nn.Linear(input_dim, output_dim, bias=bias), torch.nn.ReLU()]]
        self.logic_not_modules.append(torch.nn.Linear(input_dim, output_dim))
        self.logic_not = torch.nn.Sequential(*self.logic_not_modules)

    def forward(self, x: Tensor):
        return self.logic_not(x)


class LogicOr(torch.nn.Module):
    def __init__(self, input_dim: int = 128, output_dim: int = 64, n_layers: int = 2, bias=False):
        super().__init__()
        self.logic_or_modules = []
        self.logic_or_modules.append(torch.nn.Linear(input_dim, output_dim, bias=bias))
        self.logic_or_modules.append(torch.nn.ReLU)
        for i in range(n_layers - 2):
            self.logic_or_modules.append(torch.nn.Linear(output_dim, output_dim, bias=bias))
            self.logic_or_modules.append(torch.nn.ReLU)
        self.logic_or_modules.append(torch.nn.Linear(output_dim, output_dim, bias=bias))
        self.logic_or = torch.nn.Sequential(*self.logic_or_modules)

    def forward(self, x1: Tensor, x2: Tensor):
        assert (len(x1) == len(x2))
        vector = torch.cat((x1, x2), dim=1)  # affianca colonne
        return self.logic_or(vector)


class LogicInteraction(torch.nn.Module):
    def __init__(self, input_dim: int = 128, output_dim: int = 64, hidden_dim:int =64, n_layers: int = 2, bias=True):
        super().__init__()
        self.logic_int_modules = []
        self.logic_int_modules.append(torch.nn.Linear(input_dim, hidden_dim, bias=bias))
        self.logic_int_modules.append(torch.nn.ReLU)
        for i in range(n_layers - 2):
            self.logic_int_modules.append(torch.nn.Linear(hidden_dim, hidden_dim, bias=bias))
            self.logic_int_modules.append(torch.nn.ReLU)
        self.logic_int_modules.append(torch.nn.Linear(hidden_dim, output_dim, bias=bias))
        self.logic_int = torch.nn.Sequential(*self.logic_int_modules)

    def forward(self, ui: Tensor):
        return self.logic_int(ui)
