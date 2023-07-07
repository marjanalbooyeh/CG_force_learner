import torch
import torch.nn as nn
import rowan


class BaseNN(nn.Module):
    def __init__(self, in_dim,  hidden_dim, out_dim, n_layers, act_fn="ReLU", dropout=0.3, inp_mode="append", augment_pos="r", augment_orient="a", pool="mean"):
        super(BaseNN, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.act_fn = act_fn
        self.dropout = dropout
        self.inp_mode = inp_mode
        self.augment_pos = augment_pos
        self.augment_orient = augment_orient
        self.pool = pool
        
        if self.augment_pos == "r" and self.inp_mode =="append":
            self.in_dim += 1
        if self.augment_orient == "a" and self.inp_mode =="append":
            self.in_dim += 1

        self.net = self._get_net()

    def _get_act_fn(self):
        act = getattr(nn, self.act_fn)
        return act()
    
    def _augment_input(self, x):
        if self.augment_pos == "r" and self.inp_mode =="append":
            # include center-to-center distance as a feature 
            x = torch.cat((x, torch.norm(x[:, :3], dim=1, keepdim=True).to(x.device)), dim=1).to(x.device)

        if self.augment_orient == "a" and self.inp_mode == "append":
            angles = torch.tensor(rowan.to_axis_angle(x[:, 3:7].detach().cpu().numpy())[1]).unsqueeze(1).to(x.device)
            x = torch.cat((x, angles), dim=1).to(x.device)
        return x
        


class NN(BaseNN):
    def __init__(self, in_dim,  hidden_dim, out_dim, n_layers, **kwargs):
        super(NN, self).__init__(in_dim, hidden_dim, out_dim, n_layers, **kwargs)


    def _get_net(self):
        layers = [nn.Linear(self.in_dim, self.hidden_dim), self._get_act_fn()]
        for i in range(self.n_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            # if self.batch_dim:
            #     layers.append(nn.BatchNorm1d(self.batch_dim))
            layers.append(self._get_act_fn())
            layers.append(nn.Dropout(p=self.dropout))
        layers.append(nn.Linear(self.hidden_dim, self.out_dim))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self._augment_input(x)
        out = self.net(x)

        if self.inp_mode == "stack":
            if self.pool == "mean":
                out = torch.mean(out, dim=1)
            elif self.pool == "sum":
                out = torch.sum(out, dim=1)
            elif self.pool == "max":
                out = torch.max(out, dim=1)[0]
        return out
    
    
class NNSkipShared(BaseNN):
    def __init__(self, in_dim,  hidden_dim, out_dim, n_layers, **kwargs):
        super(NNSkipShared, self).__init__(in_dim,  hidden_dim, out_dim, n_layers, **kwargs)
        self.activations = self._get_activations()
        self.dropouts = self._get_dropouts()
        self.input_connection = nn.Linear(self.in_dim, self.hidden_dim)


    def _get_net(self):
        layers = [nn.Linear(self.in_dim, self.hidden_dim)]
        for i in range(self.n_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            # if self.batch_dim:
            #     layers.append(nn.BatchNorm1d(self.batch_dim))
            
        layers.append(nn.Linear(self.hidden_dim, self.out_dim))
        return nn.ModuleList(layers)
    
    
    def _get_activations(self):
        activations = []
        for i in range(self.n_layers):
            activations.append(self._get_act_fn())
        return nn.ModuleList(activations)
    
    def _get_dropouts(self):
        dropouts = []
        for i in range(self.n_layers):
            dropouts.append(nn.Dropout(p=self.dropout))
        return nn.ModuleList(dropouts)

    def forward(self, x):
        x = self._augment_input(x)
        # transform input to hidden dim size
        x_transform = self.input_connection(x)
        for i in range(self.n_layers): 
            # add original transformed input to each layer before activation
            x = self.activations[i](self.net[i](x) + x_transform)
            x = self.dropouts[i](x)

            
        out = self.net[-1](x)

        if self.inp_mode == "stack":
            if self.pool == "mean":
                out = torch.mean(out, dim=1)
            elif self.pool == "sum":
                out = torch.sum(out, dim=1)
            elif self.pool == "max":
                out = torch.max(out, dim=1)[0]
        return out
    
    
class NNGrow(BaseNN):
    def __init__(self, in_dim,  hidden_dim, out_dim, n_layers, **kwargs):
        super(NNGrow, self).__init__(in_dim,  hidden_dim, out_dim, n_layers, **kwargs)



    def _get_net(self):
        layers = [nn.Linear(self.in_dim, self.hidden_dim[0]), self._get_act_fn()]
        for i in range(1, len(self.hidden_dim)):
            layers.append(nn.Linear(self.hidden_dim[i-1], self.hidden_dim[i]))
            # if self.batch_dim:
            #     layers.append(nn.BatchNorm1d(self.batch_dim))
            layers.append(self._get_act_fn())
            layers.append(nn.Dropout(p=self.dropout))
        layers.append(nn.Linear(self.hidden_dim[-1], self.out_dim))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self._augment_input(x)
        out = self.net(x)

        if self.inp_mode == "stack":
            if self.pool == "mean":
                out = torch.mean(out, dim=1)
            elif self.pool == "sum":
                out = torch.sum(out, dim=1)
            elif self.pool == "max":
                out = torch.max(out, dim=1)[0]
        return out
        return out