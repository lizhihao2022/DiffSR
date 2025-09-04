import torch
import torch.nn.functional as F
from time import time


class LpLoss(object):
    '''
    loss function with rel/abs Lp loss
    '''
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y, **kwargs):
        return self.rel(x, y)


class CarCFDLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(CarCFDLoss, self).__init__()

        self.lp_loss = LpLoss(d=d, p=p, size_average=size_average, reduction=reduction)

    def compute_loss(self, x, y, batch, graph, sep=True, **kwargs):
        mask = (graph.batch == batch)
        surf = graph.surf[mask]
        press_loss = self.lp_loss(x[:, surf, -1], y[:, surf, -1])
        vol_loss = self.lp_loss(x[:, :, :-1], y[:, :, :-1])
        
        if sep:
            return [press_loss + vol_loss, press_loss, vol_loss]
        else:
            return press_loss + vol_loss
    
    def __call__(self, x, y, **kwargs):
        return self.compute_loss(x, y, **kwargs)


class MultipleLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        
        self.lp_loss = LpLoss(d=d, p=p, size_average=size_average, reduction=reduction)
    
    def compute_loss(self, x, y, sep=True, **kwargs):
        num_feature = x.size(2)
        loss_list = []
        for i in range(num_feature):
            loss_list.append(self.lp_loss(x[:, :, i], y[:, :, i]))
        
        all_loss = sum(loss_list)
        
        if sep:
            return [all_loss] + loss_list
        else:
            return all_loss
    
    def __call__(self, x, y, **kwargs):
        return self.compute_loss(x, y, **kwargs)


class AverageRecord(object):
    """Computes and stores the average and current values for multidimensional data"""

    def __init__(self):
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


class LossRecord:
    """
    A class for keeping track of loss values during training.

    Attributes:
        start_time (float): The time when the LossRecord was created.
        loss_list (list): A list of loss names to track.
        loss_dict (dict): A dictionary mapping each loss name to an AverageRecord object.
    """

    def __init__(self, loss_list):
        self.start_time = time()
        self.loss_list = loss_list
        self.loss_dict = {loss: AverageRecord() for loss in self.loss_list}
    
    def update(self, update_dict, n):
        for key, value in update_dict.items():
            self.loss_dict[key].update(value, n)
    
    def format_metrics(self):
        result = ""
        for loss in self.loss_list:
            result += "{}: {:.8f} | ".format(loss, self.loss_dict[loss].avg)
        result += "Time: {:.2f}s".format(time() - self.start_time)

        return result
    
    def to_dict(self):
        return {
            loss: self.loss_dict[loss].avg for loss in self.loss_list
        }
    
    def __str__(self):
        return self.format_metrics()
    
    def __repr__(self):
        return self.loss_dict[self.loss_list[0]].avg


def weight_sparsity_loss(weight):
    """
    weight: [B, N, 1] (经过sigmoid激活)
    """
    # L1正则化
    l1_loss = torch.abs(weight).mean()
    
    # 熵正则化 (避免单个粒子权重过高)
    entropy = - (weight * torch.log(weight + 1e-8)).sum(dim=1).mean()
    
    return 0.1 * l1_loss + 0.01 * entropy


def mu_boundary_loss(mu, domain_min, domain_max):
    """
    mu: [B, N, pos_dim]
    domain_min/max: [pos_dim]
    """
    lower_violation = torch.relu(domain_min - mu)  # mu < domain_min 的正值
    upper_violation = torch.relu(mu - domain_max)  # mu > domain_max 的正值
    loss = (lower_violation**2 + upper_violation**2).mean()
    return loss


def chamfer_loss(x, y):
    dist_sq = ((x.unsqueeze(2) - y.unsqueeze(1)) ** 2).sum(dim=-1)
    forward = dist_sq.min(dim=2).values.mean()
    backward = dist_sq.min(dim=1).values.mean()

    return forward + backward


def gaussian_params_loss(pred, target):
    crit = LpLoss(p=2, d=2)
    
    mu_pred = pred['mu']
    log_scale_pred = pred['scale']
    rot_pred = pred['rotation']
    weights_pred = pred['weights']
    
    mu_true = target['mu']
    scale_true = target['scale']
    rot_true = target['rotation']
    weights_true = target['weights']

    loss = crit(mu_pred, mu_true)
    loss += crit(log_scale_pred, scale_true)
    loss += crit(rot_pred, rot_true)
    loss += crit(weights_pred, weights_true)
    
    return loss


def kl_diag_gaussian(mu_p, sigma_p, mu_q, sigma_q, eps=1e-6):
    var_p = sigma_p ** 2
    var_q = sigma_q ** 2

    trace_term = var_p / var_q
    diff_term = (mu_q - mu_p) ** 2 / var_q
    log_term = 2 * (torch.log(sigma_q + eps) - torch.log(sigma_p + eps))
    kl = 0.5 * ((trace_term + diff_term - 1 + log_term).sum(dim=-1))  # sum over d

    return kl.mean() # mean over batch
