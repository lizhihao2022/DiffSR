import torch
import torch.nn.functional as F
from time import time
import torch.distributed as dist

_loss_dict = {
    
}


class CompositeLoss:
    """
    组合损失：传入 {name: weight}，自动求和并返回总损失与分项日志
    """
    def __init__(self, spec: dict[str, float]):  # e.g. {"l1":1.0,"l2":0.1,"physics":0.5}
        self.loss_list = ["total_loss", "l2", "l1"]
        self.init_record()

    def __call__(self, pred, target, *, batch_size: int | None = None, **batch):
        logs = {}
        total = 0.0
        for name, w in self.spec.items():
            if w == 0: 
                continue
            fn = LOSS_REGISTRY[name]
            val = fn(pred, target, **batch)  # 标量（已mean）
            total = total + w * val
            logs[name] = float(val.detach().item())
        logs["loss_total"] = float(total.detach().item())
        if record is not None and batch_size is not None:
            record.update(logs, n=batch_size)
        return total  # 用于 backward

    def init_record(self):
        self.record = LossRecord(self.loss_list)

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

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y, **kwargs):
        return self.rel(x, y)


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
    
    def update(self, update_dict, n=1):
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
        
    def dist_reduce(self, device=None):
        if not (dist.is_available() and dist.is_initialized()):
            return

        device = device if device is not None else (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available() else torch.device("cpu")
        )

        for loss in self.loss_list:
            # 打包 sum 与 count，一次 all_reduce 两次也行，这里演示两次更直观
            t_sum = torch.tensor(self.loss_dict[loss].sum, dtype=torch.float32, device=device)
            t_cnt = torch.tensor(self.loss_dict[loss].count, dtype=torch.float32, device=device)

            dist.all_reduce(t_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(t_cnt, op=dist.ReduceOp.SUM)

            global_sum = t_sum.item()
            global_cnt = t_cnt.item()

            # 防止除零（极端情况：全局没有任何样本）
            if global_cnt > 0:
                self.loss_dict[loss].sum = global_sum
                self.loss_dict[loss].count = global_cnt
                self.loss_dict[loss].avg = global_sum / global_cnt
            else:
                # 保持为 0，或设为 NaN/Inf 按需处理
                self.loss_dict[loss].sum = 0.0
                self.loss_dict[loss].count = 0
                self.loss_dict[loss].avg = 0.0
    
    def __str__(self):
        return self.format_metrics()
    
    def __repr__(self):
        return self.loss_dict[self.loss_list[0]].avg
