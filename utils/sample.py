from torch_geometric.nn.pool import fps


def global_sample(pos, ratio=0.25, batch=None):
    sampled_indices = fps(pos, ratio=ratio, batch=batch)
    
    return sampled_indices
