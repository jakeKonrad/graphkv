
from typing import NamedTuple

from .glzip import _CSR, _GraphSageSampler
import numpy as np
import torch

def _map_none(f, x):
    if x is not None:
        f(x)
    else:
        None

class CSR:
    def __init__(self, edge_index):
        self._csr = _CSR(edge_index=edge_index.numpy())

    @property
    def order(self):
        return self._csr.order

class Adj(NamedTuple):
    edge_index: torch.Tensor
    e_id: torch.Tensor
    size: torch.Tensor

    def to(self, *args, **kwargs):
        return Adj(self.edge_index.to(*args, **kwargs),
                   self.e_id.to(*args, **kwargs), self.size)

class GraphSageSampler:
    def __init__(self, csr, sizes):
        self._sampler = _GraphSageSampler(csr._csr, sizes)

    def sample(self, batch):
        batch = batch.numpy()
        (nodes, batch_size, adjs) = self._sampler.sample(batch)
        nodes = torch.from_numpy(nodes)
        adjs = [adj for adj in map(lambda a: Adj(torch.from_numpy(a[0]), torch.from_numpy(a[1]), torch.from_numpy(a[2])), adjs)]
        return (nodes, batch_size, adjs)
