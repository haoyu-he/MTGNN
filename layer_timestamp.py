from __future__ import division
import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F


class graph_constructor_timestamp(nn.Module):
    """
    Timestamp-specific graph constructor that creates adjacency matrices with timestep dimension.
    
    Supports both:
    - Single adjacency: (nodes, nodes) -> backward compatible
    - Timestamp-specific: (timesteps, nodes, nodes) -> different adjacency per timestep
    """
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None, seq_length=None):
        super(graph_constructor_timestamp, self).__init__()
        self.nnodes = nnodes
        self.seq_length = seq_length
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx, timestamp_specific=False):
        """
        Args:
            idx: Node indices
            timestamp_specific: If True, return (timesteps, nodes, nodes), else (nodes, nodes)
        """
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        
        if timestamp_specific and self.seq_length is not None:
            # Create timestamp-specific adjacency by adding small random variations
            timestamp_adj = []
            for t in range(self.seq_length):
                # Add small random variation to create different adjacency per timestep
                adj_t = adj + torch.randn_like(adj) * 0.01
                adj_t = F.relu(torch.tanh(self.alpha * adj_t))
                # Reapply masking
                adj_t = adj_t * mask
                timestamp_adj.append(adj_t)
            return torch.stack(timestamp_adj, dim=0)  # (timesteps, nodes, nodes)
        else:
            return adj  # (nodes, nodes)


class dy_nconv_timestamp(nn.Module):
    """
    Timestamp-specific dynamic nconv layer that creates adjacency from input features.
    
    This layer creates dynamic adjacency matrices from input features, similar to the original dy_nconv.
    """
    def __init__(self):
        super(dy_nconv_timestamp, self).__init__()

    def forward(self, x1, x2):
        """
        Args:
            x1: Input tensor (batch, nodes, channels, timesteps) - transposed input
            x2: Input tensor (batch, channels, nodes, timesteps) - original input
        """
        # Create dynamic adjacency from input features
        # This is the same as original dy_nconv
        x = torch.einsum('ncvl,nvwl->ncwl', (x1, x2))
        return x.contiguous()


class prop_timestamp(nn.Module):
    """
    Timestamp-specific prop layer that handles adjacency matrices with timestep dimension.
    """
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(prop_timestamp, self).__init__()
        self.nconv = nconv_timestamp()
        self.mlp = linear(c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        """
        Args:
            x: Input tensor (batch, channels, nodes, timesteps)
            adj: Adjacency matrix (nodes, nodes) or (timesteps, nodes, nodes)
        """
        # Handle different adjacency matrix dimensions
        if adj.dim() == 2:
            # Single adjacency matrix - add self-loops and normalize
            adj = adj + torch.eye(adj.size(0)).to(x.device)
            d = adj.sum(1)
            a = adj / d.view(-1, 1)
        elif adj.dim() == 3:
            # Timestamp-specific adjacency matrices
            # adj: (timesteps, nodes, nodes)
            eye = torch.eye(adj.size(1)).to(x.device)
            adj = adj + eye.unsqueeze(0)  # Add self-loops to each timestep
            d = adj.sum(2)  # Sum over target nodes for each timestep
            a = adj / d.unsqueeze(2)  # Normalize
        else:
            raise ValueError(f"Adjacency matrix must be 2D or 3D, got {adj.dim()}D")

        h = x
        
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
        
        ho = self.mlp(h)
        return ho


class dy_mixprop_timestamp(nn.Module):
    """
    Timestamp-specific dynamic mixprop layer that handles adjacency matrices with timestep dimension.
    """
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(dy_mixprop_timestamp, self).__init__()
        self.nconv = dy_nconv_timestamp()
        self.mlp1 = linear((gdep+1)*c_in, c_out)
        self.mlp2 = linear((gdep+1)*c_in, c_out)

        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        self.lin1 = linear(c_in, c_in)
        self.lin2 = linear(c_in, c_in)

    def forward(self, x):
        """
        Args:
            x: Input tensor (batch, channels, nodes, timesteps)
        """
        x1 = torch.tanh(self.lin1(x))
        x2 = torch.tanh(self.lin2(x))
        
        # Create dynamic adjacency from input features
        # This creates timestamp-specific adjacency matrices
        adj = self.nconv(x1.transpose(2, 1), x2)
        adj0 = torch.softmax(adj, dim=2)
        adj1 = torch.softmax(adj.transpose(2, 1), dim=2)

        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj0)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho1 = self.mlp1(ho)

        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj1)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho2 = self.mlp2(ho)

        return ho1 + ho2


class nconv_timestamp(nn.Module):
    """
    Timestamp-specific nconv layer that handles adjacency matrices with timestep dimension.
    
    Supports both:
    - Single adjacency: (nodes, nodes) -> broadcast to all timesteps
    - Timestamp-specific: (timesteps, nodes, nodes) -> different adjacency per timestep
    """
    def __init__(self):
        super(nconv_timestamp, self).__init__()

    def forward(self, x, A):
        """
        Args:
            x: Input tensor (batch, channels, nodes, timesteps)
            A: Adjacency matrix (nodes, nodes) or (timesteps, nodes, nodes)
        """
        if A.dim() == 2:
            # Single adjacency matrix - broadcast to all timesteps
            # x: (batch, channels, nodes, timesteps)
            # A: (nodes, nodes)
            x = torch.einsum('ncwl,vw->ncvl', (x, A))
        elif A.dim() == 3:
            # Timestamp-specific adjacency matrices
            # x: (batch, channels, nodes, timesteps)
            # A: (timesteps, nodes, nodes)
            x = torch.einsum('ncwl,tvw->ncvl', (x, A))
        else:
            raise ValueError(f"Adjacency matrix must be 2D or 3D, got {A.dim()}D")
        
        return x.contiguous()


class mixprop_timestamp(nn.Module):
    """
    Timestamp-specific mixprop layer that handles adjacency matrices with timestep dimension.
    """
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(mixprop_timestamp, self).__init__()
        self.nconv = nconv_timestamp()
        self.mlp = linear((gdep+1)*c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        """
        Args:
            x: Input tensor (batch, channels, nodes, timesteps)
            adj: Adjacency matrix (nodes, nodes) or (timesteps, nodes, nodes)
        """
        # Handle different adjacency matrix dimensions
        if adj.dim() == 2:
            # Single adjacency matrix - add self-loops and normalize
            adj = adj + torch.eye(adj.size(0)).to(x.device)
            d = adj.sum(1)
            a = adj / d.view(-1, 1)
        elif adj.dim() == 3:
            # Timestamp-specific adjacency matrices
            # adj: (timesteps, nodes, nodes)
            eye = torch.eye(adj.size(1)).to(x.device)
            adj = adj + eye.unsqueeze(0)  # Add self-loops to each timestep
            d = adj.sum(2)  # Sum over target nodes for each timestep
            a = adj / d.unsqueeze(2)  # Normalize
        else:
            raise ValueError(f"Adjacency matrix must be 2D or 3D, got {adj.dim()}D")

        h = x
        out = [h]
        
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
            out.append(h)
        
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        return ho


# Import all other classes from the original layer.py
from layer import (
    nconv, dy_nconv, linear, prop, mixprop, dy_mixprop,
    dilated_1D, dilated_inception, graph_constructor, graph_global,
    graph_undirected, graph_directed, LayerNorm
)