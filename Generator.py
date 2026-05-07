import torch
from torch import nn
from torch.nn import functional as F

def gumbel_sigmoid(logits, tau=1.0, hard=False, eps=1e-10):
    uniforms = torch.rand_like(logits)
    gumbels = -torch.log(-torch.log(uniforms + eps) + eps)
    y_soft = torch.sigmoid((logits + gumbels) / tau)

    if hard:
        y_hard = (y_soft > 0.5).float()
        ret = y_hard - y_soft.detach() + y_soft
        return ret
    else:
        return y_soft

def get_metrics(probs, labels):
    probs = torch.argmax(probs, dim=1)
    correct = 0
    for i in range(len(probs)):
        if probs[i] == labels[i]:
            correct += 1
    return correct / len(probs)

def F1_score(pred, truth):
    pred_v = torch.argmax(pred, dim=1)
    tp = (pred_v * truth).sum()
    tn = ((1 - pred_v) * (1 - truth)).sum()
    fp = (pred_v * (1 - truth)).sum()
    fn = ((1 - pred_v) * truth).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1

class AdversarialGraphGenerator(nn.Module):
    def __init__(self, node_features_dim, hidden_dim=128, dropout=0.1):
        super(AdversarialGraphGenerator, self).__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1) 
        )

    def forward(self, node_features, original_edges, num_nodes,
                add_edge_ratio=0.1, temperature=1.0):
        batch_size = node_features.size(0)
        node_embeddings = self.node_encoder(node_features)
        adversarial_edges_indices = []
        adversarial_edges_weights = []
        all_edges_list = [e.squeeze(0) if e.dim() == 3 else e for e in original_edges if e.numel() > 0]
        if len(all_edges_list) > 0:
            full_edge_index = torch.cat(all_edges_list, dim=1)
            vals = torch.ones(full_edge_index.size(1), device=node_features.device)
            adj = torch.sparse_coo_tensor(full_edge_index, vals, (num_nodes, num_nodes))
            adj_2 = torch.sparse.mm(adj, adj).coalesce()
            candidate_indices_pool = adj_2.indices()
            mask_no_self = candidate_indices_pool[0] != candidate_indices_pool[1]
            candidate_indices_pool = candidate_indices_pool[:, mask_no_self]
        else:
            candidate_indices_pool = torch.empty((2, 0), dtype=torch.long, device=node_features.device)
        for edge_type_idx, edges in enumerate(original_edges):
            edges = edges.squeeze(0) if edges.dim() == 3 else edges
            if edges.size(1) > 0:
                kept_edges = edges
                kept_weights = torch.ones(edges.size(1), device=node_features.device)
            else:
                kept_edges = edges
                kept_weights = torch.tensor([], device=node_features.device)
            num_add = int(num_nodes * add_edge_ratio)
            if num_add > 0 and candidate_indices_pool.size(1) > 0:
                total_candidates = candidate_indices_pool.size(1)
                if total_candidates <= num_add:
                    selected_candidates = candidate_indices_pool
                else:
                    perm = torch.randperm(total_candidates, device=node_features.device)[:num_add]
                    selected_candidates = candidate_indices_pool[:, perm]
                src_nodes = selected_candidates[0]
                tgt_nodes = selected_candidates[1]
                src_embeddings = node_embeddings[src_nodes]
                tgt_embeddings = node_embeddings[tgt_nodes]
                candidate_edge_embeddings = torch.cat([src_embeddings, tgt_embeddings], dim=1)
                add_logits = self.edge_predictor(candidate_edge_embeddings).squeeze(-1)
                add_weights = gumbel_sigmoid(add_logits, tau=temperature, hard=False)
                num_final_add = min(len(src_nodes), num_add)
                _, add_indices = torch.topk(add_weights.detach(), num_final_add)
                new_edges_indices = torch.stack([src_nodes[add_indices], tgt_nodes[add_indices]], dim=0)
                new_edges_weights = add_weights[add_indices]
                final_edges = torch.cat([kept_edges, new_edges_indices], dim=1)
                final_weights = torch.cat([kept_weights, new_edges_weights], dim=0)
            else:
                final_edges = kept_edges
                final_weights = kept_weights
            adversarial_edges_indices.append(final_edges.unsqueeze(0))
            adversarial_edges_weights.append(final_weights.unsqueeze(0))
            
        return adversarial_edges_indices, adversarial_edges_weights