import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

DEMOGRAPHIC_GROUPS: List[Tuple[str, str]] = [
    ("Black", "Female"),
    ("Black", "Male"),
    ("Asian", "Male"),
    ("Asian", "Female"),
    ("White", "Male"),
    ("White", "Female"),
]

TEXT_TEMPLATES = [
    "a photo of a {race} {gender} face",
    "a portrait photo of a {race} {gender}",
    "a close-up photo of a {race} {gender} face",
    "a selfie of a {race} {gender}",
]

class GatingNetwork(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int = 256, top_k: int = 2):
        super(GatingNetwork, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.network(x)

        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)

        router_probs = F.softmax(logits, dim=-1)
        sparse_mask = torch.zeros_like(logits).scatter_(-1, top_k_indices, 1.0)
        tokens_per_expert = torch.mean(sparse_mask.float(), dim=0)
        router_prob_per_expert = torch.mean(router_probs, dim=0)
        load_balancing_loss = self.num_experts * torch.sum(tokens_per_expert * router_prob_per_expert)

        return {
            'gating_logits': logits,
            'top_k_indices': top_k_indices,
            'top_k_gates': top_k_gates,
            'balance_loss': load_balancing_loss
        }
