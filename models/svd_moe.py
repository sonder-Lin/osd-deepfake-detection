import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class SVDMoeLinear(nn.Module):

    def __init__(self, in_features, out_features, r_main, num_experts, rank_per_expert, artifact_expert_idx, bias=True):
        super(SVDMoeLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r_main = r_main
        self.num_experts = num_experts
        self.rank_per_expert = rank_per_expert
        self.artifact_expert_idx = artifact_expert_idx

        self.register_buffer('weight_main', torch.zeros(out_features, in_features))
        self.register_buffer('U_r', torch.zeros(out_features, r_main))
        self.register_buffer('V_r', torch.zeros(r_main, in_features))

        self.U_experts = nn.ParameterList([nn.Parameter(torch.zeros(out_features, rank_per_expert)) for _ in range(num_experts)])
        self.S_experts = nn.ParameterList([nn.Parameter(torch.zeros(rank_per_expert)) for _ in range(num_experts)])
        self.V_experts = nn.ParameterList([nn.Parameter(torch.zeros(rank_per_expert, in_features)) for _ in range(num_experts)])


        self.register_buffer('weight_original_fnorm', torch.tensor(0.0))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

    def forward_main_only(self, x: torch.Tensor) -> torch.Tensor:

        output = F.linear(x, self.weight_main, None)
        if self.bias is not None:
            output = output + self.bias
        return output

    def forward(self, x: torch.Tensor, gating_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        output_main = F.linear(x, self.weight_main, None)

        top_k_indices = gating_outputs['top_k_indices']
        top_k_gates = gating_outputs['top_k_gates']
        k = top_k_indices.size(1)

        expert_output = torch.zeros_like(output_main)
        U_all = torch.stack([p for p in self.U_experts])
        S_all = torch.stack([p for p in self.S_experts])
        V_all = torch.stack([p for p in self.V_experts])


        original_dim = x.dim()


        if original_dim == 2:
            x = x.unsqueeze(1)

            expert_output = expert_output.unsqueeze(1)


        for i in range(k):
            chosen_expert_indices = top_k_indices[:, i]
            gate_values = top_k_gates[:, i].unsqueeze(-1)

            U_batch = U_all[chosen_expert_indices]
            S_batch = S_all[chosen_expert_indices]
            V_batch = V_all[chosen_expert_indices]


            x_v = torch.bmm(x, V_batch.transpose(1, 2))


            x_v_s = x_v * S_batch.unsqueeze(1)


            current_expert_output = torch.bmm(x_v_s, U_batch.transpose(1, 2))


            expert_output += current_expert_output * gate_values.unsqueeze(-1)


        if original_dim == 2:
            expert_output = expert_output.squeeze(1)

        final_output = output_main + expert_output
        if self.bias is not None:
            final_output = final_output + self.bias
        return final_output

    def _calculate_pairwise_loss(self, base1, base2):
        error = base1.t() @ base2
        return torch.norm(error, p='fro')

    def compute_targeted_orthogonal_loss(self, active_expert_idx: int) -> torch.Tensor:

        loss = torch.tensor(0.0, device=self.weight_main.device)
        num_pairs = 0

        active_U = F.normalize(self.U_experts[active_expert_idx], dim=0)
        active_V_t = F.normalize(self.V_experts[active_expert_idx], dim=1).t()


        loss += self._calculate_pairwise_loss(self.U_r, active_U)
        loss += self._calculate_pairwise_loss(self.V_r.t(), active_V_t)
        num_pairs += 1

        if active_expert_idx == self.artifact_expert_idx:
            return loss / num_pairs


        for i in range(active_expert_idx):

            if i == self.artifact_expert_idx:
                continue

            prev_U = F.normalize(self.U_experts[i].detach(), dim=0)
            prev_V_t = F.normalize(self.V_experts[i].detach(), dim=1).t()
            loss += self._calculate_pairwise_loss(prev_U, active_U)
            loss += self._calculate_pairwise_loss(prev_V_t, active_V_t)
            num_pairs += 1

        return loss / num_pairs if num_pairs > 0 else torch.tensor(0.0, device=self.weight_main.device)

    def compute_full_orthogonal_loss(self) -> torch.Tensor:

        all_U_bases = [self.U_r] + [F.normalize(u, dim=0) for u in self.U_experts]
        all_V_bases_t = [self.V_r.t()] + [F.normalize(v, dim=1).t() for v in self.V_experts]

        loss = torch.tensor(0.0, device=self.weight_main.device)
        num_pairs = 0


        for i in range(len(all_U_bases)):
            for j in range(i + 1, len(all_U_bases)):
                loss += self._calculate_pairwise_loss(all_U_bases[i], all_U_bases[j])
                loss += self._calculate_pairwise_loss(all_V_bases_t[i], all_V_bases_t[j])
                num_pairs += 1

        return loss / num_pairs if num_pairs > 0 else torch.tensor(0.0, device=self.weight_main.device)
