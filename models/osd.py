import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from transformers import CLIPModel, CLIPProcessor
from transformers.models.clip.modeling_clip import CLIPVisionEmbeddings

from models.clip_moe import ViTMoELayer
from models.routing import DEMOGRAPHIC_GROUPS, TEXT_TEMPLATES
from models.svd_moe import SVDMoeLinear

class OSD(nn.Module):
    def __init__(self, config=None):
        super(OSD, self).__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.rank_per_expert = config.rank_per_expert

        self.moe_lambda_orth = config.moe_lambda_orth

        self.training_mode = "standard"
        self.active_expert_idx = None
        self.use_gt_router = False
        self.use_random_router = False


        self.semantic_expert_scale = getattr(config, 'semantic_expert_scale', 1.0)
        self.artifact_expert_scale = getattr(config, 'artifact_expert_scale', 1.0)


        self.artifact_expert_idx = config.num_experts - 1
        self.num_semantic_experts = config.num_experts - 1
        if self.num_semantic_experts <= 0:
            raise ValueError("num_experts must be at least 2 (1 artifact expert and at least 1 semantic expert).")

        pretrained_path = config.CLIP_path


        self.zero_shot_tau = getattr(config, 'zero_shot_tau', 0.1)

        clip_model = CLIPModel.from_pretrained(pretrained_path)
        vision_config = clip_model.vision_model.config
        self.hidden_size = vision_config.hidden_size


        total_rank = vision_config.hidden_size
        residual_rank = self.rank_per_expert

        if residual_rank >= total_rank:
            raise ValueError(
                f"The total rank for experts ({residual_rank}) must be less than the total rank ({total_rank}). "
                f"Please reduce num_experts ({self.num_experts}) or rank_per_expert ({self.rank_per_expert})."
            )

        r_main = total_rank - residual_rank
        print(f"Rank allocation: total_rank={total_rank}, r_main={r_main}, num_experts={self.num_experts}, rank_per_expert={self.rank_per_expert}")


        self.embeddings = CLIPVisionEmbeddings(vision_config)
        self.ln_pre = nn.LayerNorm(vision_config.hidden_size)
        self.encoder_layers = nn.ModuleList([
            ViTMoELayer(vision_config, self.num_experts, r_main, self.rank_per_expert, self.artifact_expert_idx) for _ in range(vision_config.num_hidden_layers)
        ])
        self.ln_post = nn.LayerNorm(self.hidden_size, eps=vision_config.layer_norm_eps)
        self.head = nn.Linear(self.hidden_size, 2)

        self.load_and_replace_from_pretrained(clip_model)


        self.register_buffer('visual_projection_weight', clip_model.visual_projection.weight.data.clone())


        self._init_text_prototypes(clip_model, pretrained_path)

    def load_and_replace_from_pretrained(self, pretrained_model):
        vision_model = pretrained_model.vision_model

        self.embeddings.load_state_dict(vision_model.embeddings.state_dict())
        self.ln_pre.load_state_dict(vision_model.pre_layrnorm.state_dict())
        self.ln_post.load_state_dict(vision_model.post_layernorm.state_dict())

        for i, pretrained_layer in enumerate(vision_model.encoder.layers):
            moe_layer = self.encoder_layers[i]
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
                self._replace_linear_with_svd_moe(
                    getattr(pretrained_layer.self_attn, proj_name),
                    getattr(moe_layer.self_attn, proj_name)
                )
            moe_layer.layer_norm1.load_state_dict(pretrained_layer.layer_norm1.state_dict())
            moe_layer.mlp.load_state_dict(pretrained_layer.mlp.state_dict())
            moe_layer.layer_norm2.load_state_dict(pretrained_layer.layer_norm2.state_dict())


        for name, param in self.named_parameters():
            if 'experts' not in name and 'head' not in name:
                param.requires_grad = False

    @torch.no_grad()
    def _init_text_prototypes(self, clip_model, pretrained_path: str):

        device = next(self.parameters()).device


        text_model = clip_model.text_model
        text_projection = clip_model.text_projection
        processor = CLIPProcessor.from_pretrained(pretrained_path)

        text_feats = []
        num_groups = min(self.num_semantic_experts, len(DEMOGRAPHIC_GROUPS))

        for i in range(num_groups):
            race, gender = DEMOGRAPHIC_GROUPS[i]
            prompts = [tpl.format(race=race, gender=gender) for tpl in TEXT_TEMPLATES]

            inputs = processor(text=prompts, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}


            text_outputs = text_model(**inputs)

            pooled_output = text_outputs.pooler_output
            feats = text_projection(pooled_output).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)


            proto = feats.mean(dim=0)
            proto = proto / proto.norm(dim=-1, keepdim=True)
            text_feats.append(proto)


        text_prototypes = torch.stack(text_feats, dim=0)
        self.register_buffer('text_prototypes', text_prototypes)
        print(f"Initialized {num_groups} text prototypes for zero-shot routing")

    def _replace_linear_with_svd_moe(self, original_module: nn.Linear, moe_module):
        original_weight = original_module.weight.data
        moe_module.weight_original_fnorm.data.copy_(torch.norm(original_weight, p='fro'))
        if original_module.bias is not None:
            moe_module.bias.data.copy_(original_module.bias.data)

        U, S, Vh = torch.linalg.svd(original_weight, full_matrices=False)


        r = moe_module.r_main
        U_r, S_r, Vh_r = U[:, :r], S[:r], Vh[:r, :]
        moe_module.weight_main.data.copy_(U_r @ torch.diag(S_r) @ Vh_r)

        moe_module.U_r.data.copy_(U_r)
        moe_module.V_r.data.copy_(Vh_r)


        chunk_start_rank = r


        if chunk_start_rank >= len(S):

            for i in range(moe_module.num_experts):
                moe_module.U_experts[i].data.zero_()
                moe_module.S_experts[i].data.zero_()
                moe_module.V_experts[i].data.zero_()
            return

        U_chunk, S_chunk, Vh_chunk = U[:, chunk_start_rank:], S[chunk_start_rank:], Vh[chunk_start_rank:, :]


        actual_chunk_rank = U_chunk.shape[1]

        if actual_chunk_rank < moe_module.rank_per_expert:
            pad_rank = moe_module.rank_per_expert - actual_chunk_rank
            U_chunk = F.pad(U_chunk, (0, pad_rank))
            S_chunk = F.pad(S_chunk, (0, pad_rank))
            Vh_chunk = F.pad(Vh_chunk, (0, 0, 0, pad_rank))


        for i in range(moe_module.num_experts):
            moe_module.U_experts[i].data.copy_(U_chunk)
            moe_module.S_experts[i].data.copy_(S_chunk)
            moe_module.V_experts[i].data.copy_(Vh_chunk)

    def set_training_mode(self, mode: str, active_expert_idx: int = None, trainable_expert_indices: list = None):

        if mode not in ['hard_sampling', 'head_finetune', 'standard']:
            raise ValueError("Training mode must be one of 'hard_sampling', 'head_finetune', or 'standard'")

        if mode == 'hard_sampling' and (active_expert_idx is None or not isinstance(active_expert_idx, int) or active_expert_idx < 0):
            raise ValueError("A valid integer active_expert_idx must be provided for 'hard_sampling' mode.")

        if mode == 'head_finetune' and trainable_expert_indices is not None and not isinstance(trainable_expert_indices, list):
            raise ValueError("'trainable_expert_indices' must be a list of integers.")


        self.training_mode = mode
        self.active_expert_idx = active_expert_idx if mode == 'hard_sampling' else None

        for param in self.parameters():
            param.requires_grad = False

        if mode == 'standard':

            for param in self.head.parameters():
                param.requires_grad = True

            for layer in self.encoder_layers:
                for proj_name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
                    moe_linear_layer = getattr(layer.self_attn, proj_name)
                    for param in moe_linear_layer.U_experts:
                        param.requires_grad = True
                    for param in moe_linear_layer.S_experts:
                        param.requires_grad = True
                    for param in moe_linear_layer.V_experts:
                        param.requires_grad = True

        elif mode == 'head_finetune':

            print("Unfreezing: Head only (zero-shot routing mode).")
            for param in self.head.parameters():
                param.requires_grad = True


            if trainable_expert_indices:
                print(f"Additionally unfreezing experts with indices: {trainable_expert_indices}")
                for expert_idx in trainable_expert_indices:
                    if not 0 <= expert_idx < self.num_experts:
                        raise ValueError(f"trainable_expert_index ({expert_idx}) is out of bounds for num_experts ({self.num_experts}).")

                    for layer in self.encoder_layers:
                        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
                            moe_linear_layer = getattr(layer.self_attn, proj_name)
                            moe_linear_layer.U_experts[expert_idx].requires_grad = True
                            moe_linear_layer.S_experts[expert_idx].requires_grad = True
                            moe_linear_layer.V_experts[expert_idx].requires_grad = True

        elif mode == 'hard_sampling':

            if self.active_expert_idx >= self.num_experts:
                 raise ValueError(f"active_expert_idx ({self.active_expert_idx}) is out of bounds for num_experts ({self.num_experts}).")

            for layer in self.encoder_layers:
                for proj_name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
                    moe_linear_layer = getattr(layer.self_attn, proj_name)


                    moe_linear_layer.U_experts[self.active_expert_idx].requires_grad = True
                    moe_linear_layer.S_experts[self.active_expert_idx].requires_grad = True
                    moe_linear_layer.V_experts[self.active_expert_idx].requires_grad = True

            for param in self.head.parameters():
                param.requires_grad = True

        print(f"Successfully set training mode to '{self.training_mode}'" + (f" (active expert: {self.active_expert_idx})" if self.training_mode == 'hard_sampling' else ""))


    def forward_main_only(self, images: torch.Tensor) -> torch.Tensor:

        hidden_states = self.embeddings(images)
        hidden_states = self.ln_pre(hidden_states)

        for layer_module in self.encoder_layers:
            hidden_states = layer_module.forward_main_only(hidden_states)

        pooled_output = self.ln_post(hidden_states[:, 0, :])
        return pooled_output

    def zero_shot_routing(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        with torch.no_grad():

            image_features = self.forward_main_only(images)


            image_features = image_features @ self.visual_projection_weight.T
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)


            scores = (image_features @ self.text_prototypes.T) / self.zero_shot_tau
            expert_indices = scores.argmax(dim=1, keepdim=True)

        return expert_indices, scores

    def forward(self, images, category_labels=None, output_attentions: bool = False) -> dict:
        batch_size = images.size(0)

        hidden_states = self.embeddings(images)
        hidden_states = self.ln_pre(hidden_states)

        gating_outputs = {}


        if self.training_mode == 'hard_sampling':
            if self.active_expert_idx is None:
                raise ValueError("In hard_sampling mode, active_expert_idx must be set.")

            hard_sampling_target = torch.full((batch_size,), self.active_expert_idx, dtype=torch.long, device=hidden_states.device)
            gating_outputs['top_k_indices'] = hard_sampling_target.unsqueeze(1)
            gating_outputs['top_k_gates'] = torch.ones_like(gating_outputs['top_k_indices'], dtype=hidden_states.dtype)
            gating_outputs['balance_loss'] = torch.tensor(0.0, device=hidden_states.device)
            gating_outputs['gating_logits'] = None


        else:

            artifact_expert_indices = torch.full(
                (batch_size, 1),
                self.artifact_expert_idx,
                dtype=torch.long,
                device=hidden_states.device
            )
            artifact_expert_gates = torch.ones(
                (batch_size, 1),
                dtype=hidden_states.dtype,
                device=hidden_states.device
            )


            if self.use_gt_router and category_labels is not None:
                semantic_expert_indices = category_labels.view(-1, 1).long()
                semantic_expert_gates = torch.ones(
                    (batch_size, 1),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device
                )
                gating_outputs['gating_logits'] = None
                gating_outputs['balance_loss'] = torch.tensor(0.0, device=hidden_states.device)

            elif self.use_random_router:
                semantic_expert_indices = torch.randint(
                    0, self.num_semantic_experts,
                    (batch_size, 1),
                    dtype=torch.long,
                    device=hidden_states.device
                )
                semantic_expert_gates = torch.ones(
                    (batch_size, 1),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device
                )
                gating_outputs['gating_logits'] = None
                gating_outputs['balance_loss'] = torch.tensor(0.0, device=hidden_states.device)
            else:

                semantic_expert_indices, routing_scores = self.zero_shot_routing(images)
                semantic_expert_gates = torch.ones(
                    (batch_size, 1),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device
                )
                gating_outputs['gating_logits'] = routing_scores
                gating_outputs['balance_loss'] = torch.tensor(0.0, device=hidden_states.device)


            scaled_semantic_gates = semantic_expert_gates * self.semantic_expert_scale
            scaled_artifact_gates = artifact_expert_gates * self.artifact_expert_scale


            gating_outputs['top_k_indices'] = torch.cat([semantic_expert_indices, artifact_expert_indices], dim=1)
            gating_outputs['top_k_gates'] = torch.cat([scaled_semantic_gates, scaled_artifact_gates], dim=1)


        final_gates = torch.zeros(batch_size, self.num_experts, device=hidden_states.device)

        final_gates.scatter_(-1, gating_outputs['top_k_indices'], gating_outputs['top_k_gates'])


        last_attn_weights = None
        num_layers = len(self.encoder_layers)
        for i, layer_module in enumerate(self.encoder_layers):
            is_last_layer = (i == num_layers - 1)
            layer_output = layer_module(
                hidden_states,
                gating_outputs=gating_outputs,
                output_attentions=(output_attentions and is_last_layer)
            )
            hidden_states = layer_output[0]
            if is_last_layer and output_attentions:
                last_attn_weights = layer_output[1]

        pooled_output = self.ln_post(hidden_states[:, 0, :])
        pred = self.head(pooled_output)
        prob = torch.softmax(pred, dim=1)[:, 1]

        result = {
            'cls': pred,
            'prob': prob,
            'balance_loss': gating_outputs['balance_loss'],
            'gating_logits': gating_outputs['gating_logits'],
            'final_gates': final_gates,
            'features': pooled_output,
        }

        if output_attentions:
            result['attentions'] = last_attn_weights

        return result

    def get_losses(self, pred_dict: dict, labels, expert_domain_labels, criterion) -> dict:
        pred = pred_dict['cls']
        classification_loss = criterion(pred, labels)

        orth_loss = torch.tensor(0.0, device=pred.device)

        num_moe_layers = sum(1 for module in self.modules() if isinstance(module, SVDMoeLinear))


        if self.training_mode not in ['head_finetune'] and num_moe_layers > 0:
            current_orth_loss = 0.0
            for module in self.modules():
                if isinstance(module, SVDMoeLinear):
                    if self.training_mode == 'hard_sampling':
                        current_orth_loss += module.compute_targeted_orthogonal_loss(self.active_expert_idx)
                    elif self.training_mode == 'standard':
                        current_orth_loss += module.compute_full_orthogonal_loss()
            orth_loss = current_orth_loss / num_moe_layers

        total_loss = classification_loss + self.moe_lambda_orth * orth_loss

        return {
            'overall_loss': total_loss,
            'classification_loss': classification_loss.detach(),
            'orth_loss': orth_loss.detach(),
        }
