from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin


def matmul(A, B):
    return torch.matmul(A, B)


def ib_ib_bj_to_ij(pf, pp, cp):
    ll = torch.amax(cp, dim=-1)
    ll = torch.where(torch.isfinite(ll), ll, torch.zeros_like(ll))

    pp = torch.exp(pp - ll[None, :])
    cp = torch.exp(cp - ll[:, None])

    pp = torch.nan_to_num(pp, nan=0.0, posinf=0.0, neginf=0.0)
    cp = torch.nan_to_num(cp, nan=0.0, posinf=0.0, neginf=0.0)

    ratio = pf / pp
    ratio[pp == 0.0] = 0.0
    ratio = torch.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)

    af = torch.matmul(ratio, cp)
    return af


class CHMM(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        vocab_size: int,
        eos_token_id: int,
        clones_per_token: list[int] | torch.Tensor,
    ):
        super().__init__()

        clones_per_token = torch.as_tensor(clones_per_token, dtype=torch.long)
        assert len(clones_per_token) == vocab_size, "One value per token for num clones"

        hidden_states = int(clones_per_token.sum().item())
        alpha_exp = torch.softmax(torch.randn(hidden_states, hidden_states), dim=1)
        gamma = torch.log_softmax(torch.randn(hidden_states), dim=0)

        self.alpha_exp = nn.Parameter(alpha_exp, requires_grad=False)
        self.gamma = nn.Parameter(gamma, requires_grad=False)

        self.register_buffer(
            "clones_per_token",
            clones_per_token,
        )
        self.register_buffer(
            "clone_to_token",
            torch.repeat_interleave(
                torch.arange(vocab_size, dtype=torch.long),
                clones_per_token,
            ),
        )
        self.register_buffer(
            "beta",
            self._build_deterministic_beta(
                clones_per_token=clones_per_token.tolist(),
                vocab_size=vocab_size,
                dtype=alpha_exp.dtype,
            ),
        )

        self.hidden_states = hidden_states
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id

    @staticmethod
    def _build_deterministic_beta(
        clones_per_token: list[int],
        vocab_size: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        hidden_states = int(sum(clones_per_token))
        beta = torch.full((hidden_states, vocab_size), float("-inf"), dtype=dtype)

        start = 0
        for token_id, num_clones in enumerate(clones_per_token):
            end = start + num_clones
            beta[start:end, token_id] = 0.0
            start = end

        return beta

    def config_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "eos_token_id": self.eos_token_id,
            "clones_per_token": self.clones_per_token.tolist(),
        }

    @torch.no_grad()
    def save_pretrained(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "config": self.config_dict(),
            "alpha_exp": self.alpha_exp.detach().cpu(),
            "gamma": self.gamma.detach().cpu(),
        }
        torch.save(payload, output_dir / "model.pt")

        with open(output_dir / "config.json", "w", encoding="utf-8") as fout:
            json.dump(payload["config"], fout, indent=2)

    @classmethod
    def from_pretrained(cls, model_path: str | Path, map_location: str | torch.device | None = None) -> "CHMM":
        model_path = Path(model_path)
        payload = torch.load(model_path / "model.pt", map_location="cpu", weights_only=True)

        model = cls(
            vocab_size=payload["config"]["vocab_size"],
            eos_token_id=payload["config"]["eos_token_id"],
            clones_per_token=payload["config"]["clones_per_token"],
        )
        model.update_params(payload["alpha_exp"], payload["gamma"])

        if map_location is not None:
            model = model.to(map_location)
        return model

    @torch.no_grad()
    def update_params(self, alpha_exp, gamma):
        self.alpha_exp.data = alpha_exp.to(self.alpha_exp.device, dtype=self.alpha_exp.dtype)
        self.gamma.data = gamma.to(self.gamma.device, dtype=self.gamma.dtype)

    def forward(self, input_ids):
        device = self.alpha_exp.device
        alpha_exp, beta, gamma_exp = self.alpha_exp, self.beta, torch.softmax(self.gamma, dim=0)
        hidden_states = self.hidden_states
        batch_size, seq_len = input_ids.shape

        input_ids_ = torch.permute(input_ids, (1, 0)).contiguous()
        lookup_ids = input_ids_.clamp_min(0)

        input_probs = beta[
            torch.arange(0, hidden_states, device=device)[None, :, None],
            lookup_ids[:, None, :],
        ].contiguous()

        observed = (input_ids_ != -1)[:, None, :].expand(-1, hidden_states, -1)
        input_probs = torch.where(observed, input_probs, torch.zeros_like(input_probs))

        ys = []
        y = torch.zeros((hidden_states, batch_size), device=device)
        for t in range(seq_len - 1, -1, -1):
            if t != seq_len - 1:
                y_max = torch.amax(y, dim=0, keepdim=True)
                y_max = torch.where(torch.isfinite(y_max), y_max, torch.zeros_like(y_max))
                y = torch.exp(y - y_max)
                y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
                y = matmul(alpha_exp, y)
                y = torch.log(y) + y_max
            y += input_probs[t, :, :]
            ys.append(y)

        y_max = torch.amax(y, dim=0)
        y_max = torch.where(torch.isfinite(y_max), y_max, torch.zeros_like(y_max))
        y = torch.exp(y - y_max.unsqueeze(0))
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        y = matmul(gamma_exp.unsqueeze(0), y).squeeze()
        y = torch.log(y) + y_max

        ys.append(y)
        return ys

    def backward(self, input_ids, probs, alpha_flow, beta_flow, gamma_flow):
        device = self.alpha_exp.device
        alpha_exp, beta, gamma_exp = self.alpha_exp, self.beta, torch.softmax(self.gamma, dim=0)
        hidden_states = self.hidden_states
        batch_size, seq_len = input_ids.shape
        neg_inf = torch.tensor(float("-inf"), device=device, dtype=beta.dtype)

        input_ids_ = torch.permute(input_ids, (1, 0)).contiguous()
        lookup_ids = input_ids_.clamp_min(0)

        raw_input_probs = beta[
            torch.arange(0, hidden_states, device=device)[None, :, None],
            lookup_ids[:, None, :],
        ].contiguous()

        observed = (input_ids_ != -1)[:, None, :].expand(-1, hidden_states, -1)
        input_probs = torch.where(observed, raw_input_probs, torch.zeros_like(raw_input_probs))
        supported = (~observed) | torch.isfinite(raw_input_probs)

        pf = gamma_exp.unsqueeze(0) * torch.exp(
            torch.permute(probs[-2], (1, 0)).contiguous() - probs[-1][:, None]
        )
        pf = torch.nan_to_num(pf, nan=0.0, posinf=0.0, neginf=0.0)

        gamma_flow.add_(torch.sum(pf, dim=0))

        for t in range(0, seq_len - 1):
            layer_idx = seq_len - t - 1
            cp = probs[layer_idx - 1]

            supported_t = supported[t]
            emit_t = input_probs[t, :, :]
            pp = torch.where(supported_t, probs[layer_idx] - emit_t, neg_inf)
            pp = torch.nan_to_num(pp, nan=float("-inf"), posinf=float("inf"), neginf=float("-inf"))

            alpha_flow.add_(
                ib_ib_bj_to_ij(
                    torch.permute(pf, (1, 0)).contiguous(),
                    pp,
                    torch.permute(cp, (1, 0)).contiguous(),
                )
            )

            pp = torch.permute(pp, (1, 0))
            cp = torch.permute(cp, (1, 0))
            pp_max = torch.amax(pp, dim=1, keepdim=True)
            pp_max = torch.where(torch.isfinite(pp_max), pp_max, torch.zeros_like(pp_max))

            pp_ = torch.exp(pp - pp_max)
            pp_ = torch.nan_to_num(pp_, nan=0.0, posinf=0.0, neginf=0.0)

            ratio = pf / pp_
            ratio[pp_ == 0.0] = 0.0
            ratio = torch.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)

            pf = matmul(ratio, alpha_exp) * torch.exp(cp - pp_max)
            pf = torch.nan_to_num(pf, nan=0.0, posinf=0.0, neginf=0.0)

    def loglikelihood(self, input_ids, batch_size):
        device = self.alpha_exp.device
        data_size, _ = input_ids.shape

        ll = torch.tensor([0.0], device=device)
        for batch_idx in range(0, data_size, batch_size):
            batch_size_ = min(batch_size, data_size - batch_idx)
            input_ids_batch = input_ids[batch_idx: batch_idx + batch_size_].to(device)
            probs_ = self.forward(input_ids_batch)
            ll += torch.sum(probs_[-1])

        return ll



if __name__ == '__main__':
    x = CHMM(vocab_size=5, eos_token_id='\n', clones_per_token=[2 if i%4==0 else 1 for i in range(5)])
    print(x.beta, [2 if i%4==0 else 1 for i in range(5)])
    print(x.clone_to_token)
    print(x.clones_per_token)
    print(x.alpha_exp)
    print(x.gamma)