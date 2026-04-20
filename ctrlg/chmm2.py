import os

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin


def matmul(A, B):
    return torch.matmul(A, B)


def ib_ib_bj_to_ij(pf, pp, cp):
    ll = torch.amax(cp, dim=-1)
    pp = torch.exp(pp - ll[None, :])
    cp = torch.exp(cp - ll[:, None])

    ratio = pf / pp
    ratio[pp == 0.0] = 0.0
    af = torch.matmul(ratio, cp)

    return af


class CHMM(nn.Module, PyTorchModelHubMixin):
    def __init__(
            self,
            vocab_size: int,
            eos_token_id: int,
            clones_per_token: list[int],
        ):
        super().__init__()

        assert len(clones_per_token) == vocab_size, "One value per state for num clones"

        hidden_states = int(sum(clones_per_token))

        alpha_exp = torch.softmax(torch.randn(hidden_states, hidden_states), dim=1)
        gamma = torch.log_softmax(torch.randn(hidden_states), dim=0)

        self.alpha_exp = nn.Parameter(alpha_exp, requires_grad=False)
        self.gamma = nn.Parameter(gamma, requires_grad=False)

        # Fixed deterministic emission matrix:
        # each hidden state can emit exactly one token.
        self.register_buffer(
            "beta",
            self._build_deterministic_beta(
                clones_per_token=clones_per_token,
                vocab_size=vocab_size,
                dtype=alpha_exp.dtype,
            ),
        )
        self.register_buffer(
            "clones_per_token",
            torch.tensor(clones_per_token, dtype=torch.long),
        )
        self.register_buffer(
            "clone_to_token",
            torch.repeat_interleave(
                torch.arange(vocab_size, dtype=torch.long),
                torch.tensor(clones_per_token, dtype=torch.long),
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
            beta[start:end, token_id] = 0.0  # log(1)
            start = end

        return beta

    def forward(self, input_ids):
        device = self.alpha_exp.device
        alpha_exp, beta, gamma_exp = self.alpha_exp, self.beta, torch.softmax(self.gamma, dim=0)
        hidden_states, vocab_size, eos_token_id = self.hidden_states, self.vocab_size, self.eos_token_id
        batch_size, seq_len = input_ids.shape

        input_ids_ = torch.permute(input_ids, (1, 0)).contiguous()
        lookup_ids = input_ids_.clamp_min(0)  # avoid -1 indexing

        input_probs = beta[
            torch.arange(0, hidden_states, device=device)[None, :, None],
            lookup_ids[:, None, :],
        ].contiguous()  # seq_len * hidden_states * batch_size

        # Important change:
        # do NOT multiply by 0, because invalid entries can be -inf in CHMM.
        observed = (input_ids_ != -1)[:, None, :].expand(-1, hidden_states, -1)
        input_probs = torch.where(observed, input_probs, torch.zeros_like(input_probs))

        ys = []
        y = torch.zeros((hidden_states, batch_size), device=device)
        for t in range(seq_len - 1, -1, -1):
            if t != seq_len - 1:
                y_max = torch.amax(y, dim=0, keepdim=True)
                y = torch.exp(y - y_max)
                y = matmul(alpha_exp, y)
                y = torch.log(y) + y_max
            y += input_probs[t, :, :]  # hidden_states * batch_size
            ys.append(y)

        y_max = torch.amax(y, dim=0)
        y = torch.exp(y - y_max.unsqueeze(0))
        y = matmul(gamma_exp.unsqueeze(0), y).squeeze()
        y = torch.log(y) + y_max

        ys.append(y)

        return ys

    # top-down circuit pass
    def backward(self, input_ids, probs, alpha_flow, beta_flow, gamma_flow):
        device = self.alpha_exp.device
        alpha_exp, beta, gamma_exp = self.alpha_exp, self.beta, torch.softmax(self.gamma, dim=0)
        hidden_states, vocab_size, eos_token_id = self.hidden_states, self.vocab_size, self.eos_token_id
        batch_size, seq_len = input_ids.shape
        neg_inf = torch.tensor(float("-inf"), device=device, dtype=beta.dtype)

        input_ids_ = torch.permute(input_ids, (1, 0)).contiguous()  # seq_len * batch_size
        lookup_ids = input_ids_.clamp_min(0)

        # Raw CHMM emission scores:
        #   0     for compatible clone/token pairs
        #   -inf  for incompatible clone/token pairs
        raw_input_probs = beta[
            torch.arange(0, hidden_states, device=device)[None, :, None],
            lookup_ids[:, None, :],
        ].contiguous()  # seq_len * hidden_states * batch_size

        observed = (input_ids_ != -1)[:, None, :].expand(-1, hidden_states, -1)

        # For missing tokens, emission contribution should be neutral (log 1 = 0)
        input_probs = torch.where(observed, raw_input_probs, torch.zeros_like(raw_input_probs))

        # Support mask for safe subtraction:
        # - if token is observed: only finite-emission clones are supported
        # - if token is missing: all clones are supported
        supported = (~observed) | torch.isfinite(raw_input_probs)

        flows = []
        pf = gamma_exp.unsqueeze(0) * torch.exp(
            torch.permute(probs[-2], (1, 0)).contiguous() - probs[-1][:, None]
        )  # batch_size * hidden_states
        pf = torch.nan_to_num(pf, nan=0.0, posinf=0.0, neginf=0.0)
        flows.append(pf)

        # update gamma_flow
        gamma_flow.add_(torch.sum(pf, dim=0))

        for t in range(0, seq_len - 1):
            layer_idx = seq_len - t - 1
            cp = probs[layer_idx - 1]  # hidden_states * batch_size

            # IMPORTANT PATCH:
            # only subtract the emission term on supported clones.
            # unsupported clones are forced to -inf directly, instead of computing
            #   -inf - (-inf) -> NaN
            supported_t = supported[t]  # hidden_states * batch_size
            emit_t = input_probs[t, :, :]  # hidden_states * batch_size

            pp = torch.where(supported_t, probs[layer_idx] - emit_t, neg_inf)  # hidden_states * batch_size
            pp = torch.nan_to_num(pp, nan=float("-inf"), posinf=float("inf"), neginf=float("-inf"))

            alpha_flow.add_(
                ib_ib_bj_to_ij(
                    torch.permute(pf, (1, 0)).contiguous(),
                    pp,
                    torch.permute(cp, (1, 0)).contiguous(),
                )
            )

            pp = torch.permute(pp, (1, 0))  # batch_size * hidden_states
            cp = torch.permute(cp, (1, 0))  # batch_size * hidden_states

            pp_max = torch.amax(pp, dim=1, keepdim=True)  # batch_size * 1
            # guard against a pathological all -inf row
            pp_max = torch.where(torch.isfinite(pp_max), pp_max, torch.zeros_like(pp_max))

            pp_ = torch.exp(pp - pp_max)
            pp_ = torch.nan_to_num(pp_, nan=0.0, posinf=0.0, neginf=0.0)

            ratio = pf / pp_
            ratio[pp_ == 0.0] = 0.0
            ratio = torch.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)

            pf = matmul(ratio, alpha_exp) * torch.exp(cp - pp_max)
            pf = torch.nan_to_num(pf, nan=0.0, posinf=0.0, neginf=0.0)

            flows.append(pf)

        # No beta_flow update:
        # CHMM emissions are deterministic and are not learned.

    def loglikelihood(self, input_ids, batch_size):
        device = self.alpha_exp.device
        data_size, seq_len = input_ids.shape

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