"""PyTorch CHMM implementation specialized for large-vocabulary distillation.

This implementation is intentionally simple and explicit:

- Hidden states are "clones" of observed tokens.
- The model stores a next-token distribution for every clone/state.
- For tokens with more than one clone, a small router distributes the
  next-token probability mass across that token's clones.

This factorization is equivalent to a standard CHMM transition matrix but is a
better fit for text distillation, where the vocabulary is very large and only a
small subset of tokens need more than one clone.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn


NEG_INF = -1.0e9


def _dtype_from_string(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype '{dtype_name}'.")
    return mapping[dtype_name]


def rank_tokens_from_counts(
    token_counts: torch.Tensor,
    protected_token_ids: Iterable[int] | None = None,
) -> torch.Tensor:
    """Return token ids sorted by descending empirical frequency."""
    scores = token_counts.clone()
    if protected_token_ids is not None:
        protected = torch.tensor(
            list(protected_token_ids),
            dtype=torch.long,
            device=scores.device,
        )
        if protected.numel() > 0:
            scores[protected] = -1
    return torch.argsort(scores, descending=True)


def build_clone_schedule(
    vocab_size: int,
    ranked_token_ids: Sequence[int] | torch.Tensor,
    four_clone_tokens: int = 64,
    two_clone_tokens: int = 256,
    protected_token_ids: Iterable[int] | None = None,
) -> torch.Tensor:
    """Build a configurable clone schedule from token-frequency ranks."""
    clones = torch.ones(vocab_size, dtype=torch.long)

    ranked = torch.as_tensor(ranked_token_ids, dtype=torch.long)
    if protected_token_ids is not None:
        protected = set(int(x) for x in protected_token_ids)
        ranked = torch.tensor(
            [int(x) for x in ranked.tolist() if int(x) not in protected],
            dtype=torch.long,
        )

    if four_clone_tokens > 0:
        clones[ranked[:four_clone_tokens]] = 4
    if two_clone_tokens > 0:
        start = four_clone_tokens
        end = four_clone_tokens + two_clone_tokens
        clones[ranked[start:end]] = 2

    return clones


class CHMM(nn.Module):
    """Clone HMM for large-vocabulary text sequences.

    Parameters are stored in a factorized form:

    - `token_log_probs[s, x] = log P(x_{t+1}=x | z_t=s)`
    - `clone_log_probs[s, m, c] = log P(clone=c | x_{t+1}=multi_token_m, z_t=s)`

    For single-clone tokens the router is implicit.
    """

    def __init__(
        self,
        vocab_size: int,
        eos_token_id: int,
        clones_per_token: torch.Tensor,
        storage_dtype: str = "bfloat16",
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()

        device = torch.device(device)
        clones_per_token = torch.as_tensor(clones_per_token, dtype=torch.long, device=device)
        if clones_per_token.ndim != 1:
            raise ValueError("clones_per_token must be a 1D tensor.")
        if len(clones_per_token) != vocab_size:
            raise ValueError("clones_per_token must have length vocab_size.")
        if int(clones_per_token[eos_token_id].item()) != 1:
            raise ValueError("eos_token_id must keep exactly one clone.")

        self.vocab_size = int(vocab_size)
        self.eos_token_id = int(eos_token_id)
        self.storage_dtype_name = storage_dtype
        self.storage_dtype = _dtype_from_string(storage_dtype)

        token_offsets = torch.zeros(self.vocab_size + 1, dtype=torch.long, device=device)
        token_offsets[1:] = torch.cumsum(clones_per_token, dim=0)
        hidden_states = int(token_offsets[-1].item())
        max_clones = int(clones_per_token.max().item())

        state_to_token = torch.repeat_interleave(
            torch.arange(self.vocab_size, dtype=torch.long, device=device),
            clones_per_token,
        )

        state_to_clone = torch.empty(hidden_states, dtype=torch.long, device=device)
        for token_id in range(self.vocab_size):
            start = int(token_offsets[token_id].item())
            end = int(token_offsets[token_id + 1].item())
            state_to_clone[start:end] = torch.arange(end - start, dtype=torch.long, device=device)

        multi_token_ids = torch.nonzero(clones_per_token > 1, as_tuple=False).squeeze(-1)
        router_token_to_slot = torch.full((self.vocab_size,), -1, dtype=torch.long, device=device)
        if multi_token_ids.numel() > 0:
            router_token_to_slot[multi_token_ids] = torch.arange(
                multi_token_ids.numel(),
                dtype=torch.long,
                device=device,
            )
        multi_token_clone_counts = clones_per_token[multi_token_ids]

        clone_valid_mask = (
            torch.arange(max_clones, dtype=torch.long, device=device)[None, :]
            < multi_token_clone_counts[:, None]
            if multi_token_ids.numel() > 0
            else torch.zeros((0, max_clones), dtype=torch.bool, device=device)
        )

        self.hidden_states = hidden_states
        self.max_clones = max_clones
        self.num_multi_tokens = int(multi_token_ids.numel())

        self.register_buffer("clones_per_token", clones_per_token.to(device))
        self.register_buffer("token_offsets", token_offsets.to(device))
        self.register_buffer("state_to_token", state_to_token.to(device))
        self.register_buffer("state_to_clone", state_to_clone.to(device))
        self.register_buffer("multi_token_ids", multi_token_ids.to(device))
        self.register_buffer("multi_token_clone_counts", multi_token_clone_counts.to(device))
        self.register_buffer("router_token_to_slot", router_token_to_slot.to(device))
        self.register_buffer("clone_valid_mask", clone_valid_mask.to(device))

        token_log_probs = torch.full(
            (hidden_states, self.vocab_size),
            -math.log(self.vocab_size),
            dtype=self.storage_dtype,
            device=device,
        )
        clone_log_probs = torch.full(
            (hidden_states, max(1, self.num_multi_tokens), max_clones),
            NEG_INF,
            dtype=self.storage_dtype,
            device=device,
        )
        if self.num_multi_tokens > 0:
            uniform = torch.zeros(
                (hidden_states, self.num_multi_tokens, max_clones),
                dtype=torch.float32,
                device=device,
            )
            for slot, num_clones in enumerate(self.multi_token_clone_counts.tolist()):
                uniform[:, slot, :num_clones] = -math.log(num_clones)
            uniform = uniform.masked_fill(~self.clone_valid_mask[None, :, :], NEG_INF)
            clone_log_probs[:, : self.num_multi_tokens].copy_(uniform.to(self.storage_dtype))

        initial_log_probs = torch.full(
            (hidden_states,),
            -math.log(hidden_states),
            dtype=torch.float32,
            device=device,
        )

        self.register_buffer("token_log_probs", token_log_probs)
        self.register_buffer("clone_log_probs", clone_log_probs)
        self.register_buffer("initial_log_probs", initial_log_probs)

    @property
    def device(self) -> torch.device:
        return self.token_log_probs.device

    def config_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "eos_token_id": self.eos_token_id,
            "storage_dtype": self.storage_dtype_name,
        }

    @torch.no_grad()
    def save_pretrained(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "config": self.config_dict(),
            "clones_per_token": self.clones_per_token.cpu(),
            "token_log_probs": self.token_log_probs.cpu(),
            "clone_log_probs": self.clone_log_probs.cpu(),
            "initial_log_probs": self.initial_log_probs.cpu(),
        }
        torch.save(payload, output_dir / "model.pt")

        with open(output_dir / "config.json", "w", encoding="utf-8") as fout:
            json.dump(payload["config"], fout, indent=2)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        map_location: str | torch.device | None = None,
    ) -> "CHMM":
        model_path = Path(model_path)
        payload = torch.load(model_path / "model.pt", map_location="cpu", weights_only=True)

        model = cls(
            vocab_size=payload["config"]["vocab_size"],
            eos_token_id=payload["config"]["eos_token_id"],
            clones_per_token=payload["clones_per_token"],
            storage_dtype=payload["config"]["storage_dtype"],
            device="cpu",
        )
        model.token_log_probs.copy_(payload["token_log_probs"].to(model.token_log_probs.device))
        model.clone_log_probs.copy_(payload["clone_log_probs"].to(model.clone_log_probs.device))
        model.initial_log_probs.copy_(payload["initial_log_probs"].to(model.initial_log_probs.device))
        if map_location is not None:
            model = model.to(map_location)
        return model

    def empty_count_buffers(
        self,
        count_dtype: torch.dtype = torch.float32,
        device: str | torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.device if device is None else torch.device(device)
        token_counts = torch.zeros(
            (self.hidden_states, self.vocab_size),
            dtype=count_dtype,
            device=device,
        )
        clone_counts = torch.zeros(
            (self.hidden_states, max(1, self.num_multi_tokens), self.max_clones),
            dtype=count_dtype,
            device=device,
        )
        initial_counts = torch.zeros(
            (self.hidden_states,),
            dtype=count_dtype,
            device=device,
        )
        return token_counts, clone_counts, initial_counts

    def _state_views(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        clone_axis = torch.arange(self.max_clones, device=input_ids.device)
        starts = self.token_offsets[input_ids]
        clone_counts = self.clones_per_token[input_ids]
        state_ids = starts.unsqueeze(-1) + clone_axis.view(1, 1, -1)
        state_ids = state_ids.clamp(max=self.hidden_states - 1)
        state_mask = clone_axis.view(1, 1, -1) < clone_counts.unsqueeze(-1)
        return state_ids, state_mask

    def _gather_token_log_probs(
        self,
        source_states: torch.Tensor,
        next_tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_source = source_states.shape
        flat_states = source_states.reshape(-1)
        flat_tokens = next_tokens[:, None].expand(-1, num_source).reshape(-1)
        values = self.token_log_probs[flat_states, flat_tokens]
        return values.reshape(batch_size, num_source).float()

    def _gather_clone_log_probs(
        self,
        source_states: torch.Tensor,
        next_tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_source = source_states.shape
        device = source_states.device

        clone_log = torch.full(
            (batch_size, num_source, self.max_clones),
            NEG_INF,
            dtype=torch.float32,
            device=device,
        )

        slots = self.router_token_to_slot[next_tokens]
        single = slots < 0
        if single.any():
            clone_log[single, :, 0] = 0.0

        multi = slots >= 0
        if multi.any():
            multi_states = source_states[multi]
            multi_slots = slots[multi]
            flat_states = multi_states.reshape(-1)
            flat_slots = multi_slots[:, None].expand(-1, num_source).reshape(-1)
            gathered = self.clone_log_probs[flat_states, flat_slots]
            clone_log[multi] = gathered.reshape(-1, num_source, self.max_clones).float()

        return clone_log

    def _transition_cache(
        self,
        state_ids: torch.Tensor,
        state_mask: torch.Tensor,
        t: int,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source_states = state_ids[:, t]
        source_mask = state_mask[:, t]
        dest_mask = state_mask[:, t + 1]
        next_tokens = input_ids[:, t + 1]

        token_log = self._gather_token_log_probs(source_states, next_tokens)
        clone_log = self._gather_clone_log_probs(source_states, next_tokens)

        token_log = token_log.masked_fill(~source_mask, NEG_INF)
        clone_log = clone_log.masked_fill(~source_mask[:, :, None], NEG_INF)
        clone_log = clone_log.masked_fill(~dest_mask[:, None, :], NEG_INF)
        return token_log, clone_log, dest_mask

    @torch.no_grad()
    def forward_backward(
        self,
        input_ids: torch.Tensor,
        return_messages: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
        """Compute batch log-likelihoods and optionally return messages."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be a 2D tensor.")
        if input_ids.shape[1] < 1:
            raise ValueError("Sequences must have length >= 1.")

        input_ids = input_ids.to(self.device, non_blocking=True)
        batch_size, seq_len = input_ids.shape

        state_ids, state_mask = self._state_views(input_ids)

        alpha = torch.full(
            (seq_len, batch_size, self.max_clones),
            NEG_INF,
            dtype=torch.float32,
            device=self.device,
        )
        beta = torch.full_like(alpha, NEG_INF)

        first_states = state_ids[:, 0]
        first_mask = state_mask[:, 0]
        log_pi = self.initial_log_probs[first_states].float()
        log_pi = log_pi.masked_fill(~first_mask, NEG_INF)

        init_norm = torch.logsumexp(log_pi, dim=-1, keepdim=True)
        alpha[0] = log_pi - init_norm
        log_likelihoods = init_norm.squeeze(-1)

        transition_cache: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for t in range(seq_len - 1):
            token_log, clone_log, dest_mask = self._transition_cache(state_ids, state_mask, t, input_ids)
            transition_cache.append((token_log, clone_log, dest_mask))

            logits = alpha[t][:, :, None] + token_log[:, :, None] + clone_log
            logits = logits.masked_fill(~state_mask[:, t][:, :, None], NEG_INF)
            logits = logits.masked_fill(~dest_mask[:, None, :], NEG_INF)

            alpha_t = torch.logsumexp(logits, dim=1)
            alpha_t = alpha_t.masked_fill(~dest_mask, NEG_INF)
            step_norm = torch.logsumexp(alpha_t, dim=-1, keepdim=True)
            alpha[t + 1] = alpha_t - step_norm
            log_likelihoods += step_norm.squeeze(-1)

        beta[-1] = torch.where(state_mask[:, -1], torch.zeros_like(beta[-1]), torch.full_like(beta[-1], NEG_INF))
        for t in range(seq_len - 2, -1, -1):
            token_log, clone_log, dest_mask = transition_cache[t]

            logits = token_log[:, :, None] + clone_log + beta[t + 1][:, None, :]
            logits = logits.masked_fill(~state_mask[:, t][:, :, None], NEG_INF)
            logits = logits.masked_fill(~dest_mask[:, None, :], NEG_INF)

            beta_t = torch.logsumexp(logits, dim=-1)
            beta_t = beta_t.masked_fill(~state_mask[:, t], NEG_INF)
            beta_norm = torch.logsumexp(beta_t, dim=-1, keepdim=True)
            beta[t] = beta_t - beta_norm

        if not return_messages:
            return log_likelihoods, None

        return log_likelihoods, (alpha, beta, state_ids, state_mask, transition_cache)

    @torch.no_grad()
    def accumulate_expected_counts(
        self,
        input_ids: torch.Tensor,
        token_counts: torch.Tensor,
        clone_counts: torch.Tensor,
        initial_counts: torch.Tensor,
    ) -> torch.Tensor:
        """Run forward-backward and accumulate one EM E-step worth of counts."""
        input_ids = input_ids.to(self.device, non_blocking=True)
        log_likelihoods, cache = self.forward_backward(input_ids, return_messages=True)
        assert cache is not None
        alpha, beta, state_ids, state_mask, transition_cache = cache

        initial_logits = alpha[0] + beta[0]
        initial_logits = initial_logits - torch.logsumexp(initial_logits, dim=-1, keepdim=True)
        initial_probs = initial_logits.exp()

        initial_state_ids = state_ids[:, 0][state_mask[:, 0]]
        initial_values = initial_probs[state_mask[:, 0]]
        initial_counts.index_add_(0, initial_state_ids, initial_values)

        flat_token_counts = token_counts.view(-1)
        flat_clone_counts = clone_counts.view(-1)

        for t in range(input_ids.shape[1] - 1):
            token_log, clone_log, dest_mask = transition_cache[t]
            xi_logits = alpha[t][:, :, None] + token_log[:, :, None] + clone_log + beta[t + 1][:, None, :]
            valid = state_mask[:, t][:, :, None] & dest_mask[:, None, :]
            xi_logits = xi_logits.masked_fill(~valid, NEG_INF)
            xi_logits = xi_logits - torch.logsumexp(xi_logits.reshape(xi_logits.shape[0], -1), dim=-1).view(-1, 1, 1)
            xi = torch.exp(xi_logits).masked_fill(~valid, 0.0)

            token_post = xi.sum(dim=-1)
            source_states = state_ids[:, t]
            next_tokens = input_ids[:, t + 1]
            src_valid = state_mask[:, t]

            src_flat = source_states[src_valid]
            tok_flat = next_tokens[:, None].expand_as(token_post)[src_valid]
            val_flat = token_post[src_valid]
            token_indices = src_flat.long() * self.vocab_size + tok_flat.long()
            flat_token_counts.index_add_(0, token_indices, val_flat)

            if self.num_multi_tokens == 0:
                continue

            router_slots = self.router_token_to_slot[next_tokens]
            multi_rows = router_slots >= 0
            if not multi_rows.any():
                continue

            xi_multi = xi[multi_rows]
            src_multi = source_states[multi_rows]
            src_valid_multi = src_valid[multi_rows]
            dst_valid_multi = dest_mask[multi_rows]
            slot_multi = router_slots[multi_rows]

            clone_axis = torch.arange(self.max_clones, device=self.device)
            valid_multi = src_valid_multi[:, :, None] & dst_valid_multi[:, None, :]

            src_idx = src_multi[:, :, None].expand(-1, -1, self.max_clones)[valid_multi]
            slot_idx = slot_multi[:, None, None].expand(-1, src_multi.shape[1], self.max_clones)[valid_multi]
            clone_idx = clone_axis.view(1, 1, -1).expand(xi_multi.shape[0], src_multi.shape[1], -1)[valid_multi]
            clone_val = xi_multi[valid_multi]

            flat_idx = (
                src_idx.long() * clone_counts.shape[1] * self.max_clones
                + slot_idx.long() * self.max_clones
                + clone_idx.long()
            )
            flat_clone_counts.index_add_(0, flat_idx, clone_val)

        return log_likelihoods

    @torch.no_grad()
    def accumulate_hard_counts(
        self,
        input_ids: torch.Tensor,
        token_counts: torch.Tensor,
        clone_counts: torch.Tensor,
        initial_counts: torch.Tensor,
        context_mode: str = "both",
    ) -> None:
        """Initialize counts using deterministic context hashing."""
        if context_mode not in {"prev", "next", "both"}:
            raise ValueError("context_mode must be one of: prev, next, both.")

        input_ids = input_ids.to(self.device, non_blocking=True)
        batch_size, seq_len = input_ids.shape

        prev_tokens = torch.full_like(input_ids, self.eos_token_id)
        prev_tokens[:, 1:] = input_ids[:, :-1]

        next_tokens = torch.full_like(input_ids, self.eos_token_id)
        next_tokens[:, :-1] = input_ids[:, 1:]

        if context_mode == "prev":
            context_key = prev_tokens
        elif context_mode == "next":
            context_key = next_tokens
        else:
            context_key = prev_tokens * 1009 + next_tokens * 9176

        clone_counts_per_pos = self.clones_per_token[input_ids]
        assigned_clones = torch.remainder(context_key, clone_counts_per_pos.clamp_min(1))
        state_ids = self.token_offsets[input_ids] + assigned_clones

        initial_counts.index_add_(0, state_ids[:, 0].reshape(-1), torch.ones(batch_size, device=self.device))

        flat_token_counts = token_counts.view(-1)
        flat_clone_counts = clone_counts.view(-1)

        source_states = state_ids[:, :-1].reshape(-1)
        target_tokens = input_ids[:, 1:].reshape(-1)
        token_indices = source_states.long() * self.vocab_size + target_tokens.long()
        flat_token_counts.index_add_(0, token_indices, torch.ones_like(source_states, dtype=token_counts.dtype))

        if self.num_multi_tokens == 0:
            return

        target_slots = self.router_token_to_slot[target_tokens]
        multi_mask = target_slots >= 0
        if not multi_mask.any():
            return

        target_clones = assigned_clones[:, 1:].reshape(-1)[multi_mask]
        source_states = source_states[multi_mask]
        target_slots = target_slots[multi_mask]
        flat_idx = (
            source_states.long() * clone_counts.shape[1] * self.max_clones
            + target_slots.long() * self.max_clones
            + target_clones.long()
        )
        flat_clone_counts.index_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=clone_counts.dtype))

    @torch.no_grad()
    def update_from_counts(
        self,
        token_counts: torch.Tensor,
        clone_counts: torch.Tensor,
        initial_counts: torch.Tensor,
        pseudocount: float = 1.0e-3,
        clone_pseudocount: float | None = None,
        initial_pseudocount: float | None = None,
        row_chunk_size: int = 512,
    ) -> None:
        """Normalize expected counts into log-probability tables."""
        clone_pseudocount = pseudocount if clone_pseudocount is None else clone_pseudocount
        initial_pseudocount = pseudocount if initial_pseudocount is None else initial_pseudocount

        for start in range(0, self.hidden_states, row_chunk_size):
            end = min(self.hidden_states, start + row_chunk_size)
            block = token_counts[start:end].float()
            block = block + pseudocount
            denom = block.sum(dim=-1, keepdim=True)
            log_block = block.log() - denom.log()
            self.token_log_probs[start:end].copy_(log_block.to(self.storage_dtype))

        if self.num_multi_tokens > 0:
            valid_mask = self.clone_valid_mask.to(clone_counts.device)
            for start in range(0, self.hidden_states, row_chunk_size):
                end = min(self.hidden_states, start + row_chunk_size)
                block = clone_counts[start:end].float()
                block = block + clone_pseudocount * valid_mask[None, :, :].float()
                block = block.masked_fill(~valid_mask[None, :, :], 0.0)
                denom = block.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
                probs = block / denom
                log_block = torch.full_like(probs, NEG_INF)
                log_block[valid_mask[None, :, :].expand_as(probs)] = probs[valid_mask[None, :, :].expand_as(probs)].clamp_min(1.0e-12).log()
                self.clone_log_probs[start:end, : self.num_multi_tokens].copy_(log_block.to(self.storage_dtype))

        init_block = initial_counts.float() + initial_pseudocount
        init_block = init_block / init_block.sum().clamp_min(1.0e-12)
        self.initial_log_probs.copy_(init_block.clamp_min(1.0e-12).log())

    @torch.no_grad()
    def loglikelihood(self, input_ids: torch.Tensor, batch_size: int) -> torch.Tensor:
        total = torch.zeros(1, dtype=torch.float32, device=self.device)
        for start in range(0, input_ids.shape[0], batch_size):
            batch = input_ids[start : start + batch_size]
            ll, _ = self.forward_backward(batch, return_messages=False)
            total += ll.sum()
        return total
