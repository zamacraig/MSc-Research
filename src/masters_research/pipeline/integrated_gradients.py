"""Integrated Gradients for the node-level graph models.

Why not ``captum.attr.IntegratedGradients`` directly
----------------------------------------------------
Captum evaluates the model once on *all* ``n_steps`` interpolated inputs stacked
along dimension 0, i.e. on a ``[n_steps * num_nodes, F]`` tensor. The graph
models take a single fixed snapshot whose ``edge_index`` refers to ``num_nodes``
nodes, so a stacked input is not a valid graph:

* ``TemporalAttentionGNN`` concatenates its ``[history, num_nodes, H]`` buffer
  with the current features, which raises a size mismatch (8 vs 400 rows).
* ``SimpleGCN`` / ``TemporalGCN_GRU`` do not raise, but only the first ``num_nodes``
  rows receive messages and the returned single-row output is paired with the
  whole stacked batch, so the attributions are not a valid IG estimate.

Here every interpolation step is a separate forward pass on a genuine
``[num_nodes, F]`` snapshot, so the graph is always well formed.

Recurrent state
---------------
The attention model keeps a history buffer that every forward pass appends to.
It is frozen to its state at call time before each step and restored afterwards,
so the 50 interpolation steps do not pollute the buffer and every step is
explained against the same temporal context.
"""
from __future__ import annotations

import torch


def _log_probs(model, x, edge_index, edge_weight):
    out = model(x, edge_index, edge_weight)
    return out[0] if isinstance(out, tuple) else out


def integrated_gradients(
    model,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight,
    node_idx: int,
    target_class: int,
    n_steps: int = 50,
    baseline: torch.Tensor | None = None,
):
    """Integrated Gradients of ``log p(target_class)`` at ``node_idx``.

    Uses the midpoint Riemann sum with a zero baseline (as Captum does by
    default). Returns ``(attributions, delta)`` where ``attributions`` has the
    shape of ``x`` and ``delta`` is the completeness residual
    ``sum(attr) - (f(x) - f(baseline))``; a value near zero indicates the
    integral was approximated well.
    """
    x = x.detach()
    baseline = torch.zeros_like(x) if baseline is None else baseline.detach()
    diff = x - baseline

    has_history = hasattr(model, "_history")
    saved_history = None
    if has_history and model._history is not None:
        saved_history = model._history.clone()

    def _reset_state():
        if has_history:
            model._history = None if saved_history is None else saved_history.clone()

    total_grad = torch.zeros_like(x)
    try:
        for k in range(n_steps):
            alpha = (k + 0.5) / n_steps
            xi = (baseline + alpha * diff).clone().requires_grad_(True)
            _reset_state()
            score = _log_probs(model, xi, edge_index, edge_weight)[node_idx, target_class]
            grad = torch.autograd.grad(score, xi)[0]
            total_grad += grad.detach()

        attributions = diff * total_grad / n_steps

        with torch.no_grad():
            _reset_state()
            f_x = _log_probs(model, x, edge_index, edge_weight)[node_idx, target_class]
            _reset_state()
            f_b = _log_probs(model, baseline, edge_index, edge_weight)[node_idx, target_class]
        delta = float(attributions.sum() - (f_x - f_b))
    finally:
        if has_history:
            model._history = None if saved_history is None else saved_history.clone()

    return attributions.detach(), delta
