from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import torch


def get_trainable_parameters(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def grads_to_vector(grads: Sequence[torch.Tensor | None], params: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    pieces = []
    for grad, parameter in zip(grads, params):
        if grad is None:
            pieces.append(torch.zeros_like(parameter, memory_format=torch.contiguous_format).reshape(-1))
        else:
            pieces.append(grad.reshape(-1))
    if not pieces:
        return torch.tensor([], device=params[0].device if params else "cpu")
    return torch.cat(pieces)


def vector_to_grads(vector: torch.Tensor, params: Sequence[torch.nn.Parameter]) -> List[torch.Tensor]:
    grads = []
    offset = 0
    for parameter in params:
        numel = parameter.numel()
        grads.append(vector[offset: offset + numel].view_as(parameter))
        offset += numel
    return grads


def assign_gradient_vector(params: Sequence[torch.nn.Parameter], grad_vector: torch.Tensor) -> None:
    offset = 0
    for parameter in params:
        numel = parameter.numel()
        grad = grad_vector[offset: offset + numel].view_as(parameter)
        if parameter.grad is None:
            parameter.grad = grad.clone()
        else:
            parameter.grad.copy_(grad)
        offset += numel


def gradient_from_loss(loss: torch.Tensor, params: Sequence[torch.nn.Parameter], retain_graph: bool) -> torch.Tensor:
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    return grads_to_vector(grads, params)


def project_conflicting_gradient(
    source: torch.Tensor,
    reference: torch.Tensor,
    eps: float = 1e-12,
    conflict_threshold: float = 0.0,
) -> tuple[torch.Tensor, float, float, bool]:
    dot = torch.dot(source, reference)
    ref_norm_sq = torch.dot(reference, reference)
    cosine = dot / (source.norm() * reference.norm() + eps)
    surgery_applied = bool(dot < 0 and cosine < conflict_threshold and ref_norm_sq > eps)
    if surgery_applied:
        source = source - (dot / ref_norm_sq) * reference
    return source, float(dot.item()), float(cosine.item()), surgery_applied


def aggregate_gradients(class_gradients: Dict[int, torch.Tensor], classes: Iterable[int]) -> torch.Tensor | None:
    selected = [class_gradients[class_id] for class_id in classes if class_id in class_gradients]
    if not selected:
        return None
    return torch.stack(selected, dim=0).mean(dim=0)


def compute_group_cgsil_gradient(
    per_class_losses: Dict[int, torch.Tensor],
    params: Sequence[torch.nn.Parameter],
    tail_classes: Sequence[int],
    beta: float,
    conflict_threshold: float = 0.0,
) -> dict:
    class_ids = list(per_class_losses.keys())
    class_gradients = {}
    for index, class_id in enumerate(class_ids):
        retain_graph = index < len(class_ids) - 1
        class_gradients[class_id] = gradient_from_loss(per_class_losses[class_id], params, retain_graph=retain_graph)

    tail_class_set = set(int(class_id) for class_id in tail_classes)
    tail_present = [class_id for class_id in class_ids if class_id in tail_class_set]
    head_present = [class_id for class_id in class_ids if class_id not in tail_class_set]

    g_tail = aggregate_gradients(class_gradients, tail_present)
    g_head = aggregate_gradients(class_gradients, head_present)

    dot = 0.0
    cosine = 0.0
    surgery_applied = False
    if g_tail is None and g_head is None:
        raise ValueError("No class gradients were computed for CGSIL.")
    if g_tail is None:
        combined = g_head
    elif g_head is None:
        combined = g_tail
    else:
        g_head, dot, cosine, surgery_applied = project_conflicting_gradient(
            g_head,
            g_tail,
            conflict_threshold=conflict_threshold,
        )
        combined = beta * g_tail + (1.0 - beta) * g_head

    return {
        "gradient": combined,
        "class_gradients": class_gradients,
        "head_classes_present": head_present,
        "tail_classes_present": tail_present,
        "dot": dot,
        "cosine": cosine,
        "surgery_applied": surgery_applied,
    }
