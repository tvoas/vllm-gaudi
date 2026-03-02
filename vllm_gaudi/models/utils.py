import torch
import torch.nn as nn
from vllm.multimodal import NestedTensors
from vllm.model_executor.models import utils
from vllm.model_executor.models.utils import (_embedding_count_expression, _flatten_embeddings)


# TODO: Replaced masked_scatter with torch.where to avoid HPU performance issues
# with non_zero_i8 ops in TPC kernel. However, torch.where creates dynamic operations
# causing recompilation on each run. Need to find a static operation alternative.
def _merge_multimodal_embeddings(
    inputs_embeds: torch.Tensor,
    multimodal_embeddings: NestedTensors,
    is_multimodal: torch.Tensor,
) -> torch.Tensor:
    """
    Merge ``multimodal_embeddings`` into ``inputs_embeds`` by overwriting the
    positions in ``inputs_embeds`` corresponding to placeholder tokens in
    ``input_ids``.

    Note:
        This updates ``inputs_embeds`` in place.
    """
    if len(multimodal_embeddings) == 0:
        return inputs_embeds

    import habana_frameworks.torch.core as htcore
    htcore.mark_step()

    mm_embeds_flat = _flatten_embeddings(multimodal_embeddings)
    input_dtype = inputs_embeds.dtype

    if inputs_embeds.ndim == 3 and mm_embeds_flat.ndim == 2:
        original_shape = inputs_embeds.shape
        inputs_embeds = inputs_embeds.view(-1, inputs_embeds.shape[-1])
        inputs_embeds.index_copy_(0, is_multimodal, mm_embeds_flat)
        return inputs_embeds.view(original_shape)
    if is_multimodal.dtype != torch.bool:
        return inputs_embeds.index_copy_(0, is_multimodal, mm_embeds_flat)
    try:
        # For debugging
        # inputs_embeds[is_multimodal] = mm_embeds_flat.to(dtype=input_dtype)

        # NOTE: This can avoid D2H sync (#22105), but fails to
        # raise an error if is_multimodal.sum() < len(mm_embeds_flat)
        inputs_embeds.masked_scatter_(is_multimodal.unsqueeze(-1), mm_embeds_flat.to(dtype=input_dtype))
    except RuntimeError as e:
        num_actual_tokens = len(mm_embeds_flat)
        num_expected_tokens = is_multimodal.sum().item()

        if num_actual_tokens != num_expected_tokens:
            expr = _embedding_count_expression(multimodal_embeddings)

            raise ValueError(f"Attempted to assign {expr} = {num_actual_tokens} "
                             f"multimodal tokens to {num_expected_tokens} placeholders") from e

        raise ValueError("Error during masked scatter operation") from e

    return inputs_embeds


def _add_dynamo_graph_break_wrapper(module: nn.Module):
    """Wraps a module's forward method to trigger a graph break before execution."""
    if getattr(module, "_hpu_graph_breaks_installed", False):
        return

    orig_forward = module.forward

    def wrapped_forward(*args, **kwargs):
        torch._dynamo.graph_break()
        return orig_forward(*args, **kwargs)

    module.forward = wrapped_forward
    module._hpu_graph_breaks_installed = True

def apply_universal_moe_graph_breaks(model: nn.Module) -> int:
    """
    Recursively scans a model for standard LLM Decoder layers.
    If the layer's MLP is determined to be an MoE, it applies graph breaks 
    to the bounding normalization layers to drastically reduce compile times.
    """
    upgraded = 0
    for _, module in model.named_modules():
        # Identify standard Decoder layers by their core components
        if (hasattr(module, "input_layernorm") and 
            hasattr(module, "post_attention_layernorm") and 
            hasattr(module, "mlp")):
            
            mlp = module.mlp
            
            # Heuristic to detect MoE structures in vLLM (Qwen, Mixtral, Deepseek, etc.)
            is_moe = (
                hasattr(mlp, "experts") or 
                hasattr(mlp, "gate") or 
                "moe" in type(mlp).__name__.lower()
            )

            if is_moe:
                _add_dynamo_graph_break_wrapper(module.post_attention_layernorm)
                upgraded += 1
                
    return upgraded


utils._merge_multimodal_embeddings = _merge_multimodal_embeddings
