#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from omegaconf import OmegaConf

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import read_mode_config
from starVLA.training.trainer_utils.trainer_tools import resize_images


def parse_args():
    parser = argparse.ArgumentParser(description="Sanity check sdpa vs flash_attention_3 on one fixed batch.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt/.safetensors")
    parser.add_argument("--config-yaml", type=str, default="", help="Optional config override instead of checkpoint config")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset-mode", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--dataset-name", type=str, default="", help="Optional subdataset name for exact sample lookup")
    parser.add_argument("--trajectory-id", type=int, default=-1, help="Optional exact trajectory id")
    parser.add_argument("--step", type=int, default=-1, help="Optional exact step within trajectory")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--sdpa-impl", type=str, default="sdpa")
    parser.add_argument("--fa3-impl", type=str, default="flash_attention_3")
    parser.add_argument("--use-accelerator", action="store_true", help="Wrap models with Accelerator before checks")
    parser.add_argument("--hidden-mean-abs-tol", type=float, default=1e-3)
    parser.add_argument("--hidden-cosine-tol", type=float, default=0.999)
    parser.add_argument("--loss-rel-tol", type=float, default=0.05)
    parser.add_argument("--grad-rel-tol", type=float, default=0.2)
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_state_dict(checkpoint_path: str):
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(checkpoint_path))
    return torch.load(str(checkpoint_path), map_location="cpu")


def load_cfg(args):
    if args.config_yaml:
        cfg = OmegaConf.load(args.config_yaml)
    else:
        cfg_dict, _ = read_mode_config(args.checkpoint)
        cfg = OmegaConf.create(cfg_dict)
    return cfg


def build_model(base_cfg, checkpoint_path: str, attn_impl: str, device: str):
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    cfg.framework.qwenvl.attn_implementation = attn_impl
    if hasattr(cfg, "trainer"):
        cfg.trainer.pretrained_checkpoint = None
    model = build_framework(cfg)
    state_dict = load_state_dict(checkpoint_path)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    return model


def load_fixed_examples(cfg, dataset_mode: str, sample_index: int, batch_size: int):
    if cfg.datasets.vla_data.dataset_py != "lerobot_datasets":
        raise ValueError("This sanity script currently supports only lerobot_datasets.")
    dataset = get_vla_dataset(data_cfg=cfg.datasets.vla_data, mode=dataset_mode)
    if len(dataset) == 0:
        raise ValueError(f"Dataset split `{dataset_mode}` is empty.")
    indices = [(sample_index + i) % len(dataset) for i in range(batch_size)]
    examples = [dataset[idx] for idx in indices]
    return examples, indices


def load_exact_example(cfg, dataset_mode: str, dataset_name: str, trajectory_id: int, step: int):
    if cfg.datasets.vla_data.dataset_py != "lerobot_datasets":
        raise ValueError("This sanity script currently supports only lerobot_datasets.")
    mixture = get_vla_dataset(data_cfg=cfg.datasets.vla_data, mode=dataset_mode)
    for dataset in mixture.datasets:
        if dataset_name and dataset.dataset_name != dataset_name:
            continue
        trajectory_ids = set(np.asarray(dataset.trajectory_ids).tolist())
        if trajectory_id not in trajectory_ids:
            continue
        step_to_index = {
            base_index: sample_index
            for sample_index, (traj_id, base_index) in enumerate(dataset.all_steps)
            if traj_id == trajectory_id
        }
        if step not in step_to_index:
            raise KeyError(
                f"Found trajectory_id={trajectory_id} in dataset={dataset.dataset_name}, "
                f"but step={step} is unavailable."
            )
        sample = dataset[step_to_index[step]]
        meta = {
            "dataset_name": dataset.dataset_name,
            "trajectory_id": trajectory_id,
            "step": step,
            "dataset_index": step_to_index[step],
        }
        return [sample], meta
    raise KeyError(
        f"Could not find dataset_name={dataset_name or '<any>'}, trajectory_id={trajectory_id}, step={step} "
        f"in split={dataset_mode}."
    )


def prepare_qwen_inputs(model, examples):
    if type(examples) is not list:
        examples = [examples]
    batch_images = [to_pil_preserve(example["image"]) for example in examples]
    instructions = [example["lang"] for example in examples]
    state = [example["state"] for example in examples] if "state" in examples[0] else None
    instructions = model.add_discretized_state_to_instruction(instructions, state) if state is not None else instructions

    train_obs_image_size = getattr(model.config.datasets.vla_data, "obs_image_size", None)
    if train_obs_image_size:
        batch_images = resize_images(batch_images, target_size=train_obs_image_size)

    action_tokens = model.action_token * model.chunk_len
    prompt_suffix = f" Please predict the next {model.chunk_len} robot actions: <action>{action_tokens}<action>."
    instructions = [instruction + prompt_suffix for instruction in instructions]

    qwen_inputs = model.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
    return qwen_inputs, instructions


def check_predict_action_path(name: str, model, examples):
    print(f"\n[Predict action path: {name}]")
    try:
        output = model.predict_action(examples=examples)
        normalized_actions = output["normalized_actions"]
        arr = np.asarray(normalized_actions)
        print(f"predict_action succeeded: shape={arr.shape} nan={np.isnan(arr).any()} inf={np.isinf(arr).any()}")
    except Exception as exc:
        print(f"predict_action failed: {type(exc).__name__}: {exc}")


def get_action_positions(input_ids: torch.Tensor, action_token_id: int):
    counts = (input_ids == action_token_id).sum(dim=1).tolist()
    positions = [(input_ids[b] == action_token_id).nonzero(as_tuple=False).flatten().tolist() for b in range(input_ids.shape[0])]
    return counts, positions


def compare_equal(name: str, lhs: torch.Tensor, rhs: torch.Tensor):
    equal = torch.equal(lhs, rhs)
    print(f"{name}: {'OK' if equal else 'DIFF'}")
    if not equal:
        diff_count = int((lhs != rhs).sum().item())
        print(f"  differing entries: {diff_count}")
    return equal


def selected_hidden(hidden: torch.Tensor, action_mask: torch.Tensor):
    selected = hidden[action_mask]
    return selected.reshape(-1, hidden.shape[-1]).float()


def summarize_tensor(name: str, tensor: torch.Tensor, preview: int = 8):
    flat = tensor.detach().float().flatten()
    nan_count = int(torch.isnan(flat).sum().item())
    inf_count = int(torch.isinf(flat).sum().item())
    finite = flat[torch.isfinite(flat)]
    print(f"{name}: shape={tuple(tensor.shape)} nan_count={nan_count} inf_count={inf_count}")
    if finite.numel() > 0:
        print(
            f"  finite min={finite.min().item():.8f} max={finite.max().item():.8f} "
            f"mean={finite.mean().item():.8f} std={finite.std(unbiased=False).item():.8f}"
        )
    else:
        print("  no finite values")
    print(f"  preview={flat[:preview].tolist()}")


def finite_stats(name: str, tensor: torch.Tensor):
    is_nan = torch.isnan(tensor).any().item()
    is_inf = torch.isinf(tensor).any().item()
    print(f"{name}: nan={bool(is_nan)} inf={bool(is_inf)}")


def hidden_compare(h_sdpa: torch.Tensor, h_fa3: torch.Tensor):
    summarize_tensor("selected_hidden_sdpa", h_sdpa)
    summarize_tensor("selected_hidden_fa3", h_fa3)
    abs_diff = (h_sdpa - h_fa3).abs()
    mean_abs = abs_diff.mean().item()
    max_abs = abs_diff.max().item()
    cosine_flat = F.cosine_similarity(h_sdpa.flatten(), h_fa3.flatten(), dim=0).item()
    row_cosine = F.cosine_similarity(h_sdpa, h_fa3, dim=-1).mean().item()
    finite_stats("hidden_sdpa", h_sdpa)
    finite_stats("hidden_fa3", h_fa3)
    print(f"hidden mean_abs_diff: {mean_abs:.8f}")
    print(f"hidden max_abs_diff : {max_abs:.8f}")
    print(f"hidden cosine_flat  : {cosine_flat:.8f}")
    print(f"hidden cosine_mean  : {row_cosine:.8f}")
    return {
        "mean_abs_diff": mean_abs,
        "max_abs_diff": max_abs,
        "cosine_flat": cosine_flat,
        "cosine_mean": row_cosine,
        "nan_or_inf": bool(torch.isnan(h_sdpa).any() or torch.isnan(h_fa3).any() or torch.isinf(h_sdpa).any() or torch.isinf(h_fa3).any()),
    }


def module_grad_norm(module):
    if module is None:
        return None
    grads = [p.grad.detach().float().norm() for p in module.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    return torch.norm(torch.stack(grads)).item()


def named_grad_norms(model):
    stats = {
        "action_model": module_grad_norm(getattr(model, "action_model", None)),
        "qwen_vl_interface": module_grad_norm(getattr(model, "qwen_vl_interface", None)),
    }
    for attr in ["project_layers", "projector", "adapter", "layer_qformer"]:
        if hasattr(model, attr):
            stats[attr] = module_grad_norm(getattr(model, attr))
    keyword_stats = {}
    for keyword in ["lora", "adapter", "projector", "project_layers", "layer_qformer"]:
        grads = [p.grad.detach().float().norm() for n, p in model.named_parameters() if keyword in n and p.grad is not None]
        if grads:
            keyword_stats[f"kw:{keyword}"] = torch.norm(torch.stack(grads)).item()
    stats.update(keyword_stats)
    return stats


def grads_have_nan_inf(model):
    for _, p in model.named_parameters():
        if p.grad is None:
            continue
        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
            return True
    return False


def rel_diff(a, b, eps=1e-12):
    denom = max(abs(a), abs(b), eps)
    return abs(a - b) / denom


def print_dict(title: str, data: dict):
    print(f"\n[{title}]")
    for key, value in data.items():
        print(f"  {key}: {value}")


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA is required for this sanity check."
    set_seed(args.seed)

    base_cfg = load_cfg(args)
    if args.trajectory_id >= 0 and args.step >= 0:
        examples, meta = load_exact_example(
            base_cfg, args.dataset_mode, args.dataset_name, args.trajectory_id, args.step
        )
        print(f"Using exact sample: {meta}")
    else:
        examples, indices = load_fixed_examples(base_cfg, args.dataset_mode, args.sample_index, args.batch_size)
        print(f"Using dataset_mode={args.dataset_mode} sample_indices={indices}")
        print(f"Batch size={len(examples)}")

    model_sdpa = build_model(base_cfg, args.checkpoint, args.sdpa_impl, args.device)
    model_fa3 = build_model(base_cfg, args.checkpoint, args.fa3_impl, args.device)
    accelerator = None
    if args.use_accelerator:
        accelerator = Accelerator(device_placement=False)
        model_sdpa, model_fa3 = accelerator.prepare(model_sdpa, model_fa3)
        print(f"Accelerator enabled: device={accelerator.device}")

    raw_sdpa = accelerator.unwrap_model(model_sdpa) if accelerator is not None else model_sdpa
    raw_fa3 = accelerator.unwrap_model(model_fa3) if accelerator is not None else model_fa3
    raw_sdpa.eval()
    raw_fa3.eval()

    check_predict_action_path("sdpa", raw_sdpa, examples)
    check_predict_action_path("fa3", raw_fa3, examples)

    qwen_sdpa, instructions_sdpa = prepare_qwen_inputs(raw_sdpa, examples)
    qwen_fa3, instructions_fa3 = prepare_qwen_inputs(raw_fa3, examples)

    print("\n[One loss + backward check]")
    model_sdpa.train()
    model_fa3.train()
    model_sdpa.zero_grad(set_to_none=True)
    model_fa3.zero_grad(set_to_none=True)

    set_seed(args.seed)
    loss_sdpa = model_sdpa.forward(examples)["action_loss"]
    if accelerator is not None:
        accelerator.backward(loss_sdpa)
    else:
        loss_sdpa.backward()
    grads_sdpa = named_grad_norms(model_sdpa)
    sdpa_bad_grads = grads_have_nan_inf(model_sdpa)

    set_seed(args.seed)
    loss_fa3 = model_fa3.forward(examples)["action_loss"]
    if accelerator is not None:
        accelerator.backward(loss_fa3)
    else:
        loss_fa3.backward()
    grads_fa3 = named_grad_norms(model_fa3)
    fa3_bad_grads = grads_have_nan_inf(model_fa3)

    print(f"loss_sdpa: {loss_sdpa.item():.8f}")
    print(f"loss_fa3 : {loss_fa3.item():.8f}")
    print(f"loss_rel_diff: {rel_diff(loss_sdpa.item(), loss_fa3.item()):.8f}")
    print_dict("grad_norms_sdpa", grads_sdpa)
    print_dict("grad_norms_fa3", grads_fa3)
    print(f"sdpa grads nan/inf: {sdpa_bad_grads}")
    print(f"fa3  grads nan/inf: {fa3_bad_grads}")

    grad_rel_diffs = {}
    for key in sorted(set(grads_sdpa.keys()) & set(grads_fa3.keys())):
        if grads_sdpa[key] is None or grads_fa3[key] is None:
            continue
        grad_rel_diffs[key] = rel_diff(grads_sdpa[key], grads_fa3[key])
    print_dict("grad_rel_diffs", grad_rel_diffs)

    print("\n[Pre-forward check]")
    compare_equal("input_ids identical", qwen_sdpa["input_ids"], qwen_fa3["input_ids"])
    compare_equal("attention_mask identical", qwen_sdpa["attention_mask"], qwen_fa3["attention_mask"])
    counts_sdpa, positions_sdpa = get_action_positions(qwen_sdpa["input_ids"], model_sdpa.action_token_id)
    counts_fa3, positions_fa3 = get_action_positions(qwen_fa3["input_ids"], model_fa3.action_token_id)
    print(f"sdpa action counts: {counts_sdpa}")
    print(f"fa3  action counts: {counts_fa3}")
    print(f"sdpa action positions: {positions_sdpa}")
    print(f"fa3  action positions: {positions_fa3}")
    assert all(c == model_sdpa.chunk_len for c in counts_sdpa), f"sdpa action counts != chunk_len: {counts_sdpa}"
    assert all(c == model_fa3.chunk_len for c in counts_fa3), f"fa3 action counts != chunk_len: {counts_fa3}"
    assert positions_sdpa == positions_fa3, "Action token positions differ between sdpa and fa3"

    ids0_sdpa = qwen_sdpa["input_ids"].clone()
    ids0_fa3 = qwen_fa3["input_ids"].clone()
    action_mask_sdpa = (ids0_sdpa == model_sdpa.action_token_id).clone()
    action_mask_fa3 = (ids0_fa3 == model_fa3.action_token_id).clone()

    raw_sdpa.eval()
    raw_fa3.eval()
    with torch.no_grad():
        out_sdpa = raw_sdpa.qwen_vl_interface(
            **qwen_sdpa, output_attentions=False, output_hidden_states=True, return_dict=True
        )
        out_fa3 = raw_fa3.qwen_vl_interface(
            **qwen_fa3, output_attentions=False, output_hidden_states=True, return_dict=True
        )

    print("\n[Forward mutation check]")
    mutated_sdpa = not torch.equal(qwen_sdpa["input_ids"], ids0_sdpa)
    mutated_fa3 = not torch.equal(qwen_fa3["input_ids"], ids0_fa3)
    print(f"sdpa input_ids mutated after forward: {mutated_sdpa}")
    print(f"fa3  input_ids mutated after forward: {mutated_fa3}")

    hidden_sdpa = out_sdpa.hidden_states[-1].detach()
    hidden_fa3 = out_fa3.hidden_states[-1].detach()
    summarize_tensor("last_hidden_sdpa", hidden_sdpa)
    summarize_tensor("last_hidden_fa3", hidden_fa3)
    h_sdpa = selected_hidden(hidden_sdpa, action_mask_sdpa)
    h_fa3 = selected_hidden(hidden_fa3, action_mask_fa3)

    print("\n[Forward hidden/action check]")
    hidden_stats = hidden_compare(h_sdpa, h_fa3)

    forward_hidden_close = (
        not hidden_stats["nan_or_inf"]
        and hidden_stats["mean_abs_diff"] <= args.hidden_mean_abs_tol
        and hidden_stats["cosine_flat"] >= args.hidden_cosine_tol
    )
    loss_close = rel_diff(loss_sdpa.item(), loss_fa3.item()) <= args.loss_rel_tol
    grads_close = (not sdpa_bad_grads) and (not fa3_bad_grads) and all(
        diff <= args.grad_rel_tol for diff in grad_rel_diffs.values()
    )

    print("\n[Minimal conclusion table]")
    print(f"input_ids_mutate_sdpa: {mutated_sdpa}")
    print(f"input_ids_mutate_fa3 : {mutated_fa3}")
    print(f"forward_hidden_close : {forward_hidden_close}")
    print(f"loss_close           : {loss_close}")
    print(f"grads_close          : {grads_close}")

    if forward_hidden_close and loss_close and grads_close:
        conclusion = "training forward/backward probably okay"
    elif forward_hidden_close and (not mutated_fa3):
        conclusion = "training path looks okay; inference/open-loop path likely broken elsewhere"
    elif mutated_fa3:
        conclusion = "fa3 mutates input_ids under forward; clone pre-forward ids/action mask and avoid post-forward input_ids"
    else:
        conclusion = "fa3 forward and/or training path looks unsafe on this batch"
    print(f"conclusion: {conclusion}")


if __name__ == "__main__":
    main()
