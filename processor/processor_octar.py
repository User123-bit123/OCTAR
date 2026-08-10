import logging
import math
import os
import random
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda import amp
from torch.nn import functional as F

from loss.supcontrast import SupConLoss
from model.clip import clip
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval


def _base_model(model):
    return model.module if hasattr(model, "module") else model


def _sample_patch_size(height, width, area_range, aspect_range):
    image_area = height * width
    for _ in range(20):
        target_area = random.uniform(area_range[0], area_range[1]) * image_area
        aspect_ratio = random.uniform(aspect_range[0], aspect_range[1])
        patch_h = int(round((target_area * aspect_ratio) ** 0.5))
        patch_w = int(round((target_area / aspect_ratio) ** 0.5))
        if 1 <= patch_h < height and 1 <= patch_w < width:
            return patch_h, patch_w
    return max(1, min(height - 1, int(height * 0.3))), max(
        1, min(width - 1, int(width * 0.3))
    )


def _choose_patch_source(index, targets, avoid_same_id):
    batch_size = targets.size(0)
    if batch_size == 1:
        return index
    if avoid_same_id:
        candidates = torch.nonzero(targets != targets[index], as_tuple=False).flatten()
        if candidates.numel() > 0:
            selected = torch.randint(candidates.numel(), (1,), device=targets.device)
            return candidates[selected].item()
    source_index = torch.randint(batch_size - 1, (1,), device=targets.device).item()
    return source_index + 1 if source_index >= index else source_index


def _randint_from_range(min_value, max_value):
    min_value = int(round(min_value))
    max_value = int(round(max_value))
    if max_value < min_value:
        return min_value
    return random.randint(min_value, max_value)


def _sample_edge_box(height, width, patch_h, patch_w, edge_ratio):
    edge_h = max(patch_h, int(round(height * edge_ratio)))
    edge_w = max(patch_w, int(round(width * edge_ratio)))
    regions = []
    if edge_h <= height:
        regions.append((0, edge_h - patch_h, 0, width - patch_w))
        regions.append((height - edge_h, height - patch_h, 0, width - patch_w))
    if edge_w <= width:
        regions.append((0, height - patch_h, 0, edge_w - patch_w))
        regions.append((0, height - patch_h, width - edge_w, width - patch_w))
    if not regions:
        return random.randint(0, height - patch_h), random.randint(0, width - patch_w)
    y_min, y_max, x_min, x_max = random.choice(regions)
    return _randint_from_range(y_min, y_max), _randint_from_range(x_min, x_max)


def _sample_body_box(height, width, patch_h, patch_w, x_range, y_range):
    center_x_min = width * x_range[0]
    center_x_max = width * x_range[1]
    center_y_min = height * y_range[0]
    center_y_max = height * y_range[1]
    x_min = max(0, center_x_min - patch_w / 2)
    x_max = min(width - patch_w, center_x_max - patch_w / 2)
    y_min = max(0, center_y_min - patch_h / 2)
    y_max = min(height - patch_h, center_y_max - patch_h / 2)
    if x_max < x_min or y_max < y_min:
        return random.randint(0, height - patch_h), random.randint(0, width - patch_w)
    return _randint_from_range(y_min, y_max), _randint_from_range(x_min, x_max)


def _sample_target_box(height, width, patch_h, patch_w, soc_cfg):
    parts = [list(part) for part in soc_cfg.PARTS]
    y_range = random.choice(parts) if parts else list(soc_cfg.BODY_Y_RANGE)
    return _sample_body_box(
        height,
        width,
        patch_h,
        patch_w,
        list(soc_cfg.BODY_X_RANGE),
        y_range,
    )


def _encode_prompt_groups(model, device, groups):
    prompts = []
    lengths = []
    for group in groups:
        group = list(group)
        prompts.extend(group)
        lengths.append(len(group))
    if any(length == 0 for length in lengths):
        raise ValueError("OCTAR prompt groups must not be empty.")
    tokenized = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        features = _base_model(model).encode_text_tokens(tokenized)
        features = F.normalize(features.float(), dim=-1)
    outputs = []
    start = 0
    for length in lengths:
        outputs.append(features[start:start + length])
        start += length
    return outputs


def build_soc_text_features(model, device, soc_cfg):
    source_positive, source_negative, target, target_negative = _encode_prompt_groups(
        model,
        device,
        [
            soc_cfg.SOURCE_POSITIVE_PROMPTS,
            soc_cfg.SOURCE_NEGATIVE_PROMPTS,
            soc_cfg.TARGET_PROMPTS,
            soc_cfg.TARGET_NEGATIVE_PROMPTS,
        ],
    )
    return {
        "source_positive": source_positive,
        "source_negative": source_negative,
        "target": target,
        "target_negative": target_negative,
    }


def build_octar_text_features(model, device, cfg):
    positive, negative = _encode_prompt_groups(
        model,
        device,
        [
            cfg.MODEL.OCTAR.RTG_POSITIVE_PROMPTS,
            cfg.MODEL.OCTAR.RTG_NEGATIVE_PROMPTS,
        ],
    )
    return {"positive": positive, "negative": negative}


def _infer_token_grid(num_tokens, height, width):
    grid_h = max(1, int(round(math.sqrt(num_tokens * height / max(width, 1)))))
    while grid_h > 1 and num_tokens % grid_h != 0:
        grid_h -= 1
    grid_w = max(1, num_tokens // grid_h)
    if grid_h * grid_w != num_tokens:
        grid_w = max(1, int(round(math.sqrt(num_tokens * width / max(height, 1)))))
        grid_h = max(1, num_tokens // grid_w)
    return grid_h, grid_w


def _pool_box_tokens(token_features, y, x, patch_h, patch_w, height, width):
    num_tokens = token_features.shape[0]
    grid_h, grid_w = _infer_token_grid(num_tokens, height, width)
    row_start = max(0, min(grid_h - 1, int(math.floor(y * grid_h / height))))
    row_end = max(row_start + 1, min(grid_h, int(math.ceil((y + patch_h) * grid_h / height))))
    col_start = max(0, min(grid_w - 1, int(math.floor(x * grid_w / width))))
    col_end = max(col_start + 1, min(grid_w, int(math.ceil((x + patch_w) * grid_w / width))))
    tokens = token_features.reshape(grid_h, grid_w, -1)[row_start:row_end, col_start:col_end]
    return tokens.reshape(-1, token_features.shape[-1]).mean(dim=0)


def _score_soc_source_box(token_features, text_features, box, image_size):
    y, x, patch_h, patch_w = box
    height, width = image_size
    pooled = F.normalize(
        _pool_box_tokens(token_features, y, x, patch_h, patch_w, height, width).float(),
        dim=0,
    )
    positive = torch.matmul(text_features["source_positive"], pooled).mean()
    negative = torch.matmul(text_features["source_negative"], pooled).mean()
    return positive - negative, pooled


def _score_soc_target_box(token_features, target_feature, text_features, box, image_size, soc_cfg):
    y, x, patch_h, patch_w = box
    height, width = image_size
    pooled = F.normalize(
        _pool_box_tokens(token_features, y, x, patch_h, patch_w, height, width).float(),
        dim=0,
    )
    score = torch.matmul(text_features["target"], pooled).mean()
    negative = torch.matmul(text_features["target_negative"], pooled).mean()
    score = score - float(soc_cfg.TARGET_NEG_WEIGHT) * negative
    if target_feature is not None:
        target_feature = F.normalize(target_feature.float(), dim=0)
        score = score + float(soc_cfg.TARGET_ID_WEIGHT) * (pooled * target_feature).sum()
    return score


def _soc_unreal_penalty(images, index, source_index, source_box, target_box, min_visible_ratio):
    src_y, src_x, patch_h, patch_w = source_box
    dst_y, dst_x, _, _ = target_box
    patch = images[source_index, :, src_y:src_y + patch_h, src_x:src_x + patch_w]
    target_patch = images[index, :, dst_y:dst_y + patch_h, dst_x:dst_x + patch_w]
    color_penalty = (patch.float().mean(dim=(1, 2)) - target_patch.float().mean(dim=(1, 2))).abs().mean()
    visible_ratio = 1.0 - float(patch_h * patch_w) / float(images.shape[-2] * images.shape[-1])
    return color_penalty + patch.new_tensor(max(0.0, float(min_visible_ratio) - visible_ratio)).float()


def _sample_soc_pair_boxes(
    images,
    index,
    source_index,
    patch_h,
    patch_w,
    soc_cfg,
    source_tokens,
    target_tokens,
    target_feature,
    text_features,
):
    height, width = images.shape[-2:]
    source_candidates = []
    target_candidates = []
    for _ in range(max(1, int(soc_cfg.NUM_SOURCE_CANDIDATES))):
        src_y, src_x = _sample_edge_box(height, width, patch_h, patch_w, soc_cfg.EDGE_RATIO)
        box = (src_y, src_x, patch_h, patch_w)
        score, pooled = _score_soc_source_box(source_tokens, text_features, box, (height, width))
        source_candidates.append((score, pooled, box))
    for _ in range(max(1, int(soc_cfg.NUM_TARGET_CANDIDATES))):
        dst_y, dst_x = _sample_target_box(height, width, patch_h, patch_w, soc_cfg)
        box = (dst_y, dst_x, patch_h, patch_w)
        score = _score_soc_target_box(
            target_tokens, target_feature, text_features, box, (height, width), soc_cfg
        )
        target_candidates.append((score, box))

    candidates = []
    for source_score, source_feature, source_box in source_candidates:
        for target_score, target_box in target_candidates:
            hard_score = torch.relu(source_score.float()) * torch.relu(target_score.float())
            unreal_score = _soc_unreal_penalty(
                images,
                index,
                source_index,
                source_box,
                target_box,
                soc_cfg.MIN_VISIBLE_RATIO,
            )
            pair_score = (
                float(soc_cfg.SRC_WEIGHT) * source_score.float()
                + float(soc_cfg.TAR_WEIGHT) * target_score.float()
                + float(soc_cfg.HARD_WEIGHT) * hard_score
                - float(soc_cfg.UNREAL_WEIGHT) * unreal_score.float()
            )
            candidates.append(
                {
                    "pair_score": pair_score,
                    "source_score": source_score.detach().float(),
                    "target_score": target_score.detach().float(),
                    "hard_score": hard_score.detach().float(),
                    "unreal_score": unreal_score.detach().float(),
                    "source_feature": source_feature.detach(),
                    "source_box": source_box,
                    "target_box": target_box,
                }
            )

    scores = torch.stack([item["pair_score"] for item in candidates]).float()
    topk = max(1, min(int(soc_cfg.TOPK), len(candidates)))
    top_scores, top_indices = torch.topk(scores, k=topk)
    if soc_cfg.SAMPLE_MODE == "top" or topk == 1:
        selected_rank = 0
    elif soc_cfg.SAMPLE_MODE == "uniform":
        selected_rank = random.randrange(topk)
    elif soc_cfg.SAMPLE_MODE == "softmax":
        temperature = max(float(soc_cfg.TEMPERATURE), 1e-6)
        weights = torch.softmax(top_scores / temperature, dim=0).detach().cpu().tolist()
        selected_rank = random.choices(range(topk), weights=weights, k=1)[0]
    else:
        raise ValueError("INPUT.SOC.SAMPLE_MODE must be top, uniform, or softmax.")
    return candidates[top_indices[selected_rank].item()]


def compose_soc_view(
    images,
    targets,
    cfg,
    source_tokens,
    target_tokens,
    target_features,
    text_features,
    eligible_mask,
):
    soc_cfg = cfg.INPUT.SOC
    occluded = images.clone()
    batch_size, _, height, width = images.shape
    target_mask = torch.zeros((batch_size, 1, height, width), device=images.device, dtype=images.dtype)
    applied_mask = torch.zeros(batch_size, device=images.device, dtype=images.dtype)
    stats = {
        "target_mask": target_mask,
        "applied_mask": applied_mask,
        "pair_score_sum": 0.0,
        "source_score_sum": 0.0,
        "target_score_sum": 0.0,
        "hard_score_sum": 0.0,
        "unreal_score_sum": 0.0,
        "pair_count": 0,
    }

    for index in range(batch_size):
        if float(eligible_mask[index].detach().float().item()) <= 0.0:
            continue
        if random.random() > float(soc_cfg.PROB):
            continue
        patch_h, patch_w = _sample_patch_size(
            height, width, list(soc_cfg.AREA_RANGE), list(soc_cfg.ASPECT_RANGE)
        )
        source_index = _choose_patch_source(index, targets, bool(soc_cfg.AVOID_SAME_ID))
        selected = _sample_soc_pair_boxes(
            images,
            index,
            source_index,
            patch_h,
            patch_w,
            soc_cfg,
            source_tokens[source_index],
            target_tokens[index],
            target_features[index],
            text_features,
        )
        src_y, src_x, _, _ = selected["source_box"]
        dst_y, dst_x, _, _ = selected["target_box"]
        patch = images[source_index, :, src_y:src_y + patch_h, src_x:src_x + patch_w].clone()
        alpha = float(soc_cfg.BLEND_ALPHA)
        if alpha >= 1.0:
            occluded[index, :, dst_y:dst_y + patch_h, dst_x:dst_x + patch_w] = patch
        else:
            target_patch = occluded[index, :, dst_y:dst_y + patch_h, dst_x:dst_x + patch_w]
            occluded[index, :, dst_y:dst_y + patch_h, dst_x:dst_x + patch_w] = (
                alpha * patch + (1.0 - alpha) * target_patch
            )
        target_mask[index, :, dst_y:dst_y + patch_h, dst_x:dst_x + patch_w] = 1.0
        applied_mask[index] = 1.0
        stats["pair_score_sum"] += selected["pair_score"].detach().float().item()
        stats["source_score_sum"] += selected["source_score"].item()
        stats["target_score_sum"] += selected["target_score"].item()
        stats["hard_score_sum"] += selected["hard_score"].item()
        stats["unreal_score_sum"] += selected["unreal_score"].item()
        stats["pair_count"] += 1
    return occluded, stats


def _clip_token_reliability(clean_tokens, text_features):
    token_norm = F.normalize(clean_tokens.detach().float(), dim=-1)
    positive = text_features["positive"].to(token_norm)
    negative = text_features["negative"].to(token_norm)
    positive_score = torch.einsum("bnd,kd->bnk", token_norm, positive).mean(dim=-1)
    negative_score = torch.einsum("bnd,kd->bnk", token_norm, negative).mean(dim=-1)
    return positive_score - negative_score


def _rtg_image_scores(token_scores, cfg):
    ratio = min(max(float(cfg.MODEL.OCTAR.RTG_TOKEN_RATIO), 1e-6), 1.0)
    topk = max(1, int(math.ceil(token_scores.shape[1] * ratio)))
    return torch.topk(token_scores.detach().float(), k=topk, dim=1).values.mean(dim=1)


def _rtg_teacher_weights(image_scores, cfg):
    octar_cfg = cfg.MODEL.OCTAR
    selected = image_scores >= float(octar_cfg.RTG_SCORE_THRESHOLD)
    ratio = min(max(float(octar_cfg.RTG_BATCH_TOPK_RATIO), 1e-6), 1.0)
    topk = max(1, int(math.ceil(image_scores.shape[0] * ratio)))
    top_indices = torch.topk(image_scores.detach().float(), k=topk, dim=0).indices
    top_mask = torch.zeros_like(selected, dtype=torch.bool)
    top_mask.scatter_(0, top_indices, True)
    selected = selected & top_mask
    if bool(octar_cfg.RTG_ENSURE_ONE) and not bool(selected.any().item()):
        selected = selected.clone()
        selected[torch.argmax(image_scores.detach().float())] = True
    return selected.to(dtype=image_scores.dtype)


def _region_token_indices(num_tokens, cfg, device):
    height, width = list(cfg.INPUT.SIZE_TRAIN)
    grid_h, grid_w = _infer_token_grid(num_tokens, int(height), int(width))
    row_centers = (torch.arange(grid_h, device=device).float() + 0.5) / float(grid_h)
    vertical = row_centers[:, None].expand(grid_h, grid_w).reshape(-1)
    if vertical.numel() != num_tokens:
        vertical = torch.linspace(
            0.5 / float(num_tokens),
            1.0 - 0.5 / float(num_tokens),
            steps=num_tokens,
            device=device,
        )
    bins = [float(value) for value in cfg.MODEL.OCTAR.REGION_BINS]
    regions = []
    for region_id in range(len(bins) - 1):
        left, right = bins[region_id], bins[region_id + 1]
        mask = (vertical >= left) & (
            vertical <= right if region_id == len(bins) - 2 else vertical < right
        )
        regions.append(mask.nonzero(as_tuple=False).flatten())
    return regions


def _global_candidate_weights(token_scores, cfg):
    octar_cfg = cfg.MODEL.OCTAR
    selected = token_scores >= float(octar_cfg.CANDIDATE_SCORE_THRESHOLD)
    ratio = min(max(float(octar_cfg.CANDIDATE_RATIO), 1e-6), 1.0)
    topk = max(1, int(math.ceil(token_scores.shape[1] * ratio)))
    indices = torch.topk(token_scores.detach().float(), k=topk, dim=1).indices
    top_mask = torch.zeros_like(selected, dtype=torch.bool)
    top_mask.scatter_(1, indices, True)
    return (selected & top_mask).to(dtype=token_scores.dtype)


def _region_confidence(scores, regions, cfg):
    ratio = min(max(float(cfg.MODEL.OCTAR.REGION_CONF_TOPK_RATIO), 1e-6), 1.0)
    confidences = []
    for indices in regions:
        if indices.numel() == 0:
            confidences.append(torch.full((scores.shape[0],), -1e6, device=scores.device))
            continue
        region_scores = scores[:, indices]
        topk = max(1, int(math.ceil(region_scores.shape[1] * ratio)))
        confidences.append(torch.topk(region_scores, k=topk, dim=1).values.mean(dim=1))
    return torch.stack(confidences, dim=1)


def _adjust_region_quotas(quotas, valid, total_k, min_tokens):
    quotas = quotas.clone()
    if min_tokens > 0 and int(valid.sum().item()) * min_tokens <= total_k:
        quotas[valid] = torch.maximum(
            quotas[valid], torch.full_like(quotas[valid], int(min_tokens))
        )
    while int(quotas.sum().item()) > total_k:
        candidates = (quotas > min_tokens) & valid
        if not bool(candidates.any()):
            candidates = (quotas > 0) & valid
        if not bool(candidates.any()):
            break
        indices = candidates.nonzero(as_tuple=False).flatten()
        quotas[indices[torch.argmax(quotas[indices])]] -= 1
    return quotas


def _region_balanced_candidate_weights(token_scores, cfg):
    octar_cfg = cfg.MODEL.OCTAR
    scores = token_scores.detach().float()
    batch_size, num_tokens = scores.shape
    ratio = min(max(float(octar_cfg.CANDIDATE_RATIO), 1e-6), 1.0)
    total_k = max(1, int(math.ceil(num_tokens * ratio)))
    threshold = float(octar_cfg.CANDIDATE_SCORE_THRESHOLD)
    confidence_threshold = float(octar_cfg.REGION_CONF_THRESHOLD)
    min_tokens = max(0, int(octar_cfg.REGION_MIN_TOKENS))
    temperature = max(float(octar_cfg.REGION_QUOTA_TEMPERATURE), 1e-6)
    regions = _region_token_indices(num_tokens, cfg, scores.device)
    confidence = _region_confidence(scores, regions, cfg)
    selected = torch.zeros_like(scores, dtype=torch.bool)
    global_selected = _global_candidate_weights(scores, cfg).bool()

    for batch_index in range(batch_size):
        valid = confidence[batch_index] >= confidence_threshold
        if not bool(valid.any()):
            selected[batch_index] = global_selected[batch_index]
            continue
        valid_indices = valid.nonzero(as_tuple=False).flatten()
        if bool(octar_cfg.REGION_ADAPTIVE_QUOTA):
            weights = torch.softmax(confidence[batch_index, valid_indices] / temperature, dim=0)
            raw_quotas = weights * float(total_k)
            valid_quotas = torch.floor(raw_quotas).long()
            remain = total_k - int(valid_quotas.sum().item())
            if remain > 0:
                order = torch.argsort(raw_quotas - valid_quotas.float(), descending=True)
                valid_quotas[order[:remain]] += 1
            quotas = torch.zeros(len(regions), device=scores.device, dtype=torch.long)
            quotas[valid_indices] = valid_quotas
        else:
            quotas = torch.zeros(len(regions), device=scores.device, dtype=torch.long)
            quotas[valid_indices] = total_k // int(valid_indices.numel())
            remain = total_k % int(valid_indices.numel())
            if remain > 0:
                order = torch.argsort(confidence[batch_index, valid_indices], descending=True)
                quotas[valid_indices[order[:remain]]] += 1
        quotas = _adjust_region_quotas(quotas, valid, total_k, min_tokens)

        sample_selected = torch.zeros(num_tokens, device=scores.device, dtype=torch.bool)
        for region_id, indices in enumerate(regions):
            quota = int(quotas[region_id].item())
            if quota <= 0 or indices.numel() == 0:
                continue
            region_scores = scores[batch_index, indices]
            valid_tokens = region_scores >= threshold
            if not bool(valid_tokens.any()):
                continue
            candidate_indices = indices[valid_tokens]
            candidate_scores = region_scores[valid_tokens]
            quota = min(quota, int(candidate_indices.numel()))
            top_indices = torch.topk(candidate_scores, k=quota).indices
            sample_selected[candidate_indices[top_indices]] = True
        selected[batch_index] = sample_selected if bool(sample_selected.any()) else global_selected[batch_index]
    return selected.to(dtype=token_scores.dtype)


def _reliable_candidate_weights(token_scores, cfg):
    return _region_balanced_candidate_weights(token_scores, cfg)


def _pta_reliable_token_pool(clean_tokens, token_scores, cfg):
    reliable = _reliable_candidate_weights(token_scores, cfg).to(clean_tokens.device).bool()
    scores = token_scores.detach().float().to(clean_tokens.device)
    temperature = max(float(cfg.MODEL.OCTAR.PTA_TEMPERATURE), float(cfg.MODEL.OCTAR.EPS))
    logits = (scores / temperature).masked_fill(~reliable, -1e4)
    empty = ~reliable.any(dim=1)
    if bool(empty.any()):
        logits = logits.clone()
        logits[empty] = scores[empty] / temperature
    weights = torch.softmax(logits, dim=1).to(dtype=clean_tokens.dtype)
    pooled = (weights.unsqueeze(-1) * clean_tokens.detach().float()).sum(dim=1)
    return pooled.to(dtype=clean_tokens.dtype), reliable.float().mean().detach()


def _pta_anchor(clean_features, clean_tokens, token_scores, cfg):
    token_pool, reliable_rate = _pta_reliable_token_pool(clean_tokens, token_scores, cfg)
    alpha = min(max(float(cfg.MODEL.OCTAR.PTA_ALPHA), 0.0), 1.0)
    clean_global = clean_features.detach().float()
    local_feature = token_pool.float()
    if bool(cfg.MODEL.OCTAR.NORMALIZE_FEATURES):
        clean_global = F.normalize(clean_global, dim=-1)
        local_feature = F.normalize(local_feature, dim=-1)
    anchor = (1.0 - alpha) * clean_global + alpha * local_feature
    if bool(cfg.MODEL.OCTAR.NORMALIZE_FEATURES):
        anchor = F.normalize(anchor, dim=-1)
    return anchor.to(dtype=clean_features.dtype), reliable_rate


def _token_overlap_from_mask(target_mask, num_tokens, cfg):
    grid_h, grid_w = _infer_token_grid(
        num_tokens, cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1]
    )
    overlap = F.adaptive_avg_pool2d(target_mask.float(), (grid_h, grid_w)).squeeze(1)
    return overlap.reshape(overlap.shape[0], -1).clamp(0.0, 1.0)


def _rmtd_loss(clean_tokens, occluded_tokens, target_mask, token_scores, teacher_weights, cfg):
    overlap = _token_overlap_from_mask(target_mask, occluded_tokens.shape[1], cfg).to(
        occluded_tokens.device
    )
    reliable = _reliable_candidate_weights(token_scores, cfg).to(occluded_tokens)
    teachers = teacher_weights.to(occluded_tokens).view(-1, 1)
    corrupted = (overlap >= float(cfg.MODEL.OCTAR.MASK_OVERLAP_THRESHOLD)).to(
        occluded_tokens
    )
    weights = overlap.to(occluded_tokens) * corrupted * reliable * teachers
    token_count = (corrupted * reliable * teachers).float().sum(dim=1).mean()
    eps = float(cfg.MODEL.OCTAR.EPS)
    if float(weights.sum().detach().item()) <= eps:
        zero = occluded_tokens.float().sum() * 0.0
        return zero, {
            "tokens": token_count.detach(),
            "sim": torch.zeros((), device=occluded_tokens.device),
            "reliable": reliable.float().mean().detach(),
        }
    occluded_norm = F.normalize(occluded_tokens.float(), dim=-1)
    clean_norm = F.normalize(clean_tokens.detach().float(), dim=-1)
    similarity = (occluded_norm * clean_norm).sum(dim=-1).clamp(-1.0, 1.0)
    loss = ((1.0 - similarity) * weights.float()).sum() / (weights.float().sum() + eps)
    weighted_similarity = (similarity * weights.float()).sum() / (weights.float().sum() + eps)
    return loss, {
        "tokens": token_count.detach(),
        "sim": weighted_similarity.detach(),
        "reliable": reliable.float().mean().detach(),
    }


def _apr_update_prototype_bank(
    prototype_bank, prototype_filled, clean_anchors, targets, teacher_weights, cfg
):
    eps = float(cfg.MODEL.OCTAR.EPS)
    weights = teacher_weights.detach().float().to(clean_anchors.device).view(-1)
    selected = weights > eps
    if not bool(selected.any()):
        return prototype_filled.float().mean().item(), 0.0
    momentum = float(cfg.MODEL.OCTAR.APR_MOMENTUM)
    features = clean_anchors.detach().float()
    if bool(cfg.MODEL.OCTAR.NORMALIZE_FEATURES):
        features = F.normalize(features, dim=-1)
    with torch.no_grad():
        for target_id_tensor in torch.unique(targets[selected].detach()):
            target_id = int(target_id_tensor.item())
            class_mask = (targets == target_id) & selected
            class_weights = weights[class_mask].to(features)
            class_mean = (features[class_mask] * class_weights[:, None]).sum(dim=0) / (
                class_weights.sum() + eps
            )
            if bool(cfg.MODEL.OCTAR.NORMALIZE_FEATURES):
                class_mean = F.normalize(class_mean, dim=0)
            if bool(prototype_filled[target_id]):
                prototype_bank[target_id].mul_(momentum).add_(class_mean, alpha=1.0 - momentum)
            else:
                prototype_bank[target_id].copy_(class_mean)
                prototype_filled[target_id] = True
            if bool(cfg.MODEL.OCTAR.NORMALIZE_FEATURES):
                prototype_bank[target_id].copy_(F.normalize(prototype_bank[target_id], dim=0))
    return prototype_filled.float().mean().item(), float(weights[selected].mean().item())


def _apr_loss(occluded_features, targets, prototype_bank, prototype_filled, sample_weights, cfg):
    sample_weights = sample_weights.to(occluded_features)
    sample_weights = sample_weights * prototype_filled[targets].to(sample_weights)
    eps = float(cfg.MODEL.OCTAR.EPS)
    if float(sample_weights.sum().detach().item()) <= eps:
        zero = occluded_features.float().sum() * 0.0
        return zero, {
            "sim": torch.zeros((), device=occluded_features.device),
            "weight": sample_weights.detach().mean(),
        }
    features = occluded_features.float()
    prototypes = prototype_bank[targets].detach().to(features)
    if bool(cfg.MODEL.OCTAR.NORMALIZE_FEATURES):
        features = F.normalize(features, dim=-1)
        prototypes = F.normalize(prototypes, dim=-1)
    similarity = (features * prototypes).sum(dim=-1).clamp(-1.0, 1.0)
    loss = ((1.0 - similarity) * sample_weights.float()).sum() / (
        sample_weights.float().sum() + eps
    )
    return loss, {
        "sim": (similarity * sample_weights.float()).sum().detach()
        / (sample_weights.float().sum().detach() + eps),
        "weight": sample_weights.detach().mean(),
    }


def _should_eval_epoch(epoch, eval_period, eval_epochs):
    return epoch in eval_epochs or (eval_period > 0 and epoch % eval_period == 0)


def _build_identity_text_features(model, num_classes, batch_size, device):
    features = []
    with torch.no_grad():
        for start in range(0, num_classes, batch_size):
            labels = torch.arange(start, min(start + batch_size, num_classes), device=device)
            with amp.autocast(enabled=True):
                features.append(model(label=labels, get_text=True, prompt_mode="full").cpu())
    return torch.cat(features, dim=0).to(device)


def _evaluate(cfg, model, val_loader, evaluator, device):
    evaluator.reset()
    model.eval()
    for images, identities, camera_ids, camera_batch, views, _ in val_loader:
        with torch.no_grad():
            images = images.to(device)
            camera_batch = camera_batch.to(device) if cfg.MODEL.SIE_CAMERA else None
            views = views.to(device) if cfg.MODEL.SIE_VIEW else None
            features = model(images, cam_label=camera_batch, view_label=views)
            evaluator.update((features, identities, camera_ids))
    return evaluator.compute()


def do_train_stage2(
    cfg,
    model,
    center_criterion,
    train_loader_stage2,
    val_loader,
    optimizer,
    optimizer_center,
    scheduler,
    loss_fn,
    num_query,
    local_rank,
):
    if not bool(cfg.MODEL.OCTAR.ENABLED) or not bool(cfg.INPUT.SOC.ENABLED):
        raise ValueError("The OCTAR release requires MODEL.OCTAR.ENABLED and INPUT.SOC.ENABLED.")

    logger = logging.getLogger("transreid.train")
    logger.info("Start OCTAR Stage 2 training")
    device = torch.device("cuda", local_rank)
    model.to(device)
    if torch.cuda.device_count() > 1:
        logger.info("Using %d GPUs for training", torch.cuda.device_count())
        model = nn.DataParallel(model)
    num_classes = _base_model(model).num_classes

    identity_text = _build_identity_text_features(
        model, num_classes, cfg.SOLVER.STAGE2.IMS_PER_BATCH, device
    )
    soc_text = build_soc_text_features(model, device, cfg.INPUT.SOC)
    octar_text = build_octar_text_features(model, device, cfg)
    prototype_dim = int(_base_model(model).in_planes_proj)
    prototype_bank = torch.zeros(num_classes, prototype_dim, device=device)
    prototype_filled = torch.zeros(num_classes, device=device, dtype=torch.bool)

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE2.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.STAGE2.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.STAGE2.EVAL_PERIOD
    eval_epochs = set(int(epoch) for epoch in cfg.SOLVER.STAGE2.EVAL_EPOCHS)
    best_mAP = -1.0
    best_epoch = -1
    best_path = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_best.pth")
    total_start = time.monotonic()

    meter_names = [
        "loss", "occ", "acc", "teacher", "image", "soc", "rmtd", "tokens",
        "token_sim", "candidate", "pta", "apr", "apr_sim", "proto_fill", "proto_weight",
    ]
    meters = {name: AverageMeter() for name in meter_names}

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        for meter in meters.values():
            meter.reset()
        scheduler.step()
        model.train()

        for iteration, (images, identities, cameras, views) in enumerate(train_loader_stage2, start=1):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            images = images.to(device)
            identities = identities.to(device)
            cameras = cameras.to(device) if cfg.MODEL.SIE_CAMERA else None
            views = views.to(device) if cfg.MODEL.SIE_VIEW else None

            with amp.autocast(enabled=True):
                scores, features, clean_global, clean_tokens = model(
                    x=images,
                    label=identities,
                    cam_label=cameras,
                    view_label=views,
                    return_patch_tokens=True,
                )
                clean_logits = clean_global @ identity_text.t()
                loss = loss_fn(scores, features, identities, cameras, clean_logits)

                token_scores = _clip_token_reliability(clean_tokens, octar_text)
                image_scores = _rtg_image_scores(token_scores, cfg)
                teacher_weights = _rtg_teacher_weights(image_scores, cfg)
                clean_anchors, candidate_rate = _pta_anchor(
                    clean_global, clean_tokens, token_scores, cfg
                )
                prototype_fill, prototype_weight = _apr_update_prototype_bank(
                    prototype_bank,
                    prototype_filled,
                    clean_anchors,
                    identities,
                    teacher_weights,
                    cfg,
                )

                occluded_images, soc_stats = compose_soc_view(
                    images,
                    identities,
                    cfg,
                    clean_tokens.detach(),
                    clean_tokens.detach(),
                    clean_global.detach(),
                    soc_text,
                    teacher_weights,
                )
                occ_scores, occ_features, occluded_global, occluded_tokens = model(
                    x=occluded_images,
                    label=identities,
                    cam_label=cameras,
                    view_label=views,
                    return_patch_tokens=True,
                )
                occ_logits = occluded_global @ identity_text.t()
                occ_loss = loss_fn(occ_scores, occ_features, identities, cameras, occ_logits)
                loss = loss + float(cfg.INPUT.SOC.OCC_REID_WEIGHT) * occ_loss

                rmtd_loss = occluded_global.float().sum() * 0.0
                rmtd_stats = {
                    "tokens": torch.zeros((), device=device),
                    "sim": torch.zeros((), device=device),
                    "reliable": candidate_rate,
                }
                apr_loss = occluded_global.float().sum() * 0.0
                apr_stats = {
                    "sim": torch.zeros((), device=device),
                    "weight": torch.zeros((), device=device),
                }
                if epoch > int(cfg.MODEL.OCTAR.WARMUP_EPOCHS):
                    rmtd_loss, rmtd_stats = _rmtd_loss(
                        clean_tokens,
                        occluded_tokens,
                        soc_stats["target_mask"],
                        token_scores,
                        teacher_weights,
                        cfg,
                    )
                    loss = loss + float(cfg.MODEL.OCTAR.RMTD_WEIGHT) * rmtd_loss
                    apr_weights = teacher_weights * soc_stats["applied_mask"].to(teacher_weights)
                    apr_loss, apr_stats = _apr_loss(
                        occluded_global,
                        identities,
                        prototype_bank,
                        prototype_filled,
                        apr_weights,
                        cfg,
                    )
                    loss = loss + float(cfg.MODEL.OCTAR.APR_WEIGHT) * apr_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            if "center" in cfg.MODEL.METRIC_LOSS_TYPE:
                for parameter in center_criterion.parameters():
                    parameter.grad.data *= 1.0 / cfg.SOLVER.CENTER_LOSS_WEIGHT
                scaler.step(optimizer_center)
            scaler.update()

            accuracy = (clean_logits.max(1)[1] == identities).float().mean()
            batch_size = images.shape[0]
            meters["loss"].update(loss.item(), batch_size)
            meters["occ"].update(occ_loss.item(), batch_size)
            meters["acc"].update(accuracy, 1)
            meters["teacher"].update(teacher_weights.float().mean().item(), batch_size)
            meters["image"].update(image_scores.float().mean().item(), batch_size)
            if soc_stats["pair_count"] > 0:
                meters["soc"].update(
                    soc_stats["pair_score_sum"] / soc_stats["pair_count"],
                    soc_stats["pair_count"],
                )
            meters["rmtd"].update(rmtd_loss.item(), batch_size)
            meters["tokens"].update(rmtd_stats["tokens"].item(), batch_size)
            meters["token_sim"].update(rmtd_stats["sim"].item(), batch_size)
            meters["candidate"].update(rmtd_stats["reliable"].item(), batch_size)
            meters["pta"].update(candidate_rate.item(), batch_size)
            meters["apr"].update(apr_loss.item(), batch_size)
            meters["apr_sim"].update(apr_stats["sim"].item(), batch_size)
            meters["proto_fill"].update(prototype_fill, batch_size)
            meters["proto_weight"].update(prototype_weight, batch_size)

            if iteration % log_period == 0:
                logger.info(
                    "Epoch[%d] Iteration[%d/%d] Loss: %.3f, Occ: %.3f, Acc: %.3f, "
                    "RTG: %.3f, SOC: %.3f, R-MTD: %.3f, Tokens: %.1f, TokenSim: %.3f, "
                    "PTA: %.3f, APR: %.3f, APRSim: %.3f, ProtoFill: %.3f, Lr: %.2e",
                    epoch,
                    iteration,
                    len(train_loader_stage2),
                    meters["loss"].avg,
                    meters["occ"].avg,
                    meters["acc"].avg,
                    meters["teacher"].avg,
                    meters["soc"].avg,
                    meters["rmtd"].avg,
                    meters["tokens"].avg,
                    meters["token_sim"].avg,
                    meters["pta"].avg,
                    meters["apr"].avg,
                    meters["apr_sim"].avg,
                    meters["proto_fill"].avg,
                    scheduler.get_lr()[0],
                )

        elapsed = time.time() - epoch_start
        logger.info(
            "Epoch %d done. Time per batch: %.3fs, speed: %.1f samples/s",
            epoch,
            elapsed / max(1, len(train_loader_stage2)),
            train_loader_stage2.batch_size * max(1, len(train_loader_stage2)) / elapsed,
        )

        if checkpoint_period > 0 and epoch % checkpoint_period == 0:
            if not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_{}.pth".format(epoch)),
                )

        if bool(cfg.TEST.EVAL) and _should_eval_epoch(epoch, eval_period, eval_epochs):
            if not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0:
                cmc, mAP, _, _, _, _, _ = _evaluate(cfg, model, val_loader, evaluator, device)
                logger.info("Validation Results - Epoch: %d", epoch)
                logger.info("mAP: %.1f%%", mAP * 100.0)
                for rank in [1, 5, 10]:
                    logger.info("CMC curve, Rank-%-3d: %.1f%%", rank, cmc[rank - 1] * 100.0)
                if mAP > best_mAP:
                    best_mAP = mAP
                    best_epoch = epoch
                    torch.save(model.state_dict(), best_path)
                torch.cuda.empty_cache()

    logger.info(
        "Training complete in %s. Best epoch: %d, best mAP: %.1f%%",
        timedelta(seconds=time.monotonic() - total_start),
        best_epoch,
        best_mAP * 100.0,
    )


def do_inference(cfg, model, val_loader, num_query):
    device = torch.device("cuda")
    logger = logging.getLogger("transreid.test")
    logger.info("Enter OCTAR inference")
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    if torch.cuda.device_count() > 1:
        logger.info("Using %d GPUs for inference", torch.cuda.device_count())
        model = nn.DataParallel(model)
    model.to(device)
    cmc, mAP, _, _, _, _, _ = _evaluate(cfg, model, val_loader, evaluator, device)
    logger.info("mAP: %.1f%%", mAP * 100.0)
    for rank in [1, 5, 10]:
        logger.info("CMC curve, Rank-%-3d: %.1f%%", rank, cmc[rank - 1] * 100.0)
    return cmc[0], cmc[4], mAP
