# OCTAR Method and Code Mapping

OCTAR operates only during Stage 2 training. The clean and synthetically occluded views share one CLIP-ReID image encoder. Clean features used as targets are detached, so the method does not introduce a second teacher network. At inference, SOC, RTG, R-MTD, PTA, and APR are removed.

## 1. Semantic Occlusion Composer (SOC)

SOC samples source patches near the boundary of a different-identity donor image and target boxes from upper, middle, and lower body intervals. Detached CLIP-projected patch tokens score the candidates against fixed source-occluder and target-body prompt groups. The highest-scoring source-target pairs are retained, and one pair is sampled by a temperature-controlled softmax.

Code:

- `build_soc_text_features`
- `_sample_soc_pair_boxes`
- `compose_soc_view`
- `INPUT.SOC`

Final settings: application probability `0.5`, area range `[0.15, 0.35]`, 8 source candidates, 8 target candidates, top-3 pair sampling, temperature `0.2`, target identity weight `0.5`, and minimum visible ratio `0.55`.

## 2. Reliable Teacher Gate (RTG)

RTG computes each clean token's CLIP contrast score as the mean similarity to pedestrian prompts minus the mean similarity to occlusion/background prompts. The strongest 60% token scores are averaged into an image score, and the top 50% images in each mini-batch are retained as reliable teachers.

Code:

- `_clip_token_reliability`
- `_rtg_image_scores`
- `_rtg_teacher_weights`

Rejected samples remain clean and continue contributing to the standard CLIP-ReID objective, but they do not contribute to SOC, R-MTD, PTA, or APR.

## 3. Region-aware Masked Token Distillation (R-MTD)

The token grid is divided into normalized vertical intervals `[0, 0.35)`, `[0.35, 0.70)`, and `[0.70, 1.0]`. A total candidate budget of 60% is distributed across valid regions according to regional CLIP reliability. The resulting candidate set is intersected with token cells whose pasted-mask overlap is at least `0.5`. The occluded student token is then matched to the detached clean token with cosine loss.

Code:

- `_region_balanced_candidate_weights`
- `_reliable_candidate_weights`
- `_rmtd_loss`

The R-MTD loss weight is `0.03`, activated after a 10-epoch warm-up.

## 4. Purified Token Anchor (PTA)

PTA reuses the region-balanced clean candidate set before mask intersection. Candidate tokens are aggregated with a softmax over CLIP reliability scores and blended with the normalized clean CLIP-projected global token:

```text
anchor = normalize((1 - alpha) * clean_global + alpha * reliable_local)
```

Code: `_pta_reliable_token_pool` and `_pta_anchor`.

Final settings: `alpha=0.3` and aggregation temperature `0.07`.

## 5. Anchor Prototype Regularization (APR)

APR maintains one normalized prototype per training identity. Reliable detached PTA anchors update the prototype bank by momentum. Occluded projected global tokens are attracted to initialized identity prototypes with cosine loss only when RTG accepts the teacher and SOC actually applies an occlusion.

Code:

- `_apr_update_prototype_bank`
- `_apr_loss`

Final settings: momentum `0.1`, loss weight `0.05`, and a 10-epoch loss warm-up. Prototype updates are performed without back-propagation.

## 6. Objective and Inference

The Stage 2 objective is:

```text
L = L_CLIP-ReID(clean)
  + 0.5 * L_CLIP-ReID(occluded)
  + 0.03 * L_R-MTD
  + 0.05 * L_APR
```

The retrieval descriptor remains the CLIP-ReID concatenation of the unprojected and CLIP-projected global tokens before BNNeck. The prototype bank and all OCTAR auxiliary operations are training-only.
