import logging
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda import amp

from loss.supcontrast import SupConLoss
from utils.meter import AverageMeter


def do_train_stage1(cfg, model, train_loader_stage1, optimizer, scheduler, local_rank):
    checkpoint_period = cfg.SOLVER.STAGE1.CHECKPOINT_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE1.LOG_PERIOD

    logger = logging.getLogger("transreid.train")
    logger.info("Start Stage 1 prompt training")
    model.to(local_rank)
    if torch.cuda.device_count() > 1:
        logger.info("Using %d GPUs for Stage 1", torch.cuda.device_count())
        model = nn.DataParallel(model)

    loss_meter = AverageMeter()
    scaler = amp.GradScaler()
    contrastive_loss = SupConLoss(device)

    start_time = time.monotonic()
    image_features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for images, identities, _, _ in train_loader_stage1:
            images = images.to(device)
            identities = identities.to(device)
            with amp.autocast(enabled=True):
                batch_features = model(images, identities, get_image=True)
            labels.extend(identities)
            image_features.extend(batch_features.cpu())

    labels = torch.stack(labels, dim=0).to(device)
    image_features = torch.stack(image_features, dim=0).to(device)
    batch_size = cfg.SOLVER.STAGE1.IMS_PER_BATCH
    num_images = labels.shape[0]

    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        scheduler.step(epoch)
        model.train()
        permutation = torch.randperm(num_images, device=device)

        for iteration, start in enumerate(range(0, num_images, batch_size), start=1):
            indices = permutation[start:start + batch_size]
            target = labels[indices]
            batch_image_features = image_features[indices]

            optimizer.zero_grad()
            with amp.autocast(enabled=True):
                text_features = model(label=target, get_text=True, prompt_mode="full")
                loss_i2t = contrastive_loss(batch_image_features, text_features, target, target)
                loss_t2i = contrastive_loss(text_features, batch_image_features, target, target)
                loss = loss_i2t + loss_t2i

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_meter.update(loss.item(), target.shape[0])

            if iteration % log_period == 0:
                logger.info(
                    "Stage1 Epoch[%d] Iteration[%d/%d] Loss: %.3f, Base Lr: %.2e",
                    epoch,
                    iteration,
                    (num_images + batch_size - 1) // batch_size,
                    loss_meter.avg,
                    scheduler._get_lr(epoch)[0],
                )

        if epoch % checkpoint_period == 0:
            checkpoint = os.path.join(
                cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_stage1_{}.pth".format(epoch)
            )
            if not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0:
                torch.save(model.state_dict(), checkpoint)

    elapsed = timedelta(seconds=time.monotonic() - start_time)
    logger.info("Stage 1 running time: %s", elapsed)
