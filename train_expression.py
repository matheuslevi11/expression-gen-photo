"""Train the expression-conditioned diffusion adaptor on preprocessed MEAD clips.

Ported from the original ``train_bokehK.py`` with three substantive changes:

    1. ``CameraBokehK`` is replaced by :class:`genphoto.data.expression_dataset.ExpressionMEAD`.
    2. The accidental ``import pdb; pdb.set_trace()`` and the noisy ``print('loss', loss)``
       lines from the original are removed.
    3. The trainable-parameter set is the same (camera_encoder + 'merge' attention params), but
       the camera_encoder is now treated as an "expression encoder" semantically. We keep the
       internal class names (CameraAdaptor / CameraCameraEncoder) untouched to avoid touching the
       rest of the architecture.

Validation samples are written to ``{output_dir}/samples/sample-{global_step}/``.
"""

import argparse
import datetime
import inspect
import math
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Dict, Tuple

import omegaconf.listconfig
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.models.attention_processor import AttnProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from einops import rearrange
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import CLIPTextModel, CLIPTokenizer

from genphoto.data.expression_dataset import ExpressionMEAD
from genphoto.models.attention_processor import AttnProcessor as CustomizedAttnProcessor
from genphoto.models.camera_adaptor import CameraAdaptor, CameraCameraEncoder
from genphoto.models.unet import UNet3DConditionModelCameraCond
from genphoto.pipelines.pipeline_animation import GenPhotoPipeline
from genphoto.utils.util import format_time, save_videos_grid, setup_logger


def init_dist(launcher: str = "pytorch", backend: str = "nccl", port: int = 29500, **kwargs) -> int:
    """Initialise torch.distributed; supports pytorch and slurm launchers."""
    if launcher == "pytorch":
        rank = int(os.environ["RANK"])
        num_gpus = torch.cuda.device_count()
        local_rank = rank % num_gpus
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, **kwargs)

    elif launcher == "slurm":
        proc_id = int(os.environ["SLURM_PROCID"])
        ntasks = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        num_gpus = torch.cuda.device_count()
        local_rank = proc_id % num_gpus
        torch.cuda.set_device(local_rank)
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(ntasks)
        os.environ["RANK"] = str(proc_id)
        port = os.environ.get("PORT", port)
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group(backend=backend)

    else:
        raise NotImplementedError(f"Not implemented launcher type: `{launcher}`!")

    return local_rank


def main(
    name: str,
    launcher: str,
    port: int,
    output_dir: str,
    pretrained_model_path: str,
    train_data: Dict,
    validation_data: Dict,
    cfg_random_null_text: bool = True,
    cfg_random_null_text_ratio: float = 0.1,
    unet_additional_kwargs: Dict = None,
    unet_subfolder: str = "unet",
    lora_rank: int = 4,
    lora_scale: float = 1.0,
    lora_ckpt: str = None,
    motion_module_ckpt: str = "",
    motion_lora_rank: int = 0,
    motion_lora_scale: float = 1.0,
    camera_encoder_kwargs: Dict = None,
    attention_processor_kwargs: Dict = None,
    noise_scheduler_kwargs: Dict = None,
    do_sanity_check: bool = True,
    max_train_epoch: int = -1,
    max_train_steps: int = 100,
    validation_steps: int = 100,
    validation_steps_tuple: Tuple = (-1,),
    learning_rate: float = 3e-5,
    lr_warmup_steps: int = 0,
    lr_scheduler: str = "constant",
    num_workers: int = 8,
    train_batch_size: int = 1,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-8,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    checkpointing_epochs: int = 5,
    checkpointing_steps: int = -1,
    mixed_precision_training: bool = True,
    global_seed: int = 42,
    logger_interval: int = 10,
    resume_from: str = None,
    max_validation_samples: int = None,
):
    check_min_version("0.10.0.dev0")

    local_rank = init_dist(launcher=launcher, port=port)
    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()
    is_main_process = global_rank == 0

    seed = global_seed + global_rank
    torch.manual_seed(seed)

    folder_name = name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(output_dir, folder_name)

    *_, config = inspect.getargvalues(inspect.currentframe())

    logger = setup_logger(output_dir, global_rank)

    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, "config.yaml"))

    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))

    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    unet = UNet3DConditionModelCameraCond.from_pretrained_2d(
        pretrained_model_path,
        subfolder=unet_subfolder,
        unet_additional_kwargs=unet_additional_kwargs,
    )
    expression_encoder = CameraCameraEncoder(**camera_encoder_kwargs)

    logger.info("Setting the attention processors")
    unet.set_all_attn_processor(
        add_spatial_lora=lora_ckpt is not None,
        add_motion_lora=motion_lora_rank > 0,
        lora_kwargs={"lora_rank": lora_rank, "lora_scale": lora_scale},
        motion_lora_kwargs={"lora_rank": motion_lora_rank, "lora_scale": motion_lora_scale},
        **attention_processor_kwargs,
    )

    if lora_ckpt is not None:
        logger.info(f"Loading the image lora checkpoint from {lora_ckpt}")
        lora_checkpoints = torch.load(lora_ckpt, map_location=unet.device)
        if "lora_state_dict" in lora_checkpoints.keys():
            lora_checkpoints = lora_checkpoints["lora_state_dict"]
        _, lora_u = unet.load_state_dict(lora_checkpoints, strict=False)
        assert len(lora_u) == 0
        logger.info("Loading done")
    else:
        logger.info("We do not add image lora")

    if motion_module_ckpt != "":
        logger.info(f"Loading the motion module checkpoint from {motion_module_ckpt}")
        mm_checkpoints = torch.load(motion_module_ckpt, map_location=unet.device)
        if "motion_module_state_dict" in mm_checkpoints:
            mm_checkpoints = {k.replace("module.", ""): v for k, v in mm_checkpoints["motion_module_state_dict"].items()}
        _, mm_u = unet.load_state_dict(mm_checkpoints, strict=False)
        assert len(mm_u) == 0
        logger.info("Loading done")
    else:
        logger.info("We do not load pretrained motion module checkpoint")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    spatial_attn_proc_modules = torch.nn.ModuleList(
        [v for v in unet.attn_processors.values()
         if not isinstance(v, (CustomizedAttnProcessor, AttnProcessor))]
    )
    temporal_attn_proc_modules = torch.nn.ModuleList(
        [v for v in unet.mm_attn_processors.values()
         if not isinstance(v, (CustomizedAttnProcessor, AttnProcessor))]
    )
    spatial_attn_proc_modules.requires_grad_(True)
    temporal_attn_proc_modules.requires_grad_(True)
    expression_encoder.requires_grad_(True)
    for n, p in spatial_attn_proc_modules.named_parameters():
        if "lora" in n:
            p.requires_grad = False

    expression_adaptor = CameraAdaptor(unet, expression_encoder)

    encoder_trainable_params = list(filter(lambda p: p.requires_grad, expression_encoder.parameters()))
    encoder_trainable_param_names = [
        n for n, p in expression_encoder.named_parameters() if p.requires_grad
    ]
    attention_trainable_params = [
        v for k, v in unet.named_parameters() if v.requires_grad and "merge" in k and "lora" not in k
    ]
    attention_trainable_param_names = [
        k for k, v in unet.named_parameters() if v.requires_grad and "merge" in k and "lora" not in k
    ]

    trainable_params = encoder_trainable_params + attention_trainable_params
    trainable_param_names = encoder_trainable_param_names + attention_trainable_param_names

    if is_main_process:
        logger.info(f"trainable parameter number: {len(trainable_params)}")
        logger.info(f"encoder trainable number: {len(encoder_trainable_params)}")
        logger.info(f"attention processor trainable number: {len(attention_trainable_params)}")
        logger.info(f"trainable parameter names: {trainable_param_names}")
        logger.info(f"encoder trainable scale: {sum(p.numel() for p in encoder_trainable_params) / 1e6:.3f} M")
        logger.info(f"attention processor trainable scale: {sum(p.numel() for p in attention_trainable_params) / 1e6:.3f} M")
        logger.info(f"trainable parameter scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    vae.to(local_rank)
    text_encoder.to(local_rank)

    logger.info("Building training dataset (ExpressionMEAD)")
    train_dataset = ExpressionMEAD(**train_data)
    distributed_sampler = DistributedSampler(
        train_dataset,
        num_replicas=num_processes,
        rank=global_rank,
        shuffle=True,
        seed=global_seed,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        sampler=distributed_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    logger.info("Building validation dataset (ExpressionMEAD)")
    validation_dataset = ExpressionMEAD(**validation_data)
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    if max_train_steps == -1:
        assert max_train_epoch != -1
        max_train_steps = max_train_epoch * len(train_dataloader)

    if checkpointing_steps == -1:
        assert checkpointing_epochs != -1
        checkpointing_steps = checkpointing_epochs * len(train_dataloader)

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    validation_pipeline = GenPhotoPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=noise_scheduler,
        camera_encoder=expression_encoder,
    )
    validation_pipeline.enable_vae_slicing()

    expression_adaptor.to(local_rank)
    expression_adaptor = DDP(expression_adaptor, device_ids=[local_rank], output_device=local_rank)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    total_batch_size = train_batch_size * num_processes * gradient_accumulation_steps

    if is_main_process:
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    if resume_from is not None:
        logger.info(f"Resuming the training from the checkpoint: {resume_from}")
        ckpt = torch.load(resume_from, map_location=expression_adaptor.device)
        global_step = ckpt["global_step"]
        trained_iterations = global_step % len(train_dataloader)
        first_epoch = int(global_step // len(train_dataloader))
        encoder_state_dict = ckpt["camera_encoder_state_dict"]
        attention_processor_state_dict = ckpt["attention_processor_state_dict"]
        enc_m, enc_u = expression_adaptor.module.camera_encoder.load_state_dict(encoder_state_dict, strict=False)
        assert len(enc_m) == 0 and len(enc_u) == 0
        _, attention_processor_u = expression_adaptor.module.unet.load_state_dict(
            attention_processor_state_dict, strict=False
        )
        assert len(attention_processor_u) == 0
        logger.info("Loaded encoder + attention processor weights for resume.")
        lr_scheduler.last_epoch = first_epoch
    else:
        trained_iterations = 0

    scaler = torch.cuda.amp.GradScaler() if mixed_precision_training else None

    # Real gradient accumulation: only step the optimizer (and increment
    # ``global_step`` / lr / clip / checkpoint / validate) every
    # ``gradient_accumulation_steps`` micro-batches. Earlier versions of this
    # script ignored ``gradient_accumulation_steps`` entirely, so an effective
    # batch > train_batch_size was impossible.
    micro_step = 0  # counts micro-batches since the last optimizer step

    for epoch in range(first_epoch, num_train_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        expression_adaptor.train()

        data_iter = iter(train_dataloader)
        for step in range(trained_iterations, len(train_dataloader)):
            iter_start_time = time.time()

            batch = next(data_iter)
            data_end_time = time.time()

            if cfg_random_null_text:
                batch["text"] = [t if random.random() > cfg_random_null_text_ratio else "" for t in batch["text"]]

            if epoch == first_epoch and step == 0 and do_sanity_check:
                pixel_values_chk, texts = batch["pixel_values"].cpu(), batch["text"]
                pixel_values_chk = rearrange(pixel_values_chk, "b f c h w -> b c f h w")
                for idx, (pv, text) in enumerate(zip(pixel_values_chk, texts)):
                    pv = pv[None, ...]
                    label = "-".join(text.replace("/", "").split()[:10]) if text else f"{global_rank}-{idx}"
                    save_videos_grid(pv, f"{output_dir}/sanity_check/{label}.gif", rescale=True)

            pixel_values = batch["pixel_values"].to(local_rank)
            video_length = pixel_values.shape[1]
            with torch.no_grad():
                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                latents = latents * 0.18215

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.no_grad():
                prompt_ids = tokenizer(
                    batch["text"],
                    max_length=tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(latents.device)
                encoder_hidden_states = text_encoder(prompt_ids)[0]

            camera_embedding = batch["camera_embedding"].to(device=local_rank)
            camera_embedding = rearrange(camera_embedding, "b f c h w -> b c f h w")

            with torch.cuda.amp.autocast(enabled=mixed_precision_training):
                model_pred = expression_adaptor(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    camera_embedding=camera_embedding,
                )

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    raise NotImplementedError
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss = loss / gradient_accumulation_steps

            if mixed_precision_training:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            micro_step += 1
            do_optimizer_step = micro_step >= gradient_accumulation_steps
            if not do_optimizer_step:
                iter_end_time = time.time()
                # Skip optimizer/lr/global_step/checkpoint/validation/logging
                # until we've accumulated a full effective batch.
                continue

            if mixed_precision_training:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, expression_adaptor.parameters()), max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, expression_adaptor.parameters()), max_grad_norm
                )
                optimizer.step()

            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            micro_step = 0
            iter_end_time = time.time()

            if is_main_process and (global_step % checkpointing_steps == 0):
                save_path = os.path.join(output_dir, "checkpoints")
                state_dict = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "camera_encoder_state_dict": expression_adaptor.module.camera_encoder.state_dict(),
                    "attention_processor_state_dict": {
                        k: v for k, v in unet.state_dict().items() if k in attention_trainable_param_names
                    },
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                torch.save(state_dict, os.path.join(save_path, f"checkpoint-step-{global_step}.ckpt"))
                logger.info(f"Saved state to {save_path} (global_step: {global_step})")

            if is_main_process and (
                (global_step + 1) % validation_steps == 0 or (global_step + 1) in validation_steps_tuple
            ):
                generator = torch.Generator(device=latents.device)
                generator.manual_seed(global_seed)

                if isinstance(train_data, omegaconf.listconfig.ListConfig):
                    height = train_data[0].sample_size[0] if not isinstance(train_data[0].sample_size, int) else train_data[0].sample_size
                    width = train_data[0].sample_size[1] if not isinstance(train_data[0].sample_size, int) else train_data[0].sample_size
                else:
                    height = train_data.sample_size[0] if not isinstance(train_data.sample_size, int) else train_data.sample_size
                    width = train_data.sample_size[1] if not isinstance(train_data.sample_size, int) else train_data.sample_size

                validation_data_iter = iter(validation_dataloader)
                for idx, validation_batch in enumerate(validation_data_iter):
                    if max_validation_samples is not None and idx >= max_validation_samples:
                        logger.info(
                            f"Reached max_validation_samples={max_validation_samples}; "
                            f"skipping remaining {len(validation_dataloader) - idx} val clips."
                        )
                        break
                    val_camera_embedding = validation_batch["camera_embedding"].to(device=unet.device)
                    val_camera_embedding = rearrange(val_camera_embedding, "b f c h w -> b c f h w")

                    sample = validation_pipeline(
                        prompt=validation_batch["text"],
                        camera_embedding=val_camera_embedding,
                        video_length=video_length,
                        height=height,
                        width=width,
                        num_inference_steps=25,
                        guidance_scale=8.0,
                        generator=generator,
                    ).videos[0]

                    if "clip_name" in validation_batch:
                        clip_label = validation_batch["clip_name"][0]
                    else:
                        clip_label = f"{idx}"
                    sample_save_path = f"{output_dir}/samples/sample-{global_step}/{clip_label}_sample.gif"
                    reference_save_path = f"{output_dir}/samples/sample-{global_step}/{clip_label}_reference.gif"

                    save_videos_grid(sample[None, ...], sample_save_path)
                    logger.info(f"Saved generated sample to {sample_save_path}")

                    sample_reference = (validation_batch["pixel_values"][0].permute(1, 0, 2, 3) + 1.0) / 2.0
                    save_videos_grid(sample_reference[None, ...], reference_save_path)
                    logger.info(f"Saved reference video to {reference_save_path}")

            if (global_step % logger_interval) == 0 or global_step == 0:
                gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
                msg = (
                    f"Iter: {global_step}/{max_train_steps}, Loss: {loss.detach().item(): .4f}, "
                    f"lr: {lr_scheduler.get_last_lr()}, Data time: {format_time(data_end_time - iter_start_time)}, "
                    f"Iter time: {format_time(iter_end_time - data_end_time)}, "
                    f"ETA: {format_time((iter_end_time - iter_start_time) * (max_train_steps - global_step))}, "
                    f"GPU memory: {gpu_memory: .2f} G"
                )
                logger.info(msg)

            if global_step >= max_train_steps:
                break
        trained_iterations = 0  # only the resume epoch starts mid-way

    dist.destroy_process_group()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    parser.add_argument("--port", type=int, default=29500)
    args = parser.parse_args()

    name = Path(args.config).stem
    config = OmegaConf.load(args.config)

    main(name=name, launcher=args.launcher, port=args.port, **config)
