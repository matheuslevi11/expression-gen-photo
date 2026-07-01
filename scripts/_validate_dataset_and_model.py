"""Static validation: load the production expression.yaml-equivalent config,
build the dataset, fetch one sample, build the encoder + UNet + adaptor, run a
single forward pass with the real conditioning tensor, and exit.

Goal: catch shape / path / import errors without launching the full distributed
training script.
"""

import sys
import time
from pathlib import Path

import torch
from einops import rearrange
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CONFIG = REPO / "configs/train_genphoto/expression_validation.yaml"


def main() -> int:
    cfg = OmegaConf.load(CONFIG)

    print("[1/6] config loaded:", CONFIG.name)
    print("       pretrained_model_path exists:",
          Path(cfg.pretrained_model_path).exists())
    print("       lora_ckpt exists           :", Path(cfg.lora_ckpt).exists())
    print("       motion_module_ckpt exists  :", Path(cfg.motion_module_ckpt).exists())
    print("       train root exists          :", Path(cfg.train_data.root_path).exists())
    train_ann = Path(cfg.train_data.root_path) / cfg.train_data.annotation_json
    val_ann = Path(cfg.validation_data.root_path) / cfg.validation_data.annotation_json
    print("       train annotations exist    :", train_ann.exists())
    print("       val annotations exist      :", val_ann.exists())

    from genphoto.data.expression_dataset import ExpressionMEAD

    print("[2/6] building ExpressionMEAD (train) ...")
    t0 = time.time()
    train_ds = ExpressionMEAD(**OmegaConf.to_container(cfg.train_data))
    print(f"       ok ({len(train_ds)} clips, build {time.time()-t0:.1f}s)")

    print("[3/6] loading one training sample (with CCL CLIP encode) ...")
    t0 = time.time()
    sample = train_ds[0]
    print(f"       ok ({time.time()-t0:.1f}s)")
    pv = sample["pixel_values"]
    ce = sample["camera_embedding"]
    iv = sample["intensity_values"]
    print(f"       pixel_values    : {tuple(pv.shape)} dtype={pv.dtype} "
          f"range=[{pv.min():.3f}, {pv.max():.3f}]")
    print(f"       camera_embedding: {tuple(ce.shape)} dtype={ce.dtype} "
          f"phys_mean={ce[:, :3].mean():.3f} ccl_mean={ce[:, 3:].mean():.3e}")
    print(f"       intensity_values: {iv.flatten().tolist()}")
    print(f"       text            : {sample['text']!r}")
    print(f"       clip_name       : {sample['clip_name']}")

    expected_cin = 6 * cfg.camera_encoder_kwargs.downscale_factor ** 2
    assert expected_cin == cfg.camera_encoder_kwargs.cin, (
        f"encoder cin mismatch: {cfg.camera_encoder_kwargs.cin} vs expected {expected_cin}"
    )
    print(f"       encoder cin check: {cfg.camera_encoder_kwargs.cin} == 6 * "
          f"{cfg.camera_encoder_kwargs.downscale_factor}^2 OK")

    print("[4/6] building UNet3DConditionModelCameraCond + CameraCameraEncoder ...")
    from genphoto.models.camera_adaptor import CameraAdaptor, CameraCameraEncoder
    from genphoto.models.unet import UNet3DConditionModelCameraCond

    t0 = time.time()
    unet = UNet3DConditionModelCameraCond.from_pretrained_2d(
        cfg.pretrained_model_path,
        subfolder=cfg.unet_subfolder,
        unet_additional_kwargs=OmegaConf.to_container(cfg.unet_additional_kwargs),
    )
    enc = CameraCameraEncoder(**OmegaConf.to_container(cfg.camera_encoder_kwargs))
    unet.set_all_attn_processor(
        add_spatial_lora=True,
        add_motion_lora=False,
        lora_kwargs={"lora_rank": cfg.lora_rank, "lora_scale": cfg.lora_scale},
        motion_lora_kwargs={"lora_rank": 0, "lora_scale": 1.0},
        **OmegaConf.to_container(cfg.attention_processor_kwargs),
    )
    adaptor = CameraAdaptor(unet, enc)
    print(f"       ok (build {time.time()-t0:.1f}s)")

    n_train_enc = sum(p.numel() for p in enc.parameters())
    n_merge = sum(
        v.numel() for k, v in unet.named_parameters()
        if "merge" in k and "lora" not in k
    )
    print(f"       encoder params      : {n_train_enc/1e6:.2f} M")
    print(f"       'merge' attn params : {n_merge/1e6:.2f} M (trained alongside)")

    print("[5/6] one fp16 forward pass on cuda:0 (batch=1, frames=5, 256x384) ...")
    from diffusers import AutoencoderKL
    from transformers import CLIPTextModel, CLIPTokenizer

    device = torch.device("cuda:0")
    vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_path, subfolder="vae").to(device).eval()
    tok = CLIPTokenizer.from_pretrained(cfg.pretrained_model_path, subfolder="tokenizer")
    txt = CLIPTextModel.from_pretrained(cfg.pretrained_model_path, subfolder="text_encoder").to(device).eval()

    adaptor.to(device)

    pv_b = pv.unsqueeze(0).to(device)
    ce_b = ce.unsqueeze(0).to(device)
    with torch.no_grad():
        f = pv_b.shape[1]
        latents = vae.encode(rearrange(pv_b, "b f c h w -> (b f) c h w")).latent_dist.sample()
        latents = rearrange(latents, "(b f) c h w -> b c f h w", f=f) * 0.18215
        noise = torch.randn_like(latents)
        ts = torch.randint(0, 1000, (1,), device=device).long()
        ids = tok([sample["text"]], max_length=tok.model_max_length, padding="max_length",
                  truncation=True, return_tensors="pt").input_ids.to(device)
        ehs = txt(ids)[0]

    ce_b_perm = rearrange(ce_b, "b f c h w -> b c f h w")
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    with torch.cuda.amp.autocast(enabled=True):
        pred = adaptor(latents, ts, encoder_hidden_states=ehs, camera_embedding=ce_b_perm)
        loss = torch.nn.functional.mse_loss(pred.float(), noise.float())
    loss.backward()
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    print(f"       pred shape   : {tuple(pred.shape)} (matches noise {tuple(noise.shape)})")
    print(f"       loss         : {loss.item():.4f}")
    print(f"       fwd+bwd time : {dt:.2f}s")
    print(f"       peak VRAM    : {peak:.2f} GiB (batch=1)")

    print("[6/6] all checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
