"""Sample expression-controlled portraits with a trained ExpressionAdaptor.

Mirrors ``inference_bokehK.py`` from the original codebase:

    python inference_expression.py \\
        --config configs/inference_genphoto/expression.yaml \\
        --base_scene "A portrait photograph of a young woman, frontal view, neutral background." \\
        --intensity_list "[0.0, 0.25, 0.5, 0.75, 1.0]"

The intensity list is interpreted as smile / AU12 strength in roughly ``[0, 1]``. Any 5-element
list works (the architecture is hard-coded to 5 frames in the released checkpoints).
"""

import argparse
import json
import logging
import os

import torch
from diffusers import AutoencoderKL, DDIMScheduler
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from transformers import CLIPTextModel, CLIPTokenizer

from genphoto.data.expression_dataset import build_ccl_embedding, create_intensity_embedding
from genphoto.models.camera_adaptor import CameraAdaptor, CameraCameraEncoder
from genphoto.models.unet import UNet3DConditionModelCameraCond
from genphoto.pipelines.pipeline_animation import GenPhotoPipeline
from genphoto.utils.util import save_videos_grid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntensityEmbedding(Dataset):
    """Builds the 6-channel ``[f, 6, H, W]`` conditioning tensor for a list of smile intensities."""

    def __init__(
        self,
        intensity_values: torch.Tensor,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel,
        device: torch.device,
        sample_size=(256, 384),
        prompt_template: str = "<smile intensity: {value:.3f}>",
    ) -> None:
        self.intensity_values = intensity_values.to(device)
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device
        self.sample_size = tuple(sample_size)
        self.prompt_template = prompt_template

    def load(self) -> torch.Tensor:
        if len(self.intensity_values) != 5:
            raise ValueError(
                f"Expected 5 intensity values to match the released checkpoint config, "
                f"got {len(self.intensity_values)}."
            )

        H, W = self.sample_size
        intensity_embedding = create_intensity_embedding(self.intensity_values, H, W).to(self.device)
        ccl_embedding = build_ccl_embedding(
            intensity_values=self.intensity_values,
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            target_height=H,
            target_width=W,
            prompt_template=self.prompt_template,
            device=self.device,
        ).to(self.device)
        return torch.cat((intensity_embedding, ccl_embedding), dim=1)  # [f, 6, H, W]


def load_models(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(cfg.noise_scheduler_kwargs))
    vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_path, subfolder="vae").to(device)
    vae.requires_grad_(False)
    tokenizer = CLIPTokenizer.from_pretrained(cfg.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.pretrained_model_path, subfolder="text_encoder").to(device)
    text_encoder.requires_grad_(False)

    unet = UNet3DConditionModelCameraCond.from_pretrained_2d(
        cfg.pretrained_model_path,
        subfolder=cfg.unet_subfolder,
        unet_additional_kwargs=cfg.unet_additional_kwargs,
    ).to(device)
    unet.requires_grad_(False)

    expression_encoder = CameraCameraEncoder(**cfg.camera_encoder_kwargs).to(device)
    expression_encoder.requires_grad_(False)
    expression_adaptor = CameraAdaptor(unet, expression_encoder)
    expression_adaptor.requires_grad_(False)
    expression_adaptor.to(device)

    add_temporal = cfg.attention_processor_kwargs.get("add_temporal", True)
    if not add_temporal and not cfg.attention_processor_kwargs.get("add_spatial", False):
        logger.info(
            "Baseline mode: no camera-conditioned attention processors will be installed; "
            "the expression embedding is computed but ignored by every attention layer."
        )
    logger.info("Setting the attention processors")
    unet.set_all_attn_processor(
        add_spatial_lora=cfg.lora_ckpt is not None,
        add_motion_lora=cfg.motion_lora_rank > 0,
        lora_kwargs={"lora_rank": cfg.lora_rank, "lora_scale": cfg.lora_scale},
        motion_lora_kwargs={"lora_rank": cfg.motion_lora_rank, "lora_scale": cfg.motion_lora_scale},
        **cfg.attention_processor_kwargs,
    )

    if cfg.get("lora_ckpt") is not None:
        logger.info(f"Loading the LoRA checkpoint from {cfg.lora_ckpt}")
        lora_checkpoints = torch.load(cfg.lora_ckpt, map_location=unet.device)
        if "lora_state_dict" in lora_checkpoints.keys():
            lora_checkpoints = lora_checkpoints["lora_state_dict"]
        _, lora_u = unet.load_state_dict(lora_checkpoints, strict=False)
        assert len(lora_u) == 0, f"Unexpected LoRA keys: {lora_u}"

    if cfg.get("motion_module_ckpt") is not None:
        logger.info(f"Loading the motion module checkpoint from {cfg.motion_module_ckpt}")
        mm_checkpoints = torch.load(cfg.motion_module_ckpt, map_location=unet.device)
        _, mm_u = unet.load_state_dict(mm_checkpoints, strict=False)
        assert len(mm_u) == 0

    if cfg.get("expression_adaptor_ckpt") is not None:
        ckpt_path = cfg.expression_adaptor_ckpt
        logger.info(f"Loading expression adaptor from {ckpt_path}")
        adaptor_checkpoint = torch.load(ckpt_path, map_location=device)
        encoder_state_dict = adaptor_checkpoint["camera_encoder_state_dict"]
        attention_processor_state_dict = adaptor_checkpoint["attention_processor_state_dict"]
        enc_m, enc_u = expression_adaptor.camera_encoder.load_state_dict(encoder_state_dict, strict=False)
        assert len(enc_m) == 0 and len(enc_u) == 0
        _, attention_processor_u = expression_adaptor.unet.load_state_dict(
            attention_processor_state_dict, strict=False
        )
        assert len(attention_processor_u) == 0
        logger.info("Expression Adaptor loading done")
    else:
        logger.info("No Expression Adaptor checkpoint provided")

    pipeline = GenPhotoPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=noise_scheduler,
        camera_encoder=expression_encoder,
    ).to(device)
    pipeline.enable_vae_slicing()

    return pipeline, device


def run_inference(
    pipeline,
    tokenizer,
    text_encoder,
    base_scene: str,
    intensity_list: str,
    output_dir: str,
    device,
    video_length: int = 5,
    height: int = 256,
    width: int = 384,
    prompt_template: str = "<smile intensity: {value:.3f}>",
    seed: int = 42,
):
    os.makedirs(output_dir, exist_ok=True)

    intensity_values = torch.tensor(json.loads(intensity_list), dtype=torch.float32).unsqueeze(1)
    camera_embedding = IntensityEmbedding(
        intensity_values=intensity_values,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        device=device,
        sample_size=(height, width),
        prompt_template=prompt_template,
    ).load()
    camera_embedding = rearrange(camera_embedding.unsqueeze(0), "b f c h w -> b c f h w")

    # Re-seed immediately before sampling so different model variants (baseline vs.
    # trained adaptor) draw identical initial noise even though they consume a
    # different number of RNG draws during module construction.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    with torch.no_grad():
        sample = pipeline(
            prompt=base_scene,
            camera_embedding=camera_embedding,
            video_length=video_length,
            height=height,
            width=width,
            num_inference_steps=25,
            guidance_scale=8.0,
        ).videos[0]

    sample_save_path = os.path.join(output_dir, "sample.gif")
    save_videos_grid(sample[None, ...], sample_save_path)
    logger.info(f"Saved generated sample to {sample_save_path}")


def main(config_path: str, base_scene: str, intensity_list: str, seed: int = 42):
    torch.manual_seed(seed)
    cfg = OmegaConf.load(config_path)
    logger.info("Loading models...")
    pipeline, device = load_models(cfg)
    logger.info("Starting inference...")
    run_inference(
        pipeline,
        pipeline.tokenizer,
        pipeline.text_encoder,
        base_scene,
        intensity_list,
        cfg.output_dir,
        device=device,
        seed=seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--base_scene", type=str, required=True, help="Identity / scene caption")
    parser.add_argument("--intensity_list", type=str, required=True,
                        help="JSON list of 5 smile intensity values, e.g. '[0.0, 0.25, 0.5, 0.75, 1.0]'")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed applied at start and re-applied right before sampling")
    args = parser.parse_args()
    main(args.config, args.base_scene, args.intensity_list, args.seed)
