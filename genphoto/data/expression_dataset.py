"""Expression-conditioned dataset for adapting Generative Photography to facial expression control.

Mirrors the structure of the original camera-conditioned datasets (e.g. ``CameraShutterSpeed``):
each training sample is a sequence of ``sample_n_frames`` frames showing the same identity at
different expression intensities, plus a 6-channel ``camera_embedding`` consisting of:

    - 3 channels of *physical* intensity signal (scalar broadcast to spatial map), and
    - 3 channels of *Contrastive Camera Learning (CCL)* signal (CLIP text-embedding differences
      between consecutive intensity prompts).

The 6-channel layout keeps the encoder configuration ``cin: 384`` (= 6 * downscale_factor**2)
unchanged so the rest of the architecture is reused as-is.
"""

import json
import math
import os
import random
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from einops import rearrange
from torch.utils.data.dataset import Dataset
from transformers import CLIPTextModel, CLIPTokenizer


def create_intensity_embedding(
    intensity_values: torch.Tensor,
    target_height: int,
    target_width: int,
    num_channels: int = 3,
) -> torch.Tensor:
    """Broadcast a scalar expression intensity per frame into a ``[f, num_channels, H, W]`` map.

    Direct analog of ``create_shutter_speed_embedding`` from the camera codebase: encode the
    parameter as a constant spatial channel per frame so the encoder receives a value that is
    spatially uniform but temporally varying.

    Args:
        intensity_values: ``[f, 1]`` tensor of scalar intensities (typically AU12 in ``[0, 1]``).
        target_height: spatial height ``H`` for the output map.
        target_width: spatial width ``W`` for the output map.
        num_channels: number of physical channels to emit (kept at 3 to keep the encoder's
            6-channel input layout).

    Returns:
        Tensor of shape ``[f, num_channels, H, W]`` with the intensity broadcast across space.
    """
    f = intensity_values.shape[0]
    scales = intensity_values.view(f, 1, 1, 1).expand(f, num_channels, target_height, target_width)
    return scales.contiguous().float()


def build_ccl_embedding(
    intensity_values: torch.Tensor,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    target_height: int,
    target_width: int,
    prompt_template: str = "<smile intensity: {value:.3f}>",
    pad_dim: int = 128,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Build the CCL (Contrastive Camera Learning) channel from per-frame intensity prompts.

    For each frame ``i`` we form a prompt encoding the intensity value, encode it with CLIP, and
    take the difference to the previous frame's hidden state. The trailing entry uses the
    last-minus-first difference (matches the original implementation). The resulting per-frame
    difference tensor of shape ``[f, 77, 768]`` is right-padded to ``[f, 128, 768]`` and reshaped
    to ``[f, H, W]`` (H=256, W=384 by default → 128*768 = 256*384).

    Returns a ``[f, 3, H, W]`` tensor (channel-broadcast).
    """
    f = intensity_values.shape[0]
    prompts = [prompt_template.format(value=v.item()) for v in intensity_values]

    with torch.no_grad():
        prompt_ids = tokenizer(
            prompts,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        encoder_hidden_states = text_encoder(input_ids=prompt_ids).last_hidden_state  # [f, 77, 768]

    differences: List[torch.Tensor] = []
    for i in range(1, encoder_hidden_states.size(0)):
        differences.append((encoder_hidden_states[i] - encoder_hidden_states[i - 1]).unsqueeze(0))
    differences.append((encoder_hidden_states[-1] - encoder_hidden_states[0]).unsqueeze(0))
    diffs = torch.cat(differences, dim=0)  # [f, 77, 768]

    pad_length = pad_dim - diffs.size(1)
    if pad_length > 0:
        diffs = F.pad(diffs, (0, 0, 0, pad_length))  # pad sequence dim to 128
    elif pad_length < 0:
        diffs = diffs[:, :pad_dim, :]

    expected = target_height * target_width
    actual = diffs.size(1) * diffs.size(2)
    if actual != expected:
        raise ValueError(
            f"CCL flat size {actual} (= {diffs.size(1)} * {diffs.size(2)}) does not match "
            f"target H*W = {target_height} * {target_width} = {expected}. Adjust pad_dim or "
            "sample_size so they multiply consistently."
        )

    ccl = diffs.reshape(f, target_height, target_width).unsqueeze(1)  # [f, 1, H, W]
    ccl = ccl.expand(-1, 3, -1, -1).contiguous()  # [f, 3, H, W]
    return ccl


def _select_intensity_progression(
    frame_intensities: Sequence[float],
    sample_n_frames: int,
) -> List[int]:
    """Pick ``sample_n_frames`` frame indices forming a monotonically increasing intensity ramp.

    Strategy: sort all frames by their intensity (ascending) and then take ``sample_n_frames``
    indices spaced evenly through the sorted list. This yields a clean low → high progression
    while still drawing from a single source clip (so identity / lighting / pose stay consistent).
    """
    if len(frame_intensities) < sample_n_frames:
        raise ValueError(
            f"Clip has only {len(frame_intensities)} frames, need at least {sample_n_frames}."
        )
    order = np.argsort(np.asarray(frame_intensities, dtype=np.float64))
    picks = np.linspace(0, len(order) - 1, sample_n_frames).round().astype(int)
    return [int(order[p]) for p in picks]


class ExpressionMEAD(Dataset):
    """Same-identity, varying-expression-intensity dataset built from preprocessed MEAD clips.

    The expected ``annotation_json`` is produced by ``scripts/preprocess_mead.py`` and is a list
    of clip records of the form::

        {
            "clip_id": "M003_happy_level_3_001",
            "actor": "M003",
            "emotion": "happy",
            "intensity_level": 3,
            "caption": "A portrait photograph of a man.",
            "frames": [
                {"path": "M003/happy/level_3/001/frame_00000.jpg", "au12": 0.13},
                ...
            ]
        }

    For training, frames within each clip are reordered into a monotonic intensity ramp. For
    validation with an explicit ``intensity_list``, pixel frames still come from a monotonic
    AU12 ramp in the clip (so reference videos are visually distinct), while the scalar targets
    passed to the encoder use ``intensity_list`` (so evaluation uses fixed targets across runs).
    """

    def __init__(
        self,
        root_path: str,
        annotation_json: str,
        clip_model_path: str,
        sample_n_frames: int = 5,
        sample_size: Tuple[int, int] = (256, 384),
        is_Train: bool = True,
        prompt_template: str = "<smile intensity: {value:.3f}>",
        emotion_filter: str = "happy",
    ) -> None:
        self.root_path = root_path
        self.sample_n_frames = sample_n_frames
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.sample_size = sample_size
        self.is_Train = is_Train
        self.prompt_template = prompt_template

        with open(os.path.join(root_path, annotation_json), "r") as fh:
            full = json.load(fh)
        if emotion_filter and emotion_filter.lower() != "all":
            self.dataset = [c for c in full if c.get("emotion", "").lower() == emotion_filter.lower()]
        else:
            self.dataset = full
        if not self.dataset:
            raise ValueError(
                f"No clips found in {annotation_json} after emotion_filter={emotion_filter!r}."
            )
        self.length = len(self.dataset)

        self.pixel_transforms = [
            transforms.Resize(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]

        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(clip_model_path, subfolder="text_encoder")
        self.text_encoder.eval()
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

    def __len__(self) -> int:
        return self.length

    def _load_clip(self, idx: int):
        clip = self.dataset[idx]
        frames_meta = clip["frames"]
        frame_paths = [f["path"] for f in frames_meta]
        frame_intensities = [float(f.get("au12", f.get("intensity", 0.0))) for f in frames_meta]
        caption = clip.get("caption", "A portrait photograph of a person.")

        if self.is_Train:
            chosen_indices = _select_intensity_progression(frame_intensities, self.sample_n_frames)
            chosen_intensities = [frame_intensities[i] for i in chosen_indices]
        else:
            target_list = clip.get("intensity_list")
            if target_list is None:
                chosen_indices = _select_intensity_progression(frame_intensities, self.sample_n_frames)
                chosen_intensities = [frame_intensities[i] for i in chosen_indices]
            else:
                if isinstance(target_list, str):
                    target_list = json.loads(target_list)
                if len(target_list) != self.sample_n_frames:
                    raise ValueError(
                        f"Clip {clip.get('clip_id')} has intensity_list of length "
                        f"{len(target_list)}, expected {self.sample_n_frames}."
                    )
                # MEAD happy clips rarely reach AU12 ≈ 0; argmin-matching targets to frames
                # collapses several slots onto the same image. Keep a visual low→high ramp for
                # pixels, but feed the fixed target_list into the conditioning channels.
                chosen_indices = _select_intensity_progression(
                    frame_intensities, self.sample_n_frames
                )
                chosen_intensities = [float(t) for t in target_list]

        chosen_paths = [os.path.join(self.root_path, frame_paths[i]) for i in chosen_indices]
        images = []
        for path in chosen_paths:
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(f"Could not read frame at {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
        pixel_values = np.stack(images, axis=0)
        pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous() / 255.0

        intensity_values = torch.tensor(chosen_intensities, dtype=torch.float32).unsqueeze(1)
        return pixel_values, caption, intensity_values, clip.get("clip_id", f"clip_{idx}")

    def get_batch(self, idx: int):
        pixel_values, caption, intensity_values, clip_id = self._load_clip(idx)

        H, W = self.sample_size
        intensity_embedding = create_intensity_embedding(intensity_values, H, W)
        ccl_embedding = build_ccl_embedding(
            intensity_values=intensity_values,
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            target_height=H,
            target_width=W,
            prompt_template=self.prompt_template,
            device=intensity_values.device,
        )
        camera_embedding = torch.cat((intensity_embedding, ccl_embedding), dim=1)  # [f, 6, H, W]

        return pixel_values, caption, camera_embedding, intensity_values, clip_id

    def __getitem__(self, idx: int):
        attempt_idx = idx
        last_err = None
        for _ in range(8):
            try:
                pixel_values, caption, camera_embedding, intensity_values, clip_id = self.get_batch(attempt_idx)
                break
            except Exception as e:
                last_err = e
                attempt_idx = random.randint(0, self.length - 1)
        else:
            raise RuntimeError(f"Failed to load any sample after retries; last error: {last_err}")

        for transform in self.pixel_transforms:
            pixel_values = transform(pixel_values)

        return dict(
            pixel_values=pixel_values,
            text=caption,
            camera_embedding=camera_embedding,
            intensity_values=intensity_values,
            clip_name=clip_id,
        )
