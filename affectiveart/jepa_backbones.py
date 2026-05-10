from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    model_id: str
    backend: str
    modality: str
    family: str
    static_image_safe: bool
    recommended_for_track2_full: bool
    local_notes: str


BACKBONES: tuple[BackboneSpec, ...] = (
    BackboneSpec(
        name="clip-vit-b-32",
        model_id="ViT-B/32",
        backend="clip",
        modality="image",
        family="clip",
        static_image_safe=True,
        recommended_for_track2_full=True,
        local_notes="Fast baseline; useful for leakage audit and retrieval sanity checks.",
    ),
    BackboneSpec(
        name="siglip2-base-patch16-224",
        model_id="google/siglip2-base-patch16-224",
        backend="hf",
        modality="image",
        family="siglip",
        static_image_safe=True,
        recommended_for_track2_full=True,
        local_notes="Current strongest Track2 single backbone in local holdout.",
    ),
    BackboneSpec(
        name="dinov2-base",
        model_id="facebook/dinov2-base",
        backend="hf",
        modality="image",
        family="dino",
        static_image_safe=True,
        recommended_for_track2_full=True,
        local_notes="Good structural vision baseline; weaker than SigLIP2 on Track2 emotion labels.",
    ),
    BackboneSpec(
        name="ijepa-vith16-1k",
        model_id="facebook/ijepa_vith16_1k",
        backend="hf",
        modality="image",
        family="jepa",
        static_image_safe=True,
        recommended_for_track2_full=False,
        local_notes="ViT-H 448px I-JEPA; smoke works but local full Track2 is too slow on MPS.",
    ),
    BackboneSpec(
        name="ijepa-vith14-1k",
        model_id="facebook/ijepa_vith14_1k",
        backend="hf",
        modality="image",
        family="jepa",
        static_image_safe=True,
        recommended_for_track2_full=False,
        local_notes="ViT-H 224px I-JEPA; lower resolution, not lower parameter count.",
    ),
    BackboneSpec(
        name="vjepa2-vitl",
        model_id="facebook/vjepa2-vitl-fpc64-256",
        backend="hf",
        modality="video",
        family="jepa",
        static_image_safe=False,
        recommended_for_track2_full=False,
        local_notes="Video model; repeated still frames are not a meaningful Track2 signal.",
    ),
)


def lookup_backbone(name: str) -> BackboneSpec:
    for backbone in BACKBONES:
        if backbone.name == name or backbone.model_id == name:
            return backbone
    raise KeyError(f"unknown backbone: {name}")


def image_backbones() -> list[BackboneSpec]:
    return [backbone for backbone in BACKBONES if backbone.modality == "image"]


def recommended_track2_backbones() -> list[BackboneSpec]:
    return [backbone for backbone in BACKBONES if backbone.recommended_for_track2_full]
