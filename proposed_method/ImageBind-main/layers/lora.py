"""
Parameter-efficient adaptation for ImageBind audio encoder.

Provides two mechanisms:
  1. BlockAdapter: low-rank adapter added after each transformer block
     (Houlsby et al., 2019; He et al., 2022)
  2. PhaseAudioLoRA: wraps the frozen audio encoder pipeline, injects
     adapters at each block, and optionally uses a new patch embedding
     stem for phase/IF input.

Usage:
    phase_encoder = PhaseAudioLoRA(model, rank=8, alpha=16)
    phase_feas = phase_encoder(phase_mel_input)  # [B, 10, 1024]

Only the adapter parameters + optional new stem are trainable.
Frozen encoder weights are referenced, not copied.
"""
import math
import torch
import torch.nn as nn


class BlockAdapter(nn.Module):
    """
    Low-rank adapter added after each transformer block.

    Functionally equivalent to LoRA but applied at the block output level,
    avoiding the need to modify nn.MultiheadAttention internals.

    Parameters: 2 * dim * rank (e.g., 2 * 768 * 8 = 12,288 per block)
    """
    def __init__(self, dim=768, rank=8, alpha=16):
        super().__init__()
        self.scaling = alpha / rank
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        # Init: down with small random, up with zeros → starts as identity
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.down(x)) * self.scaling


class PhaseAudioLoRA(nn.Module):
    """
    Runs phase/IF input through the frozen ImageBind audio encoder
    with trainable low-rank adapters at each transformer block.

    Architecture:
        phase_stem (trainable Conv2d) → frozen audio preprocessor pos_embed
        → 12 frozen transformer blocks + 12 trainable adapters
        → frozen audio head (LayerNorm → CLS select → Linear)
        → frozen audio postprocessor (Normalize → LogitScaling)

    The frozen weights are REFERENCED from the main model (not copied).
    Only phase_stem + adapters are trainable (~160K params).

    Args:
        model: the ImageBindModel instance (frozen)
        rank: adapter bottleneck rank (default 8)
        alpha: LoRA-style scaling factor (default 16)
        init_stem_from_pretrained: if True, initialize phase stem from
            pretrained audio stem weights (recommended)
        stem_init_scale: scale factor for pretrained stem init (default 0.1,
            smaller = starts closer to pretrained behavior)
    """
    def __init__(self, model, rank=8, alpha=16,
                 init_stem_from_pretrained=True, stem_init_scale=0.1):
        super().__init__()

        # References to frozen components (NOT copies)
        self.preprocessor = model.modality_preprocessors['audio']
        self.trunk = model.modality_trunks['audio']
        self.head = model.modality_heads['audio']
        self.postprocessor = model.modality_postprocessors['audio']

        # Audio encoder config
        audio_embed_dim = self.trunk.blocks[0].attn.embed_dim  # 768
        num_blocks = len(self.trunk.blocks)  # 12

        # Trainable: new patch embedding stem for phase/IF input
        # Phase mel has different statistics from magnitude mel,
        # so we need a separate stem. Shape: [1, 128, 204] → patches → 768-dim
        # AudioPreprocessor inherits from RGBDTPreprocessor, stem is at .rgbt_stem
        pretrained_stem = self.preprocessor.rgbt_stem  # PatchEmbedGeneric
        # PatchEmbedGeneric.proj is either nn.Sequential or a single Conv2d
        if isinstance(pretrained_stem.proj, nn.Sequential):
            pretrained_stem_conv = pretrained_stem.proj[0]  # first layer is Conv2d
        else:
            pretrained_stem_conv = pretrained_stem.proj  # single Conv2d
        self.phase_stem = nn.Conv2d(
            in_channels=1,
            out_channels=audio_embed_dim,
            kernel_size=pretrained_stem_conv.kernel_size,
            stride=pretrained_stem_conv.stride,
            bias=False,
        )
        if init_stem_from_pretrained:
            with torch.no_grad():
                self.phase_stem.weight.copy_(
                    pretrained_stem_conv.weight * stem_init_scale
                )

        # Trainable: norm layer for new stem (matching pretrained structure)
        self.phase_stem_norm = nn.LayerNorm(audio_embed_dim)
        pretrained_norm = pretrained_stem.norm_layer
        if pretrained_norm is not None and isinstance(pretrained_norm, nn.LayerNorm):
            with torch.no_grad():
                self.phase_stem_norm.weight.copy_(pretrained_norm.weight)
                self.phase_stem_norm.bias.copy_(pretrained_norm.bias)

        # Trainable: adapters for each transformer block
        self.adapters = nn.ModuleList([
            BlockAdapter(dim=audio_embed_dim, rank=rank, alpha=alpha)
            for _ in range(num_blocks)
        ])

        # Freeze all referenced components
        for component in [self.preprocessor, self.trunk, self.head, self.postprocessor]:
            for p in component.parameters():
                p.requires_grad = False

    def _stem_forward(self, x):
        """Run phase input through the new trainable stem.
        Input: [B*T, 1, 128, 204]
        Output: [B*T, num_patches, 768]
        """
        # Patch embedding
        x = self.phase_stem(x)  # [B*T, 768, H', W']
        # Flatten spatial dims
        x = x.flatten(2).transpose(1, 2)  # [B*T, num_patches, 768]
        # Norm
        x = self.phase_stem_norm(x)
        return x

    def forward(self, phase_input):
        """
        Args:
            phase_input: [B, T, 1, 128, 204] phase mel spectrogram (IF projected)

        Returns:
            phase_feas: [B, T, 1024] phase embeddings in ImageBind's shared space
        """
        B, T = phase_input.shape[:2]
        x = phase_input.reshape(B * T, *phase_input.shape[2:])  # [B*T, 1, 128, 204]

        # Step 1: Trainable stem → patch tokens
        patch_tokens = self._stem_forward(x)  # [B*T, num_patches, 768]

        # Step 2: Add CLS token + positional embedding (from frozen preprocessor)
        cls_token = self.preprocessor.cls_token.expand(
            patch_tokens.shape[0], -1, -1
        )  # [B*T, 1, 768]
        tokens = torch.cat([cls_token, patch_tokens], dim=1)  # [B*T, 1+num_patches, 768]

        # Add positional embedding
        pos_embed = self.preprocessor.pos_embedding_helper.get_pos_embedding(x, tokens)
        tokens = tokens + pos_embed

        # Step 3: Pre-transformer layer (frozen)
        if self.trunk.pre_transformer_layer:
            tokens = self.trunk.pre_transformer_layer(tokens)

        # Step 4: Transformer blocks (frozen) + adapters (trainable)
        # Per-block LoRA: gradients flow through frozen blocks back to stem
        for blk, adapter in zip(self.trunk.blocks, self.adapters):
            tokens = blk(tokens, attn_mask=None)
            tokens = tokens + adapter(tokens)  # adapter residual

        # Step 5: Post-transformer layer (frozen)
        if self.trunk.post_transformer_layer:
            tokens = self.trunk.post_transformer_layer(tokens)

        # Step 6: Head (frozen): LayerNorm → CLS select → Linear → 1024
        output = self.head(tokens)

        # Step 7: Postprocessor (frozen): Normalize → LogitScaling
        output = self.postprocessor(output)

        # Reshape back
        output = output.reshape(B, T, -1)  # [B, T, 1024]
        return output

    def trainable_parameters(self):
        """Return only trainable parameters (for optimizer)."""
        params = []
        params.extend(self.phase_stem.parameters())
        params.extend(self.phase_stem_norm.parameters())
        params.extend(self.adapters.parameters())
        return params

    def num_trainable_params(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.trainable_parameters())
