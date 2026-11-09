#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
from functools import partial
from types import SimpleNamespace
import copy

import torch

# Default ImageBind checkpoint location: <ImageBind-main>/.checkpoints/imagebind_huge.pth
# Override with the IMAGEBIND_CKPT environment variable.
IMAGEBIND_CKPT_PATH = os.path.normpath(os.environ.get(
    "IMAGEBIND_CKPT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".checkpoints", "imagebind_huge.pth"),
))
import torch.nn as nn
import torch.nn.functional as F 

from imagebind_finetune.models.helpers import (EinOpsRearrange, LearnableLogitScaling, Normalize,
                            SelectElement, SelectEOSAndProject)
from imagebind_finetune.models.multimodal_preprocessors import (AudioPreprocessor,
                                             IMUPreprocessor, PadIm2Video,
                                             PatchEmbedGeneric,
                                             RGBDTPreprocessor,
                                             SpatioTemporalPosEmbeddingHelper,
                                             TextPreprocessor,
                                             ThermalPreprocessor)
from imagebind_finetune.models.transformer import MultiheadAttention, SimpleTransformer

from layers.temporal_av_attn_layer import (
    TemporalAttentionModule)

import pdb

ModalityType = SimpleNamespace(
    VISION="vision",
    VISION_IMAG="vision_imag",
    TEXT="text",
    AUDIO="audio",
    THERMAL="thermal",
    DEPTH="depth",
    IMU="imu",
)



class ImageBindModel(nn.Module):
    def __init__(
        self,
        video_frames=2,
        kernel_size=(2, 14, 14),
        audio_kernel_size=16,
        audio_stride=10,
        out_embed_dim=768,
        vision_embed_dim=1024,
        vision_num_blocks=24,
        vision_num_heads=16,
        audio_embed_dim=768,
        audio_num_blocks=12,
        audio_num_heads=12,
        audio_num_mel_bins=128,
        audio_target_len=204,
        audio_drop_path=0.1,
        text_embed_dim=768,
        text_num_blocks=12,
        text_num_heads=12,
        depth_embed_dim=384,
        depth_kernel_size=16,
        depth_num_blocks=12,
        depth_num_heads=8,
        depth_drop_path=0.0,
        thermal_embed_dim=768,
        thermal_kernel_size=16,
        thermal_num_blocks=12,
        thermal_num_heads=12,
        thermal_drop_path=0.0,
        imu_embed_dim=512,
        imu_kernel_size=8,
        imu_num_blocks=6,
        imu_num_heads=8,
        imu_drop_path=0.7,
        # spatial attention
        spatial_av_attn_layer_ids=([], []),
        sattn_flag='none',
        # temporal attention
        tattn_flag=False,
        sa_layer_num=1,
        xa_layer_num=1,
        feat_dim=1024,
        hid_dim=256,
        d_ff=512,
        head_num=1,
        dropout=0.1,
        use_adj_in_attn=True,
        gamma=0.6,
        bias=0.2,
        use_mask_in_attn=True,
        win_size=4,
        norm_flag=None,
    ):
        super().__init__()

        self.modality_preprocessors = self._create_modality_preprocessors(
            video_frames,
            vision_embed_dim,
            kernel_size,
            text_embed_dim,
            audio_embed_dim,
            audio_kernel_size,
            audio_stride,
            audio_num_mel_bins,
            audio_target_len,
            depth_embed_dim,
            depth_kernel_size,
            thermal_embed_dim,
            thermal_kernel_size,
            imu_embed_dim,
        )

        self.modality_trunks = self._create_modality_trunks(
            vision_embed_dim,
            vision_num_blocks,
            vision_num_heads,
            text_embed_dim,
            text_num_blocks,
            text_num_heads,
            audio_embed_dim,
            audio_num_blocks,
            audio_num_heads,
            audio_drop_path,
            depth_embed_dim,
            depth_num_blocks,
            depth_num_heads,
            depth_drop_path,
            thermal_embed_dim,
            thermal_num_blocks,
            thermal_num_heads,
            thermal_drop_path,
            imu_embed_dim,
            imu_num_blocks,
            imu_num_heads,
            imu_drop_path,
        )

        self.modality_heads = self._create_modality_heads(
            out_embed_dim,
            vision_embed_dim,
            text_embed_dim,
            audio_embed_dim,
            depth_embed_dim,
            thermal_embed_dim,
            imu_embed_dim,
        )

        self.modality_postprocessors = self._create_modality_postprocessors(
            out_embed_dim
        )

        # spatial attention
        self.spatial_av_attn_layer_ids = spatial_av_attn_layer_ids
        self.sattn_flag = sattn_flag
        # else:
        #     # self.spatial_av_layers = None
        #     print("==> not performing spatial attention")


        # temporal attention
        self.tattn_flag = tattn_flag

        print("==> building temporal attention layers")
        base_args = (sa_layer_num, xa_layer_num, feat_dim, hid_dim, d_ff,
                     head_num, dropout, use_adj_in_attn, gamma, bias,
                     use_mask_in_attn, win_size, norm_flag)
        self.temporal_av_layer = TemporalAttentionModule(*base_args)


    def _create_spatial_av_layers(
        self,
        spatial_av_attn_layer_ids,
        audio_embed_dim=768,
        vision_embed_dim=1024,
    ):
        audio_lids, vision_lids = spatial_av_attn_layer_ids
        module_a2v = nn.Linear(audio_embed_dim, vision_embed_dim, bias=False)
        module_v2a = nn.Linear(vision_embed_dim, audio_embed_dim, bias=False)
        layers_a2v = nn.ModuleList([copy.deepcopy(module_a2v) for i in range(len(audio_lids))])
        layers_v2a = nn.ModuleList([copy.deepcopy(module_v2a) for i in range(len(vision_lids))])
        
        modality_spatial_attn_layers = {
            ModalityType.AUDIO: layers_a2v,
            ModalityType.VISION: layers_v2a
        }
        return nn.ModuleDict(modality_spatial_attn_layers)



    def _create_modality_preprocessors(
        self,
        video_frames=2,
        vision_embed_dim=1024,
        kernel_size=(2, 14, 14),
        text_embed_dim=768,
        audio_embed_dim=768,
        audio_kernel_size=16,
        audio_stride=10,
        audio_num_mel_bins=128,
        audio_target_len=204,
        depth_embed_dim=768,
        depth_kernel_size=16,
        thermal_embed_dim=768,
        thermal_kernel_size=16,
        imu_embed_dim=512,
    ):
        rgbt_stem = PatchEmbedGeneric(
            proj_stem=[
                PadIm2Video(pad_type="repeat", ntimes=2),
                nn.Conv3d(
                    in_channels=3,
                    kernel_size=kernel_size,
                    out_channels=vision_embed_dim,
                    stride=kernel_size,
                    bias=False,
                ),
            ]
        )
        rgbt_preprocessor = RGBDTPreprocessor(
            img_size=[3, video_frames, 224, 224],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            rgbt_stem=rgbt_stem,
            depth_stem=None,
        )

        text_preprocessor = TextPreprocessor(
            context_length=77,
            vocab_size=49408,
            embed_dim=text_embed_dim,
            causal_masking=True,
        )

        audio_stem = PatchEmbedGeneric(
            proj_stem=[
                nn.Conv2d(
                    in_channels=1,
                    kernel_size=audio_kernel_size,
                    stride=audio_stride,
                    out_channels=audio_embed_dim,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=audio_embed_dim),
        )
        audio_preprocessor = AudioPreprocessor(
            img_size=[1, audio_num_mel_bins, audio_target_len],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            audio_stem=audio_stem,
        )

        depth_stem = PatchEmbedGeneric(
            [
                nn.Conv2d(
                    kernel_size=depth_kernel_size,
                    in_channels=1,
                    out_channels=depth_embed_dim,
                    stride=depth_kernel_size,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=depth_embed_dim),
        )

        depth_preprocessor = RGBDTPreprocessor(
            img_size=[1, 224, 224],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            rgbt_stem=None,
            depth_stem=depth_stem,
        )

        thermal_stem = PatchEmbedGeneric(
            [
                nn.Conv2d(
                    kernel_size=thermal_kernel_size,
                    in_channels=1,
                    out_channels=thermal_embed_dim,
                    stride=thermal_kernel_size,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=thermal_embed_dim),
        )
        thermal_preprocessor = ThermalPreprocessor(
            img_size=[1, 224, 224],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            thermal_stem=thermal_stem,
        )

        imu_stem = PatchEmbedGeneric(
            [
                nn.Linear(
                    in_features=48,
                    out_features=imu_embed_dim,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=imu_embed_dim),
        )

        imu_preprocessor = IMUPreprocessor(
            img_size=[6, 2000],
            num_cls_tokens=1,
            kernel_size=8,
            embed_dim=imu_embed_dim,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            imu_stem=imu_stem,
        )

        modality_preprocessors = {
            ModalityType.VISION: rgbt_preprocessor,
            ModalityType.TEXT: text_preprocessor,
            ModalityType.AUDIO: audio_preprocessor,
            ModalityType.DEPTH: depth_preprocessor,
            ModalityType.THERMAL: thermal_preprocessor,
            ModalityType.IMU: imu_preprocessor,
        }

        return nn.ModuleDict(modality_preprocessors)

    def _create_modality_trunks(
        self,
        vision_embed_dim=1024,
        vision_num_blocks=24,
        vision_num_heads=16,
        text_embed_dim=768,
        text_num_blocks=12,
        text_num_heads=12,
        audio_embed_dim=768,
        audio_num_blocks=12,
        audio_num_heads=12,
        audio_drop_path=0.0,
        depth_embed_dim=768,
        depth_num_blocks=12,
        depth_num_heads=12,
        depth_drop_path=0.0,
        thermal_embed_dim=768,
        thermal_num_blocks=12,
        thermal_num_heads=12,
        thermal_drop_path=0.0,
        imu_embed_dim=512,
        imu_num_blocks=6,
        imu_num_heads=8,
        imu_drop_path=0.7,
    ):
        def instantiate_trunk(
            embed_dim, num_blocks, num_heads, pre_transformer_ln, add_bias_kv, drop_path,
        ):
            return SimpleTransformer(
                embed_dim=embed_dim,
                num_blocks=num_blocks,
                ffn_dropout_rate=0.0,
                drop_path_rate=drop_path,
                attn_target=partial(
                    MultiheadAttention,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    bias=True,
                    add_bias_kv=add_bias_kv,
                ),
                pre_transformer_layer=nn.Sequential(
                    nn.LayerNorm(embed_dim, eps=1e-6)
                    if pre_transformer_ln
                    else nn.Identity(),
                    EinOpsRearrange("b l d -> l b d"),
                ),
                post_transformer_layer=EinOpsRearrange("l b d -> b l d"),
            )

        modality_trunks = {}
        modality_trunks[ModalityType.VISION] = instantiate_trunk(
            vision_embed_dim,
            vision_num_blocks,
            vision_num_heads,
            pre_transformer_ln=True,
            add_bias_kv=False,
            drop_path=0.0,
        )
        modality_trunks[ModalityType.TEXT] = instantiate_trunk(
            text_embed_dim,
            text_num_blocks,
            text_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=False,
            drop_path=0.0,
        )
        modality_trunks[ModalityType.AUDIO] = instantiate_trunk(
            audio_embed_dim,
            audio_num_blocks,
            audio_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=audio_drop_path,
        )
        modality_trunks[ModalityType.DEPTH] = instantiate_trunk(
            depth_embed_dim,
            depth_num_blocks,
            depth_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=depth_drop_path,
        )
        modality_trunks[ModalityType.THERMAL] = instantiate_trunk(
            thermal_embed_dim,
            thermal_num_blocks,
            thermal_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=thermal_drop_path,
        )
        modality_trunks[ModalityType.IMU] = instantiate_trunk(
            imu_embed_dim,
            imu_num_blocks,
            imu_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=imu_drop_path,
        )

        return nn.ModuleDict(modality_trunks)

    def _create_modality_heads(
        self,
        out_embed_dim,
        vision_embed_dim,
        text_embed_dim,
        audio_embed_dim,
        depth_embed_dim,
        thermal_embed_dim,
        imu_embed_dim,
    ):
        modality_heads = {}

        modality_heads[ModalityType.VISION] = nn.Sequential(
            nn.LayerNorm(normalized_shape=vision_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(vision_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.TEXT] = SelectEOSAndProject(
            proj=nn.Sequential(
                nn.LayerNorm(normalized_shape=text_embed_dim, eps=1e-6),
                nn.Linear(text_embed_dim, out_embed_dim, bias=False),
            )
        )

        modality_heads[ModalityType.AUDIO] = nn.Sequential(
            nn.LayerNorm(normalized_shape=audio_embed_dim, eps=1e-6),
            SelectElement(index=0), #! 0 denoting selecting first [cls] token
            nn.Linear(audio_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.DEPTH] = nn.Sequential(
            nn.LayerNorm(normalized_shape=depth_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(depth_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.THERMAL] = nn.Sequential(
            nn.LayerNorm(normalized_shape=thermal_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(thermal_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.IMU] = nn.Sequential(
            nn.LayerNorm(normalized_shape=imu_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Dropout(p=0.5),
            nn.Linear(imu_embed_dim, out_embed_dim, bias=False),
        )

        return nn.ModuleDict(modality_heads)


    def _create_modality_postprocessors(self, out_embed_dim):
        modality_postprocessors = {}

        modality_postprocessors[ModalityType.VISION] = Normalize(dim=-1)
        modality_postprocessors[ModalityType.TEXT] = nn.Sequential(
            Normalize(dim=-1), LearnableLogitScaling(learnable=True)
        )
        modality_postprocessors[ModalityType.AUDIO] = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=20.0, learnable=False),
        )
        modality_postprocessors[ModalityType.DEPTH] = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=5.0, learnable=False),
        )
        modality_postprocessors[ModalityType.THERMAL] = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=10.0, learnable=False),
        )
        modality_postprocessors[ModalityType.IMU] = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=5.0, learnable=False),
        )

        return nn.ModuleDict(modality_postprocessors)


    def spatial_attention(self, audio_tokens, vision_tokens, layer_a2v, layer_v2a):
        # audio_tokens: [229, bs*10, 768]
        # vision_tokens: [257, bs*10, 1024]
        def process_sattn(a_cls_token, v_patch_tokens, layer_a2v):
            a_cls_token = layer_a2v(a_cls_token) # [1, bs*10, 768->1024]
            norm_a_cls_token = F.normalize(a_cls_token, dim=-1) # [1, bs*10, 1024]
            norm_v_patch_tokens = F.normalize(v_patch_tokens, dim=-1) # [256, bs*10, 1024]
            av_simm = torch.sum(torch.mul(norm_a_cls_token, norm_v_patch_tokens), dim=-1) # [256, bs*10]
            # soft_av_simm = F.softmax(av_simm, dim=0).unsqueeze(-1) # [256, bs*10, 1] #! may also try using 'sigmoid'
            # updated_v_patch_tokens = v_patch_tokens + torch.mul(v_patch_tokens, soft_av_simm)
            updated_v_patch_tokens = v_patch_tokens + torch.mul(v_patch_tokens, av_simm.unsqueeze(-1))
            # pdb.set_trace()
            return updated_v_patch_tokens

        a_cls_token = audio_tokens[0, :, :].unsqueeze(0) # [1, bs*10, 768]
        a_patch_tokens = audio_tokens[1:, :, :] # [228, bs*10, 768]
        v_cls_token = vision_tokens[0, :, :].unsqueeze(0) # [1, bs*10, 1024]
        v_patch_tokens = vision_tokens[1:, :, :] # [256, bs*10, 1024]

        updated_a_patch_tokens = process_sattn(v_cls_token, a_patch_tokens, layer_v2a) # [228, bs*10, 768]
        updated_v_patch_tokens = process_sattn(a_cls_token, v_patch_tokens, layer_a2v) # [256, bs*10, 1024]

        updated_a_tokens = torch.cat([a_cls_token, updated_a_patch_tokens], dim=0) # [229, bs*10, 768]
        updated_v_tokens = torch.cat([v_cls_token, updated_v_patch_tokens], dim=0) # [257, bs*10, 1024]
        return updated_a_tokens, updated_v_tokens


    def forward(self, inputs):
        # pdb.set_trace()
        outputs = {}
        # for modality_key, modality_value in inputs.items():
        inputs_temp = {}
        reduce_flag = {}

        # iHSV: extract imaginary visual input before main loop (shares vision preprocessor)
        has_vision_imag = ModalityType.VISION_IMAG in inputs
        vision_imag_value = None
        if has_vision_imag:
            vision_imag_value = inputs.pop(ModalityType.VISION_IMAG)

        for modality_key, modality_value in inputs.items():
            reduce_list = (
                modality_value.ndim >= 5
            )  #! Audio and Video inputs consist of multiple clips
            if reduce_list:
                B, S = modality_value.shape[:2]
                modality_value = modality_value.reshape(
                    B * S, *modality_value.shape[2:]
                )
                reduce_flag[modality_key] = True
            else:
                reduce_flag[modality_key] = False
            if modality_value is not None:
                modality_value = self.modality_preprocessors[modality_key](
                    **{modality_key: modality_value}
                )
                inputs_temp[modality_key] = modality_value
                # trunk_inputs = modality_value["trunk"]
                # head_inputs = modality_value["head"]
                # inputs_temp[modality_key]['trunk'] = trunk_inputs
                # inputs_temp[modality_key]['head'] = head_inputs
            # pdb.set_trace()
        ###################! processing text embedding ####################
        text_modality_key = 'text'
        text_modality_value = None
        if text_modality_key in inputs_temp:
            text_trunk_inputs = inputs_temp[text_modality_key]['trunk']
            text_head_inputs = inputs_temp[text_modality_key]['head']
            text_transformer_blocks = self.modality_trunks[text_modality_key].blocks
            text_tokens = text_trunk_inputs['tokens'] #! Transformer encoders runs here
            #! add several learnable tokens here
            # TODO
            if self.modality_trunks[text_modality_key].pre_transformer_layer:
                text_tokens = self.modality_trunks[text_modality_key].pre_transformer_layer(text_tokens)
            # pdb.set_trace()
            for blk_id, blk in enumerate(text_transformer_blocks):
                #! processing each Transformer block
                text_tokens = blk(text_tokens, attn_mask=text_trunk_inputs['attn_mask'])
            # pdb.set_trace()
            if self.modality_trunks[text_modality_key].post_transformer_layer:
                text_tokens = self.modality_trunks[text_modality_key].post_transformer_layer(text_tokens)
            text_modality_value = text_tokens

            text_modality_value = self.modality_heads[text_modality_key](
                text_modality_value, **text_head_inputs
            )
            text_modality_value = self.modality_postprocessors[text_modality_key](
                text_modality_value
            )
            if reduce_flag.get(text_modality_key, False):
                text_modality_value = text_modality_value.reshape(B, S, -1)
                # text_modality_value = text_modality_value.mean(dim=1)
            # text_modality_value = text_modality_value + 0.5 * self.task_res_text_learner(text_modality_value) # fixed version


        ###################! pre-processing audio and visual embedding ####################
        audio_modality_key = 'audio'
        vision_modality_key = 'vision'
        audio_modality_value = None
        vision_modality_value = None
        has_audio = audio_modality_key in inputs_temp
        has_vision = vision_modality_key in inputs_temp

        if has_audio:
            audio_trunk_inputs = inputs_temp[audio_modality_key]['trunk']
            audio_head_inputs = inputs_temp[audio_modality_key]['head']
            audio_transformer_blocks = self.modality_trunks[audio_modality_key].blocks
            audio_tokens = audio_trunk_inputs['tokens']
            if self.modality_trunks[audio_modality_key].pre_transformer_layer:
                audio_tokens = self.modality_trunks[audio_modality_key].pre_transformer_layer(audio_tokens)

        if has_vision:
            vision_trunk_inputs = inputs_temp[vision_modality_key]['trunk']
            vision_head_inputs = inputs_temp[vision_modality_key]['head']
            vison_transformer_blocks = self.modality_trunks[vision_modality_key].blocks
            vision_tokens = vision_trunk_inputs['tokens']
            if self.modality_trunks[vision_modality_key].pre_transformer_layer:
                vision_tokens = self.modality_trunks[vision_modality_key].pre_transformer_layer(vision_tokens)

        if has_audio and has_vision:
            # pdb.set_trace()
            sattn_type = self.sattn_flag
            if sattn_type == 'none':
                for audio_blk_id, audio_blk in enumerate(audio_transformer_blocks):
                    audio_tokens = audio_blk(audio_tokens, attn_mask=None)
                for vision_blk_id, vision_blk in enumerate(vison_transformer_blocks):
                    vision_tokens = vision_blk(vision_tokens, attn_mask=None)
            else: #! learnable mutual audio-vision spatial attention
                a2v_modulelist, v2a_modulelist = self.spatial_av_layers[audio_modality_key], self.spatial_av_layers[vision_modality_key]
                a2v_layer_ids, v2a_layer_ids = self.spatial_av_attn_layer_ids
                audio_blocks_num, vision_blocks_num = len(audio_transformer_blocks), len(vison_transformer_blocks)
                if sattn_type in ['bothFirst', 'bothLast', 'evenFirst', 'evenLast', 'fixedBlkids']:
                    a_blk_id, v_blk_id = 0, 0
                    for i in range(len(a2v_layer_ids)):
                        while(a_blk_id <= a2v_layer_ids[i]):
                            audio_tokens = audio_transformer_blocks[a_blk_id](audio_tokens, attn_mask=None)
                            a_blk_id += 1
                        while(v_blk_id <= v2a_layer_ids[i]):
                            vision_tokens = vison_transformer_blocks[v_blk_id](vision_tokens, attn_mask=None)
                            v_blk_id += 1
                        audio_tokens, vision_tokens = self.spatial_attention(audio_tokens, vision_tokens, a2v_modulelist[i], v2a_modulelist[i])
                    for ai in range(a_blk_id, audio_blocks_num):
                        audio_tokens = audio_transformer_blocks[ai](audio_tokens, attn_mask=None)
                    for vi in range(v_blk_id, vision_blocks_num):
                        vision_tokens = vison_transformer_blocks[vi](vision_tokens, attn_mask=None)
                else:
                    raise NotImplementedError
        elif has_audio:
            # Audio-only pass (e.g., phase through audio encoder)
            for audio_blk in self.modality_trunks[audio_modality_key].blocks:
                audio_tokens = audio_blk(audio_tokens, attn_mask=None)
        elif has_vision:
            # Vision-only pass
            for vision_blk in self.modality_trunks[vision_modality_key].blocks:
                vision_tokens = vision_blk(vision_tokens, attn_mask=None)

        # audio token post processing
        if has_audio:
            if self.modality_trunks[audio_modality_key].post_transformer_layer:
                audio_tokens = self.modality_trunks[audio_modality_key].post_transformer_layer(audio_tokens)
            audio_modality_value = audio_tokens
            audio_modality_value = self.modality_heads[audio_modality_key](
                audio_modality_value, **audio_head_inputs
            )
            audio_modality_value = self.modality_postprocessors[audio_modality_key](
                audio_modality_value
            )
            if reduce_flag.get(audio_modality_key, False):
                audio_modality_value = audio_modality_value.reshape(B, S, -1)

        # vision token post processing
        if has_vision:
            if self.modality_trunks[vision_modality_key].post_transformer_layer:
                vision_tokens = self.modality_trunks[vision_modality_key].post_transformer_layer(vision_tokens)
            vision_modality_value = vision_tokens
            vision_modality_value = self.modality_heads[vision_modality_key](
                vision_modality_value, **vision_head_inputs
            )
            vision_modality_value = self.modality_postprocessors[vision_modality_key](
                vision_modality_value
            )
            if reduce_flag.get(vision_modality_key, False):
                vision_modality_value = vision_modality_value.reshape(B, S, -1)

        if vision_modality_value is not None and audio_modality_value is not None:
            if torch.sum(vision_modality_value.isnan()) > 0 or torch.sum(audio_modality_value.isnan()) > 0:
                pdb.set_trace()

        #! iHSV: process imaginary visual through the SAME frozen vision encoder
        imag_vision_modality_value = None
        if has_vision_imag and vision_imag_value is not None:
            # Reshape: [B, S, 3, 224, 224] -> [B*S, 3, 224, 224]
            vi = vision_imag_value
            if vi.ndim >= 5:
                vi = vi.reshape(B * S, *vi.shape[2:])
            # Reuse the same frozen vision preprocessor, trunk, head, postprocessor
            vi_preprocessed = self.modality_preprocessors[vision_modality_key](
                **{vision_modality_key: vi}
            )
            vi_trunk_inputs = vi_preprocessed['trunk']
            vi_head_inputs = vi_preprocessed['head']
            vi_tokens = vi_trunk_inputs['tokens']
            if self.modality_trunks[vision_modality_key].pre_transformer_layer:
                vi_tokens = self.modality_trunks[vision_modality_key].pre_transformer_layer(vi_tokens)
            for blk in self.modality_trunks[vision_modality_key].blocks:
                vi_tokens = blk(vi_tokens, attn_mask=None)
            if self.modality_trunks[vision_modality_key].post_transformer_layer:
                vi_tokens = self.modality_trunks[vision_modality_key].post_transformer_layer(vi_tokens)
            imag_vision_modality_value = self.modality_heads[vision_modality_key](
                vi_tokens, **vi_head_inputs
            )
            imag_vision_modality_value = self.modality_postprocessors[vision_modality_key](
                imag_vision_modality_value
            )
            if vision_imag_value.ndim >= 5:
                imag_vision_modality_value = imag_vision_modality_value.reshape(B, S, -1)

        #! learnable temporal audio-vision attention
        if has_audio and has_vision:
            audio_modality_value, vision_modality_value = self.temporal_av_layer(
                audio_modality_value, vision_modality_value
            )

        if has_vision and has_audio:
            if torch.sum(vision_modality_value.isnan()) > 0 or torch.sum(audio_modality_value.isnan()) > 0:
                pdb.set_trace()

        if text_modality_value is not None:
            outputs[text_modality_key] = text_modality_value
        if audio_modality_value is not None:
            outputs[audio_modality_key] = audio_modality_value
        if vision_modality_value is not None:
            outputs[vision_modality_key] = vision_modality_value
        if imag_vision_modality_value is not None:
            outputs[ModalityType.VISION_IMAG] = imag_vision_modality_value

        return outputs


def imagebind_huge(
    pretrained=False,
    # spatial attention
    spatial_av_attn_layer_ids=([0], [0]),
    sattn_flag='none',
    # temporal attention
    tattn_flag=False,
    sa_layer_num=1,
    xa_layer_num=1,
    feat_dim=1024,
    hid_dim=256,
    d_ff=512,
    head_num=1,
    dropout=0.1,
    use_adj_in_attn=False,
    gamma=0.6,
    bias=0.2,
    use_mask_in_attn=False,
    win_size=4,
    norm_flag=None,
    # text tuning
):
    model = ImageBindModel(
        vision_embed_dim=1280,
        vision_num_blocks=32,
        vision_num_heads=16,
        text_embed_dim=1024,
        text_num_blocks=24,
        text_num_heads=16,
        out_embed_dim=1024,
        audio_drop_path=0.1,
        imu_drop_path=0.7,
        # spatial attention
        spatial_av_attn_layer_ids=spatial_av_attn_layer_ids,
        sattn_flag=sattn_flag,
        # temporal attention
        tattn_flag=tattn_flag,
        sa_layer_num=sa_layer_num,
        xa_layer_num=xa_layer_num,
        feat_dim=feat_dim,
        hid_dim=hid_dim,
        d_ff=d_ff,
        head_num=head_num,
        dropout=dropout,
        use_adj_in_attn=use_adj_in_attn,
        gamma=gamma,
        bias=bias,
        use_mask_in_attn=use_mask_in_attn,
        win_size=win_size,
        norm_flag=norm_flag,
    )

    def initialize_imagebind_weights(model):
        imagebind_model_dict = model.state_dict()
        pretrained_path = IMAGEBIND_CKPT_PATH
        pretrained_state_dicts = torch.load(pretrained_path)
        state_dict = {k : v for k, v in pretrained_state_dicts.items() if k in imagebind_model_dict.keys()}
        imagebind_model_dict.update(state_dict)
        model.load_state_dict(imagebind_model_dict)
        # pdb.set_trace()
        print("==> Load pretrained Imagemodel parameters")
        return model

    if pretrained:
        pretrained_path = IMAGEBIND_CKPT_PATH
        if not os.path.exists(pretrained_path):
        # if not os.path.exists(".checkpoints/imagebind_huge.pth"):
            print(
                f"Downloading imagebind weights to {pretrained_path} ..."
            )
            os.makedirs(os.path.dirname(pretrained_path), exist_ok=True)
            torch.hub.download_url_to_file(
                "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth",
                pretrained_path,
                progress=True,
            )

        # model.load_state_dict(torch.load(".checkpoints/imagebind_huge.pth"))
        model = initialize_imagebind_weights(model)

    return model
