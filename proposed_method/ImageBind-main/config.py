from easydict import EasyDict as edict
import os as _os
import yaml
import pdb

"""
default config
"""
cfg = edict()

# Dataset selection: "ovavel", "ave", or "ave_ov".
# Override at launch time without editing this file:  OVAVEL_DATASET=ave_ov bash ...
cfg.DATASET = _os.environ.get("OVAVEL_DATASET", "ovavel")

# -----------------------------------------------------------------------------
# Paths. Everything is resolved relative to this repository so a fresh clone
# runs without edits. Override with environment variables if needed:
#   OVAVEL_DATA_DIR  OV-AVEBench preprocessed data (default <repo>/ovave_dataset_preprocessed)
#   OVAVEL_META_DIR  OV-AVEBench meta/annotation files (default <repo>/meta_anno_files)
#   OVAVEL_AVE_DIR   AVE dataset root, for OVAVEL_DATASET=ave|ave_ov (default <repo>/AVE_Dataset)
# -----------------------------------------------------------------------------
CODE_DIR = _os.path.dirname(_os.path.abspath(__file__))
REPO_ROOT = _os.path.abspath(_os.path.join(CODE_DIR, "..", ".."))
_OVAVEL_DATA = _os.environ.get("OVAVEL_DATA_DIR", _os.path.join(REPO_ROOT, "ovave_dataset_preprocessed"))
_OVAVEL_META = _os.environ.get("OVAVEL_META_DIR", _os.path.join(REPO_ROOT, "meta_anno_files"))
_AVE_DIR = _os.environ.get("OVAVEL_AVE_DIR", _os.path.join(REPO_ROOT, "AVE_Dataset"))

# =============================================================================
# BACKBONE selection — pre-extracted feature source.
# Toggle cfg.BACKBONE.NAME; feat_dim and features subdir auto-populate from the
# preset table. The training script's TemporalAttentionModel uses
# cfg.BACKBONE.FEAT_DIM, and dataloader_ave_features.py reads from
# <DATA_BASE_DIR>/<cfg.BACKBONE.FEATURES_SUBDIR>/.
#
# To add a new backbone: add an entry to _BACKBONE_PRESETS below.
# =============================================================================
cfg.BACKBONE = edict()
cfg.BACKBONE.NAME = _os.environ.get("OVAVEL_BACKBONE", "imagebind")   # 'imagebind' | 'languagebind' | ...

_BACKBONE_PRESETS = {
    "imagebind":    {"feat_dim": 1024, "features_subdir": "imagebind_features"},
    "languagebind": {"feat_dim": 768,  "features_subdir": "languagebind_features"},
}
_preset = _BACKBONE_PRESETS[cfg.BACKBONE.NAME]
cfg.BACKBONE.FEAT_DIM = _preset["feat_dim"]
cfg.BACKBONE.FEATURES_SUBDIR = _preset["features_subdir"]

if cfg.DATASET == "ave":
    cfg.TRAIN_BG_CLASS_ID = 28 # train (28 categories, BG = index 28)
    cfg.TOTAL_BG_CLASS_ID = 28 # total (same 28 categories for AVE, no open-set split)
    cfg.DATA_BASE_DIR = _AVE_DIR
    cfg.META_CSV_PATH = _os.path.join(_AVE_DIR, "ave_dataset_meta_combined.csv")
    cfg.ANNO_JSON_PATH = _os.path.join(_AVE_DIR, "generated_annotations.json")
    cfg.TOTAL_CLOSE_OPEN_CATEGORY_CSV_PATH = _os.path.join(_AVE_DIR, "ave_categories.csv")
    cfg.TRAIN_CLOSE_CATEGORY_CSV_PATH = _os.path.join(_AVE_DIR, "ave_categories.csv")
elif cfg.DATASET == "ave_ov":
    # AVE with 8 unseen classes: 20 close + 8 open + "other"
    # Close (train): 21 text entries (indices 0-19 close + 20 "other")
    # Total (test):  29 text entries (indices 0-19 close + 20-27 open + 28 "other")
    # BG_CLASS_ID = index of "other" (same role as "Background" in original AVE)
    cfg.TRAIN_BG_CLASS_ID = 20 # "other" at index 20 in train CSV (21 entries)
    cfg.TOTAL_BG_CLASS_ID = 28 # "other" at index 28 in total CSV (29 entries)
    cfg.DATA_BASE_DIR = _AVE_DIR
    cfg.META_CSV_PATH = _os.path.join(_AVE_DIR, "ave_ov_dataset_meta.csv")
    cfg.ANNO_JSON_PATH = _os.path.join(_AVE_DIR, "generated_annotations.json")
    cfg.TOTAL_CLOSE_OPEN_CATEGORY_CSV_PATH = _os.path.join(_AVE_DIR, "ave_ov_total_close_open_categories.csv")
    cfg.TRAIN_CLOSE_CATEGORY_CSV_PATH = _os.path.join(_AVE_DIR, "ave_ov_train_close_categories.csv")
else:
    cfg.TRAIN_BG_CLASS_ID = 46 # train
    cfg.TOTAL_BG_CLASS_ID = 67 # total
    cfg.DATA_BASE_DIR = _OVAVEL_DATA
    cfg.META_CSV_PATH = _os.path.join(_OVAVEL_META, "ovave_dataset_meta.csv")
    cfg.ANNO_JSON_PATH = _os.path.join(_OVAVEL_META, "released_ovavel_dataset_anno.json")
    cfg.TOTAL_CLOSE_OPEN_CATEGORY_CSV_PATH = _os.path.join(_OVAVEL_META, "ovave_total_close_open_categories.csv")
    cfg.TRAIN_CLOSE_CATEGORY_CSV_PATH = _os.path.join(_OVAVEL_META, "ovave_train_close_categories.csv")

#### spatial attention
cfg.SA = edict()
cfg.SA.SA_ATTN_FLAG = False #!
cfg.SA.SA_ATTN_TYPE = "bothLast" # select from [bothFirst, bothLast, evenFirst, evenLast, fixedBlkids]
cfg.SA.SA_BOTH_LAYER_K = 1
cfg.SA.SA_EVEN_LAYER_K = 1
cfg.SA.SA_FIXED_LAYER_A = [0, 3]
cfg.SA.SA_FIXED_LAYER_V = [8, 16]

#### temporal attention
cfg.TA = edict()
cfg.TA.TA_ATTN_FLAG = True #! Enable temporal attention (uses PhaseAwareComplexAttention when USE_PHASE_AWARE_COMPLEX_ATTN=True)
cfg.TA.SA_LAYER_NUM = 1  # Self-attention layers
cfg.TA.XA_LAYER_NUM = 1  # Cross-attention layers (allocated at init for RNG-equivalence with original repo; not used in forward())
cfg.TA.HIDDEN_DIM = 256
cfg.TA.FF_DIM = 512
cfg.TA.HEAD_NUM = 1
cfg.TA.DROPOUT = 0.1
cfg.TA.USE_ADJ_IN_ATTN = False
cfg.TA.GAMMA = 0.6
cfg.TA.BIAS = 0.2
cfg.TA.USE_MASK_IN_ATTN = False
cfg.TA.WIN_SIZE = 5
cfg.TA.NORM_FLAG = None

# === ALL COMPLEX VARIANTS DISABLED - Using real TemporalAttentionModule ===
cfg.TA.USE_COMPLEX_ATTN = False  #! Use ComplexTemporalAttentionModule (audio=real, visual=imag)
cfg.TA.USE_DUAL_STREAM_ATTN = False  #! Use SeparateComplexTemporalAttentionModule (separate complex attention for audio and visual)
cfg.TA.USE_CROSS_MODAL_ATTN = False  #! Use ComplexCrossModalAttentionModule (complex self-attn + cross-attn)
cfg.TA.USE_COMPLEX_PROJ_ATTN = False  #! Use ComplexProjectionSelfAttentionModule (complex 1x1 conv projection + real self-attn)
cfg.TA.USE_COMPLEX_TEMPORAL_PROJ_ATTN = False  #! Use ComplexTemporalProjectionSelfAttentionModule (complex 5x1 conv projection + real self-attn)
cfg.TA.TEMPORAL_KERNEL = 5  #! Kernel size for ComplexTemporalProjectionSelfAttentionModule
cfg.TA.USE_TEXT_HEADED_ATTN = False  #! Use TextHeadedTemporalAttention (text categories as query heads)
cfg.TA.USE_TEXT_HEADED_COMPLEX_ATTN = False  #! Use TextHeadedComplexTemporalAttention (complex version)
cfg.TA.TEXT_HEADED_REFINEMENT_LAYERS = 1  #! Number of refinement layers after text-headed attention

#### Dual Complex Self-Attention with Text as Imaginary
cfg.TA.USE_DUAL_COMPLEX_TEXT_IMAG = False  #! Use DualComplexSelfAttentionWithTextImag (text=imag, audio/visual=real)
cfg.TA.DUAL_COMPLEX_NUM_LAYERS = 1  #! Number of complex transformer layers
cfg.TA.DUAL_COMPLEX_TEXT_BROADCAST = 'attention'  #! How to broadcast text: 'attention', 'mean', 'project'
cfg.TA.DUAL_COMPLEX_OUTPUT_MODE = 'real'  #! How to extract real from complex: 'real', 'magnitude', 'concat', 'gate'

#### Identity Complex Ablation (real=imag=modality, no text)
cfg.TA.USE_IDENTITY_COMPLEX_ABLATION = False  #! Ablation: use modality as both real and imag (z = x + i*x)

#### Split Complex Attention (split 1024 -> 512 real + 512 imag, then concat back)
cfg.TA.USE_SPLIT_COMPLEX_ATTN = False  #! Split feature into two halves for real/imag, preserves all information

#### Projected Complex Attention (linear projection from real to complex, full 1024-dim)
cfg.TA.USE_PROJECTED_COMPLEX_ATTN = False  #! Project 1024 real -> 1024 real + 1024 imag via learned linear layers
cfg.TA.PROJECTED_COMPLEX_OUTPUT_MODE = 'real'  #! Output mode: 'real', 'magnitude', or 'concat'

#### Temporal Diff Complex Attention (real=frame, imag=temporal difference)
cfg.TA.USE_TEMPORAL_DIFF_COMPLEX_ATTN = False  #! real=static features, imag=temporal differences (motion/dynamics)
cfg.TA.TEMPORAL_DIFF_PADDING_MODE = 'copy'  #! Padding mode for last frame: 'copy', 'zero', 'circular', 'reverse'

#### Aligned Temporal Diff Complex Attention (For Paper - with alignment loss)
cfg.TA.USE_ALIGNED_TEMPORAL_DIFF_ATTN = False  #! Temporal diff + residual + alignment loss for fair comparison
cfg.TA.ALIGNED_RESIDUAL_WEIGHT = 0.5  #! Initial gate value (0.5=balanced, higher=more original preservation)
cfg.TA.ALIGNED_LOSS_WEIGHT = 0.1  #! Weight for alignment loss in total loss

#### Rotary Complex Attention (phase = temporal position, like RoPE)
cfg.TA.USE_ROTARY_COMPLEX_ATTN = False  #! Phase encodes temporal position, magnitude encodes content
cfg.TA.ROTARY_LEARNABLE_FREQ = True  #! Whether to learn frequency parameters
cfg.TA.ROTARY_NUM_FREQ_BANDS = None  #! Multi-scale: set to 4 or 8 for different frequency bands, None for standard RoPE

#### Phase-Aware Complex Attention (Hilbert Projection + ModReLU) - DISABLED
cfg.TA.USE_PHASE_AWARE_COMPLEX_ATTN = False  #! Use PhaseAwareComplexAttention with Hilbert projection and ModReLU
cfg.TA.PHASE_AWARE_USE_TEMPORAL_BIAS = True  #! Add temporal position encoding to imaginary part
cfg.TA.PHASE_AWARE_OUTPUT_MODE = 'real'  #! Output mode: 'real', 'magnitude', or 'concat'
cfg.TA.PHASE_AWARE_MODRELU_BIAS_INIT = -0.1  #! Initial bias for ModReLU (negative creates dead zone)

#### Phase Coherence Loss (for F1 improvement via local_mean in phase domain)
cfg.TA.PHASE_COHERENCE_LOSS_ENABLED = False  #! Enable phase coherence loss
cfg.TA.PHASE_COHERENCE_LOSS_WEIGHT = 0.1  #! Weight for phase coherence loss in total loss
cfg.TA.PHASE_COHERENCE_WINDOW = 3  #! Window size for local phase coherence (like local_mean)

#### TAPA: Token-Aware Phase Attention (from arXiv:2509.12635)
cfg.TA.USE_TAPA = False  #! Use TAPA positional encoding in complex attention
cfg.TA.TAPA_THETA = 0.1  #! Split ratio: θD for amplitude, (1-θ)D for phase (paper default: 0.1)
cfg.TA.TAPA_ALPHA = 0.5  #! Distance scaling exponent |m-n|^α (paper default: 0.5)
cfg.TA.TAPA_IMAG_MODE = 'temporal_diff'  #! How to initialize imaginary part:
                                         #! 'learned': network learns projection from real
                                         #! 'zero': pure zeros
                                         #! 'temporal_diff': temporal gradient as imaginary (ABLATION: proper attn + temporal_diff)

#### Fusion CNN Control
cfg.FUSION_CNN_ENABLED = False  #! BASELINE: no fusion CNN; geomean fusion `sqrt(relu(simm_at * simm_vt))` directly on RAW similarities
cfg.FUSION_CNN_TYPE = 'complex' #! Choice when FUSION_CNN_ENABLED=True:
                                #!   'complex'   ComplexCNN, in=1 h=32   (31,874 params)
                                #!   'real'      stacked RealCNN, h=45   (31,411)
                                #!   'real_dual' DualPathRealCNN, h=32   (31,841)
                                #!   'mlp'       pointwise RealCNN tk=1, h=100 (31,501)  [2-stream only]
                                #!   'gated'     GatedFusionCNN (GMU), h=45     (31,951)  [2-stream only]
                                #!   'bilinear'  BilinearFusionCNN (MLB), h=45  (31,456)  [2-stream only]
                                #! mlp/gated/bilinear are alternative learnable fusion-family controls
cfg.MLP_CNN_HIDDEN = 100        #! width for FUSION_CNN_TYPE='mlp', chosen to match ComplexCNN's param count
cfg.GATED_CNN_HIDDEN = 45       #! width for FUSION_CNN_TYPE='gated'
cfg.BILINEAR_CNN_HIDDEN = 45    #! width for FUSION_CNN_TYPE='bilinear'

#### ComplexCNN activation choice
#! 'crelu'    = ComplexReLU (F.relu on real and imag separately).
#!              WARNING: restricts hidden phases to first quadrant [0, π/2],
#!              destroys phase information at every hidden layer. Phase variation
#!              at the output exists only because conv5 has no activation after it.
#! 'modrelu'  = ComplexModReLU (Arjovsky 2016; ReLU(|z|+b)·z/|z|). Phase-
#!              preserving — modulates magnitude, leaves phase intact. Lets
#!              phase structure build up across all 5 layers.
cfg.COMPLEX_CNN_ACTIVATION = 'crelu'
cfg.COMPLEX_CNN_MODRELU_BIAS_INIT = -0.1   # initial b; <0 creates a dead zone

# =============================================================================
# Phase-Aware Logit (exploits the Q3 decomposition: phase = event/bg binary,
# magnitude = per-class). Gates ComplexCNN's magnitude by cos(phase - θ_y).
# Operates on conv5 (mag, phase) BEFORE per-segment min-max normalization.
# Only active when FUSION_CNN_TYPE='complex'.
# Two learnable scalars (event_phase, bg_phase) — total +2 params vs baseline.
# Init from Q3 diagnostic measurements on a trained ComplexCNN.
# =============================================================================
cfg.PHASE_AWARE_LOGIT = edict()
cfg.PHASE_AWARE_LOGIT.ENABLED          = False
cfg.PHASE_AWARE_LOGIT.LEARNABLE        = True
cfg.PHASE_AWARE_LOGIT.EVENT_PHASE_INIT = 3.07     # rad (~+176°)
cfg.PHASE_AWARE_LOGIT.BG_PHASE_INIT    = -2.88    # rad (~-165°)

#### ComplexCNN Channel Arrangement
cfg.FUSION_CHANNEL_MODE = 'complex_pair'
#! 'content_structure' = ch0:(vision+i·audio),      ch1:(iHSV+i·phase)       [original]
#! 'modality'          = ch0:(audio+i·phase),        ch1:(vision+i·iHSV)      [group by source]
#! 'complex_pair'      = ch0:(vis_real+i·vis_imag),  ch1:(aud_real+i·aud_imag) [real+imag per domain]
#!                     —— with IHSV disabled: ch0 = (simm_vt, 0), ch1 = (simm_at, simm_pt)

#### Real<->Imaginary swap ablation
#! Swaps which similarity is the real vs imaginary part of a complex channel,
#! WITHOUT changing streams, params, or training. Tests whether the gain depends
#! on the real/imag *assignment* or only on the complex coupling. Applied inside
#! ComplexCNN.forward on the semantic inputs, so it is identical in train & eval.
#!   'none'   = paper config (2s: z=S_vt+i·S_at ; 4s: ch0=S_vt+i·S_iHSV, ch1=S_at+i·S_phase)
#!   'visual' = swap re<->im in the visual channel  (4s: ch0=S_iHSV+i·S_vt)
#!   'audio'  = swap re<->im in the audio channel   (4s: ch1=S_phase+i·S_at)
#!   'both'   = swap both channels (and the ONLY meaningful value for the 2-stream
#!              single-channel model: z=S_at+i·S_vt)
cfg.SWAP_REIM = 'none'

# =============================================================================
# PHASE INFORMATION FROM AUDIO (extracted from kaldi's FFT, zero cost)
# Enable ONE of these at a time, or both False to disable phase entirely.
# =============================================================================
cfg.PHASE = edict()
cfg.PHASE.ENABLED = False                  #! Master switch — set True to use phase at all

cfg.PHASE.MODE = 'cyclegan'              #! 'encoder' = separate PhaseEncoder
                                          #! 'imagebind' = PhaseToMelProjector (IF → mel-shape) → ImageBind audio encoder (frozen) + own temporal attn → simm_pt
                                          #! 'lora' = LoRA-adapted audio encoder (trainable stem + block adapters)
                                          #! 'hilbert' = Hilbert analytic signal imag of mel → ImageBind audio encoder (no projector needed)
                                          #! 'pix2pix' = Pix2Pix GAN translates IF → mel-like, then frozen audio encoder
                                          #! 'cyclegan' = CycleGAN translates IF → mel-like (unpaired, cycle-consistent), then frozen audio encoder
                                          #! 'cond_cyclegan' = Conditional CycleGAN: G_IF2Mel(IF, mel_cond) → mel-like, then frozen audio encoder

cfg.PHASE.INPUT_TYPE = 'if'              #! 'raw' = phase angles [-π, π]
                                          #! 'if' = instantaneous frequency only (∂phase/∂time, single channel)
                                          #! 'if_gd' = instantaneous frequency + group delay (derivatives, 2-channel)

cfg.PHASE.ENCODER_CHANNELS = 32          #! Hidden channels in PhaseEncoder CNN
cfg.PHASE.ENCODER_LAYERS = 4             #! Number of conv layers in PhaseEncoder
cfg.PHASE.LORA_RANK = 4                  #! LoRA adapter bottleneck rank (mode='lora')
cfg.PHASE.LORA_ALPHA = 8                 #! LoRA scaling factor alpha/rank (mode='lora')
cfg.PHASE.PIX2PIX_CKPT = _os.path.join(CODE_DIR, 'checkpoints', 'pix2pix', 'pix2pix_best.pt')  #! Path to trained Pix2Pix generator (mode='pix2pix')
cfg.PHASE.CYCLEGAN_CKPT = _os.environ.get('CYCLEGAN_CKPT', _os.path.join(CODE_DIR, 'checkpoints', 'cyclegan_if2mel.pt'))  #! Path to trained vanilla (unconditional) CycleGAN generator (mode='cyclegan')
cfg.PHASE.COND_CYCLEGAN_CKPT = _os.path.join(CODE_DIR, 'checkpoints', 'cyclegan_cond', 'cyclegan_best.pt')  #! Path to trained conditional CycleGAN generator (mode='cond_cyclegan')

#### Phase eval-time corruption controls (mode='cyclegan' only; eval path, no retraining)
#! Control: is the translated phase stream clip-specific content, or a
#! generic mel-like texture that acts as augmentation? See run_phase_mismatch.sh.
#!   'none'     paper: clip i gets its own phase
#!   'foreign'  clip i gets an unrelated clip's phase (still a genuine CycleGAN input)
#!   'noise'    phase ~ U[-pi, pi]: does G's output depend on its input at all?
#!   'timerev'  clip i gets its own phase, time-reversed (same marginals, alignment destroyed)
#!   'mute'     PLUMBING CONTROL: phase similarity zeroed downstream. If this does not
#!              move the metrics, the trained model ignores the phase stream entirely
#!              and no phase-content experiment can conclude anything.
cfg.PHASE.EVAL_MISMATCH = 'none'

#! TRAIN-time phase corruption. 'noise' trains the phase stream on G(uniform random
#! phase) instead of G(IF). The eval-time controls show the converged model ignores
#! phase content; this asks whether the +1.5 Eve-F1 the stream contributes during
#! TRAINING needs the phase to be the clip's own, or whether any well-conditioned
#! channel in the encoder's similarity space would do (i.e. a regulariser).
#!   'none' | 'noise'
cfg.PHASE.TRAIN_MISMATCH = 'none'

#### iHSV: imaginary visual companion stream (HSV→complex→imag→RGB encoder, or phase-IFFT)
cfg.IHSV = edict()
cfg.IHSV.ENABLED = False   #! Enabled: visual imag stream
cfg.IHSV.MODE = 'ihsv'  #! 'ihsv' = HSV→complex→imag→RGB; 'phase_ifft' = 2D-FFT phase-only IFFT + CLAHE (3ch broadcast)

#### iHSV eval-time corruption control (eval path, no retraining)
#! Same content-destruction test as PHASE.EVAL_MISMATCH, applied to the iHSV stream,
#! so both auxiliary streams can be compared on ONE 4-stream checkpoint.
#!   'none' | 'foreign' (an unrelated clip's iHSV embedding) | 'mute' (zeroed)
cfg.IHSV.EVAL_MISMATCH = 'none'
cfg.IHSV.RESCALE_MODE = 'minmax'  #! How to map _hsv_to_rgb output into [0,1] for the imag branch:
                                 #! 'clamp'  = clamp(0,1)                        [original — boundary spikes]
                                 #! 'minmax' = per-image min-max to [0,1]        [reversible: matches 'clamp' bit-for-bit when output is already in [0,1]]

#### Complex Temporal Attention Mode: DISABLED — use real TransformerEncoder for all 4 streams
cfg.COMPLEX_TEMPORAL = edict()
cfg.COMPLEX_TEMPORAL.ENABLED = False  #! Disabled: per-stream real TransformerEncoder
                                      #! Visual real: RGB  → ViT-L → backbone real temporal attn  → simm_vt
                                      #! Visual imag: iHSV → ViT-L → ihsv_temporal_attn (real)    → simm_imag_vt
                                      #! Audio  real: mel  → audio enc → backbone real temporal attn → simm_at
                                      #! Audio  imag: phase IF + mel cond → cond CycleGAN G_IF2Mel → audio enc → phase_temporal_attn (real) → simm_pt

#### DEPTH: depth maps from Depth-Anything-V2 fed through ImageBind depth encoder
cfg.DEPTH = edict()
cfg.DEPTH.ENABLED = False  #! Master switch: load depth PNGs -> ImageBind depth encoder -> depth_feas

# =============================================================================
# COMPLEX REPRESENTATION MODE (from FCCNs)
# When enabled, replaces the standard pipelines:
#   Vision: RGB → HSV → Complex → hsv_to_rgb(real) & hsv_to_rgb(imag) → ViT-L
#   Audio:  controlled by AUDIO_MODE below
# Requires IHSV.ENABLED=True and PHASE.ENABLED=True to have all 4 streams.
# =============================================================================
cfg.COMPLEX_REPR = edict()
cfg.COMPLEX_REPR.ENABLED = False  #! Master switch for FCCNs-style complex decomposition
cfg.COMPLEX_REPR.AUDIO_MODE = 'magnitude_phase'
#! 'fft_real_imag'    = |FFT.real| → mel & |FFT.imag| → mel (original, distribution mismatch with frozen encoder)
#! 'magnitude_phase'  = standard magnitude mel (encoder understands) + phase IF/GD as imaginary (natural complex pair)

#### text projection tuning
cfg.TEXT_TUNE_FLAG = False

#### CycleGAN / pix2pix output -> ImageBind audio encoder calibration
# False (default) -- pre-patch behaviour: raw mel-scale output is fed to
#                    ImageBind without the Normalize(-4.268, 9.138) that
#                    load_and_transform_audio_data applies to real mels.
# True            -- apply the normalization at every CycleGAN/pix2pix
#                    output site so ImageBind sees inputs on the same scale
#                    as during its own training.
cfg.APPLY_IMAGEBIND_AUDIO_NORM = False

# =============================================================================
# FCCN-style auxiliary phase loss (adapted from the FCCNs reference implementation).
# Applied at the ComplexCNN fusion output. The complex output z = real + i*imag
# of the final fusion conv layer yields magnitude |z| and phase angle(z).
#   loss = (CE(|z|, y) - LAMBDA * mean(phase[b, t, y_{b,t}])) / 2
# i.e. CE on magnitude (existing path) minus the average phase at the true class
# (encourages phase at true class toward +pi).
# Reversibility: ENABLED=False keeps the original CE-on-magnitude loss.
# =============================================================================
cfg.FCCN_LOSS = edict()
cfg.FCCN_LOSS.ENABLED = False
cfg.FCCN_LOSS.LAMBDA = 1.0
cfg.FCCN_LOSS.MASK_BG = False    #! If True, exclude bg-labeled segments from the aux term.
cfg.FCCN_LOSS.MODE = 'fccn'      #! 'fccn' | 'cosine' | 'coherence'
#! 'fccn'      = original: phase_term = mean θ[y]; loss = (CE - λ·phase_term)/2
#!               (pushes θ at true class toward +π; treats phase as linear, not circular)
#! 'cosine'    = circular-aware: phase_term = mean(1 - cos(θ[y])); loss = CE + λ·phase_term
#!               (pushes θ at true class toward 0; bounded ∈ [0,2]; scale-stable across C)
#! 'coherence' = temporal: per-clip |R̄| of θ[y] over event segments; loss = CE + λ·(1 - mean |R̄|)
#!               (rewards phase being consistent across time within events; needs ≥2 FG segs)

# =============================================================================
# Modality-balance loss (NEW for the BMVC submission).
# Encodes the §3 observation into a training-time loss on the fused complex output:
#   - For event segments (y != bg): push arg(z_y) toward EVENT_TARGET (default π/4
#     = the empirical modality-balance angle measured on raw similarities).
#   - For bg segments: optionally push arg(z_y) toward BG_TARGET (default 0
#     = bg's empirical mean phase on raw similarities) AND/OR penalize magnitude.
#
# Implementation: applies to fusion_phase (the ComplexCNN output's atan2(im, re))
# at the ground-truth class index. Uses circular-aware 1-cos loss to avoid the
# wrap-around problem of linear MSE. Combines additively with CE-on-magnitude.
# =============================================================================
cfg.MODALITY_BALANCE_LOSS = edict()
cfg.MODALITY_BALANCE_LOSS.ENABLED = False
cfg.MODALITY_BALANCE_LOSS.LAMBDA_EVENT = 0.5
cfg.MODALITY_BALANCE_LOSS.LAMBDA_BG    = 0.0      #! 0.0 = bg unconstrained (recommended start)
cfg.MODALITY_BALANCE_LOSS.EVENT_TARGET = 0.7854   #! π/4 = 0.7853981...  (modality balance)
cfg.MODALITY_BALANCE_LOSS.BG_TARGET    = 0.0      #! origin
cfg.MODALITY_BALANCE_LOSS.MODE         = 'fixed'  #! 'fixed' = constants above
                                                  #! 'learnable' = θ_evt, θ_bg are scalar params
                                                  #! 'per_class' = one learnable θ_c per event class

# =============================================================================
# Phase-Spread Loss (NEW). O3-faithful angular-margin loss combining vMF
# concentration (intra-class) and hyperspherical uniformity (inter-class) on
# the fused complex phase. Open-vocab safe — no per-class learnable params.
# =============================================================================
cfg.PHASE_SPREAD_LOSS = edict()
cfg.PHASE_SPREAD_LOSS.ENABLED          = False
cfg.PHASE_SPREAD_LOSS.LAMBDA_INTRA     = 0.5      #! weight on (1 − R̄_c) per-class concentration
cfg.PHASE_SPREAD_LOSS.LAMBDA_SPREAD    = 0.5      #! weight on inter-class hinge
cfg.PHASE_SPREAD_LOSS.MARGIN           = 0.5      #! rad; pairs with angular dist ≥ MARGIN incur no spread penalty
cfg.PHASE_SPREAD_LOSS.BG_EVENT_WEIGHT  = 2.0      #! multiplier on bg-event pairs in the spread term

# =============================================================================
# MOE: Mixture-of-Experts Temporal Attention (boundary-aware expert routing)
# Replaces standard TemporalAttentionModule with multiple specialized experts:
#   - Steady-state expert: wide temporal context for interior frames
#   - Onset expert: causal temporal diff for detecting event starts
#   - Offset expert: anti-causal temporal diff for detecting event ends
#   - Boundary-refine expert: narrow context for sharp per-frame decisions
# A learned router dynamically allocates expert weights per frame.
# =============================================================================
cfg.MOE = edict()
cfg.MOE.ENABLED = False              #! Master switch — set True to use MoE temporal attention
cfg.MOE.NUM_EXPERTS = 4              #! Number of temporal experts (default: 4 = steady/onset/offset/boundary)
cfg.MOE.ROUTER_HIDDEN_DIM = 128     #! Router MLP hidden dimension
cfg.MOE.EXPERT_HIDDEN_DIM = 256     #! Hidden dimension inside each expert's temporal conv
cfg.MOE.LOAD_BALANCE_WEIGHT = 0.01  #! Weight for load-balancing loss (KL div toward uniform)
cfg.MOE.TOP_K = 0                   #! Top-K expert selection (0 = soft routing, all experts weighted)
cfg.MOE.USE_TEMPORAL_DIFF_ROUTING = True  #! Feed temporal differences to router as routing signal

#### Cross-modal AV phase coherence loss (Idea 1, novel for OV-AVEL).
#### Operates on RAW similarity tensors BEFORE ComplexCNN/CReLU — phase intact.
#### Forms z[t,c] = simm_vt[t,c] + i·simm_at[t,c] and pushes the temporal phase
#### at the GT class to be coherent within events while leaving non-GT classes
#### unconstrained. Zero added trainable parameters (just an auxiliary loss).
#### When ENABLED=False the loss term is skipped — fully reversible.
cfg.AV_PHASE_COHERENCE = edict()
cfg.AV_PHASE_COHERENCE.ENABLED = False     #! Master switch
cfg.AV_PHASE_COHERENCE.WEIGHT  = 0.1       #! Multiplier for the auxiliary loss
cfg.AV_PHASE_COHERENCE.WINDOW  = 3         #! Temporal window for circular mean
cfg.AV_PHASE_COHERENCE.LAMBDA_BG = 0.5     #! Weight on background coherence penalty
cfg.AV_PHASE_COHERENCE.MARGIN  = 0.0       #! Optional margin in the gap loss

#### Phase-variance fg/bg gate (Idea 4, novel for OV-AVEL).
#### Inference-time only: computes circular variance of arg(simm_vt + i·simm_at)
#### across the class axis. High variance ⇒ no class achieves coherent AV
#### phase ⇒ likely background. Trainable (α, β) ≈ 2 scalars.
#### When ENABLED=False the gate is skipped — fully reversible.
cfg.PHASE_VARIANCE_GATE = edict()
cfg.PHASE_VARIANCE_GATE.ENABLED    = False  #! Master switch
cfg.PHASE_VARIANCE_GATE.LEARNABLE  = True   #! If False, use fixed (α, β)
cfg.PHASE_VARIANCE_GATE.ALPHA_INIT = 4.0    #! Initial slope on -α·var
cfg.PHASE_VARIANCE_GATE.BETA_INIT  = 2.0    #! Initial bias

#### Boundary-aware loss weighting (upweight frames at event transitions)
cfg.BOUNDARY = edict()
cfg.BOUNDARY.ENABLED = False   #! Upweight boundary frames in CE loss
cfg.BOUNDARY.ALPHA = 2.0       #! Boundary weight multiplier (frame weight = 1 + alpha * is_boundary)

#### Temporal smoothing on prediction logits
cfg.SMOOTH = edict()
cfg.SMOOTH.ENABLED = False
cfg.SMOOTH.KERNEL_SIZE = 3  # odd integer, 3 or 5 recommended
cfg.SMOOTH.SIGMA = 1.0  # Gaussian std

#### Classification loss: 'ce' (default) or 'focal'
cfg.LOSS = edict()
cfg.LOSS.TYPE = 'ce'           # 'ce' | 'focal'
cfg.LOSS.FOCAL_GAMMA = 2.0     # focusing parameter; γ=0 reduces to CE
cfg.LOSS.FOCAL_ALPHA = None    # None or list/tuple of length C (per-class weights)

#### Logit adjustment (Menon et al. 2021): post-hoc subtract tau*log_pi from logits
#### at evaluation time only. Reduces bg-collapse caused by long-tail prior bias.
#### Train CE loss is unchanged; this only affects argmax/F1 at val/test.
cfg.LOGIT_ADJ = edict()
cfg.LOGIT_ADJ.ENABLED = False           # toggle
cfg.LOGIT_ADJ.TAU = 1.0                 # adjustment strength (grid-search; logits are min-max in [0,1] so try 0.02–0.2 first)
cfg.LOGIT_ADJ.PRIOR_SMOOTHING_EPS = 1.0 # Laplace smoothing on top of count-based priors
cfg.LOGIT_ADJ.UNSEEN_FG_MODE = 'mean_fg'  # how to set prior for open-set classes never seen at train:
                                          #   'mean_fg' = use mean of seen-fg counts (default; neutral wrt other fg)
                                          #   'min_fg'  = use min seen-fg count   (treat as rarest seen class)
                                          #   'laplace' = pure Laplace eps only   (artificially huge boost — be careful)


# =============================================================================
# Override mechanism for sweep / experiment scripts.
# -----------------------------------------------------------------------------
# WHEN TO USE THIS:
#   - You are running a SWEEP across configurations (multi-seed, ablation,
#     architecture comparison) and need each invocation to use a different
#     config WITHOUT mutating config.py on disk.
#   - You are launching multiple training jobs in parallel that must use
#     different settings (e.g., GPU 0 = ComplexCNN, GPU 1 = RealCNN).
#
# WHEN TO JUST EDIT THIS FILE INSTEAD:
#   - You are running ONE experiment with a stable configuration.
#   - You want to permanently change the default for everyone.
#
# HOW IT WORKS:
#   Set the env var CFG_OVERRIDES_JSON to a JSON dict of dotted keys → values
#   BEFORE invoking python. Each python process reads this once at import,
#   so concurrent processes with different env vars get independent configs.
#   No global file mutation, no race conditions, no stale state.
#
# EXAMPLE:
#   CFG_OVERRIDES_JSON='{"FUSION_CNN_ENABLED": true, "FUSION_CNN_TYPE": "real"}' \
#     bash run_baseline_v1_train_fully_av.sh
#
# Missing parent keys are reported but do not error — useful for backward
# compatibility when removing flags.
# =============================================================================
def _apply_env_overrides(_cfg):
    import os as _os
    import json as _json
    raw = _os.environ.get("CFG_OVERRIDES_JSON", "").strip()
    if not raw:
        return
    try:
        overrides = _json.loads(raw)
    except Exception as exc:
        raise ValueError(f"CFG_OVERRIDES_JSON is not valid JSON: {exc}")
    applied, skipped = {}, {}
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        target = _cfg
        ok = True
        for p in parts[:-1]:
            if p not in target:
                skipped[dotted_key] = f"missing parent key '{p}'"
                ok = False
                break
            target = target[p]
        if ok:
            target[parts[-1]] = value
            applied[dotted_key] = value
    print(f"[config.py] applied CFG_OVERRIDES_JSON: {applied}")
    if skipped:
        print(f"[config.py] skipped (key path not in cfg): {skipped}")

_apply_env_overrides(cfg)
