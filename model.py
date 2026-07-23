"""
EGWT: EfficientNet Convolutional Group-Wise Transformer
Reproduction of: J. Feng et al., "Enhanced Crop Disease Detection With EfficientNet
Convolutional Group-Wise Transformer", IEEE Access, 2024.

=====================================================================================
IMPORTANT -- Table 2 is underspecified; here is the reading this file uses, and why
=====================================================================================
Table 2 does not, by itself, provide enough information to uniquely determine the
architecture. Two things ARE independently verifiable, though:

1. Stage 3 lists n_groups=4, H3=12 heads, dim=1024. 1024/4 groups = 256 per group;
   256 is not divisible by 12, so a per-head dimension cannot be derived as written,
   under any grouping convention we tried (heads-per-group or heads-total).

2. Taking "768" (stage 1 and 2) / "1024" (stage 3) literally as the transformer's
   working dimension is NOT reproducible within the paper's own reported 23.04M
   parameter budget -- and this holds regardless of how DWTE's output (64/192) is
   wired into that 768-dim space. A plausible bridging design (an explicit upward
   projection from each stage's DWTE output into a shared 768-dim block width,
   combined with the *most* parameter-efficient reading of the group-wise attention
   sharing possible -- see point 3 below) still costs ~42.9M params for stage 1+2
   alone (measured: 2 blocks x ~3.58M + 10 blocks x ~3.58M), already 1.9x the entire
   model's reported size before stage 3, DWTE, or the classifier head are counted.
   The bottleneck is not the DWTE-to-block dimension bridge; it's Eq. 8's G-MLP
   first layer (H = GELU(X W1 + b1), W1 in R^{d x d*R}), which the paper states
   explicitly operates on the FULL (ungrouped) width -- at d=768 repeated over 12
   blocks (2+10), this alone is incompatible with a 23M-parameter model under any
   standard parameterization we could construct.
   We cannot rule out that the paper uses some other, unstated mechanism to keep
   this cheap (e.g. a grouped fc1, contrary to how Eq. 8 reads) -- this is a real
   gap in the paper's description, not a claim that the paper is definitively wrong.

Given that gap, this file uses the reading that stays internally consistent and
lands closest to the stated 23.04M:

  - The per-stage transformer working dimension equals the DWTE output channel count
    directly (64 / 192 / 1024 for stages 1/2/3), matching standard CvT convention
    (this paper's own stated architectural basis) and stage 3's row, where the DWTE
    output and block dimension already agree with no bridging needed.
  - "768" happens to equal stage 2's G-MLP hidden width under Eq. 9 (192 x R2=4 =
    768, exact match); stage 1's row shows the identical bracket contents (down to
    a "R2" subscript appearing under a nominal "stage 1" heading), which is at least
    consistent with the same 768 hidden-width figure being reused/misplaced there --
    one plausible explanation among others, not a verified fact about the authors'
    intent.
  - "Parameters shared within each group" (paper's Section III.B wording) and
    Figure 7(a)'s caption ("attention computations within each group share weight
    matrix parameters") are both consistent with, and are implemented as: within a
    group, ONE small head_dim x head_dim weight matrix is reused across every head
    in that group, rather than each head having independent weights. This is the
    single most parameter-saving reading available and is still not enough to make
    a literal dim=768 fit the budget (see point 2 above) -- i.e. this is not the
    source of the discrepancy, it's already applied and still insufficient.
  - Head counts are adjusted to the nearest values that divide evenly (stage 1: 2
    heads/group, unstated by the paper; stage 2: 6 heads/group, as literally stated,
    divides cleanly; stage 3: 16 heads instead of the stated 12, restoring a valid
    split while keeping head_dim=64 consistent with the rest of the network).
  - "EfficientNet projection" (stage 3) is written in Table 2 only as "3x3, 1024" --
    the paper does not expand what internal structure this projection uses beyond
    "EfficientNet-B0 architecture ... incorporated ... for fine-grained feature
    extraction" in prose, and Figure 1 shows it only as an opaque block parallel to
    "Convolutional Projection". We implement it as the same depthwise conv
    projection used elsewhere, augmented with a Squeeze-and-Excite gate (the
    signature EfficientNet component) -- NOT a full inverted-bottleneck expansion,
    which at 1024 channels would alone cost ~10M params per Q/K/V conv (~30M/block),
    far exceeding the entire reported model size. This is a best-effort fill-in for
    a genuine information gap in the source, not a correction of an error.
  - Figure 6(b) explicitly depicts stride=1 for ALL THREE (query/key/value)
    convolutional projections, at every stage; this is what we use (kv_stride=1
    everywhere). A first attempt at this literally measured ~35.5s/training-step
    (>4.5h/epoch) at stage 1's 56x56=3136 tokens with a manual attention
    implementation -- not because stride=1 is inherently too expensive in FLOPs, but
    because materializing the full N x N attention matrix (measured ~7.6GB for a
    single group's attention call) is memory-bandwidth-bound. Switching to PyTorch's
    `F.scaled_dot_product_attention` (flash-attention backend) in GroupWiseMHA -- the
    exact same mathematical operation (Eq. 4's softmax(QK^T/sqrt(d))V), just a
    fused/memory-efficient kernel -- measured 15.6x faster and 2.9x less memory for
    the identical computation, making literal stride=1 tractable without any
    fidelity compromise.

With this reading the model totals ~19-20M params, within ~15% of the paper's 23.04M.
This is reported as a best-effort reconstruction under genuine information gaps in
the source paper, not as a verified match to the authors' actual implementation.
=====================================================================================

Other documented deviations (see also project-level report):
- No separate ImageNet pretraining stage (infeasible in the compute budget available;
  see the DWTE/EfficientNet-projection note below and the top-level report). The
  stage-3 EfficientNet-style projection blocks are seeded with weights sourced from a
  real torchvision ImageNet-pretrained EfficientNet-B0 where the depthwise-conv
  kernel shapes align (a heuristic warm start); the group-wise transformer components
  (novel to this paper, no pretrained weights exist anywhere) are randomly initialized
  and trained directly on each crop dataset.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# --------------------------------------------------------------------------------------
# 1) Depth-Wise split Token Embedding (DWTE)
# --------------------------------------------------------------------------------------
class DWTE(nn.Module):
    """Depth-wise separable conv token embedding (patchify / downsample between stages)."""

    def __init__(self, in_ch, out_ch, kernel_size, stride):
        super().__init__()
        padding = kernel_size // 2
        self.dwconv = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding,
                                 groups=in_ch, bias=False)
        self.bn = nn.BatchNorm2d(in_ch)
        self.pwconv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x):
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.pwconv(x)
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)
        return tokens, H, W


# --------------------------------------------------------------------------------------
# 2) Convolutional projection (plain depthwise, used in stage 1 & 2), Eq. 3
# --------------------------------------------------------------------------------------
class ConvProjection(nn.Module):
    def __init__(self, dim, kernel_size=3, q_stride=1, kv_stride=1, se=False):
        super().__init__()
        self.dim = dim
        self.conv_q = self._build(dim, kernel_size, q_stride, se)
        self.conv_k = self._build(dim, kernel_size, kv_stride, se)
        self.conv_v = self._build(dim, kernel_size, kv_stride, se)

    @staticmethod
    def _build(dim, kernel_size, stride, se):
        padding = kernel_size // 2
        layers = [
            nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
        ]
        if se:
            layers.append(SqueezeExcite(dim, dim))
        return nn.Sequential(*layers)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x2d = x.transpose(1, 2).reshape(B, C, H, W)
        q = self.conv_q(x2d).flatten(2).transpose(1, 2)
        k = self.conv_k(x2d).flatten(2).transpose(1, 2)
        v = self.conv_v(x2d).flatten(2).transpose(1, 2)
        return q, k, v


class SqueezeExcite(nn.Module):
    """EfficientNet's signature squeeze-excite gate; used to flavor stage 3's
    'EfficientNet projection' without the param cost of a full inverted bottleneck."""

    def __init__(self, dim, in_dim, se_ratio=0.0625):
        super().__init__()
        se_dim = max(8, int(in_dim * se_ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(dim, se_dim, 1)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(se_dim, dim, 1)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        s = self.pool(x)
        s = self.act(self.fc1(s))
        s = self.gate(self.fc2(s))
        return x * s


class EfficientNetProjection(ConvProjection):
    """Stage-3 'EfficientNet projection': depthwise conv projection + Squeeze-Excite
    (see module docstring for why a full MBConv expansion is not used here), seeded
    from a real ImageNet-pretrained EfficientNet-B0 where shapes allow."""

    def __init__(self, dim, kernel_size=3, q_stride=1, kv_stride=1, pretrained=True):
        super().__init__(dim, kernel_size, q_stride, kv_stride, se=True)
        if pretrained:
            _seed_depthwise_from_effnet_b0(self, dim, kernel_size)


_EFFNET_B0_CACHE = {}


def _seed_depthwise_from_effnet_b0(module, dim, kernel_size):
    """Best-effort weight transplant: copy a real ImageNet-pretrained depthwise 3x3
    conv kernel from torchvision's EfficientNet-B0 into this projection's depthwise
    convs, when channel counts allow a tile/repeat. This is a heuristic warm start,
    not an exact architectural match -- documented as such in the report."""
    if "model" not in _EFFNET_B0_CACHE:
        weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
        _EFFNET_B0_CACHE["model"] = torchvision.models.efficientnet_b0(weights=weights)
    src = _EFFNET_B0_CACHE["model"]

    src_dw = None
    for m in src.features.modules():
        if isinstance(m, nn.Conv2d) and m.groups == m.in_channels and m.in_channels > 1 \
                and m.kernel_size == (kernel_size, kernel_size):
            src_dw = m
    if src_dw is None:
        return
    with torch.no_grad():
        for conv_block in (module.conv_q, module.conv_k, module.conv_v):
            dw = conv_block[0]
            src_w = src_dw.weight  # (C_src, 1, k, k)
            reps = (dim + src_w.shape[0] - 1) // src_w.shape[0]
            tiled = src_w.repeat(reps, 1, 1, 1)[:dim]
            dw.weight.copy_(tiled)


# --------------------------------------------------------------------------------------
# 3) Group-wise Multi-Head Attention (G-MHA), Eq. 4-7
#    Weight sharing WITHIN a group: one small head_dim x head_dim matrix is reused
#    across all heads inside that group (see module docstring).
# --------------------------------------------------------------------------------------
class GroupWiseMHA(nn.Module):
    def __init__(self, dim, n_groups, heads_per_group, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % n_groups == 0
        self.n_groups = n_groups
        self.group_dim = dim // n_groups
        assert self.group_dim % heads_per_group == 0
        self.heads_per_group = heads_per_group
        self.head_dim = self.group_dim // heads_per_group
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(n_groups)])
        self.k_proj = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(n_groups)])
        self.v_proj = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(n_groups)])
        self.out_proj = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(n_groups)])
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def _attn_one_group(self, q, k, v, g):
        B, N, _ = q.shape
        Nk = k.shape[1]
        q = q.reshape(B, N, self.heads_per_group, self.head_dim)
        k = k.reshape(B, Nk, self.heads_per_group, self.head_dim)
        v = v.reshape(B, Nk, self.heads_per_group, self.head_dim)
        # shared linear applied identically to every head in the group (broadcast over dim=2)
        q = self.q_proj[g](q).transpose(1, 2)  # B, heads, N, head_dim
        k = self.k_proj[g](k).transpose(1, 2)
        v = self.v_proj[g](v).transpose(1, 2)
        # Mathematically identical to manual softmax(QK^T/sqrt(d))V (same op as Eq. 4,
        # just a fused/memory-efficient kernel -- not a fidelity change). Needed to
        # keep stride=1 (Figure 6b, literal) tractable: stage 1 has N=3136 tokens, and
        # materializing the full N x N attention matrix manually measured ~7.6GB per
        # call and dominated wall-clock time; SDPA's flash-attention backend measured
        # 15.6x faster and 2.9x less memory for the identical computation.
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        out = out.transpose(1, 2)  # B, N, heads, head_dim
        out = self.out_proj[g](out).reshape(B, N, self.group_dim)
        return out

    def forward(self, q, k, v):
        q_groups = q.chunk(self.n_groups, dim=-1)
        k_groups = k.chunk(self.n_groups, dim=-1)
        v_groups = v.chunk(self.n_groups, dim=-1)
        outs = [self._attn_one_group(q_groups[g], k_groups[g], v_groups[g], g)
                for g in range(self.n_groups)]
        out = torch.cat(outs, dim=-1)
        return self.proj_drop(out)


# --------------------------------------------------------------------------------------
# 4) Group-wise MLP (G-MLP), Eq. 8-9. fc1 expands the FULL (ungrouped) width per Eq. 8;
#    fc2 is grouped/shared per Eq. 9.
# --------------------------------------------------------------------------------------
class GroupWiseMLP(nn.Module):
    def __init__(self, dim, n_groups, hidden_dim, drop=0.0):
        super().__init__()
        assert dim % n_groups == 0
        assert hidden_dim % n_groups == 0
        self.n_groups = n_groups
        self.group_dim = dim // n_groups
        hidden_group = hidden_dim // n_groups

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.ModuleList([nn.Linear(hidden_group, self.group_dim) for _ in range(n_groups)])
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        h = self.act(self.fc1(x))
        h_groups = h.chunk(self.n_groups, dim=-1)
        outs = [self.fc2[g](h_groups[g]) for g in range(self.n_groups)]
        out = torch.cat(outs, dim=-1)
        return self.drop(out)


# --------------------------------------------------------------------------------------
# 5) EGWT Transformer block (projection + G-MHA + G-MLP, pre-norm residual)
# --------------------------------------------------------------------------------------
class EGWTBlock(nn.Module):
    def __init__(self, dim, n_groups, heads_per_group, mlp_hidden, use_effnet_proj=False,
                 pretrained_effnet=True, kv_stride=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        # kv_stride left at its default (1) here -- Figure 6(b) explicitly depicts
        # stride=1 for query/key/value alike, so this stays literal. Attention's O(N^2)
        # cost at stage 1's 56x56=3136 tokens is instead handled via SDPA's
        # flash-attention kernel in GroupWiseMHA (mathematically identical to manual
        # softmax(QK^T)V, just faster/more memory-efficient), which made stride=1
        # tractable without any fidelity compromise. See GroupWiseMHA.
        if use_effnet_proj:
            self.proj = EfficientNetProjection(dim, kv_stride=kv_stride, pretrained=pretrained_effnet)
        else:
            self.proj = ConvProjection(dim, kv_stride=kv_stride)
        self.attn = GroupWiseMHA(dim, n_groups, heads_per_group)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = GroupWiseMLP(dim, n_groups, mlp_hidden)

    def forward(self, x, H, W):
        residual = x
        x_n = self.norm1(x)
        q, k, v = self.proj(x_n, H, W)
        x = residual + self.attn(q, k, v)
        x = x + self.mlp(self.norm2(x))
        return x


# --------------------------------------------------------------------------------------
# 6) EGWT stage: DWTE -> N blocks (dim == DWTE output channels, no bridge needed)
# --------------------------------------------------------------------------------------
class EGWTStage(nn.Module):
    def __init__(self, in_ch, dim, dwte_kernel, dwte_stride, n_groups, heads_per_group,
                 depth, mlp_ratio=4, use_effnet_proj=False, pretrained_effnet=True,
                 kv_stride=1):
        super().__init__()
        self.dwte = DWTE(in_ch, dim, dwte_kernel, dwte_stride)
        mlp_hidden = dim * mlp_ratio
        self.blocks = nn.ModuleList([
            EGWTBlock(dim, n_groups, heads_per_group, mlp_hidden,
                      use_effnet_proj=use_effnet_proj, pretrained_effnet=pretrained_effnet,
                      kv_stride=kv_stride)
            for _ in range(depth)
        ])
        self.out_ch = dim

    def forward(self, x):
        tokens, H, W = self.dwte(x)
        for blk in self.blocks:
            tokens = blk(tokens, H, W)
        B, N, C = tokens.shape
        return tokens.transpose(1, 2).reshape(B, C, H, W)


# --------------------------------------------------------------------------------------
# 7) Full EGWT model
# --------------------------------------------------------------------------------------
class EGWT(nn.Module):
    def __init__(self, num_classes, in_ch=3, pretrained_effnet=True):
        super().__init__()
        self.stage1 = EGWTStage(in_ch, 64, 7, 4, n_groups=2, heads_per_group=2,
                                 depth=2, mlp_ratio=4, use_effnet_proj=False, kv_stride=1)
        self.stage2 = EGWTStage(64, 192, 3, 2, n_groups=2, heads_per_group=6,
                                 depth=10, mlp_ratio=4, use_effnet_proj=False, kv_stride=1)
        self.stage3 = EGWTStage(192, 1024, 3, 2, n_groups=4, heads_per_group=4,
                                 depth=3, mlp_ratio=4, use_effnet_proj=True,
                                 pretrained_effnet=pretrained_effnet)
        self.norm = nn.LayerNorm(1024)
        self.head = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)
        pooled = tokens.mean(dim=1)
        return self.head(pooled)

    def freeze_stage12(self):
        """Per the paper's fine-tuning protocol (Section IV.C): freeze stage 1 & 2,
        fine-tune from stage 3 onward."""
        for p in self.stage1.parameters():
            p.requires_grad = False
        for p in self.stage2.parameters():
            p.requires_grad = False


def count_params(model, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = EGWT(num_classes=38, pretrained_effnet=False)
    n_params = count_params(m)
    print(f"EGWT params: {n_params/1e6:.2f}M  (paper reports 23.04M)")
    for name, mod in [("stage1", m.stage1), ("stage2", m.stage2), ("stage3", m.stage3)]:
        print(f"  {name}: {count_params(mod)/1e6:.2f}M")
    x = torch.randn(2, 3, 224, 224)
    y = m(x)
    print("output shape:", y.shape)
