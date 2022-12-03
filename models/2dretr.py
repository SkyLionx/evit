from collections import namedtuple
from inspect import isfunction

from einops import rearrange
import torch
import torchmetrics
from torch import nn, Tensor, einsum
import torch.nn.functional as F

import pytorch_lightning as pl

from timm.models.vision_transformer import (
    trunc_normal_,
    partial,
    PatchEmbed,
    Block,
)

from timm.models.vision_transformer_hybrid import HybridEmbed


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
        causal: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.causal = causal

        inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.attn_fn = F.softmax
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(
        self,
        x: Tensor,
        context: Tensor = None,
        mask: Tensor = None,
        context_mask: Tensor = None,
    ) -> Tensor:
        (
            b,
            n,
            _,
        ) = x.shape
        h, device = self.heads, x.device
        kv_input = default(context, x)

        q_input = x
        k_input = kv_input
        v_input = kv_input

        q = self.to_q(q_input)
        k = self.to_k(k_input)
        v = self.to_v(v_input)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        input_mask = None
        if any(map(exists, (mask, context_mask))):
            q_mask = default(mask, lambda: torch.ones((b, n), device=device).bool())
            k_mask = q_mask if not exists(context) else context_mask
            k_mask = default(
                k_mask, lambda: torch.ones((b, k.shape[-2]), device=device).bool()
            )
            q_mask = rearrange(q_mask, "b i -> b () i ()")
            k_mask = rearrange(k_mask, "b j -> b () () j")
            input_mask = q_mask * k_mask

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        mask_value = max_neg_value(dots)

        pre_softmax_attn = dots

        if exists(input_mask):
            dots.masked_fill_(~input_mask, mask_value)
            del input_mask

        if self.causal:
            i, j = dots.shape[-2:]
            mask = torch.ones((i, j), device=device).triu_(j - i + 1).bool()
            dots.masked_fill_(mask, mask_value)
            del mask

        attn = self.attn_fn(dots, dim=-1)
        post_softmax_attn = attn

        attn = self.dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        intermediates = Intermediates(
            pre_softmax_attn=pre_softmax_attn, post_softmax_attn=post_softmax_attn
        )

        return self.to_out(out), intermediates


# constants

Intermediates = namedtuple("Intermediates", ["pre_softmax_attn", "post_softmax_attn"])

LayerIntermediates = namedtuple("Intermediates", ["hiddens", "attn_intermediates"])


# helpers


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def equals(val):
    def inner(x):
        return x == val

    return inner


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


# keyword argument helpers


def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)


def string_begins_with(prefix, str):
    return str.startswith(prefix)


def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(
        partial(string_begins_with, prefix), d
    )
    kwargs_without_prefix = dict(
        map(lambda x: (x[0][len(prefix) :], x[1]), tuple(kwargs_with_prefix.items()))
    )
    return kwargs_without_prefix, kwargs


# classes


class Scale(nn.Module):
    def __init__(self, value, fn):
        super().__init__()
        self.value = value
        self.fn = fn

    def forward(self, x, **kwargs):
        x, *rest = self.fn(x, **kwargs)
        return (x * self.value, *rest)


class Residual(nn.Module):
    def forward(self, x, residual):
        return x + residual


# feedforward


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class AttentionLayers(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        causal: bool = False,
        cross_attend: bool = False,
        only_cross: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.layers = nn.ModuleList([])

        norm_fn = partial(nn.LayerNorm, dim)

        ff_kwargs, kwargs = groupby_prefix_and_trim("ff_", kwargs)
        attn_kwargs, _ = groupby_prefix_and_trim("attn_", kwargs)

        if cross_attend and not only_cross:
            default_block = ("a", "c", "f")
        elif cross_attend and only_cross:
            default_block = ("c", "f")
        else:
            default_block = ("a", "f")

        layer_types = default_block * depth

        self.layer_types = layer_types
        self.num_attn_layers = len(list(filter(equals("a"), layer_types)))

        for layer_type in self.layer_types:
            if layer_type == "a":
                layer = Attention(dim, heads=heads, causal=causal, **attn_kwargs)
            elif layer_type == "c":
                layer = Attention(dim, heads=heads, **attn_kwargs)
            elif layer_type == "f":
                layer = FeedForward(dim, **ff_kwargs)
            else:
                raise Exception(f"invalid layer type {layer_type}")

            residual_fn = Residual()

            self.layers.append(nn.ModuleList([norm_fn(), layer, residual_fn]))

    def forward(
        self, x, context=None, mask=None, context_mask=None, return_hiddens=False
    ):
        hiddens = []
        intermediates = []

        for ind, (layer_type, (norm, block, residual_fn)) in enumerate(
            zip(self.layer_types, self.layers)
        ):
            is_last = ind == (len(self.layers) - 1)

            if layer_type == "a":
                hiddens.append(x)

            residual = x
            x = norm(x)

            if layer_type == "a":
                out, inter = block(x, mask=mask)
            elif layer_type == "c":
                out, inter = block(
                    x, context=context, mask=mask, context_mask=context_mask
                )
            elif layer_type == "f":
                out = block(x)

            x = residual_fn(out, residual)

            if layer_type in ("a", "c"):
                intermediates.append(inter)

        if return_hiddens:
            intermediates = LayerIntermediates(
                hiddens=hiddens, attn_intermediates=intermediates
            )

            return x, intermediates

        return x


class MyVisionTransformer(nn.Module):
    """Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        representation_size=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        hybrid_backbone=None,
        norm_layer=None,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone,
                img_size=img_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        x = x + self.pos_embed.repeat(B, 1, 1)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 1),
        )

    def forward(self, x):
        return self.net(x) + x


class ImageDecoderCNN(nn.Module):
    def __init__(
        self,
        patch_num: int = 4,
        num_cnn_layers: int = 3,
        num_resnet_blocks: int = 2,
        cnn_hidden_dim: int = 64,
        voxel_size: int = 32,
        dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        dim_head: int = 64,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
    ):
        super().__init__()

        if voxel_size % patch_num != 0:
            raise ValueError("voxel_size must be dividable by patch_num")

        self.patch_num = patch_num
        self.voxel_size = voxel_size
        self.patch_size = voxel_size // patch_num
        self.emb = nn.Embedding(patch_num**3, dim)
        self.transformer = AttentionLayers(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            causal=False,
            cross_attend=True,
        )

        has_resblocks = num_resnet_blocks > 0
        dec_chans = [cnn_hidden_dim] * num_cnn_layers
        dec_init_chan = dim if not has_resblocks else dec_chans[0]
        dec_chans = [dec_init_chan, *dec_chans]

        dec_chans_io = list(zip(dec_chans[:-1], dec_chans[1:]))

        dec_layers = []

        for (dec_in, dec_out) in dec_chans_io:
            dec_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(dec_in, dec_out, 4, stride=2, padding=1),
                    nn.ReLU(),
                )
            )

        for _ in range(num_resnet_blocks):
            dec_layers.insert(0, ResBlock(dec_chans[1]))

        if num_resnet_blocks > 0:
            dec_layers.insert(0, nn.Conv2d(dim, dec_chans[1], 1))

        dec_layers.append(nn.Conv2d(dec_chans[-1], 3, 1))

        self.decoder = nn.Sequential(*dec_layers)

        self.layer_norm = nn.LayerNorm(dim)

    def generate(self, context: Tensor, context_mask: Tensor = None, **kwargs):
        out = self(context, context_mask)
        return torch.sigmoid(out)

    def forward(self, context: Tensor, context_mask: Tensor = None) -> Tensor:
        x = self.emb(torch.arange(self.patch_num**2, device=context.device))
        x = x.unsqueeze(0).repeat(context.shape[0], 1, 1)

        out = self.transformer(x=x, context=context, context_mask=context_mask)
        out = self.layer_norm(out)
        out = rearrange(
            out,
            "b (h w) d -> b d h w",
            h=self.patch_num,
            w=self.patch_num,
        )
        out = self.decoder(out)

        return out


class Events2Image(pl.LightningModule):
    def __init__(self, input_shape, patch_size, embed_dim, num_layers, num_heads, lr):
        super().__init__()
        self.save_hyperparameters()

        self.input_shape = input_shape
        self.bins, self.h, self.w = self.input_shape
        self.patch_size = patch_size
        self.ph, self.pw = self.patch_size
        self.lr = lr

        self.encoder = MyVisionTransformer(
            self.h, self.ph, self.bins, embed_dim, num_layers, num_heads
        )

        patch_num = self.h // self.ph
        self.decoder = ImageDecoderCNN(patch_num, dim=embed_dim, num_cnn_layers=4)

        # Normalize should be True if images are in [0, 1] (False -> [-1, +1])
        self.lpips = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(
            net_type="vgg", normalize=True
        )
        self.ssim = torchmetrics.functional.structural_similarity_index_measure
        self.mse = torchmetrics.functional.mean_squared_error

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x

    def training_step(self, train_batch, batch_idx):
        X, y = train_batch
        y = torch.einsum("bhwc -> bchw", y)

        model_images = self(X)

        criterion = torch.nn.MSELoss()

        loss = criterion(model_images, y)

        self.log("train_loss", loss)

        # Compute metrics
        mse = self.mse(model_images, y)
        ssim = self.ssim(model_images, y, data_range=1)
        self.log("train_MSE", mse)
        self.log("train_SSIM", ssim)

        return loss

    def validation_step(self, val_batch, batch_idx):
        X, y = val_batch
        y = torch.einsum("bhwc -> bchw", y)

        model_images = self(X)

        criterion = torch.nn.MSELoss()

        loss = criterion(model_images, y)

        self.log("val_loss", loss)

        # Compute metrics
        mse = self.mse(model_images, y)
        ssim = self.ssim(model_images, y, data_range=1)
        self.lpips(model_images, y)

        self.log("val_MSE", mse)
        self.log("val_SSIM", ssim)
        self.log("val_LPIPS", self.lpips)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def predict_images(self, batch):
        events, images = batch
        return self(events)
