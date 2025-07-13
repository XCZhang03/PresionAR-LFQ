import math
from typing import List, Tuple, Union, Optional
from omegaconf import ListConfig

import torch
from torch import nn
from einops import rearrange

from timm.models.vision_transformer import DropPath
from timm.layers import SwiGLU

from modeling.modules import BaseModel, get_codebook_config

from modeling.attn_block import CondTransformerEncoder


class CondBert(BaseModel):
    def __init__(
            self,
            stage=1,
            ## token params
            img_size=256,
            input_stride=16,
            token_size=10,  ## number of bits
            codebook_size=None,
            codebook_splits=1,
            variants=None,
            ## context params
            ## transformer params
            hidden_dim=768,
            depth=20,
            heads=12,
            mlp_ratio=4.0,
            dropout=0.1,
            drop_path=0.1,
            attn_l2_norm=False,
            flash_if_available=True,
            fused_if_available=True,
            ## conditioning params
            context_conditioning="adaln",
            label_conditioning="adaln",
            num_classes=1000,
            ## Bert params
            tie_embeddings=False,
            tie_context_pos_embeddings=False,
    ):
        super().__init__()
        self.register_buffer("stage", torch.tensor(stage, dtype=torch.int32))
        
        self.codebook_size, self.token_size, self.variants = get_codebook_config(
            codebook_size=codebook_size,
            bits=token_size,
            variants=variants
        )
        self.splits = codebook_splits
        self.effective_codebook_size = int(self.variants ** (self.token_size // self.splits))
        assert self.effective_codebook_size ** self.splits == self.codebook_size, \
            f"Effective codebook size should match the codebook size" 
        self.mask_token = self.effective_codebook_size  # Mask token is the last token in the codebook

        self.img_size = img_size
        self.input_stride = input_stride
        self.seq_len = (img_size // input_stride) ** 2

        assert context_conditioning in ["control", "cross", "concat", "adaln", "embed", "channel"],  \
            f'context_conditioning must be one of ["control", "cross", "concat", "adaln", "embed"], but got {context_conditioning}'
        assert label_conditioning in ["adaln", "concat", "both"], \
            f'label_conditioning must be one of ["adaln", "concat", "both"], but got {label_conditioning}'
        self.context_conditioning = context_conditioning
        self.label_conditioning = label_conditioning
        self.num_classes = num_classes
        self.drop_label = num_classes

        self.hidden_dim = hidden_dim
        self.context_dim = token_size
        if self.context_conditioning == "channel":
            self.emb_dim = hidden_dim // 2
        else:
            self.emb_dim = hidden_dim

        
        ### define transformer 
        self.transformer = CondTransformerEncoder(
            dim=hidden_dim,
            depth=depth,
            num_heads=heads,
            attn_drop=dropout,
            proj_drop=dropout,
            drop_path=drop_path,
            mlp_ratio=mlp_ratio,
            attn_l2_norm=attn_l2_norm,
            flash_if_available=flash_if_available,
            fused_if_available=fused_if_available,
            context_conditioning=context_conditioning,
            label_conditioning=label_conditioning,
        )

        ### define embedding layers
        self.tok_emb_list = torch.nn.ModuleList()
        for _ in range(self.splits):
            self.tok_emb_list.append(torch.nn.Embedding(self.effective_codebook_size + 1, self.emb_dim))  # +1 for mask token
        
        self.proj_weight = nn.ParameterList()
        if tie_embeddings:
            for i in range(self.splits):
                self.proj_weight.append(self.tok_emb_list[i].weight[:self.effective_codebook_size, :])
        else:
            for i in range(self.splits):
                self.proj_weight.append(torch.nn.Parameter(
                    torch.empty(self.effective_codebook_size, self.emb_dim)
                ))
                nn.init.trunc_normal_(self.proj_weight[i].data, mean=0.0, std=0.02)
        self.proj_bias = torch.nn.ParameterList()
        for _ in range(self.splits):
            self.proj_bias.append(torch.nn.Parameter(torch.zeros((self.seq_len), self.effective_codebook_size)))

        self.class_emb = nn.Embedding(self.num_classes + 1, hidden_dim)
        self.pos_emb = torch.nn.init.trunc_normal_(torch.nn.Parameter(torch.zeros(1, self.seq_len, self.emb_dim)), mean=0., std=0.02) 
        if tie_context_pos_embeddings:
            self.context_pos_emb = self.pos_emb
        else:
            self.context_pos_emb = torch.nn.init.trunc_normal_(torch.nn.Parameter(torch.zeros(1, self.seq_len, self.emb_dim)), mean=0., std=0.02)
        self.context_proj = nn.Linear(self.context_dim, self.emb_dim)

        # Last layer after the Transformer block
        self.last_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.hidden_dim, out_features=self.emb_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(self.emb_dim, eps=1e-12),
        )

        self.apply(self._init_weights)
        

    def _init_weights(self, module: torch.nn.Module):
        """ Initialize the weights.

            Args:
                module -> torch.nn.Module: The module to initialize.
        """
        if isinstance(module, torch.nn.Linear):
            module.weight.data = torch.nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data = torch.nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, torch.nn.LayerNorm):
            if module.bias is not None:
                module.bias.data.zero_()
            if module.weight is not None:
                module.weight.data.fill_(1.0)
        
    def forward(
            self,
            token_indices: torch.Tensor,   ## [B, H, W, G] or [B, L, G]
            context: torch.Tensor,    ## [B, C, H, W]
            class_labels: torch.Tensor,
            drop_label_mask: Optional[torch.Tensor] = None,
    ):
        b = token_indices.size(0)
        splits = token_indices.shape[-1]
        assert splits == self.splits, f"Expected {self.splits} splits, but got {splits}"
        token_indices = token_indices.view(b, -1, splits)  # [B, L, G]


        context = context.clone()
        context = rearrange(context, 'b c h w -> b (h w) c').contiguous()  # [B, L, C]
       
        cls_token = class_labels.view(b, -1)
        if drop_label_mask is not None:
            cls_token[drop_label_mask] = self.drop_label  # Drop condition
        cls_embedding = self.class_emb(cls_token.view(b, -1))

        tok_emb = self.tok_emb_list[0](token_indices[:, :, 0])  # [B, L, D]
        for i in range(1, self.splits):
            tok_emb += self.tok_emb_list[i](token_indices[:, :, i])
        context_emb = self.context_proj(context)
        tok_emb = tok_emb + self.pos_emb
        context_emb = context_emb + self.context_pos_emb

        if self.context_conditioning in ['adaln', 'cross', 'control']:
            if self.context_conditioning == 'adaln' and self.label_conditioning == 'concat':
                raise ValueError(
                    "can not use 'concat' for label when using 'adaln' for context, "
                    "use 'adaln' for label_conditioning instead"
                )
            x = tok_emb
            context = context_emb
        elif self.context_conditioning == 'concat':
            x = torch.cat([tok_emb, context_emb], dim=1)
            context = None
        elif self.context_conditioning == 'embed':
            x = tok_emb[:, :self.seq_len, :] + context_emb
            context = None
        elif self.context_conditioning == 'channel':
            x = torch.cat([tok_emb, context_emb], dim=-1)
            context = None
        else:
            raise ValueError(f'Unknown context_conditioning: {self.context_conditioning}')

        if self.label_conditioning != 'adaln':
            x = torch.cat([x, cls_embedding], dim=1)

        x, context = self.transformer(
            x,
            context=context,
            cond=cls_embedding if self.label_conditioning != 'concat' else None,
        )

        x = x[:, :self.seq_len, :]  
        x = self.last_layer(x)
        logits = []
        for i in range(self.splits):
            logits.append(torch.matmul(x, self.proj_weight[i].t()) + self.proj_bias[i]) # [B, L, D]
        logits = torch.stack(logits, dim=2)  # [B, L, G, D]

        return logits
    
class CondLFQBert(BaseModel):
    def __init__(
            self,
            stage=1,
            ## token params
            img_size=256,
            input_stride=16,
            token_size=10,  ## number of bits
            codebook_size=None,
            codebook_splits=1,
            variants=None,
            scales=None,
            ## context params
            ## transformer params
            hidden_dim=768,
            depth=20,
            heads=12,
            mlp_ratio=4.0,
            dropout=0.1,
            drop_path=0.1,
            attn_l2_norm=False,
            flash_if_available=True,
            fused_if_available=True,
            ## conditioning params
            label_conditioning="adaln",
            num_classes=1000,
            ## masking condition
            mask_pos_embedding=False,
            mask_token_embedding=True,
    ):
        super().__init__()
        self.register_buffer("stage", torch.tensor(stage, dtype=torch.int32))
        
        self.codebook_size, self.token_size, self.variants = get_codebook_config(
            codebook_size=codebook_size,
            bits=token_size,
            variants=variants
        )
        self.splits = codebook_splits
        self.effective_token_size = self.token_size // self.splits
        self.effective_codebook_size = int(self.variants ** (self.effective_token_size))
        assert self.effective_codebook_size ** self.splits == self.codebook_size, \
            f"Effective codebook size should match the codebook size" 
        self.mask_token = self.effective_codebook_size  # Mask token is the last token in the codebook
        if mask_token_embedding:
            self.mask_embed = torch.nn.init.trunc_normal_(torch.nn.Parameter(torch.zeros(1, self.effective_token_size)), 0., 0.02)
        else:
            self.mask_embed = torch.nn.Parameter(torch.zeros(1, self.effective_token_size), requires_grad=False)
        self.mask_pos_embed = None
        if mask_pos_embedding:
            assert self.splits == 1, "Mask position embedding does not support codebook splits > 1"
            self.mask_pos_embed = torch.nn.init.trunc_normal_(torch.nn.Parameter(torch.zeros(1, hidden_dim)), 0., 0.02)
        from modeling.quantizer.mutivariante_lfq import MultivariantLFQ
        if isinstance(scales, (ListConfig, list)):
            self.scale = scales[stage]
        elif isinstance(scales, int):
            self.scale = scales ** -stage
        else:
            self.scale = 2 ** -stage
        print(f"quantizer scale: {self.scale}")
        self.quantizer = MultivariantLFQ(
            token_size=self.effective_token_size,
            variants=self.variants,
            scale=self.scale
        )


        self.img_size = img_size
        self.input_stride = input_stride
        self.seq_len = (img_size // input_stride) ** 2

        self.label_conditioning = label_conditioning
        self.num_classes = num_classes
        self.drop_label = num_classes

        self.hidden_dim = hidden_dim
        self.context_dim = token_size

        
        ### define transformer 
        self.transformer = CondTransformerEncoder(
            dim=hidden_dim,
            depth=depth,
            num_heads=heads,
            attn_drop=dropout,
            proj_drop=dropout,
            drop_path=drop_path,
            mlp_ratio=mlp_ratio,
            attn_l2_norm=attn_l2_norm,
            flash_if_available=flash_if_available,
            fused_if_available=fused_if_available,
            context_conditioning='none',
            label_conditioning=label_conditioning,
        )

        ### define embedding layers
        self.input_proj = nn.Linear(self.token_size, self.hidden_dim)
        self.class_emb = nn.Embedding(self.num_classes + 1, self.hidden_dim)
        self.pos_emb = torch.nn.init.trunc_normal_(torch.nn.Parameter(torch.zeros(1, self.seq_len, self.hidden_dim)), mean=0., std=0.02) 
        # First layer before the Transformer block
        self.first_layer = torch.nn.Sequential(
            torch.nn.LayerNorm(self.hidden_dim, eps=1e-12),
            torch.nn.Dropout(p=dropout)
        )
        # Last layer after the Transformer block
        self.last_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(self.hidden_dim, eps=1e-12),
        )
        self.prediction_layer = torch.nn.Linear(self.hidden_dim, self.splits * self.effective_codebook_size)

        self.apply(self._init_weights)

    def _init_weights(self, module: torch.nn.Module):
        """ Initialize the weights.

            Args:
                module -> torch.nn.Module: The module to initialize.
        """
        if isinstance(module, torch.nn.Linear):
            module.weight.data = torch.nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data = torch.nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, torch.nn.LayerNorm):
            if module.bias is not None:
                module.bias.data.zero_()
            if module.weight is not None:
                module.weight.data.fill_(1.0)

    def preprocess_tokens(self, token_indices: torch.Tensor):
        mask = (token_indices == self.mask_token)
        tokens = self.quantizer.get_codebook_entry(token_indices)
        tokens[mask, :] = self.mask_embed
        tokens = rearrange(tokens, 'b l g d -> b l (g d)')
        return tokens, mask
    
    def forward(
            self,
            token_indices: torch.Tensor,   ## [B, H, W, G] or [B, L, G]
            context: torch.Tensor,    ## [B, C, H, W]
            class_labels: torch.Tensor,
            drop_label_mask: Optional[torch.Tensor] = None,
    ):
        b = token_indices.size(0)
        splits = token_indices.shape[-1]
        assert splits == self.splits, f"Expected {self.splits} splits, but got {splits}"
        
        token_indices = token_indices.view(b, -1, splits)  # [B, L, G]
        tokens, mask = self.preprocess_tokens(token_indices)  # [B, L, D]
        
        context = context.clone()
        context = rearrange(context, 'b c h w -> b (h w) c').contiguous()  # [B, L, C]
        
        x = tokens + context
        x = self.input_proj(x)  # [B, L, D]
        x = x + self.pos_emb  # [B, L, D]
        if self.mask_pos_embed is not None:
            mask = mask.squeeze(-1)  # [B, L]
            x[mask, :] = x[mask, :] + self.mask_pos_embed  # Add mask position embedding
        
        cls_token = class_labels.view(b, -1)
        if drop_label_mask is not None:
            cls_token[drop_label_mask] = self.drop_label  # Drop condition
        cls_embedding = self.class_emb(cls_token.view(b, -1))
        if self.label_conditioning == 'concat':
            x = torch.cat([x, cls_embedding], dim=1)
        
        x = self.first_layer(x)  # [B, L, D]
        x, _ = self.transformer(  
            x,
            context=None,
            cond=cls_embedding if self.label_conditioning == 'adaln' else None,
        )
        x = x[:, :self.seq_len, :]  
        x = self.last_layer(x)
        logits = rearrange(self.prediction_layer(x), 'b l (g d) -> b l g d', g=self.splits, d=self.effective_codebook_size)  # [B, L, G, D]

        return logits

    



if __name__ == "__main__":
    device = torch.device("cuda")
    dtype = torch.float16
    stage = 2
    model = CondLFQBert(
        stage=stage,
        img_size=256,
        variants=3,
        codebook_splits=2,
        scales=None,
        input_stride=16,
        token_size=10,
        hidden_dim=16,
        depth=2,
        heads=1,
        mlp_ratio=4.0,
        dropout=0.0,
        drop_path=0.0,
        attn_l2_norm=False,
        flash_if_available=True,
        fused_if_available=True,
        label_conditioning="concat",
        num_classes=1000,
        mask_token_embedding=True,
        mask_pos_embedding=False,
    ).to(device=device, dtype=dtype)
    print(model)
    print(f"Model parameters: {model.num_parameters() / 1e6:.2f}M")
    from modeling.rqgan import RQModel
    from omegaconf import OmegaConf
    config = OmegaConf.load("maskbit/configs/tokenizer/rqbit_tokenizer_10bit_4lvl.yaml").model.vq_model
    vae = RQModel(
        config=config,
    ).to(device=device, dtype=dtype)
    image = torch.randn(2, 3, 256, 256).to(device=device, dtype=dtype) * 1e2
    z_quantized, result_dict = vae.encode(image,num_levels=stage)
    token_indices = result_dict['min_encoding_indices'][stage]
    from modeling.modules.factorization import split_factorized_tokens
    token_indices = split_factorized_tokens(
        token_indices.reshape(token_indices.shape[0], -1),
        codebook_size=model.codebook_size,
        splits=model.splits,
        bits=model.token_size,
        variants=model.variants
    )
    tokens = model.preprocess_tokens(token_indices)
    decoded_tokens = vae.get_codebook_entry(result_dict['min_encoding_indices'][stage], num_level=stage).permute(0, 2, 3, 1).contiguous()
    decoded_tokens = rearrange(decoded_tokens, 'b h w d -> b (h w) d')
    assert (decoded_tokens == tokens[0]).all(), "Decoded tokens do not match the preprocessed tokens"
    class_labels = torch.randint(0, 1000, (2,)).to(device)
    drop_label_mask = torch.zeros(2, dtype=torch.bool).to(device)
    from modeling.modules.masking import get_mask_tokens
    masked_tokens, masks = get_mask_tokens(
        tokens=token_indices,
        mask_token=model.mask_token,
    )
    logits = model(
        token_indices=masked_tokens,
        context=z_quantized,
        class_labels=class_labels,
        drop_label_mask=drop_label_mask
    )
    print(logits.shape)
    from modeling.modules.sampling import conditional_sample
    image = conditional_sample(
        model=model,
        vqgan_model=vae,
        stage=stage,
        context=z_quantized,
        num_samples=z_quantized.shape[0],
        labels=class_labels,
        mask_token=model.mask_token,
        codebook_splits=model.splits,
        codebook_size=model.codebook_size,
        bits=model.token_size,
        variants=model.variants,
        num_steps=2,
    )[0]
    print(image.shape)

    
        
    






