import math
from typing import List, Tuple, Union, Optional

import torch
from torch import nn
from einops import rearrange

from timm.models.vision_transformer import DropPath
from timm.layers import SwiGLU

from modeling.modules import BaseModel

from modeling.attn_block import CondTransformerEncoder


class CondBert(BaseModel):
    def __init__(
            self,
            stage=1,
            ## token params
            img_size=256,
            codebook_size=3**10,
            patch_size=16,
            context_dim=10,  ## number of bits
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
            tie_context_embeddings=False,
    ):
        super().__init__()
        self.register_buffer("stage", torch.tensor(stage, dtype=torch.int32))
        self.img_size = img_size
        self.codebook_size = codebook_size
        self.patch_size = patch_size
        self.seq_len = (img_size // patch_size) ** 2

        assert context_conditioning in ["control", "cross", "concat", "adaln", "embed", "channel"],  \
            f'context_conditioning must be one of ["control", "cross", "concat", "adaln", "embed"], but got {context_conditioning}'
        assert label_conditioning in ["adaln", "concat"], \
            f'label_conditioning must be one of ["adaln", "concat"], but got {label_conditioning}'
        self.context_conditioning = context_conditioning
        self.label_conditioning = label_conditioning
        self.num_classes = num_classes
        self.drop_label = num_classes

        
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
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
        self.tok_emb = nn.Embedding(
            codebook_size + 1, self.emb_dim
        )
        if tie_embeddings:
            self.proj_weight = self.tok_emb.weight[:codebook_size, :]
        else:
            self.proj_weight = nn.Parameter(
                torch.empty(codebook_size, self.emb_dim)
            )
            nn.init.trunc_normal_(self.proj_weight.data, mean=0.0, std=0.02)
        self.proj_bias = nn.Parameter(torch.zeros(self.seq_len, codebook_size))
        self.class_emb = nn.Embedding(self.num_classes + 1, hidden_dim)
        self.pos_emb = torch.nn.init.trunc_normal_(torch.nn.Parameter(torch.zeros(1, self.seq_len, self.emb_dim)), mean=0., std=0.02) 
        if tie_context_embeddings:
            self.context_pos_emb = self.pos_emb[:, :self.seq_len, :]
        else:
            self.context_pos_emb = torch.nn.init.trunc_normal_(torch.nn.Parameter(torch.zeros(1, self.seq_len, self.emb_dim)), mean=0., std=0.02)
        self.context_proj = nn.Linear(context_dim, self.emb_dim)

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
            token_indices: torch.Tensor,   ## [B, H, W] or [B, L]
            context: torch.Tensor,    ## [B, C, H, W]
            class_labels: torch.Tensor,
            drop_label_mask: Optional[torch.Tensor] = None,
    ):
        b = token_indices.size(0)
        token_indices = token_indices.view(b, -1)  # [B, L]

        from einops import rearrange
        context = rearrange(context, 'b c h w -> b (h w) c').contiguous()  # [B, L, C]
       
        cls_token = class_labels.view(b, -1)
        if drop_label_mask is not None:
            cls_token[drop_label_mask] = self.drop_label  # Drop condition
        cls_embedding = self.class_emb(cls_token.view(b, -1))

        tok_emb = self.tok_emb(token_indices)
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

        if self.label_conditioning == 'concat':
            x = torch.cat([x, cls_embedding], dim=1)

        x, context = self.transformer(
            x,
            context=context,
            cond=cls_embedding if self.label_conditioning == 'adaln' else None,
        )

        x = x[:, :self.seq_len, :]  
        x = self.last_layer(x)

        logits = torch.matmul(x, self.proj_weight.t()) + self.proj_bias  # [B, L, C]

        return logits
    


if __name__ == "__main__":
    device = torch.device("cuda")
    dtype = torch.float16
    stage = 2
    model = CondBert(
        stage=stage,
        img_size=256,
        codebook_size=3**10,
        patch_size=16,
        context_dim=10,
        hidden_dim=16,
        depth=2,
        heads=1,
        mlp_ratio=4.0,
        dropout=0.0,
        drop_path=0.0,
        attn_l2_norm=False,
        flash_if_available=True,
        fused_if_available=True,
        context_conditioning="channel",
        label_conditioning="concat",
        num_classes=1000,
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
    class_labels = torch.randint(0, 1000, (2,)).to(device)
    drop_label_mask = torch.zeros(2, dtype=torch.bool).to(device)
    logits = model(
        token_indices=token_indices,
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
        mask_token=model.codebook_size,
    )[0]
    print(image.shape)

    
        
    






