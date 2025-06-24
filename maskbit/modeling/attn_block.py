import math

import torch
import torch.nn as nn
import torch.nn.functional as F
 
from timm.layers import SwiGLU, DropPath




# automatically import fused operators
dropout_add_layer_norm = fused_mlp_func = memory_efficient_attention = flash_attn_func = None
try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
    from flash_attn.ops.fused_dense import fused_mlp_func
except ImportError: pass
# automatically import faster attention implementations
try: from xformers.ops import memory_efficient_attention
except ImportError: pass
try: from flash_attn import flash_attn_func              # qkv: BLHc, ret: BLHcq
except ImportError: pass
try: from torch.nn.functional import scaled_dot_product_attention as slow_attn    # q, k, v: BHLc
except ImportError:
    def slow_attn(query, key, value, scale: float, attn_mask=None, dropout_p=0.0):
        attn = query.mul(scale) @ key.transpose(-2, -1) # BHLc @ BHcL => BHLL
        if attn_mask is not None: attn.add_(attn_mask)
        return (F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn.softmax(dim=-1)) @ value


def get_default_modulation(x: torch.Tensor, dtype=None):
    shape = (1, 1, x.shape[-1])
    device = x.device
    gamma1 = torch.ones(shape, device=device, dtype=dtype or x.dtype)
    gamma2 = torch.ones(shape, device=device, dtype=dtype or x.dtype)
    scale1 = torch.zeros(shape, device=device, dtype=dtype or x.dtype)
    scale2 = torch.zeros(shape, device=device, dtype=dtype or x.dtype)
    shift1 = torch.zeros(shape, device=device, dtype=dtype or x.dtype)
    shift2 = torch.zeros(shape, device=device, dtype=dtype or x.dtype)

    return gamma1, gamma2, scale1, scale2, shift1, shift2


class FeedForward(nn.Module):
    def __init__(
            self, 
            in_features, 
            hidden_features=None, 
            out_features=None, 
            drop=0., 
            fused_if_available=True
        ):
        super().__init__()
        self.fused_mlp_func = fused_mlp_func if fused_if_available else None
        out_features = out_features or in_features
        hidden_features = hidden_features or (in_features * 4)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop, inplace=True) if drop > 0 else nn.Identity()
    
    def forward(self, x):
        if self.fused_mlp_func is not None:
            return self.drop(self.fused_mlp_func(
                x=x, weight1=self.fc1.weight, weight2=self.fc2.weight, bias1=self.fc1.bias, bias2=self.fc2.bias,
                activation='gelu_approx', save_pre_act=self.training, return_residual=False, checkpoint_lvl=0,
                heuristic=0, process_group=None,
            ))
        else:
            return self.drop(self.fc2( self.act(self.fc1(x)) ))
    
    def extra_repr(self) -> str:
        return f'fused_mlp_func={self.fused_mlp_func is not None}'


class SelfAttention(nn.Module):
    def __init__(
        self, 
        embed_dim=768, 
        num_heads=12,
        attn_drop=0., 
        proj_drop=0., 
        attn_l2_norm=False, 
        flash_if_available=True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads  # =64
        self.attn_l2_norm = attn_l2_norm
        if self.attn_l2_norm: 
            self.scale = 1
            self.scale_mul_1H11 = nn.Parameter(torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 0.25 / math.sqrt(self.head_dim)
        
        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias, self.v_bias = nn.Parameter(torch.zeros(embed_dim)), nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else nn.Identity()
        self.attn_drop: float = attn_drop
        self.using_flash = flash_if_available and flash_attn_func is not None and next(self.parameters()).device == torch.device('cuda')  # xformers only supports cuda
        self.using_xform = flash_if_available and memory_efficient_attention is not None and next(self.parameters()).device == torch.device('cuda')  # xformers only supports cuda
        

    
    
    
    # NOTE: attn_bias is None during inference because kv cache is enabled
    def forward(self, x, attn_bias=None):
        B, L, C = x.shape
        
        qkv = F.linear(input=x, weight=self.mat_qkv.weight, bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))).view(B, L, 3, self.num_heads, self.head_dim)
        main_type = qkv.dtype
        # qkv: BL3Hc
        
        using_flash = self.using_flash and attn_bias is None and qkv.dtype != torch.float32
        print(f'using_flash={self.using_flash}, using_xform={self.using_xform}, main_type={main_type}')
        if using_flash or self.using_xform: q, k, v = qkv.unbind(dim=2); dim_cat = 1   # q or k or v: BLHc
        else: q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0); dim_cat = 2               # q or k or v: BHLc
        
        if self.attn_l2_norm:
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            if using_flash or self.using_xform: scale_mul = scale_mul.transpose(1, 2)  # 1H11 to 11H1
            q = F.normalize(q, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)
        
        
        
        dropout_p = self.attn_drop if self.training else 0.0
        if using_flash:
            oup = flash_attn_func(q.to(dtype=main_type), k.to(dtype=main_type), v.to(dtype=main_type), dropout_p=dropout_p, softmax_scale=self.scale).view(B, L, C)
        elif self.using_xform:
            oup = memory_efficient_attention(q.to(dtype=main_type), k.to(dtype=main_type), v.to(dtype=main_type), attn_bias=None, p=dropout_p, scale=self.scale).view(B, L, C)
        else:
            oup = slow_attn(query=q, key=k, value=v, scale=self.scale, attn_mask=attn_bias, dropout_p=dropout_p).transpose(1, 2).reshape(B, L, C)
        
        return self.proj_drop(self.proj(oup))
        # attn = (q @ k.transpose(-2, -1)).add_(attn_bias + self.local_rpb())  # BHLc @ BHcL => BHLL
        # attn = self.attn_drop(attn.softmax(dim=-1))
        # oup = (attn @ v).transpose_(1, 2).reshape(B, L, -1)     # BHLL @ BHLc = BHLc => BLHc => BLC
    
    def extra_repr(self) -> str:
        return f'using_flash={self.using_flash}, using_xform={self.using_xform}, attn_l2_norm={self.attn_l2_norm}'
    
class CrossAttention(nn.Module):
    def __init__(
            self,
            embed_dim=768,
            num_heads=12,
            attn_drop=0., 
            proj_drop=0., 
            attn_l2_norm=False, 
            flash_if_available=True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads  # =64
        self.attn_l2_norm = attn_l2_norm
        if self.attn_l2_norm: 
            self.scale = 1
            self.scale_mul_1H11 = nn.Parameter(torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 0.25 / math.sqrt(self.head_dim)
        
        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, embed_dim, bias=False)

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else nn.Identity()
        self.attn_drop: float = attn_drop
        self.using_flash = flash_if_available and flash_attn_func is not None
        self.using_xform = flash_if_available and memory_efficient_attention is not None
        
        

    def forward(self, x, context, attn_bias=None):
        B, L, C = x.shape

        q = self.q(x).reshape(B, -1, self.num_heads, self.head_dim)  # BLHc 
        k = self.k(context).reshape(B, -1, self.num_heads, self.head_dim)  # BLHc
        v = self.v(context).reshape(B, -1, self.num_heads, self.head_dim)  # BLHc
        main_type = q.dtype

        using_flash = self.using_flash and attn_bias is None and main_type != torch.float32
        if not using_flash and not self.using_xform:
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)  # BHLc

        if self.attn_l2_norm:
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            if using_flash or self.using_xform: scale_mul = scale_mul.transpose(1, 2)  # 1H11 to 11H1
            q = F.normalize(q, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)
        
        dropout_p = self.attn_drop if self.training else 0.0    
        if using_flash:
            oup = flash_attn_func(q.to(dtype=main_type), k.to(dtype=main_type), v.to(dtype=main_type), dropout_p=dropout_p, softmax_scale=self.scale).view(B, L, C)
        elif self.using_xform:
            oup = memory_efficient_attention(q.to(dtype=main_type), k.to(dtype=main_type), v.to(dtype=main_type), attn_bias=None, p=dropout_p, scale=self.scale).view(B, L, C)
        else:
            oup = slow_attn(query=q, key=k, value=v, scale=self.scale, attn_mask=attn_bias, dropout_p=dropout_p).transpose(1, 2).reshape(B, L, C)
        
        return self.proj_drop(self.proj(oup))
    

class SpatialAdaLNAttnBlock(nn.Module):
    def __init__(
            self,
            embed_dim=768,
            num_heads=12,
            attn_drop=0., 
            proj_drop=0.,
            drop_path=0.,
            attn_l2_norm=False,
            flash_if_available=True,
            fused_if_available=True,
            use_ada_ln=True,
            mlp_ratio=4.0,  ## ratio of hidden dim to embed dim
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn = SelfAttention(
            embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=proj_drop, 
            attn_l2_norm=attn_l2_norm, flash_if_available=flash_if_available
        )
        self.ffn = FeedForward(in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio), drop=proj_drop, fused_if_available=fused_if_available)
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ada_lin = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, embed_dim * 6))
    
    def forward(self, x, context, cond=None, attn_bias=None):
        B, L, C = x.shape
        if cond is not None:
            context = cond + context  # BLC
        gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(context).view(B, L, 6, self.embed_dim).unbind(2)
        x = x + self.drop_path(self.attn(self.norm1(x).mul(scale1.add(1)).add_(shift1), attn_bias=attn_bias ).mul_(gamma1))
        x = x + self.drop_path(self.ffn(self.norm2(x).mul(scale2.add(1)).add_(shift2)).mul_(gamma2))
        return x, context

class ControlAttnBlock(nn.Module):
    def __init__(
            self,
            embed_dim=768,
            num_heads=12,
            attn_drop=0., 
            proj_drop=0.,
            drop_path=0.,
            attn_l2_norm=False,
            flash_if_available=True,
            fused_if_available=True,
            use_ada_ln=False,  ## whether use AdaLN for class label
            mlp_ratio=4.0,  ## ratio of hidden dim to embed dim
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_x = SelfAttention(
            embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=proj_drop, 
            attn_l2_norm=attn_l2_norm, flash_if_available=flash_if_available
        )
        self.attn_c = SelfAttention(
            embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=proj_drop, 
            attn_l2_norm=attn_l2_norm, flash_if_available=flash_if_available
        )
        self.ffn_x = self.ffn = FeedForward(in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio), drop=proj_drop, fused_if_available=fused_if_available)
        self.ffn_c = self.ffn = FeedForward(in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio), drop=proj_drop, fused_if_available=fused_if_available)
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.ada_lin = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim * 6) 
        ) if use_ada_ln else None


    def forward(self, x, context, cond=None, attn_bias=None):
        B, L, C = x.shape

        if (self.ada_lin is not None) and (cond is not None):
            gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond).view(B, 1, 6, self.embed_dim).unbind(2)
        else:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = get_default_modulation(x)

        x = x + self.drop_path(self.attn_x(self.norm1(x).mul(scale1.add(1)).add_(shift1), attn_bias=attn_bias).mul_(gamma1))
        context = context + self.drop_path(self.attn_c(self.norm1(context).mul(scale1.add(1)).add_(shift1), attn_bias=attn_bias).mul_(gamma1))

        x[:, :context.shape[1], :] = x[:, :context.shape[1], :] + context   ## control net style conditioning
        
        x = x + self.drop_path(self.ffn_x(self.norm2(x).mul(scale2.add(1)).add_(shift2)).mul_(gamma2))
        context = context + self.drop_path(self.ffn_c(self.norm2(context).mul(scale2.add(1)).add_(shift2)).mul_(gamma2))
        
        return x, context


class BasicAttnBlock(nn.Module):
    def __init__(
            self,
            embed_dim=768,
            num_heads=12,
            attn_drop=0., 
            proj_drop=0.,
            drop_path=0.,
            attn_l2_norm=False,
            flash_if_available=True,
            fused_if_available=True,
            use_ada_ln=False,  ## whether use AdaLN for class label
            mlp_ratio=4.0,  ## ratio of hidden dim to embed dim
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn = SelfAttention(
            embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=proj_drop, 
            attn_l2_norm=attn_l2_norm, flash_if_available=flash_if_available
        )
        self.ffn = self.ffn = FeedForward(in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio), drop=proj_drop, fused_if_available=fused_if_available)
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ada_lin = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, embed_dim * 6)) if use_ada_ln else None
    
    def forward(self, x, context=None, cond=None, attn_bias=None):
        B, L, C = x.shape
        if (self.ada_lin is not None) and (cond is not None):
            gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond).view(B, 1, 6, self.embed_dim).unbind(2)
        else:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = get_default_modulation(x)

        x = x + self.drop_path(self.attn(self.norm1(x).mul(scale1.add(1)).add_(shift1), attn_bias=attn_bias).mul_(gamma1))
        x = x + self.drop_path(self.ffn(self.norm2(x).mul(scale2.add(1)).add_(shift2)).mul_(gamma2))
        return x, context
    
class CrossAttnBlock(nn.Module):
    def __init__(
            self,
            embed_dim=768,
            num_heads=12,
            attn_drop=0., 
            proj_drop=0.,
            drop_path=0.,
            attn_l2_norm=False,
            flash_if_available=True,
            fused_if_available=True,
            use_ada_ln=False,  ## whether use AdaLN for class label
            mlp_ratio=4.0,  ## ratio of hidden dim to embed dim
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.self_attn = SelfAttention(
            embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=proj_drop, 
            attn_l2_norm=attn_l2_norm, flash_if_available=flash_if_available
        )
        self.cross_attn = CrossAttention(
            embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=proj_drop, 
            attn_l2_norm=attn_l2_norm, flash_if_available=flash_if_available
        )
        self.ffn = FeedForward(in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio), drop=proj_drop, fused_if_available=fused_if_available)
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.norm2_1 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.norm2_2 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ada_lin = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, embed_dim * 6)) if use_ada_ln else None

    def forward(self, x, context, cond=None, attn_bias=None):
        B, L, C = x.shape
        if (self.ada_lin is not None) and (cond is not None):
            gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond).view(B, 1, 6, self.embed_dim).unbind(2)
        else:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = get_default_modulation(x)
        
        x = x + self.drop_path(self.self_attn(self.norm1(x).mul(scale1.add(1)).add_(shift1), attn_bias=attn_bias).mul_(gamma1))
        x = x + self.cross_attn(self.norm2_1(x), self.norm2_2(context), attn_bias=attn_bias)
        x = x + self.drop_path(self.ffn(self.norm3(x).mul(scale2.add(1)).add_(shift2)).mul_(gamma2))
        return x, context


class CondTransformerEncoder(nn.Module):
    def __init__(
            self,
            dim=768,
            depth=12,
            num_heads=12,
            attn_drop=0.,
            proj_drop=0.,
            drop_path=0.,
            mlp_ratio=4.0,  ## ratio of hidden dim to embed dim
            attn_l2_norm=False,
            flash_if_available=True,
            fused_if_available=True,
            context_conditioning="control",
            label_conditioning="adaln",
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.drop_path = drop_path
        self.attn_l2_norm = attn_l2_norm
        self.flash_if_available = flash_if_available
        self.fused_if_available = fused_if_available
        self.mlp_ratio = mlp_ratio

        self.context_conditioning = context_conditioning
        self.label_conditioning = label_conditioning
        
        if context_conditioning == "control":
            block_cls = ControlAttnBlock
        elif context_conditioning == "cross":
            block_cls = CrossAttnBlock
        elif context_conditioning in ['embed', 'channel', 'concat']:
            block_cls = BasicAttnBlock
        elif context_conditioning == "adaln":
            block_cls = SpatialAdaLNAttnBlock
        else:
            raise ValueError(f'Unknown context_conditioning: {context_conditioning}')
        
        # if context_conditioning == "channel":
        #     embed_dim = dim * 2
        # else:
        #     embed_dim = dim
        
        
        drop_path_rate = [x.item() for x in torch.linspace(0, drop_path, depth)]

        self.blocks = nn.ModuleList([
            block_cls(
                embed_dim=dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=proj_drop,
                drop_path=drop_path_rate[i],
                attn_l2_norm=attn_l2_norm, flash_if_available=flash_if_available,
                fused_if_available=fused_if_available, use_ada_ln=(label_conditioning == "adaln"),
                mlp_ratio=mlp_ratio
            ) for i in range(depth)
        ])

    def forward(self, x, context=None, cond=None, attn_bias=None):
        B, L, C = x.shape
        if context is None:
            assert self.context_conditioning in ['embed', 'concat', 'channel'], \
                f'context must be provided when context_conditioning is {self.context_conditioning}'
        if cond is not None:
            assert self.label_conditioning == "adaln", \
                f'cond must be None when label_conditioning is {self.label_conditioning}'
        
        for block in self.blocks:
            x, context = block(x, context=context, cond=cond, attn_bias=attn_bias)
        
        return x, context