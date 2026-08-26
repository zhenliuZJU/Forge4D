# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

import xformers.ops as xops

import psutil
import time
from typing import Dict, List
import gc


XFORMERS_AVAILABLE = True
class FlashAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.drop_prob = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, pos=None, attn_mask=None) -> Tensor:
        DTYPE = x.dtype
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)

        if q.dtype in [torch.float32]:
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)

        x = xops.memory_efficient_attention(
            q, k, v,
            attn_bias=None,
            p=self.drop_prob if self.training else 0.0,
            op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
        ).to(DTYPE)

        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FlashCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.drop_prob = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x1: Tensor, x2: Tensor, pos1=None, pos2=None, attn_mask=None) -> Tensor:
        DTYPE = x1.dtype
        B, N, C = x1.shape
        _, N2, _ = x2.shape
        q = self.q(x1).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(x2).reshape(B, N2, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos1)
            k = self.rope(k, pos2)

        q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)

        if q.dtype in [torch.float32]:
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)

        x = xops.memory_efficient_attention(
            q, k, v,
            attn_bias=None,
            p=self.drop_prob if self.training else 0.0,
            op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
        ).to(DTYPE)

        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LinearAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.drop_prob = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
        self.init_weights()

    def init_weights(self):
        nn.init.constant_(self.proj.weight, 0)
        nn.init.constant_(self.proj.bias, 0)

    def forward(self, x: Tensor, pos=None, attn_mask=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)

        if q.dtype in [torch.float32]:
            q = q.to(torch.float16)
            k = k.to(torch.float16)
            v = v.to(torch.float16)
        
        q = q.permute(0,2,3,1)
        k = k.permute(0,2,3,1)
        v = v.permute(0,2,3,1)
        self.kernel_func = nn.Softmax(dim=-2)
        # self.kernel_func = nn.ReLU(inplace=False)
        # lightweight linear attention
        q = self.kernel_func(q)  # B, h, h_d, N
        k = self.kernel_func(k.transpose(-1, -2))

        use_fp32_attention = getattr(self, "fp32_attention", False)  # necessary for NAN loss
        if use_fp32_attention:
            q, k, v = q.float(), k.float(), v.float()

        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=1)
        vk = torch.matmul(v, k)
        out = torch.matmul(vk, q)

        if out.dtype in [torch.float16, torch.bfloat16]:
            out = out.float()
        out = out[:, :, :-1] / (out[:, :, -1:] + 1e-15)
        out = out.permute(0,3,1,2)

        x = out.reshape(B, N, C)
        # x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, pos=None, attn_mask=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)

            # Mask
            if attn_mask is not None:
                assert attn_mask.shape[-2:] == (N, N), f"Expected mask shape [..., {N}, {N}], got {attn_mask.shape}"
                attn = attn + attn_mask
                
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, pos=None, attn_mask=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_mask is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = qkv.unbind(2)

        x = xops.memory_efficient_attention(q, k, v, attn_bias=attn_mask)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def print_memory_table(headers, data):
    """打印内存占用表格"""
    print("\n" + "="*80)
    print("显存占用对比表")
    print("="*80)
    
    # 使用简单格式打印表格
    col_widths = [max(len(str(row[i])) for row in data + [headers]) for i in range(len(headers))]
    
    # 打印表头
    header_row = " | ".join(f"{headers[i]:^{col_widths[i]}}" for i in range(len(headers)))
    print(header_row)
    print("-" * len(header_row))
    
    # 打印数据行
    for row in data:
        row_str = " | ".join(f"{str(row[i]):^{col_widths[i]}}" for i in range(len(row)))
        print(row_str)

def format_memory(mb_value):
    """格式化内存显示"""
    if mb_value >= 1024:
        return f"{mb_value/1024:.1f} GB"
    else:
        return f"{mb_value:.0f} MB"

def measure_attention_memory(attention_class, batch_size, seq_len, dim, num_heads, 
                           forward_only=True, mixed_precision=False):
    """测量Attention模块的内存占用"""
    try:
        # 清理内存
        torch.cuda.empty_cache()
        gc.collect()
        
        # 记录初始内存
        initial_memory = torch.cuda.memory_allocated() / 1024**2
        
        # 创建模块
        attention = attention_class(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=True,
            proj_bias=True,
            attn_drop=0.0,
            proj_drop=0.0,
            fused_attn=True
        ).cuda()
        
        # 切换到评估模式
        attention.eval()
        
        # 创建测试数据
        dtype = torch.bfloat16 if mixed_precision else torch.float32
        x = torch.randn(batch_size, seq_len, dim, device='cuda', dtype=dtype)
        pos = torch.arange(seq_len, device='cuda').unsqueeze(0).expand(batch_size, -1)
        
        # 前向传播
        with torch.no_grad():
            torch.cuda.reset_peak_memory_stats()
            
            if attention_class.__name__ == 'FlashCrossAttention':
                x2 = torch.randn(batch_size, seq_len, dim, device='cuda', dtype=dtype)
                pos2 = torch.arange(seq_len, device='cuda').unsqueeze(0).expand(batch_size, -1)
                output = attention(x, x2, pos, pos2)
            else:
                output = attention(x, pos)
            
            peak_memory_forward = torch.cuda.max_memory_allocated() / 1024**2
        
        # 后向传播（如果需要）
        peak_memory_backward = 0
        if not forward_only:
            attention.train()
            
            if attention_class.__name__ == 'FlashCrossAttention':
                output = attention(x, x2, pos, pos2)
            else:
                output = attention(x, pos)
            
            loss = output.sum()
            
            torch.cuda.reset_peak_memory_stats()
            loss.backward()
            peak_memory_backward = torch.cuda.max_memory_allocated() / 1024**2
        
        # 最终内存
        final_memory = torch.cuda.memory_allocated() / 1024**2
        memory_increase = final_memory - initial_memory
        
        # 清理
        del attention, x, pos, output
        if 'x2' in locals():
            del x2, pos2
        
        torch.cuda.empty_cache()
        gc.collect()
        
        return {
            'peak_forward': peak_memory_forward,
            'peak_backward': peak_memory_backward,
            'memory_increase': memory_increase,
            'status': '成功'
        }
        
    except torch.cuda.OutOfMemoryError:
        return {
            'peak_forward': float('inf'),
            'peak_backward': float('inf'),
            'memory_increase': float('inf'),
            'status': 'OOM'
        }
    except Exception as e:
        return {
            'peak_forward': 0,
            'peak_backward': 0,
            'memory_increase': 0,
            'status': f'错误: {str(e)[:30]}'
        }

def print_model_comparison():
    """打印不同模型的显存占用对比"""
    
    # 定义要测试的模型
    attention_classes = [
        ('FlashAttention', FlashAttention),
        ('LinearAttention', LinearAttention),
        ('Attention', Attention),
        ('MemEffAttention', MemEffAttention),
    ]
    
    # 定义测试配置
    test_configs = [
        ('配置1', 1, 5604, 1024, 8, False),
        ('配置2', 1, 6500, 1024, 8, False),
        ('配置3', 1, 7524, 1024, 8, False),
        ('配置3', 1, 10952, 1024, 8, False),
    ]
    
    print("\n" + "="*100)
    print("不同Attention模型显存占用对比测试")
    print("="*100)
    
    # 测试前向传播
    print("\n📊 前向传播显存占用对比:")
    print("-"*100)
    
    table_data = []
    headers = ["配置名称", "Batch", "SeqLen", "Dim", "Heads", "精度"] + \
              [model_name for model_name, _ in attention_classes]
    
    for config_name, bs, sl, dim, heads, mix_prec in test_configs:
        row = [
            config_name,
            bs,
            sl,
            dim,
            heads,
            "BF16" if mix_prec else "FP32"
        ]
        
        for model_name, model_class in attention_classes:
            result = measure_attention_memory(
                model_class, bs, sl, dim, heads, 
                forward_only=True,
                mixed_precision=mix_prec
            )
            
            if result['status'] == '成功':
                memory_str = format_memory(result['peak_forward'])
                row.append(memory_str)
            else:
                row.append(result['status'])
        
        table_data.append(row)
    
    # 打印表格
    print(print_memory_table(headers, table_data))
    
    # 测试前向+反向传播
    print("\n\n📊 前向+反向传播显存占用对比:")
    print("-"*100)
    
    # 使用较小配置以避免OOM
    train_configs = [
        ('训练-小配置', 2, 128, 512, 8, False),
        ('训练-中配置', 2, 256, 512, 8, False),
    ]
    
    table_data_train = []
    headers_train = ["配置名称", "Batch", "SeqLen", "Dim", "Heads", "精度"] + \
                   [model_name for model_name, _ in attention_classes]
    
    for config_name, bs, sl, dim, heads, mix_prec in train_configs:
        row = [
            config_name,
            bs,
            sl,
            dim,
            heads,
            "BF16" if mix_prec else "FP32"
        ]
        
        for model_name, model_class in attention_classes:
            result = measure_attention_memory(
                model_class, bs, sl, dim, heads, 
                forward_only=False,
                mixed_precision=mix_prec
            )
            
            if result['status'] == '成功':
                memory_str = format_memory(result['peak_backward'])
                row.append(memory_str)
            else:
                row.append(result['status'])
        
        table_data_train.append(row)
    
    print(print_memory_table(headers, table_data))

def print_config_scaling():
    """打印不同配置下的显存缩放情况"""
    
    print("\n" + "="*100)
    print("配置缩放显存占用分析")
    print("="*100)
    
    # 测试单个模型在不同配置下的表现
    model_name = "FlashAttention"
    model_class = FlashAttention
    
    # Batch Size 缩放
    print("\n📈 Batch Size 缩放 (SeqLen=256, Dim=512, Heads=8):")
    print("-"*80)
    
    batch_sizes = [1, 2, 4, 8, 16]
    table_data = []
    headers = ["Batch Size", "前向峰值内存", "内存增长倍数", "状态"]
    
    prev_memory = None
    for bs in batch_sizes:
        result = measure_attention_memory(model_class, bs, 256, 512, 8, True, False)
        
        if result['status'] == '成功':
            memory = result['peak_forward']
            growth = memory / prev_memory if prev_memory and prev_memory > 0 else 1.0
            table_data.append([
                bs,
                format_memory(memory),
                f"{growth:.2f}x",
                "成功"
            ])
            prev_memory = memory
        else:
            table_data.append([bs, "-", "-", result['status']])
            break
    
    print(print_memory_table(headers, table_data))
    
    # Sequence Length 缩放
    print("\n\n📈 Sequence Length 缩放 (Batch=2, Dim=512, Heads=8):")
    print("-"*80)
    
    seq_lengths = [64, 128, 256, 512, 1024]
    table_data = []
    headers = ["Seq Length", "前向峰值内存", "内存增长倍数", "状态"]
    
    prev_memory = None
    for sl in seq_lengths:
        result = measure_attention_memory(model_class, 2, sl, 512, 8, True, False)
        
        if result['status'] == '成功':
            memory = result['peak_forward']
            growth = memory / prev_memory if prev_memory and prev_memory > 0 else 1.0
            table_data.append([
                sl,
                format_memory(memory),
                f"{growth:.2f}x",
                "成功"
            ])
            prev_memory = memory
        else:
            table_data.append([sl, "-", "-", result['status']])
            break
    
    print(print_memory_table(headers, table_data))
    
    # Model Dimension 缩放
    print("\n\n📈 Model Dimension 缩放 (Batch=2, SeqLen=256, Heads=8):")
    print("-"*80)
    
    dimensions = [256, 512, 768, 1024, 1536]
    table_data = []
    headers = ["Model Dim", "前向峰值内存", "内存增长倍数", "状态"]
    
    prev_memory = None
    for dim in dimensions:
        heads = 8 if dim >= 512 else 4  # 调整head数
        result = measure_attention_memory(model_class, 2, 256, dim, heads, True, False)
        
        if result['status'] == '成功':
            memory = result['peak_forward']
            growth = memory / prev_memory if prev_memory and prev_memory > 0 else 1.0
            table_data.append([
                dim,
                format_memory(memory),
                f"{growth:.2f}x",
                "成功"
            ])
            prev_memory = memory
        else:
            table_data.append([dim, "-", "-", result['status']])
            break
    
    print(print_memory_table(headers, table_data))

def print_mixed_precision_benefit():
    """打印混合精度的收益"""
    
    print("\n" + "="*100)
    print("混合精度内存节省分析")
    print("="*100)
    
    configs = [
        ('配置1', 1, 5604, 1024, 8),
        ('配置2', 1, 6500, 1024, 8),
        ('配置3', 1, 7524, 1024, 8),
    ]
    
    table_data = []
    headers = ["配置", "Batch", "SeqLen", "Dim", "Heads", "FP32内存", "BF16内存", "节省比例", "状态"]
    
    for config_name, bs, sl, dim, heads in configs:
        # FP32测试
        result_fp32 = measure_attention_memory(FlashAttention, bs, sl, dim, heads, True, False)
        # BF16测试
        result_bf16 = measure_attention_memory(FlashAttention, bs, sl, dim, heads, True, True)
        
        if result_fp32['status'] == '成功' and result_bf16['status'] == '成功':
            mem_fp32 = result_fp32['peak_forward']
            mem_bf16 = result_bf16['peak_forward']
            savings = (mem_fp32 - mem_bf16) / mem_fp32 * 100
            
            table_data.append([
                config_name,
                bs,
                sl,
                dim,
                heads,
                format_memory(mem_fp32),
                format_memory(mem_bf16),
                f"{savings:.1f}%",
                "成功"
            ])
        else:
            status = "FP32失败" if result_fp32['status'] != '成功' else "BF16失败"
            table_data.append([
                config_name,
                bs,
                sl,
                dim,
                heads,
                "-",
                "-",
                "-",
                status
            ])
    
    print(print_memory_table(headers, table_data))

def print_summary_statistics():
    """打印总结统计信息"""
    
    print("\n" + "="*100)
    print("显存占用总结统计")
    print("="*100)
    
    # 典型配置测试
    typical_config = (2, 256, 512, 8)  # (batch, seq_len, dim, heads)
    models = [
        ('FlashAttention', FlashAttention),
        # ('LinearAttention', LinearAttention),
        # ('Attention', Attention),
        # ('MemEffAttention', MemEffAttention),
    ]
    
    print("\n🔍 典型配置 (Batch=2, SeqLen=256, Dim=512, Heads=8) 内存占用:")
    print("-"*80)
    
    summary_data = []
    for model_name, model_class in models:
        result = measure_attention_memory(model_class, *typical_config, True, False)
        
        if result['status'] == '成功':
            summary_data.append([
                model_name,
                format_memory(result['peak_forward']),
                format_memory(result['memory_increase']),
                "✅"
            ])
        else:
            summary_data.append([
                model_name,
                "-",
                "-",
                "❌ " + result['status']
            ])
    
    headers = ["模型", "峰值内存", "内存增量", "状态"]
    print(print_memory_table(headers, summary_data))
    
    # 内存效率排名
    print("\n🏆 内存效率排名 (从低到高):")
    print("-"*80)
    
    efficiency_data = []
    for model_name, model_class in models:
        result = measure_attention_memory(model_class, *typical_config, True, False)
        if result['status'] == '成功':
            efficiency_data.append((model_name, result['peak_forward']))
    
    # 按内存占用排序
    efficiency_data.sort(key=lambda x: x[1])
    
    rank_data = []
    for i, (model_name, memory) in enumerate(efficiency_data):
        rank_data.append([i+1, model_name, format_memory(memory)])
    
    print(print_memory_table(["排名", "模型", "内存占用"], rank_data))

def main():
    """主函数"""
    if not torch.cuda.is_available():
        print("错误: 需要GPU环境运行显存测试")
        return
    
    print(f"🏃 开始显存占用测试...")
    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾  总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"🔧  CUDA版本: {torch.version.cuda}")
    print(f"⚡  PyTorch版本: {torch.__version__}")
    
    try:
        # 1. 不同模型对比
        print_model_comparison()
        
        # 2. 配置缩放分析
        print_config_scaling()
        
        # 3. 混合精度收益
        print_mixed_precision_benefit()
        
        # 4. 总结统计
        print_summary_statistics()
        
        print("\n" + "✅"*50)
        print("所有测试完成!")
        print("✅"*50)
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 运行主测试
    main()
        