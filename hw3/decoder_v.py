import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math
import collections
import loralib as lora  # 导入 loralib

class Config:
    def __init__(self, checkpoint=None):
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.vocab_size = 50257
        self.block_size = 1024
        self.checkpoint = checkpoint

class Attention(nn.Module):
    def __init__(self, cfg, rank=32):
        super().__init__()
        self.c_attn = lora.Linear(cfg.n_embd, 3 * cfg.n_embd, r=rank, lora_alpha=128, lora_dropout=0.1)
        self.c_proj = lora.Linear(cfg.n_embd, cfg.n_embd, r=rank, lora_alpha=128, lora_dropout=0.1)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        size = cfg.block_size
        self.register_buffer('bias', torch.tril(torch.ones(size, size)).view(1, 1, size, size))

    def forward(self, x, return_attention=False):
        B, T, C = x.size()
        #print(f"Input sequence length T: {T}")  # 调试用

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        #print("att shape", att.shape)
        output = self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))
        if return_attention:
            return output, att
        return output

class Block(nn.Module):
    def __init__(self, cfg, layer, rank=32):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.mlp = nn.Sequential(collections.OrderedDict([
            ('c_fc', lora.Linear(cfg.n_embd, 4 * cfg.n_embd, r=rank, lora_alpha=128, lora_dropout=0.1)),
            ('act', nn.GELU(approximate='tanh')),
            ('c_proj', lora.Linear(4 * cfg.n_embd, cfg.n_embd, r=rank, lora_alpha=128, lora_dropout=0.1))
        ]))

    def forward(self, x, return_attention=False):
        if return_attention:
            attn_out, attn_weights = self.attn(self.ln_1(x), return_attention=True)
            x = x + attn_out
            x = x + self.mlp(self.ln_2(x))
            return x, attn_weights
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
            h=nn.Sequential(*[Block(cfg, layer) for layer in range(cfg.n_layer)]),
            ln_f=nn.LayerNorm(cfg.n_embd)
        ))
        self.lm_head = lora.Linear(cfg.n_embd, cfg.vocab_size, r=32, bias=False, lora_alpha=128, lora_dropout=0.1)
        self.transformer.wte.weight = self.lm_head.weight
        
        if self.cfg.checkpoint is not None:
            state_dict = torch.load(self.cfg.checkpoint)
            transposed = ['.c_attn.weight', '.c_fc.weight', '.c_proj.weight']
            for key, value in state_dict.items():
                if any(key.endswith(w) for w in transposed):
                    state_dict[key] = value.t()
            self.transformer.load_state_dict(state_dict, strict=False)

    def forward(self, text_id: Tensor, image_emb: Tensor, return_attention=False, layer_idx=11): #11
        image_emb = image_emb.float()
        text_id = torch.narrow(text_id, 1, 0, min(text_id.size(1), self.block_size))
        pos = torch.arange(text_id.size()[1], dtype=torch.long, device=text_id.device).unsqueeze(0)
        text_emb = self.transformer.wte(text_id) + self.transformer.wpe(pos)
        combined_emb = torch.cat((image_emb, text_emb), dim=1)
        
        if return_attention:
            #attn_weights = []
            result = combined_emb
            for idx, layer in enumerate(self.transformer.h):
                if idx == layer_idx:
                    #print("here")
                    result, att = layer(result, return_attention=True)
                else:
                    result = layer(result)
            text_output = result[:, -text_emb.size(1):, :]
            #print("text_emb.size(1)", text_emb.size(1))  70
            text_output = self.lm_head(self.transformer.ln_f(text_output))
            return text_output, att #attn_weights
        else:
            result = self.transformer.h(combined_emb)
            text_output = result[:, -text_emb.size(1):, :]
            text_output = self.lm_head(self.transformer.ln_f(text_output))
            return text_output
