import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# Hiperparametreler (Küçük veri seti ve 6GB VRAM optimizasyonu)
N_EMBD = 256      # Gömme (Embedding) boyutu
N_HEAD = 8        # Attention head sayısı (256 / 8 = 32 boyutlu vektörler)
N_LAYER = 4       # Transformer bloğu sayısı
DROPOUT = 0.2
MAX_SEQ_LEN = 256 
VOCAB_SIZE = 10000

class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        assert N_EMBD % N_HEAD == 0
        
        # Q, K, V matrislerini tek bir Linear katmanda birleştirip hesaplıyoruz (Optimizasyon)
        self.c_attn = nn.Linear(N_EMBD, 3 * N_EMBD)
        self.c_proj = nn.Linear(N_EMBD, N_EMBD)
        self.attn_dropout = nn.Dropout(DROPOUT)
        self.resid_dropout = nn.Dropout(DROPOUT)
        
        # Gelecekteki tokenleri maskelemek için alt üçgensel matris (Lower Triangular)
        # register_buffer ile GPU'ya taşınır ama ağırlık (weight) olarak güncellenmez
        mask = torch.tril(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN)).view(1, 1, MAX_SEQ_LEN, MAX_SEQ_LEN)
        self.register_buffer("bias", mask)

    def forward(self, x):
        B, T, C = x.size() # Batch, Time (Seq_len), Channel (N_embd)
        
        # Q, K, V hesaplaması ve (B, T, C) -> (B, N_HEAD, T, HEAD_SIZE) dönüşümü
        qkv = self.c_attn(x)
        q, k, v = qkv.split(N_EMBD, dim=2)
        
        k = k.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2)
        q = q.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2)
        v = v.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2)
        
        # Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v # (B, N_HEAD, T, HEAD_SIZE)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.resid_dropout(self.c_proj(y))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        # Transformer standartı: FFN katmanında boyut 4 katına çıkarılıp tekrar küçültülür
        self.net = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.GELU(), # ReLU yerine daha pürüzsüz türev veren GELU (Gaussian Error Linear Unit)
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # Modern mimarilerde LayerNorm, Self-Attention ve FFN'den önce uygulanır (Pre-Norm)
        self.ln_1 = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ln_2 = nn.LayerNorm(N_EMBD)
        self.mlp = FeedForward()

    def forward(self, x):
        # Residual (Artık) bağlantılar: Degradasyonu önlemek için girdi (x) doğrudan eklenir
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class CustomGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, N_EMBD)
        
        self.position_embedding = nn.Embedding(MAX_SEQ_LEN, N_EMBD)
        
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE, bias=False)
        
        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * N_LAYER))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Normal dağılım ile başlatma (mean=0.0, std=0.02)
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Embedding matrisleri de aynı standart sapma ile başlatılır
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            # LayerNorm her zaman bias=0, weight=1 ile başlar
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        
        tok_emb = self.token_embedding(idx) # (B, T, N_EMBD)
        pos_emb = self.position_embedding(pos) # (T, N_EMBD)
        x = tok_emb + pos_emb
        
        x = self.blocks(x)
        x = self.ln_f(x)
        
        logits = self.lm_head(x) # (B, T, VOCAB_SIZE)
        
        loss = None
        if targets is not None:
            # PyTorch'un CrossEntropyLoss fonksiyonu B,C,T veya (B*T, C) boyutu bekler
            B, T, C = logits.size()
            logits_view = logits.view(B * T, C)
            targets_view = targets.view(B * T)
            loss = F.cross_entropy(logits_view, targets_view)
            
        return logits, loss

model = CustomGPT()
print(f"Modeldeki toplam parametre sayısı: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} Milyon")