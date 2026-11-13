import torch
import torch.nn.functional as F
from tokenizers.implementations import ByteLevelBPETokenizer
from model import CustomGPT
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
CONTEXT_WINDOW = 256

print("Model ve Tokenizer yükleniyor...")

vocab_path = "custom_tokenizer/vocab.json"
merges_path = "custom_tokenizer/merges.txt"
tokenizer = ByteLevelBPETokenizer(vocab=vocab_path, merges=merges_path)

model = CustomGPT()
model.load_state_dict(torch.load("custom_gpt_turkish.pth", map_location=device, weights_only=True))
model.to(device)
model.eval()

@torch.no_grad()
def generate_text(prompt, max_new_tokens, temperature=0.8, top_k=40):
    encoded = tokenizer.encode(prompt)
    
    # Modelin beklediği batch boyutuna (B, T) uygun hale getirmek için tensöre çevir
    # (1, T) boyutunda bir tensör elde ediyoruz
    idx = torch.tensor(encoded.ids, dtype=torch.long, device=device).unsqueeze(0)
    
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -CONTEXT_WINDOW:]
        
        logits, _ = model(idx_cond)
        
        # Sadece en son adımdaki (tahmin edilecek kelimenin) olasılıklarını al
        # logits boyutu: (Batch, Time, Vocab) -> (1, Vocab)
        logits = logits[:, -1, :] 
        
        # Temperature uygulaması: Değer 1'den küçükse model daha kesin, büyükse daha rastgele davranır
        logits = logits / temperature
        
        # Top-K Filtrelemesi: Yalnızca en yüksek olasılıklı K kelimeyi tut, gerisini -Sonsuz yap
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            # En olası K kelimeden en düşük olanın değerini bul
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        probs = F.softmax(logits, dim=-1)
        
        idx_next = torch.multinomial(probs, num_samples=1)
        
        idx = torch.cat((idx, idx_next), dim=1)
        
    # Üretilen tensörü düzleştir (1D) ve Python listesine çevir
    generated_ids = idx[0].tolist()
    
    # Tokenları tekrar okunabilir metne dönüştür
    return tokenizer.decode(generated_ids)

# --- Kullanım ve Test ---
if __name__ == "__main__":
    start_prompt = "Gönül gurbet ele"
    
    print(f"\nBaşlangıç Metni: '{start_prompt}'")
    print("-" * 50)
    
    # Yeni metin üretimi
    generated_output = generate_text(
        prompt=start_prompt,
        max_new_tokens=100,  # 100 yeni kelime/hece üretilecek
        temperature=0.8,     # 0.7 - 0.9 arası yaratıcılık için idealdir
        top_k=50             # Olasılık evrenini en mantıklı 50 kelime ile sınırla
    )
    
    print(generated_output)
    print("-" * 50)