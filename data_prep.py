from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
import os

print("Veri seti indiriliyor...")
dataset = load_dataset("PiotrSty/trwikisource-literature-turkish", split="train")

# Metinleri tek bir dosyada topla (Tokenizer eğitimi için)
corpus_path = "corpus.txt"
text_data = dataset["text"]

print("Corpus dosyası oluşturuluyor...")
with open(corpus_path, "w", encoding="utf-8") as f:
    for text in text_data:
        # Boş metinleri atla ve satır sonu ekleyerek yaz
        if text.strip():
            f.write(text + "\n")

print("Tokenizer eğitiliyor...")
tokenizer = ByteLevelBPETokenizer()

tokenizer.train(
    files=[corpus_path],
    vocab_size=10000,          # Küçük dataset için ideal sözlük boyutu
    min_frequency=2,        
    special_tokens=[
        "<|endoftext|>",       # Metin bitiş token'i
        "<|padding|>"          # Padding token'i
    ]
)

save_dir = "custom_tokenizer"
os.makedirs(save_dir, exist_ok=True)

tokenizer.save_model(save_dir)
tokenizer.save(os.path.join(save_dir, "tokenizer.json")) # <-- Bu satırın olması şart

print(f"Tokenizer başarıyla eğitildi ve '{save_dir}' klasörüne kaydedildi!")

# Test edelim
encoded = tokenizer.encode("Gönül gurbet ele varma, ya gelinir ya gelinmez.")
print(f"Örnek Tokenizasyon: {encoded.tokens}")
print(f"Token ID'leri: {encoded.ids}")