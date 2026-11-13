import torch
from torch.utils.data import Dataset, DataLoader
#from tokenizers import Tokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
import os

class TurkishTextDataset(Dataset):
    def __init__(self, corpus_path, tokenizer_path, max_seq_len):
        self.max_seq_len = max_seq_len
        
        vocab_path = os.path.join(tokenizer_path, "vocab.json")
        merges_path = os.path.join(tokenizer_path, "merges.txt")
        
        self.tokenizer = ByteLevelBPETokenizer(
            vocab=vocab_path,
            merges=merges_path
        )
        
        print("Metinler token'lara dönüştürülüyor...")
        with open(corpus_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        encoded = self.tokenizer.encode(text)
        
        self.tokens = torch.tensor(encoded.ids, dtype=torch.long)
        print(f"Toplam Token Sayısı: {len(self.tokens)}")

    def __len__(self):
        return len(self.tokens) - self.max_seq_len

    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.max_seq_len]
        y = self.tokens[idx + 1 : idx + self.max_seq_len + 1]
        return x, y

# --- Kullanım ve Test ---
if __name__ == "__main__":
    # Hiperparametreler (6GB VRAM için güvenli başlangıç değerleri)
    CONTEXT_WINDOW = 256  # Model bir seferde 256 token geriye bakacak
    BATCH_SIZE = 32       # Her iterasyonda 32 farklı X-Y çifti işlenecek
    
    dataset = TurkishTextDataset(
        corpus_path="corpus.txt", 
        tokenizer_path="custom_tokenizer",
        max_seq_len=CONTEXT_WINDOW
    )
    
    # num_workers=2 veya 4 yapılarak CPU->GPU veri transferi hızlandırılabilir
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # DataLoader'ın ürettiği tensörlerin boyutlarını test edelim
    x_batch, y_batch = next(iter(dataloader))
    
    print("\n--- Tensör Boyutları ---")
    print(f"X (Girdi) Batch boyutu: {x_batch.shape}  -> [Batch Size, Context Window]")
    print(f"Y (Hedef) Batch boyutu: {y_batch.shape}  -> [Batch Size, Context Window]")
    
    print("\n--- İlk Örnekteki Mantık ---")
    print(f"X'in ilk 5 tokeni: {x_batch[0][:5].tolist()}")
    print(f"Y'nin ilk 5 tokeni: {y_batch[0][:5].tolist()}")