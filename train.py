import torch
from torch.utils.data import DataLoader
from model import CustomGPT
from dataset_loader import TurkishTextDataset
import time
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Eğitim {device.upper()} üzerinde yapılacak.")

BATCH_SIZE = 16
CONTEXT_WINDOW = 256
ACCUMULATION_STEPS = 4
MAX_ITERS = 50000
EVAL_INTERVAL = 500      
LEARNING_RATE = 3e-4

print("Veri seti yükleniyor...")
dataset = TurkishTextDataset("corpus.txt", "custom_tokenizer", CONTEXT_WINDOW)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
data_iter = iter(dataloader)

model = CustomGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
scaler = torch.amp.GradScaler('cuda')

@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = torch.zeros(10)
    temp_iter = iter(dataloader)
    
    for i in range(10):
        try:
            X, Y = next(temp_iter)
        except StopIteration:
            temp_iter = iter(dataloader)
            X, Y = next(temp_iter)
            
        X, Y = X.to(device), Y.to(device)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            _, loss = model(X, Y)
        losses[i] = loss.item()
        
    model.train()
    return losses.mean().item()

tracked_iters = []
tracked_losses = []

print(f"Modeldeki toplam parametre sayısı: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} Milyon")
print("Eğitim başlıyor...")
start_time = time.time()

for iter_num in range(MAX_ITERS):
    try:
        X, Y = next(data_iter)
    except StopIteration:
        data_iter = iter(dataloader)
        X, Y = next(data_iter)
        
    X, Y = X.to(device), Y.to(device)
    
    with torch.amp.autocast('cuda', dtype=torch.float16):
        logits, loss = model(X, Y)
        loss = loss / ACCUMULATION_STEPS 
        
    scaler.scale(loss).backward()
    
    if (iter_num + 1) % ACCUMULATION_STEPS == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True) 
        
    if iter_num % EVAL_INTERVAL == 0 or iter_num == MAX_ITERS - 1:
        val_loss = estimate_loss()
        elapsed = time.time() - start_time
        
        tracked_iters.append(iter_num)
        tracked_losses.append(val_loss)
        
        yuzde = (iter_num / MAX_ITERS) * 100
        print(f"Adım {iter_num:5d}/{MAX_ITERS} | İlerleme: %{yuzde:.1f} | Loss: {val_loss:.4f} | Süre: {elapsed:.2f} sn")
        
torch.save(model.state_dict(), "custom_gpt_turkish.pth")
print("\nEğitim tamamlandı! Ağırlıklar kaydedildi.")

# --- Loss Grafiğini Çizdirme ve Kaydetme ---
print("Loss grafiği oluşturuluyor...")
plt.figure(figsize=(10, 6))
plt.plot(tracked_iters, tracked_losses, label='Validation Loss', color='blue', linewidth=2)
plt.title('Eğitim Sürecinde Loss Düşüşü (50.000 İterasyon)')
plt.xlabel('İterasyon (Adım)')
plt.ylabel('Kayıp (Loss)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.savefig("loss_graph.png", dpi=300, bbox_inches='tight')
print("Grafik 'loss_graph.png' olarak proje klasörüne kaydedildi.")