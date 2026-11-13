import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from tokenizers.implementations import ByteLevelBPETokenizer
from model import CustomGPT

app = FastAPI(title="Türkçe Custom GPT API")
templates = Jinja2Templates(directory="templates")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
CONTEXT_WINDOW = 256

print("Model ve Tokenizer belleğe yükleniyor...")
tokenizer = ByteLevelBPETokenizer(
    vocab="custom_tokenizer/vocab.json", 
    merges="custom_tokenizer/merges.txt"
)

model = CustomGPT()
try:
    model.load_state_dict(torch.load("custom_gpt_turkish.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print("Model başarıyla yüklendi!")
except FileNotFoundError:
    print("DİKKAT: Ağırlık dosyası bulunamadı.")

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 40

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request, "index.html", {"request": request})

# Yeni Streaming Endpoint'i
@app.post("/stream")
async def stream_generate(req: GenerateRequest):
    async def text_generator():
        encoded = tokenizer.encode(req.prompt)
        idx = torch.tensor(encoded.ids, dtype=torch.long, device=device).unsqueeze(0)
        current_text = tokenizer.decode(idx[0].tolist())
        
        # Ceza çarpanı: 1.0 (ceza yok) ile 2.0 (ağır ceza) arası idealdir
        repetition_penalty = 1.3 
        
        for _ in range(req.max_new_tokens):
            idx_cond = idx[:, -CONTEXT_WINDOW:]
            with torch.no_grad():
                logits, _ = model(idx_cond)
            
            logits = logits[:, -1, :] 
            logits = logits / req.temperature
            
            # Mevcut dizide (idx) daha önce geçmiş olan kelimelerin (token) puanını düşür
            for past_token in set(idx[0].tolist()):
                if logits[0, past_token] > 0:
                    logits[0, past_token] /= repetition_penalty
                else:
                    logits[0, past_token] *= repetition_penalty
            # -----------------------------------------------
            
            if req.top_k is not None and req.top_k > 0:
                v, _ = torch.topk(logits, min(req.top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            new_text = tokenizer.decode(idx[0].tolist())
            chunk = new_text[len(current_text):]
            current_text = new_text
            
            yield chunk 
            await asyncio.sleep(0.03) 
            
    return StreamingResponse(text_generator(), media_type="text/plain")