# Türkçe Custom GPT & Asenkron Web Uygulaması 🚀

Bu proje, Derin Öğrenme (Deep Learning) çalışmalarımın kapanış eseri olarak, PyTorch kullanılarak tamamen ilk prensiplerden (from scratch) inşa edilmiş özbağlanımlı (autoregressive) bir dil modelidir. Kapsamlı Türkçe Wikipedia derlemi üzerinde eğitilen model, FastAPI ve Server-Sent Events (SSE) teknolojileri ile tarayıcıda canlı metin akışı (streaming) sunan uçtan uca bir yapay zeka ürününe dönüştürülmüştür.

## Mimari ve Geliştirme Özeti
Hazır transformer soyutlamaları (örn. Hugging Face) kullanılmamış; Transformer blokları, Causal Masking ve matris çarpımları doğrudan PyTorch tensör operasyonlarıyla kodlanmıştır.

* **Veri Seti:** Türkçe Wikipedia derlemi (`musabg/wikipedia-tr`).
* **Özel Tokenizer:** Türkçe yapıya uygun olarak eğitilmiş Byte-Level BPE (Byte-Pair Encoding).
* **Model Ölçeği:** Yaklaşık 30.02 Milyon parametreli, GPT-2 tabanlı özel mimari.
* **Ağırlık Başlatma (Weight Initialization):** Derin ağlarda varyans birikimini önlemek için GPT mimarisine özgü ölçeklendirilmiş başlangıç değerleri.

## Donanım ve Eğitim Optimizasyonları
Eğitim süreci, 6GB VRAM (RTX 3060) gibi kısıtlı donanım kaynaklarında darboğaz yaratmayacak şekilde dizayn edilmiştir:
* **Automatic Mixed Precision (AMP):** Tensör operasyonları FP16'ya indirilerek Tensor Core'lar aktif edilmiş, bellek tüketimi minimize edilmiştir.
* **Gradient Accumulation:** Büyük veri setini düşük bellek ile işleyebilmek için gradyanlar biriktirilerek sanal batch optimizasyonu yapılmıştır.

## Web Arayüzü ve Inference (FastAPI)
* **Canlı Akış (Streaming):** Üretilen token'lar Python'ın `yield` jeneratörleri ve JavaScript `ReadableStream` ile daktilo efektinde ekrana aktarılır.
* **Algoritmik Kontrol:** Küçük modellerin kronik sorunu olan sonsuz döngü (repetition loop) hatalarını kırmak için logit'ler üzerinden anlık ceza uygulayan **Repetition Penalty** algoritması yazılmıştır.
* **Dinamik Parametreler:** UI üzerinden `Temperature`, `Max Tokens` ve `Top-K` filtreleri canlı olarak ayarlanabilmekte ve hazır prompt menüsü ile hızlı üretim yapılabilmektedir.

## Kurulum ve Çalıştırma

```bash
# 1. Repoyu klonlayın
git clone [https://github.com/yefeakkaya/turkish-wiki-custom-gpt.git](https://github.com/yefeakkaya/turkish-wiki-custom-gpt.git)
cd turkish-wiki-custom-gpt

# 2. Gerekli bağımlılıkları yükleyin
pip install torch fastapi uvicorn tokenizers pydantic jinja2

# 3. Sunucuyu başlatın
uvicorn app:app --reload
