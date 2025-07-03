# 🚀 FitTürkAI VM Kurulum Rehberi - SSH Bağlantısı Sonrası

Tebrikler! VM'nize başarıyla bağlandınız. Şimdi FitTürkAI sistemini kuralım.

## 🔍 **Adım 1: Sistem Durumunu Kontrol Et**

```bash
# Sistem bilgilerini kontrol et
sysinfo

# RAM durumunu kontrol et
free -h

# Disk alanını kontrol et
df -h

# Startup script'in çalışıp çalışmadığını kontrol et
cat /var/log/fitturkrai-setup.log | tail -20
```

## 📦 **Adım 2: Repository'yi Klonla ve Dosyaları Kopyala**

### Seçenek A: GitHub'dan klonla (eğer repository'niz varsa)
```bash
cd /home/ubuntu
git clone https://github.com/YOUR_USERNAME/FitTurkAI-RAG.git
cd FitTurkAI-RAG
```

### Seçenek B: Dosyaları manuel olarak oluştur
```bash
cd /home/ubuntu/FitTurkAI-RAG

# Ana dosyaları oluştur
nano test.py
```

**test.py içeriğini kopyalayın** (mevcut CPU optimize edilmiş kodu)

```bash
# Requirements dosyasını oluştur
nano requirements.txt
```

**requirements.txt içeriğini kopyalayın**

```bash
# Colab setup scriptini oluştur (opsiyonel)
nano colab_setup_and_run.py
```

## 🐍 **Adım 3: Python Environment ve Paketleri Kontrol Et**

```bash
# Python versiyonunu kontrol et
python3 --version

# Pip'i güncelle
python3 -m pip install --upgrade pip

# PyTorch'un kurulduğunu kontrol et
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CPU Available: {torch.cuda.is_available() == False}')"

# Gerekli paketleri kur (eksik varsa)
pip3 install -r requirements.txt
```

## 📚 **Adım 4: NLTK Verilerini Kontrol Et**

```bash
# NLTK verilerinin kurulduğunu kontrol et
python3 -c "
import nltk
try:
    from nltk.tokenize import sent_tokenize
    test = sent_tokenize('Bu bir test. Bu ikinci cümle.', language='turkish')
    print(f'✅ NLTK Turkish tokenization working: {len(test)} sentences')
except:
    print('❌ NLTK needs setup')
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
"
```

## 📁 **Adım 5: Veri Klasörlerini Hazırla**

```bash
# Klasör yapısını kontrol et
ls -la

# Gerekli klasörlerin varlığını kontrol et
mkdir -p indirilen_pdfler DATA fitness_rag_store_merged fine_tuned_FitTurkAI_QLoRA

# Demo verinin oluşturulduğunu kontrol et
cat DATA/demo_fitness_data.json
```

## 🧪 **Adım 6: Performance Test Çalıştır**

```bash
# Sistem performansını test et
python3 /home/ubuntu/performance_test.py
```

Bu test şunları kontrol eder:
- ✅ RAM yeterliliği (>2GB)
- ✅ CPU durumu
- ✅ Sentence Transformer yüklenmesi

## 🎯 **Adım 7: FitTürkAI Sistemini İlk Kez Çalıştır**

```bash
# Ana dizine git
cd /home/ubuntu/FitTurkAI-RAG

# İlk çalıştırma (modelleri indirecek)
python3 test.py
```

**İlk çalıştırmada**:
- Turkish LLaMA model indirilecek (~15-20 dakika)
- Sentence transformer model indirilecek (~2-5 dakika)
- Demo bilgi tabanı oluşturulacak

## 🔧 **Adım 8: Sorun Giderme (Gerekirse)**

### Model indirme sorunu varsa:
```bash
# Hugging Face cache temizle
rm -rf ~/.cache/huggingface/

# Manuel model indirme
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = 'ytu-ce-cosmos/Turkish-Llama-8b-v0.1'
print('Downloading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(model_name)
print('Downloading model...')
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='float32')
print('Models downloaded successfully!')
"
```

### RAM sorunu varsa:
```bash
# Swap'i kontrol et
free -h

# RAM kullanımını izle
watch -n 2 free -h
```

### Konfigürasyonu değiştir (test.py'de):
```python
# Düşük RAM için
config = RAGConfig(
    chunk_size=200,
    retrieval_k=3,
    max_context_length=2000
)

# Yüksek RAM için (64GB+)
config = RAGConfig(
    chunk_size=500,
    retrieval_k=10,
    max_context_length=8000
)
```

## 📊 **Adım 9: Sistem İzleme ve Optimizasyon**

```bash
# RAM kullanımını izle
rammon

# CPU kullanımını izle
htop

# GPU kullanımının olmadığını kontrol et (CPU mode)
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

## 🎮 **Adım 10: İnteraktif Kullanım**

Sistem çalıştığında:

```bash
# İnteraktif chat başlat
python3 test.py
```

**Örnek sorular**:
```
🤔 Sorunuz: Sağlıklı kahvaltı için ne önerirsiniz?
🤔 Sorunuz: Günde kaç bardak su içmeliyim?
🤔 Sorunuz: Egzersiz sonrası ne yemeli?
```

## 📥 **Adım 11: Kendi Verilerinizi Ekleyin**

### PDF dosyaları için:
```bash
# PDF'lerinizi yükleyin (scp veya başka yöntemle)
# Örnek:
# scp your_fitness_pdfs/*.pdf username@VM_IP:/home/ubuntu/FitTurkAI-RAG/indirilen_pdfler/

# Veya wget ile indirin
cd indirilen_pdfler
wget https://example.com/your_fitness_pdf.pdf
```

### JSON verileri için:
```bash
# JSON verilerinizi DATA klasörüne koyun
cd DATA
nano your_fitness_data.json

# Örnek format:
# [
#   {
#     "soru": "Protein kaynakları nelerdir?",
#     "cevap": "Protein kaynakları arasında et, balık, yumurta, baklagiller, kuruyemişler bulunur..."
#   }
# ]
```

### Bilgi tabanını yeniden oluştur:
```bash
# Yeni verilerle bilgi tabanını güncelle
python3 -c "
from test import FitnessRAG, RAGConfig
config = RAGConfig()
rag = FitnessRAG(config)
rag.build_knowledge_base(pdf_dir='./indirilen_pdfler', json_dir='./DATA')
print('Knowledge base updated!')
"
```

## 🔥 **Adım 12: Gelişmiş Kullanım**

### Script olarak çalıştır:
```bash
# Otomatik sorular
python3 -c "
from test import FitnessRAG, RAGConfig
config = RAGConfig()
rag = FitnessRAG(config)

questions = [
    'Sağlıklı kahvaltı önerileri',
    'Egzersiz programı nasıl olmalı',
    'Su tüketimi ne kadar olmalı'
]

for q in questions:
    print(f'Soru: {q}')
    answer = rag.ask(q)
    print(f'Cevap: {answer}\n')
"
```

### API olarak kullan:
```bash
# Basit Flask API
nano fitness_api.py
```

```python
from flask import Flask, request, jsonify
from test import FitnessRAG, RAGConfig

app = Flask(__name__)
config = RAGConfig()
rag = FitnessRAG(config)

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question', '')
    answer = rag.ask(question)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

```bash
# API'yi çalıştır
pip3 install flask
python3 fitness_api.py
```

## 🎯 **Başarı Kontrol Listesi**

- [ ] ✅ SSH bağlantısı kuruldu
- [ ] ✅ Sistem durumu kontrol edildi
- [ ] ✅ Python paketleri kuruldu
- [ ] ✅ NLTK verileri hazır
- [ ] ✅ Performance test geçti
- [ ] ✅ İlk model indirmesi tamamlandı
- [ ] ✅ Demo sorular çalışıyor
- [ ] ✅ Kendi veriler eklendi (opsiyonel)
- [ ] ✅ Sistem izleme kuruldu

## 🆘 **Acil Durum Komutları**

```bash
# Sistem restart (gerekirse)
sudo reboot

# Python process'leri öldür
pkill -f python3

# Disk alanı temizle
sudo apt autoremove
docker system prune -a  # Eğer Docker varsa

# Log'ları kontrol et
tail -f /var/log/syslog
tail -f /var/log/fitturkrai-setup.log
```

---

**🎉 Kurulum tamamlandığında FitTürkAI sisteminiz maksimum RAM ile CPU'da çalışıyor olacak!** 