# FitTürkAI RAG Sistemi

Türkçe fitness ve beslenme konularında özelleştirilmiş Retrieval-Augmented Generation (RAG) sistemi.

## 🚀 Özellikler

- **Fine-tuned Turkish LLM**: Fitness ve beslenme alanında özelleştirilmiş model
- **RAG Sistemi**: PDF ve JSON belgelerinden bilgi çekme
- **İnteraktif Chat**: Terminal üzerinden sohbet arayüzü
- **CUDA Desteği**: GPU varsa otomatik hızlandırma

## 📋 Gereksinimler

- Python 3.8+
- 8GB+ RAM (CPU için)
- İsteğe bağlı: NVIDIA GPU (hızlandırma için)

## 🛠️ Kurulum

### 1. Repository'yi Klonlayın
```bash
git clone https://github.com/KULLANICI_ADINIZ/REPO_ADI.git
cd REPO_ADI
```

### 2. Python Environment Oluşturun
```bash
python3 -m venv fitturkrai_env
source fitturkrai_env/bin/activate  # Linux/Mac
# veya
fitturkrai_env\Scripts\activate     # Windows
```

### 3. Dependencies Kurun
```bash
pip install -r requirements.txt
```

### 4. Büyük Model Dosyalarını İndirin

**Önemli**: Fine-tuned model ve vector store büyük olduğu için GitHub'a yüklenmemiştir.

#### Seçenek A: Google Drive'dan İndirme (Önerilen)
1. Model dosyalarını Google Drive'a yükledikten sonra
2. `download_models.py` dosyasındaki file ID'leri güncelleyin
3. İndirme scriptini çalıştırın:
```bash
python download_models.py
```

#### Seçenek B: Manuel İndirme
1. [Google Drive Link 1](https://drive.google.com/your-fine-tuned-model-link) - Fine-tuned Model
2. [Google Drive Link 2](https://drive.google.com/your-vector-store-link) - Vector Store
3. Zip dosyalarını proje klasörüne çıkarın

## 🎯 Kullanım

### Sistemi Başlatın
```bash
python test.py
```

### İlk Çalıştırma
1. Sistem otomatik olarak gerekli NLTK verilerini indirecek
2. Model dosyaları kontrol edilecek
3. Vector store yüklenecek veya oluşturulacak
4. İnteraktif chat başlayacak

### Örnek Sorgular
```
🤔 Sorunuz: Protein ihtiyacımı nasıl hesaplayabilirim?
🤔 Sorunuz: Kilo vermek için hangi egzersizleri yapmalıyım?
🤔 Sorunuz: Günlük kalori ihtiyacım nedir?
```

## 🔧 Konfigürasyon

`test.py` dosyasındaki `RAGConfig` sınıfından ayarları değiştirebilirsiniz:

```python
config = RAGConfig(
    chunk_size=300,           # Metin chunk boyutu
    retrieval_k=5,           # Kaç belge getirilecek
    max_context_length=3000, # Maksimum context uzunluğu
    generator_model_name="ytu-ce-cosmos/Turkish-Llama-8b-v0.1"
)
```

## 📁 Proje Yapısı

```
├── test.py                          # Ana uygulama
├── download_models.py               # Model indirme scripti
├── requirements.txt                 # Python bağımlılıkları
├── README.md                       # Bu dosya
├── .gitignore                      # Git ignore kuralları
├── fine_tuned_FitTurkAI_QLoRA/    # Fine-tuned model (indirilecek)
└── fitness_rag_store_merged/       # Vector store (indirilecek)
```

## 🚀 Google Cloud VM'de Çalıştırma

### VM Oluşturma
```bash
gcloud compute instances create fitturkrai-vm \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud
```

### VM'ye Bağlanma
```bash
gcloud compute ssh fitturkrai-vm --zone=us-central1-a
```

### Kurulum
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv git
git clone https://github.com/KULLANICI_ADINIZ/REPO_ADI.git
cd REPO_ADI
python3 -m venv fitturkrai_env
source fitturkrai_env/bin/activate
pip install -r requirements.txt
python download_models.py
python test.py
```

## 🛟 Sorun Giderme

### CUDA Hatası
```python
# Model CPU'da çalışırsa:
RuntimeError: CUDA not available
```
**Çözüm**: Sistem otomatik olarak CPU moduna geçecektir.

### Memory Hatası
```python
# Yetersiz RAM
RuntimeError: out of memory
```
**Çözüm**: Daha küçük model kullanın veya chunk_size'ı azaltın.

### Model Bulunamıyor
```
⚠️ Fine-tuned model not found
```
**Çözüm**: `python download_models.py` çalıştırın.

## 📞 Destek

Sorularınız için issue açabilirsiniz.

## 📄 Lisans

MIT License 