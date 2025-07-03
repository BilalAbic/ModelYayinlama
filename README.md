# FitTürkAI RAG Sistemi - CPU Optimize Linux Versiyonu

🏋️ **FitTürkAI**, Türkçe fitness ve sağlık danışmanlığı için geliştirilmiş Retrieval-Augmented Generation (RAG) sistemidir. Bu versiyon **CPU kullanımı için optimize edilmiştir** ve **Google Colab** gibi Linux ortamlarında sorunsuz çalışır.

## 🚀 Özellikler

- **CPU Optimize Edilmiş**: GPU gerektirmez, CPU'da verimli çalışır
- **Türkçe Dil Desteği**: Türkçe dokümanları işler ve yanıtlar
- **RAG Sistemi**: PDF ve JSON belgelerinden bilgi çıkarır ve kullanır
- **Interactive Chat**: Gerçek zamanlı soru-cevap sistemi
- **LoRA Adapter Desteği**: Fine-tune edilmiş modelleri destekler
- **Linux Uyumlu**: Google Colab, Ubuntu, ve diğer Linux dağıtımlarında çalışır

## 📋 Sistem Gereksinimleri

### Minimum Gereksinimler
- **RAM**: En az 4 GB (8 GB önerilir)
- **Disk**: 5 GB boş alan
- **İşletim Sistemi**: Linux (Ubuntu, Google Colab vb.)
- **Python**: 3.8 veya üzeri

### Google Colab İçin
- Ücretsiz Google Colab hesabı yeterlidir
- GPU gerekmez (CPU modunda çalışır)

## 🛠️ Kurulum

### 1. Google Colab'da Hızlı Kurulum

```python
# Google Colab'da yeni bir notebook oluşturun ve şunu çalıştırın:

# 1. Repository'yi klonlayın
!git clone https://github.com/YOUR_USERNAME/FitTurkAI-RAG.git
%cd FitTurkAI-RAG

# 2. Otomatik kurulum scriptini çalıştırın
!python colab_setup_and_run.py
```

### 2. Manuel Kurulum

```bash
# 1. Repository'yi klonlayın
git clone https://github.com/YOUR_USERNAME/FitTurkAI-RAG.git
cd FitTurkAI-RAG

# 2. Python sanal ortamı oluşturun (opsiyonel)
python -m venv venv
source venv/bin/activate  # Linux/Mac

# 3. Gerekli paketleri yükleyin
pip install -r requirements.txt

# 4. NLTK verilerini indirin
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## 📁 Klasör Yapısı

```
FitTurkAI-RAG/
├── test.py                    # Ana RAG sistemi (CPU optimize)
├── colab_setup_and_run.py     # Google Colab kurulum scripti
├── requirements.txt           # CPU için paket listesi
├── README.md                  # Bu dosya
├── indirilen_pdfler/          # PDF dosyalarınız (oluşturulacak)
├── DATA/                      # JSON veri dosyalarınız (oluşturulacak)
├── fitness_rag_store_merged/  # Vector store (oluşturulacak)
└── fine_tuned_FitTurkAI_QLoRA/ # LoRA adapter (opsiyonel)
```

## 🚴‍♀️ Kullanım

### 1. Temel Kullanım

```python
# Ana scripti çalıştırın
python test.py
```

### 2. Google Colab'da Programmatik Kullanım

```python
from test import FitnessRAG, RAGConfig

# Konfigürasyon oluşturun
config = RAGConfig(
    peft_model_path=None  # Base model kullanmak için None yapın
)

# RAG sistemini başlatın
rag_system = FitnessRAG(config)

# Bilgi tabanını oluşturun (ilk çalıştırmada)
rag_system.build_knowledge_base(
    pdf_dir="./indirilen_pdfler",
    json_dir="./DATA"
)

# Soru sorun
answer = rag_system.ask("Sağlıklı kahvaltı için ne önerirsiniz?")
print(answer)

# Interactive mode başlatın
rag_system.interactive_chat()
```

### 3. Veri Ekleme

#### PDF Dosyaları
```bash
# PDF'lerinizi bu klasöre koyun
cp your_fitness_pdfs/*.pdf ./indirilen_pdfler/
```

#### JSON Verileri
```python
# Örnek JSON formatı
{
  "soru": "Egzersiz öncesi ne yemeli?",
  "cevap": "Egzersiz öncesi hafif, karbonhidrat ağırlıklı besinler tercih edin..."
}
```

## ⚙️ Konfigürasyon

`RAGConfig` sınıfında özelleştirilebilir parametreler:

```python
config = RAGConfig(
    # Veri parametreleri
    chunk_size=300,                    # Kelime başına chunk boyutu
    chunk_overlap_sentences=2,         # Overlap cümle sayısı
    retrieval_k=5,                     # Kaç belge getirilecek
    
    # Model parametreleri
    generator_model_name="ytu-ce-cosmos/Turkish-Llama-8b-v0.1",
    peft_model_path=None,              # LoRA adapter yolu
    
    # Performans parametreleri
    max_context_length=3000            # Maksimum context uzunluğu
)
```

## 🔧 Sorun Giderme

### Yaygın Sorunlar

#### 1. Bellek Hatası (OOM)
```python
# Chunk boyutunu küçültün
config = RAGConfig(chunk_size=200, max_context_length=2000)
```

#### 2. Model Yükleme Hatası
```python
# Base model kullanın (LoRA olmadan)
config = RAGConfig(peft_model_path=None)
```

#### 3. NLTK Veri Hatası
```python
import nltk
nltk.download('punkt_tab')  # Yeni NLTK sürümü için
nltk.download('punkt')      # Eski NLTK sürümü için
nltk.download('stopwords')
```

### Google Colab Özel Çözümler

#### GPU'yu Kapatın
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # GPU'yu devre dışı bırak
```

#### Bellek Optimizasyonu
```python
# Büyük modeller için
import torch
torch.set_num_threads(2)  # CPU thread sayısını sınırla
```

## 📊 Performans Optimizasyonları

### CPU İçin İpuçları

1. **Model Boyutu**: Daha küçük embedding modelleri kullanın
2. **Chunk Size**: Daha küçük chunk'lar daha hızlıdır
3. **Thread Sayısı**: CPU çekirdek sayınıza göre ayarlayın
4. **Batch Size**: Tek seferde işlenecek belge sayısını sınırlayın

```python
# Hızlı konfigürasyon
fast_config = RAGConfig(
    embedding_model_name="paraphrase-multilingual-MiniLM-L6-v2",  # Daha küçük
    chunk_size=200,
    retrieval_k=3,
    max_context_length=2000
)
```

## 🤝 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Commit yapın (`git commit -m 'Add some AmazingFeature'`)
4. Push edin (`git push origin feature/AmazingFeature`)
5. Pull Request açın

## 📝 Değişiklik Notları

### v2.0 - CPU Optimized
- ✅ GPU bağımlılığı kaldırıldı
- ✅ CPU için optimize edildi
- ✅ Google Colab desteği eklendi
- ✅ Otomatik kurulum scripti eklendi
- ✅ Bellek optimizasyonları

### v1.0 - GPU Version
- ⚠️ Bu versiyon GPU gerektiriyordu (deprecated)

## 📞 Destek

- **Issues**: GitHub Issues bölümünü kullanın
- **Discussions**: Genel sorular için GitHub Discussions
- **Email**: [your-email@domain.com]

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır - detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 🙏 Teşekkürler

- Hugging Face Transformers ekibine
- Turkish LLaMA projesine
- Sentence Transformers ekibine
- FAISS ekibine

---

⭐ **Bu projeyi beğendiyseniz, lütfen yıldızlamayı unutmayın!** 