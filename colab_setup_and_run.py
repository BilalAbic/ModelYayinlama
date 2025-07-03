#!/usr/bin/env python3
"""
FitTürkAI RAG System - Google Colab Setup and Demo Script
CPU Optimized for Linux Environment

Bu script Google Colab ortamında FitTürkAI RAG sistemini kurar ve çalıştırır.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def install_requirements():
    """Gerekli kütüphaneleri yükle"""
    print("🔧 Gerekli kütüphaneler yükleniyor...")
    
    # Önce pip'i güncelle
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    
    # Ana kütüphaneleri yükle
    requirements = [
        "torch>=2.0.0,<2.2.0",
        "transformers>=4.36.0,<4.40.0", 
        "sentence-transformers>=2.2.2,<3.0.0",
        "accelerate>=0.24.0,<0.26.0",
        "peft>=0.7.0,<0.8.0",
        "nltk>=3.8",
        "regex>=2022.10.31",
        "faiss-cpu>=1.7.4",
        "numpy>=1.24.0,<2.0.0",
        "scipy>=1.9.0",
        "PyPDF2>=3.0.0",
        "PyMuPDF>=1.23.0",
        "tqdm>=4.64.0",
        "requests>=2.28.0",
        "gdown>=4.7.0",
        "datasets>=2.14.0",
        "tokenizers>=0.15.0",
        "safetensors>=0.3.0",
        "psutil>=5.9.0"
    ]
    
    for requirement in requirements:
        try:
            print(f"📦 Yükleniyor: {requirement}")
            subprocess.run([sys.executable, "-m", "pip", "install", requirement], 
                         check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Hata: {requirement} yüklenemedi: {e}")
            continue
    
    print("✅ Kütüphane kurulumu tamamlandı!")

def download_nltk_data():
    """NLTK verilerini indir"""
    print("📚 NLTK verileri indiriliyor...")
    import nltk
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True) 
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK verileri başarıyla indirildi!")
    except Exception as e:
        print(f"⚠️ NLTK veri indirme uyarısı: {e}")

def setup_directories():
    """Gerekli klasörleri oluştur"""
    print("📁 Klasör yapısı oluşturuluyor...")
    
    directories = [
        "./indirilen_pdfler",
        "./DATA", 
        "./fitness_rag_store_merged",
        "./fine_tuned_FitTurkAI_QLoRA"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Klasör oluşturuldu: {directory}")

def create_demo_data():
    """Demo için örnek veri oluştur"""
    print("📝 Demo verileri oluşturuluyor...")
    
    # Örnek JSON veri oluştur
    demo_data = [
        {
            "soru": "Sağlıklı kahvaltı için ne önerirsiniz?", 
            "cevap": "Sağlıklı bir kahvaltı protein, kompleks karbonhidrat ve healthy yağlar içermelidir. Yumurta, tam tahıllı ekmek, avokado, meyveler iyi seçeneklerdir."
        },
        {
            "soru": "Günde kaç bardak su içmeliyim?",
            "cevap": "Genel olarak günde 8-10 bardak (2-2.5 litre) su içmek önerilir. Aktivite düzeyinize ve hava durumuna göre bu miktar artabilir."
        },
        {
            "soru": "Egzersiz sonrası ne yemeli?",
            "cevap": "Egzersiz sonrası 30-60 dakika içinde protein ve karbonhidrat içeren besinler tüketin. Örneğin protein smoothie veya yoğurt ile meyve."
        }
    ]
    
    import json
    with open("./DATA/demo_fitness_data.json", "w", encoding="utf-8") as f:
        json.dump(demo_data, f, ensure_ascii=False, indent=2)
    
    print("✅ Demo JSON verileri oluşturuldu!")

def check_system_resources():
    """Sistem kaynaklarını kontrol et"""
    print("💻 Sistem kaynakları kontrol ediliyor...")
    try:
        import psutil
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        
        print(f"🔍 CPU Çekirdek Sayısı: {cpu_count}")
        print(f"🔍 Toplam RAM: {memory.total / (1024**3):.1f} GB")
        print(f"🔍 Kullanılabilir RAM: {memory.available / (1024**3):.1f} GB") 
        
        if memory.available < 2 * (1024**3):  # 2GB'dan az
            print("⚠️ Uyarı: RAM düşük. Model yükleme sırasında sorun yaşayabilirsiniz.")
        else:
            print("✅ Sistem kaynakları yeterli görünüyor!")
            
    except ImportError:
        print("⚠️ psutil yüklü değil, sistem kontrolü atlanıyor...")

def main():
    """Ana kurulum ve demo fonksiyonu"""
    print("🚀 FitTürkAI RAG Sistemi - Google Colab Kurulumu Başlıyor...")
    print("="*60)
    
    try:
        # 1. Sistem kontrolü
        check_system_resources()
        
        # 2. Kütüphane kurulumu
        install_requirements()
        
        # 3. NLTK verileri
        download_nltk_data()
        
        # 4. Klasör yapısı
        setup_directories()
        
        # 5. Demo verileri
        create_demo_data()
        
        print("\n" + "="*60)
        print("✅ Kurulum tamamlandı!")
        print("="*60)
        print("\n📋 Sonraki adımlar:")
        print("1. test.py dosyasını çalıştırın:")
        print("   python test.py")
        print("\n2. Veya interaktif modda başlatın:")
        print("   from test import FitnessRAG, RAGConfig")
        print("   config = RAGConfig()")
        print("   rag = FitnessRAG(config)")
        print("   rag.interactive_chat()")
        
        print("\n💡 İpuçları:")
        print("- İlk çalıştırmada model indirileceği için biraz zaman alabilir")
        print("- PDF dosyalarınızı './indirilen_pdfler' klasörüne koyun")
        print("- JSON verilerinizi './DATA' klasörüne koyun")
        print("- LoRA adapter'ınız varsa './fine_tuned_FitTurkAI_QLoRA' klasörüne koyun")
        
    except Exception as e:
        print(f"\n❌ Kurulum hatası: {e}")
        print("Lütfen hataları kontrol edin ve tekrar deneyin.")
        return False
    
    return True

# Demo fonksiyonu
def run_quick_demo():
    """Hızlı demo çalıştır"""
    print("\n🎯 Hızlı Demo Başlatılıyor...")
    
    try:
        # test.py modülünü import et
        from test import FitnessRAG, RAGConfig
        
        # Konfigürasyon oluştur (PEFT olmadan)
        config = RAGConfig(peft_model_path=None)
        
        print("🤖 Model yükleniyor... (Bu işlem biraz zaman alabilir)")
        rag_system = FitnessRAG(config)
        
        # Bilgi tabanı oluştur
        print("📚 Demo bilgi tabanı oluşturuluyor...")
        rag_system.build_knowledge_base(json_dir="./DATA")
        
        # Örnek soru sor
        demo_question = "Sağlıklı kahvaltı için ne önerirsiniz?"
        print(f"\n🤔 Demo Sorusu: {demo_question}")
        
        answer = rag_system.ask(demo_question)
        print(f"\n🤖 FitTürkAI Cevabı:\n{answer}")
        
        print("\n✅ Demo tamamlandı! Artık interactive_chat() ile tam sürümü kullanabilirsiniz.")
        
    except Exception as e:
        print(f"❌ Demo hatası: {e}")
        print("Manuel olarak test.py'yi çalıştırmayı deneyin.")

if __name__ == "__main__":
    # Kurulumu çalıştır
    success = main()
    
    # Başarılıysa demo sor
    if success:
        print("\n" + "="*60)
        demo_choice = input("Hızlı demo çalıştırmak ister misiniz? (y/N): ").strip().lower()
        if demo_choice == 'y':
            run_quick_demo()
        else:
            print("Demo atlandı. Manuel olarak test.py'yi çalıştırabilirsiniz.") 