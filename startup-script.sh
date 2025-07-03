#!/bin/bash

# FitTürkAI RAG System - VM Startup Script
# CPU Optimized with High RAM Configuration

set -e  # Exit on any error

LOG_FILE="/var/log/fitturkrai-setup.log"
exec > >(tee -a $LOG_FILE)
exec 2>&1

echo "=================================================="
echo "🚀 FitTürkAI RAG System Setup Starting..."
echo "Timestamp: $(date)"
echo "=================================================="

# Update system
echo "📦 Updating system packages..."
apt-get update && apt-get upgrade -y

# Install essential packages
echo "🔧 Installing essential packages..."
apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    curl \
    wget \
    htop \
    tree \
    unzip \
    vim \
    build-essential \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release

# Install monitoring tools
echo "📊 Installing monitoring tools..."
apt-get install -y \
    iotop \
    nethogs \
    nload \
    glances \
    ncdu

# Upgrade pip
echo "🐍 Setting up Python environment..."
python3 -m pip install --upgrade pip

# Install CPU-optimized PyTorch
echo "🔥 Installing PyTorch (CPU version)..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install core ML packages for CPU
echo "🤖 Installing ML packages..."
pip3 install \
    transformers>=4.36.0,\<4.40.0 \
    sentence-transformers>=2.2.2,\<3.0.0 \
    accelerate>=0.24.0,\<0.26.0 \
    peft>=0.7.0,\<0.8.0 \
    nltk>=3.8 \
    faiss-cpu>=1.7.4 \
    numpy>=1.24.0,\<2.0.0 \
    scipy>=1.9.0 \
    PyPDF2>=3.0.0 \
    PyMuPDF>=1.23.0 \
    tqdm>=4.64.0 \
    requests>=2.28.0 \
    datasets>=2.14.0

# System optimizations for high RAM
echo "⚡ Configuring system for high RAM usage..."

# Swap settings (minimize swap usage)
echo 'vm.swappiness=1' >> /etc/sysctl.conf
echo 'vm.vfs_cache_pressure=50' >> /etc/sysctl.conf
echo 'vm.dirty_ratio=15' >> /etc/sysctl.conf
echo 'vm.dirty_background_ratio=5' >> /etc/sysctl.conf

# Memory overcommit (allow more memory allocation)
echo 'vm.overcommit_memory=1' >> /etc/sysctl.conf
echo 'vm.overcommit_ratio=80' >> /etc/sysctl.conf

# Network optimizations
echo 'net.core.rmem_max=16777216' >> /etc/sysctl.conf
echo 'net.core.wmem_max=16777216' >> /etc/sysctl.conf

# Apply system settings
sysctl -p

# Configure user limits for high memory usage
echo "🔒 Configuring user limits..."
cat >> /etc/security/limits.conf << EOF
ubuntu soft memlock unlimited
ubuntu hard memlock unlimited
ubuntu soft stack unlimited
ubuntu hard stack unlimited
EOF

# Create FitTürkAI directory structure
echo "📁 Creating directory structure..."
sudo -u ubuntu mkdir -p /home/ubuntu/FitTurkAI-RAG
sudo -u ubuntu mkdir -p /home/ubuntu/FitTurkAI-RAG/indirilen_pdfler
sudo -u ubuntu mkdir -p /home/ubuntu/FitTurkAI-RAG/DATA
sudo -u ubuntu mkdir -p /home/ubuntu/FitTurkAI-RAG/fitness_rag_store_merged
sudo -u ubuntu mkdir -p /home/ubuntu/FitTurkAI-RAG/fine_tuned_FitTurkAI_QLoRA

# Set ownership
chown -R ubuntu:ubuntu /home/ubuntu/FitTurkAI-RAG

# Create demo data
echo "📝 Creating demo data..."
sudo -u ubuntu cat > /home/ubuntu/FitTurkAI-RAG/DATA/demo_fitness_data.json << 'EOF'
[
  {
    "soru": "Sağlıklı kahvaltı için ne önerirsiniz?",
    "cevap": "Sağlıklı bir kahvaltı protein, kompleks karbonhidrat ve sağlıklı yağlar içermelidir. Yumurta, tam tahıllı ekmek, avokado, meyveler ve kuruyemişler iyi seçeneklerdir. Ayrıca bol su içmeyi unutmayın."
  },
  {
    "soru": "Günde kaç bardak su içmeliyim?",
    "cevap": "Genel olarak günde 8-10 bardak (2-2.5 litre) su içmek önerilir. Aktivite düzeyinize, hava durumuna ve vücut ağırlığınıza göre bu miktar artabilir. Egzersiz yaptığınız günlerde daha fazla su tüketmelisiniz."
  },
  {
    "soru": "Egzersiz sonrası ne yemeli?",
    "cevap": "Egzersiz sonrası 30-60 dakika içinde protein ve karbonhidrat içeren besinler tüketin. Protein kas onarımı için, karbonhidrat ise enerji depolarını yenilemek için gereklidir. Örneğin protein smoothie, yoğurt ile meyve veya tavuk göğsü ile pirinç iyi seçeneklerdir."
  },
  {
    "soru": "Kilo vermek için hangi egzersizleri yapmalıyım?",
    "cevap": "Kilo vermek için kardiyovasküler egzersizler (yürüyüş, koşu, bisiklet) ve direnç antrenmanlarını (ağırlık kaldırma) birleştirin. Haftada en az 150 dakika orta yoğunlukta kardiyovasküler aktivite yapın. Ayrıca kas kütlesini korumak için haftada 2-3 gün kuvvet antrenmanı ekleyin."
  }
]
EOF

# Create system info script
echo "ℹ️ Creating system info script..."
sudo -u ubuntu cat > /home/ubuntu/system_info.sh << 'EOF'
#!/bin/bash
echo "=== FitTürkAI VM System Information ==="
echo "Date: $(date)"
echo "Uptime: $(uptime)"
echo ""
echo "=== CPU Information ==="
lscpu | grep -E 'Model name|CPU\(s\)|Thread'
echo ""
echo "=== Memory Information ==="
free -h
echo ""
echo "=== Disk Usage ==="
df -h
echo ""
echo "=== GPU Information ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "No GPU detected (CPU-only configuration)"
fi
echo ""
echo "=== Python Packages ==="
pip3 list | grep -E 'torch|transformers|sentence|faiss|peft'
echo ""
echo "=== System Load ==="
top -n 1 -b | head -20
EOF

chmod +x /home/ubuntu/system_info.sh
chown ubuntu:ubuntu /home/ubuntu/system_info.sh

# Create RAM monitoring script
echo "📊 Creating RAM monitoring script..."
sudo -u ubuntu cat > /home/ubuntu/monitor_ram.sh << 'EOF'
#!/bin/bash
echo "=== Real-time RAM Monitoring ==="
while true; do
    clear
    echo "FitTürkAI RAM Monitor - $(date)"
    echo "================================"
    free -h
    echo ""
    echo "Top 10 Memory Consuming Processes:"
    ps aux --sort=-%mem | head -11
    echo ""
    echo "Press Ctrl+C to exit"
    sleep 5
done
EOF

chmod +x /home/ubuntu/monitor_ram.sh
chown ubuntu:ubuntu /home/ubuntu/monitor_ram.sh

# Install performance monitoring alias
echo "📈 Setting up monitoring aliases..."
sudo -u ubuntu cat >> /home/ubuntu/.bashrc << 'EOF'

# FitTürkAI Monitoring Aliases
alias sysinfo='/home/ubuntu/system_info.sh'
alias rammon='/home/ubuntu/monitor_ram.sh'
alias fitturkrai='cd /home/ubuntu/FitTurkAI-RAG'
alias pyfree='python3 -c "import psutil; print(f\"Available RAM: {psutil.virtual_memory().available/1024**3:.1f} GB\")"'

# Python environment
export PYTHONPATH=/home/ubuntu/FitTurkAI-RAG:$PYTHONPATH

echo "🏋️ Welcome to FitTürkAI High-RAM VM!"
echo "Available commands:"
echo "  sysinfo  - System information"
echo "  rammon   - RAM monitoring"
echo "  fitturkrai - Go to project directory"
echo "  pyfree   - Check available Python memory"
EOF

# Download NLTK data
echo "📚 Downloading NLTK data..."
sudo -u ubuntu python3 -c "
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    print('NLTK data downloaded successfully!')
except Exception as e:
    print(f'NLTK download warning: {e}')
"

# Create performance test script
echo "🧪 Creating performance test script..."
sudo -u ubuntu cat > /home/ubuntu/performance_test.py << 'EOF'
#!/usr/bin/env python3
"""
FitTürkAI Performance Test Script
Tests system capabilities for CPU-based RAG operations
"""

import time
import psutil
import numpy as np
from sentence_transformers import SentenceTransformer

def test_memory():
    """Test available memory"""
    memory = psutil.virtual_memory()
    print(f"💾 Total RAM: {memory.total / 1024**3:.1f} GB")
    print(f"💾 Available RAM: {memory.available / 1024**3:.1f} GB")
    print(f"💾 Used RAM: {memory.percent}%")
    return memory.available > 2 * 1024**3  # At least 2GB free

def test_cpu():
    """Test CPU performance"""
    print(f"🔧 CPU Cores: {psutil.cpu_count()}")
    print(f"🔧 CPU Usage: {psutil.cpu_percent(interval=1)}%")
    return True

def test_sentence_transformer():
    """Test sentence transformer loading"""
    try:
        print("🤖 Testing Sentence Transformer...")
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Test encoding
        texts = ["Bu bir test cümlesidir.", "FitTürkAI sistemi çalışıyor."]
        start_time = time.time()
        embeddings = model.encode(texts)
        encoding_time = time.time() - start_time
        
        print(f"✅ Sentence Transformer loaded successfully!")
        print(f"⏱️ Encoding time: {encoding_time:.2f}s")
        print(f"📏 Embedding shape: {embeddings.shape}")
        return True
    except Exception as e:
        print(f"❌ Sentence Transformer test failed: {e}")
        return False

def main():
    print("🚀 FitTürkAI Performance Test")
    print("=" * 40)
    
    # Test system components
    memory_ok = test_memory()
    cpu_ok = test_cpu()
    transformer_ok = test_sentence_transformer()
    
    print("\n" + "=" * 40)
    print("📊 Test Results:")
    print(f"💾 Memory: {'✅ OK' if memory_ok else '❌ INSUFFICIENT'}")
    print(f"🔧 CPU: {'✅ OK' if cpu_ok else '❌ PROBLEM'}")
    print(f"🤖 Transformers: {'✅ OK' if transformer_ok else '❌ FAILED'}")
    
    if all([memory_ok, cpu_ok, transformer_ok]):
        print("\n🎉 System ready for FitTürkAI!")
    else:
        print("\n⚠️ System needs optimization!")

if __name__ == "__main__":
    main()
EOF

chmod +x /home/ubuntu/performance_test.py
chown ubuntu:ubuntu /home/ubuntu/performance_test.py

# Configure automatic updates
echo "🔄 Configuring automatic security updates..."
echo 'Unattended-Upgrade::Automatic-Reboot "false";' >> /etc/apt/apt.conf.d/50unattended-upgrades

# Final system status
echo ""
echo "=================================================="
echo "✅ FitTürkAI VM Setup Completed!"
echo "=================================================="
echo "📊 System Status:"
echo "  - OS: $(lsb_release -d | cut -f2)"
echo "  - Memory: $(free -h | awk 'NR==2{printf \"%.1f GB total, %.1f GB available\", $2/1024/1024, $7/1024/1024}')"
echo "  - CPU: $(nproc) cores"
echo "  - Python: $(python3 --version)"
echo ""
echo "🎯 Next Steps:"
echo "1. SSH to the VM: gcloud compute ssh [VM_NAME] --zone=[ZONE]"
echo "2. Run performance test: python3 /home/ubuntu/performance_test.py"
echo "3. Clone your FitTürkAI repository"
echo "4. Start the RAG system: python3 test.py"
echo ""
echo "📚 Useful Commands:"
echo "  - sysinfo: System information"
echo "  - rammon: RAM monitoring"
echo "  - fitturkrai: Go to project directory"
echo ""
echo "Setup completed at: $(date)"
echo "==================================================" 