#!/bin/bash

# FitTürkAI RAG System - High RAM Google Cloud VM Creator
# CPU Optimized for Maximum Performance

echo "🚀 FitTürkAI için Yüksek RAM'li Google Cloud VM Oluşturucu"
echo "============================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration variables
PROJECT_ID=""
VM_NAME="fitturkrai-high-ram"
ZONE="us-central1-a"
REGION="us-central1"

# RAM Seçenekleri (CPU optimize)
echo -e "${BLUE}📊 Mevcut RAM Seçenekleri:${NC}"
echo "1. n2-highmem-2:  2 vCPU,  16 GB RAM  (~$100/ay)  [Temel]"
echo "2. n2-highmem-4:  4 vCPU,  32 GB RAM  (~$200/ay)  [Önerilen]"
echo "3. n2-highmem-8:  8 vCPU,  64 GB RAM  (~$400/ay)  [Yüksek Performans]"
echo "4. n2-highmem-16: 16 vCPU, 128 GB RAM (~$800/ay)  [Maksimum]"
echo "5. n2-highmem-32: 32 vCPU, 256 GB RAM (~$1600/ay) [Ultra]"
echo "6. m2-ultramem-416: 416 vCPU, 5.9 TB RAM (~$30000/ay) [Extreme - Sadece özel durumlar]"

echo ""
read -p "🔢 Seçiminizi yapın (1-6): " choice

case $choice in
    1)
        MACHINE_TYPE="n2-highmem-2"
        RAM_GB="16"
        ;;
    2)
        MACHINE_TYPE="n2-highmem-4"
        RAM_GB="32"
        ;;
    3)
        MACHINE_TYPE="n2-highmem-8"
        RAM_GB="64"
        ;;
    4)
        MACHINE_TYPE="n2-highmem-16"
        RAM_GB="128"
        ;;
    5)
        MACHINE_TYPE="n2-highmem-32"
        RAM_GB="256"
        ;;
    6)
        MACHINE_TYPE="m2-ultramem-416"
        RAM_GB="5888"
        echo -e "${RED}⚠️  UYARI: Bu çok pahalı bir seçenek! Sadece özel durumlar için.${NC}"
        read -p "Devam etmek istediğinizden emin misiniz? (y/N): " confirm
        if [[ $confirm != "y" ]]; then
            echo "İşlem iptal edildi."
            exit 1
        fi
        ;;
    *)
        echo -e "${RED}❌ Geçersiz seçim! Varsayılan olarak n2-highmem-4 kullanılacak.${NC}"
        MACHINE_TYPE="n2-highmem-4"
        RAM_GB="32"
        ;;
esac

echo -e "${GREEN}✅ Seçilen: ${MACHINE_TYPE} (${RAM_GB} GB RAM)${NC}"

# Project ID kontrolü
if [ -z "$PROJECT_ID" ]; then
    echo ""
    echo -e "${YELLOW}🏗️  Google Cloud Project ID gerekli${NC}"
    read -p "Project ID'nizi girin: " PROJECT_ID
fi

# Disk boyutu seçimi
echo ""
echo -e "${BLUE}💽 Disk Boyutu Seçimi:${NC}"
echo "1. 50 GB   [Temel - Sadece sistem]"
echo "2. 100 GB  [Önerilen - Modeller + veri]"
echo "3. 200 GB  [Yüksek - Büyük veri setleri]"
echo "4. 500 GB  [Maksimum - Çok büyük modeller]"

read -p "Disk boyutunu seçin (1-4): " disk_choice

case $disk_choice in
    1) DISK_SIZE="50" ;;
    2) DISK_SIZE="100" ;;
    3) DISK_SIZE="200" ;;
    4) DISK_SIZE="500" ;;
    *) 
        echo -e "${YELLOW}⚠️  Varsayılan 100GB kullanılacak${NC}"
        DISK_SIZE="100"
        ;;
esac

# VM oluşturma komutu
echo ""
echo -e "${BLUE}🚀 VM oluşturuluyor...${NC}"

# Startup script içeriği
STARTUP_SCRIPT="#!/bin/bash
# FitTürkAI Setup Script
apt-get update && apt-get upgrade -y
apt-get install -y python3 python3-pip python3-venv git htop tree curl wget

# Python environment setup
python3 -m pip install --upgrade pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Clone repository (replace with your repo)
# git clone https://github.com/YOUR_USERNAME/FitTurkAI-RAG.git /home/ubuntu/FitTurkAI-RAG
# chown -R ubuntu:ubuntu /home/ubuntu/FitTurkAI-RAG

# System optimizations for high RAM
echo 'vm.swappiness=10' >> /etc/sysctl.conf
echo 'vm.vfs_cache_pressure=50' >> /etc/sysctl.conf
sysctl -p

# Install monitoring tools
apt-get install -y iotop nethogs

echo 'FitTürkAI VM setup completed!' >> /var/log/startup-script.log
"

# VM oluştur
gcloud compute instances create $VM_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
    --metadata="startup-script=$STARTUP_SCRIPT" \
    --maintenance-policy=MIGRATE \
    --provisioning-model=STANDARD \
    --service-account=default \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --tags=fitturkrai,http-server,https-server \
    --create-disk=auto-delete=yes,boot=yes,device-name=$VM_NAME,image=projects/ubuntu-os-cloud/global/images/ubuntu-2204-jammy-v20240319,mode=rw,size=$DISK_SIZE,type=projects/$PROJECT_ID/zones/$ZONE/diskTypes/pd-balanced \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --labels=purpose=ml,project=fitturkrai \
    --reservation-affinity=any

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ VM başarıyla oluşturuldu!${NC}"
    echo ""
    echo -e "${BLUE}📋 VM Bilgileri:${NC}"
    echo "  Ad: $VM_NAME"
    echo "  Tip: $MACHINE_TYPE"
    echo "  RAM: ${RAM_GB} GB"
    echo "  Disk: ${DISK_SIZE} GB"
    echo "  Zone: $ZONE"
    echo ""
    
    # IP adresini al
    EXTERNAL_IP=$(gcloud compute instances describe $VM_NAME --zone=$ZONE --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
    echo -e "${GREEN}🌐 External IP: $EXTERNAL_IP${NC}"
    echo ""
    
    echo -e "${BLUE}🔗 Bağlantı Komutları:${NC}"
    echo "  SSH: gcloud compute ssh $VM_NAME --zone=$ZONE"
    echo "  SSH Direct: ssh ubuntu@$EXTERNAL_IP"
    echo ""
    
    echo -e "${YELLOW}⏳ VM'nin hazır olması için ~2-3 dakika bekleyin${NC}"
    echo -e "${BLUE}📦 Sonraki adımlar:${NC}"
    echo "1. SSH ile bağlanın"
    echo "2. Repository'yi klonlayın"
    echo "3. FitTürkAI sistemini kurun"
    
    # SSH bağlantısı sor
    echo ""
    read -p "🔌 Şimdi SSH ile bağlanmak ister misiniz? (y/N): " connect_now
    if [[ $connect_now == "y" ]]; then
        echo "SSH bağlantısı kuruluyor..."
        gcloud compute ssh $VM_NAME --zone=$ZONE
    fi
    
else
    echo -e "${RED}❌ VM oluşturma başarısız!${NC}"
    exit 1
fi 