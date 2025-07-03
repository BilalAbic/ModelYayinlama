# 🚀 FitTürkAI - Yüksek RAM'li Google Cloud VM Hızlı Komutları

## 💰 RAM ve Maliyet Tablosu

| VM Tipi | vCPU | RAM | Aylık Maliyet* | Kullanım Durumu |
|---------|------|-----|----------------|-----------------|
| **n2-highmem-2** | 2 | 16 GB | ~$100 | Küçük modeller, test |
| **n2-highmem-4** | 4 | 32 GB | ~$200 | **Önerilen** - Orta boyut |
| **n2-highmem-8** | 8 | 64 GB | ~$400 | Büyük modeller |
| **n2-highmem-16** | 16 | 128 GB | ~$800 | **Maksimum** pratik |
| **n2-highmem-32** | 32 | 256 GB | ~$1600 | Çok büyük modeller |
| **m1-ultramem-40** | 40 | 961 GB | ~$6000 | Extreme durum |

*Yaklaşık maliyetler, gerçek fiyatlar değişebilir

## 🎯 Önerilen Konfigürasyonlar

### 1. **En Çok Önerilen**: n2-highmem-8 (64 GB RAM)
```bash
# Maksimum performans için ideal
gcloud compute instances create fitturkrai-vm \
    --machine-type=n2-highmem-8 \
    --zone=us-central1-a \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-balanced \
    --tags=fitturkrai \
    --metadata-from-file startup-script=startup-script.sh
```

### 2. **Maksimum CPU RAM**: n2-highmem-16 (128 GB RAM)
```bash
# En yüksek pratik RAM
gcloud compute instances create fitturkrai-max-vm \
    --machine-type=n2-highmem-16 \
    --zone=us-central1-a \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-balanced \
    --tags=fitturkrai \
    --metadata-from-file startup-script=startup-script.sh
```

### 3. **Ultra RAM**: n2-highmem-32 (256 GB RAM)
```bash
# En yüksek RAM (pahalı!)
gcloud compute instances create fitturkrai-ultra-vm \
    --machine-type=n2-highmem-32 \
    --zone=us-central1-a \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=500GB \
    --boot-disk-type=pd-balanced \
    --tags=fitturkrai \
    --metadata-from-file startup-script=startup-script.sh
```

## 💡 CPU Optimize Ayarları

### test.py için RAM optimizasyonu:
```python
# Yüksek RAM'li VM'ler için config
config = RAGConfig(
    chunk_size=500,           # Daha büyük chunk'lar
    retrieval_k=10,           # Daha fazla belge
    max_context_length=8000   # Daha uzun context
)
```

### Sistem ayarları (VM'de çalıştırın):
```bash
# Yüksek RAM için optimizasyon
sudo sysctl -w vm.swappiness=1
sudo sysctl -w vm.vfs_cache_pressure=50
echo 'vm.overcommit_memory=1' | sudo tee -a /etc/sysctl.conf
```

## 🚀 Hızlı Kurulum

### 1. VM oluştur (interactive script)
```bash
chmod +x create_high_ram_vm.sh
./create_high_ram_vm.sh
```

### 2. Manuel hızlı komut (64GB RAM)
```bash
export PROJECT_ID="your-project-id"
export VM_NAME="fitturkrai-vm"

gcloud compute instances create $VM_NAME \
    --project=$PROJECT_ID \
    --zone=us-central1-a \
    --machine-type=n2-highmem-8 \
    --network-interface=network-tier=PREMIUM,subnet=default \
    --maintenance-policy=MIGRATE \
    --provisioning-model=STANDARD \
    --service-account=default \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --tags=fitturkrai \
    --create-disk=auto-delete=yes,boot=yes,device-name=$VM_NAME,image=projects/ubuntu-os-cloud/global/images/ubuntu-2204-jammy-v20240319,mode=rw,size=100,type=projects/$PROJECT_ID/zones/us-central1-a/diskTypes/pd-balanced \
    --reservation-affinity=any
```

### 3. SSH bağlantı
```bash
gcloud compute ssh fitturkrai-vm --zone=us-central1-a
```

## 📊 RAM Kullanım İzleme

### VM'de RAM durumunu kontrol:
```bash
# Genel sistem durumu
htop

# RAM kullanımı
free -h

# Detaylı RAM istatistikleri  
cat /proc/meminfo

# Python process RAM kullanımı
ps aux | grep python | awk '{print $4, $11}'
```

## 🔧 Troubleshooting

### RAM yetersizse:
1. **Chunk size küçült**: `chunk_size=200`
2. **Context length azalt**: `max_context_length=2000`
3. **Retrieval sayısı azalt**: `retrieval_k=3`

### VM boyutunu artır:
```bash
# VM'yi durdur
gcloud compute instances stop fitturkrai-vm --zone=us-central1-a

# Makine tipini değiştir
gcloud compute instances set-machine-type fitturkrai-vm \
    --machine-type=n2-highmem-16 \
    --zone=us-central1-a

# VM'yi başlat
gcloud compute instances start fitturkrai-vm --zone=us-central1-a
```

## 💰 Maliyet Optimizasyonu

### 1. Preemptible Instance (75% indirim):
```bash
--preemptible \
--maintenance-policy=TERMINATE
```

### 2. Spot Instance (80% indirim):
```bash
--provisioning-model=SPOT \
--instance-termination-action=STOP
```

### 3. Committed Use Discounts:
- 1 yıl: %20 indirim
- 3 yıl: %30 indirim

## 🎯 En İyi Performans için Öneriler

### RAM için:
- **n2-highmem-8** (64GB) - En optimal
- **n2-highmem-16** (128GB) - Maksimum pratik

### Disk için:
- **pd-balanced** - Performans/fiyat dengesi
- **pd-ssd** - Maksimum hız (pahalı)

### Zone için:
- **us-central1-a** - En stabil
- **europe-west1-b** - Avrupa için 