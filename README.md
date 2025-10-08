# YOLO Object Detection Project / YOLO Nesne Tespiti Projesi

[English](#english) | [Türkçe](#turkish)

---

<a name="english"></a>
## 🇬🇧 English

### 📋 Project Overview
This project implements a YOLO (You Only Look Once) object detection model for detecting cars and pedestrians using the KITTI dataset. The project includes data preparation, model training, testing, and comprehensive evaluation scripts.

### 🗂️ Project Structure
```
├── prepare_kitti_yolo.py    # KITTI dataset conversion to YOLO format
├── train2_yolov8m.py        # Model training script
├── test_2.py                # Random 10 images testing and visualization
├── all_data_test.py         # Full dataset evaluation and top-10 selection
├── result.png               # Final model performance results
└── images/                  # Example output visualizations
    ├── Figure_1.png         
    ├── Figure_2.png         
    ├── Figure_3.png         
    ├── Figure_4.png         
    └── Figure_5.png         
```

### 🚀 Getting Started

#### Prerequisites
```bash
pip install ultralytics opencv-python matplotlib numpy pillow
```

#### Dataset Preparation
The KITTI dataset needs to be converted to YOLO format before training:

```bash
python prepare_kitti_yolo.py
```

**What this script does:**
- Converts KITTI annotation format to YOLO format
- Splits data into training and validation sets
- Creates the required directory structure
- Generates `kitti.yaml` configuration file

### 🎯 Usage

#### 1. Training the Model
Train the YOLOv8 model on the KITTI dataset:

```bash
python train2_yolov8m.py
```

**Training Configuration (Used in this project):**
- Model: YOLOv8m (medium)
- Image size: 640x640
- Epochs: 25
- Batch size: 4

**⚠️ IMPORTANT: You can adjust the model, epochs, batch size, and image size according to your computer's capabilities. This configuration is provided as a baseline example.**

#### 2. Quick Testing (Random 10 Images)
Test the trained model on 10 random images:

```bash
python test_2.py
```

**Features:**
- Randomly selects 10 images from test set
- Performs object detection
- Visualizes predictions with bounding boxes
- Displays confidence scores and class labels

#### 3. Full Evaluation (Top 10 Best Predictions)
Evaluate the model on entire test set and find best predictions:

```bash
python all_data_test.py
```

**Features:**
- Evaluates model on all test images
- Calculates precision, recall, F1-score, and mAP50
- Identifies top 10 images with highest confidence scores
- Saves results to output folder
- Generates comprehensive visualization plots

### 📊 Model Performance

The model is evaluated using multiple metrics:
- **Precision**: Accuracy of positive predictions
- **Recall**: Coverage of actual positive cases
- **F1 Score**: Harmonic mean of precision and recall
- **mAP50**: Mean Average Precision at IoU threshold 0.5

**Final results can be seen in `result.png`**

### 🎨 Output Visualizations

All images in the `images/` folder are **example outputs** showing what the model can produce:

- **Figure_1.png**: Example of class-based evaluation metrics (Precision, Recall, F1-Score, mAP50)
- **Figure_2.png**: Example of top 10 predictions grid view
- **Figure_3.png**: Example of individual detection results
- **Figure_4.png**: Example of performance analysis charts
- **Figure_5.png**: Example of additional visualizations
- **result.png**: Final model performance metrics and results

### 📁 Output Files

**Saved Predictions:**
```
top10_predictions/
├── top1_image_name.png      # Best prediction
├── top2_image_name.png      # Second best
├── ...
├── top10_image_name.png     # 10th best
└── top10_summary.png        # Grid view of all top 10
```

### 🔧 Configuration

**Model Path:**
```python
model_path = "runs/detect/car_pedestrian_model_full_v8m3/weights/best.pt"
```

**Dataset Configuration:**
```python
data_yaml = "path/to/YOLO_dataset/kitti.yaml"
test_images = "path/to/testing/image_2"
```

### 📈 Key Features

✅ KITTI dataset support  
✅ YOLOv8 architecture  
✅ Multi-class detection (Car, Pedestrian)  
✅ Comprehensive evaluation metrics  
✅ Top predictions selection  
✅ Automatic visualization and saving  
✅ Confidence-based ranking  

### 📝 Notes

- The model is trained specifically for car and pedestrian detection
- Confidence threshold is set to 0.25 for predictions
- All visualizations are automatically saved
- The project uses multiprocessing for efficient batch processing
- **Training parameters can be adjusted based on hardware capabilities**

---

<a name="turkish"></a>
## 🇹🇷 Türkçe

### 📋 Proje Hakkında
Bu proje, KITTI veri seti kullanılarak araç ve yaya tespiti yapmak için YOLO (You Only Look Once) nesne tespit modeli uygulamaktadır. Proje, veri hazırlama, model eğitimi, test etme ve kapsamlı değerlendirme scriptlerini içermektedir.

### 🗂️ Proje Yapısı
```
├── prepare_kitti_yolo.py    # KITTI veri setini YOLO formatına dönüştürme
├── train2_yolov8m.py        # Model eğitim scripti
├── test_2.py                # Rastgele 10 görsel testi ve görselleştirme
├── all_data_test.py         # Tam veri seti değerlendirmesi ve en iyi 10 seçimi
├── result.png               # Final model performans sonuçları
└── images/                  # Örnek çıktı görselleri
    ├── Figure_1.png         
    ├── Figure_2.png         
    ├── Figure_3.png         
    ├── Figure_4.png         
    └── Figure_5.png         
```

### 🚀 Başlangıç

#### Gereksinimler
```bash
pip install ultralytics opencv-python matplotlib numpy pillow
```

#### Veri Seti Hazırlığı
KITTI veri setinin eğitimden önce YOLO formatına dönüştürülmesi gerekmektedir:

```bash
python prepare_kitti_yolo.py
```

**Bu script ne yapar:**
- KITTI anotasyon formatını YOLO formatına dönüştürür
- Veriyi eğitim ve doğrulama setlerine böler
- Gerekli klasör yapısını oluşturur
- `kitti.yaml` yapılandırma dosyasını oluşturur

### 🎯 Kullanım

#### 1. Modeli Eğitme
YOLOv8 modelini KITTI veri seti üzerinde eğitin:

```bash
python train2_yolov8m.py
```

**Eğitim Yapılandırması (Bu projede kullanılan):**
- Model: YOLOv8m (orta)
- Görsel boyutu: 640x640
- Epoch: 25
- Batch boyutu: 4

**⚠️ ÖNEMLİ: Bilgisayarınızın gücüne göre model, epochs, batch boyutu ve görsel boyutu değiştirilebilir. Bu yapılandırma temel bir örnek olması için verilmiştir.**

#### 2. Hızlı Test (Rastgele 10 Görsel)
Eğitilmiş modeli rastgele 10 görsel üzerinde test edin:

```bash
python test_2.py
```

**Özellikler:**
- Test setinden rastgele 10 görsel seçer
- Nesne tespiti yapar
- Tahminleri sınırlayıcı kutularla görselleştirir
- Güven skorlarını ve sınıf etiketlerini gösterir

#### 3. Tam Değerlendirme (En İyi 10 Tahmin)
Modeli tüm test seti üzerinde değerlendirin ve en iyi tahminleri bulun:

```bash
python all_data_test.py
```

**Özellikler:**
- Tüm test görsellerinde modeli değerlendirir
- Precision, recall, F1-score ve mAP50 hesaplar
- En yüksek güven skorlarına sahip en iyi 10 görseli belirler
- Sonuçları çıktı klasörüne kaydeder
- Kapsamlı görselleştirme grafikleri oluşturur

### 📊 Model Performansı

Model birden fazla metrik kullanılarak değerlendirilir:
- **Precision (Kesinlik)**: Pozitif tahminlerin doğruluğu
- **Recall (Duyarlılık)**: Gerçek pozitif durumların kapsanması
- **F1 Score**: Precision ve recall'ın harmonik ortalaması
- **mAP50**: IoU eşiği 0.5'te Ortalama Ortalama Kesinlik

**Final sonuçlar `result.png` dosyasında görülebilir**

### 🎨 Çıktı Görselleri

`images/` klasöründeki tüm görseller, modelin üretebileceği şeyleri gösteren **örnek çıktılardır**:

- **Figure_1.png**: Sınıf bazlı değerlendirme metrikleri örneği (Precision, Recall, F1-Score, mAP50)
- **Figure_2.png**: En iyi 10 tahmin ızgara görünümü örneği
- **Figure_3.png**: Tekil tespit sonuçları örneği
- **Figure_4.png**: Performans analiz grafikleri örneği
- **Figure_5.png**: Ek görselleştirmeler örneği
- **result.png**: Final model performans metrikleri ve sonuçları

### 📁 Çıktı Dosyaları

**Kaydedilen Tahminler:**
```
top10_predictions/
├── top1_image_name.png      # En iyi tahmin
├── top2_image_name.png      # İkinci en iyi
├── ...
├── top10_image_name.png     # 10. en iyi
└── top10_summary.png        # Tüm en iyi 10'un ızgara görünümü
```

### 🔧 Yapılandırma

**Model Yolu:**
```python
model_path = "runs/detect/car_pedestrian_model_full_v8m3/weights/best.pt"
```

**Veri Seti Yapılandırması:**
```python
data_yaml = "path/to/YOLO_dataset/kitti.yaml"
test_images = "path/to/testing/image_2"
```

### 📈 Ana Özellikler

✅ KITTI veri seti desteği  
✅ YOLOv8 mimarisi  
✅ Çok sınıflı tespit (Araç, Yaya)  
✅ Kapsamlı değerlendirme metrikleri  
✅ En iyi tahminleri seçme  
✅ Otomatik görselleştirme ve kaydetme  
✅ Güven tabanlı sıralama  

### 📝 Notlar

- Model özellikle araç ve yaya tespiti için eğitilmiştir
- Tahminler için güven eşiği 0.25 olarak ayarlanmıştır
- Tüm görselleştirmeler otomatik olarak kaydedilir
- Proje, verimli toplu işleme için multiprocessing kullanır
- **Eğitim parametreleri donanım yeteneklerine göre ayarlanabilir**

---

### 👨‍💻 Author / Yazar
**Ahmed Bekmezci**

### 📄 License / Lisans
This project is open source and available under the MIT License.  
Bu proje açık kaynaklıdır ve MIT Lisansı altında mevcuttur.

### 🤝 Contributing / Katkıda Bulunma
Contributions, issues, and feature requests are welcome!  
Katkılar, sorun bildirimleri ve özellik istekleri memnuniyetle karşılanır!

### ⭐ Show your support / Desteğinizi Gösterin
Give a ⭐️ if this project helped you!  
Bu proje size yardımcı olduysa bir ⭐️ verin!
