import os
import shutil
import random

# --------------------
# Klasör yolları
# --------------------
images_train_dir = r"C:\Users\Ahmed\Desktop\image\training\image_2"
images_test_dir = r"C:\Users\Ahmed\Desktop\image\testing\image_2"
labels_train_dir = r"C:\Users\Ahmed\Desktop\label\training\label_2"

# YOLO klasör yapısı
yolo_dataset_dir = r"C:\Users\Ahmed\Desktop\YOLO_dataset"
train_images_dir = os.path.join(yolo_dataset_dir, "train", "images")
train_labels_dir = os.path.join(yolo_dataset_dir, "train", "labels")
val_images_dir = os.path.join(yolo_dataset_dir, "val", "images")
val_labels_dir = os.path.join(yolo_dataset_dir, "val", "labels")

# Klasörleri oluştur
for folder in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
    os.makedirs(folder, exist_ok=True)

# --------------------
# Sınıf mapping
# --------------------
class_mapping = {
    "Car": 0,
    "Pedestrian": 1
}

# --------------------
# Label dönüşümü fonksiyonu
# --------------------
def kitti_to_yolo(label_file, image_width=1242, image_height=375):
    yolo_lines = []
    with open(label_file, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            label = parts[0]
            if label not in class_mapping:
                continue  # sadece Car ve Pedestrian al
            bbox_left = float(parts[4])
            bbox_top = float(parts[5])
            bbox_right = float(parts[6])
            bbox_bottom = float(parts[7])

            x_center = ((bbox_left + bbox_right) / 2) / image_width
            y_center = ((bbox_top + bbox_bottom) / 2) / image_height
            width = (bbox_right - bbox_left) / image_width
            height = (bbox_bottom - bbox_top) / image_height

            class_id = class_mapping[label]
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return yolo_lines

# --------------------
# Görselleri ve labelları train/val olarak ayır
# --------------------
all_images = os.listdir(images_train_dir)
random.shuffle(all_images)
split_idx = int(0.8 * len(all_images))

train_images = all_images[:split_idx]
val_images = all_images[split_idx:]

def process_images(image_list, images_src_dir, images_dst_dir, labels_dst_dir):
    for img_file in image_list:
        img_name = os.path.splitext(img_file)[0]
        label_file = os.path.join(labels_train_dir, img_name + ".txt")
        yolo_label_file = os.path.join(labels_dst_dir, img_name + ".txt")
        img_src_path = os.path.join(images_src_dir, img_file)
        img_dst_path = os.path.join(images_dst_dir, img_file)

        # YOLO format label yaz
        if os.path.exists(label_file):
            yolo_lines = kitti_to_yolo(label_file)
            with open(yolo_label_file, 'w') as f:
                for l in yolo_lines:
                    f.write(l + "\n")

        # Görseli kopyala
        shutil.copy(img_src_path, img_dst_path)

# Train
process_images(train_images, images_train_dir, train_images_dir, train_labels_dir)
# Validation
process_images(val_images, images_train_dir, val_images_dir, val_labels_dir)

print("Veri hazırlığı tamamlandı! ✅")
print(f"Train örnekleri: {len(train_images)}")
print(f"Validation örnekleri: {len(val_images)}")
