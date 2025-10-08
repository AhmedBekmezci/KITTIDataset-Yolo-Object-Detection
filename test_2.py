import random
import os
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from collections import Counter

def main():
    # Eğitilen modeli yükle
    model = YOLO(r"C:\Users\Ahmed\Desktop\Visual Studio Code Projects\runs\detect\car_pedestrian_model_full_v8m3\weights\best.pt")

    # Test dataset yolunu tanımla
    test_dir = r"C:\Users\Ahmed\Desktop\image\testing\image_2"
    image_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Rastgele 10 görsel seç
    sample_images = random.sample(image_files, min(10, len(image_files)))

    # Her görseli test et
    for img_path in sample_images:
        results = model.predict(source=img_path, imgsz=640, conf=0.25, save=False, show=False)

        for r in results:
            # Orijinal görsel
            orig_img = cv2.imread(img_path)
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

            # Kutucuklarla işlenmiş görsel
            plotted_img = r.plot()
            plotted_img = cv2.cvtColor(plotted_img, cv2.COLOR_BGR2RGB)

            # Nesne sayımı
            labels = r.boxes.cls.cpu().numpy().astype(int)
            class_names = [model.names[int(cls)] for cls in labels]
            counts = Counter(class_names)
            count_text = ", ".join([f"{k}: {v}" for k, v in counts.items()]) if counts else "No objects detected"

            # Görselleştirme (yan yana 2 subplot)
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(orig_img)
            axes[0].axis("off")
            axes[0].set_title("Orijinal Görsel")

            axes[1].imshow(plotted_img)
            axes[1].axis("off")
            axes[1].set_title(f"Tahminler\n{count_text}")

            plt.suptitle(os.path.basename(img_path), fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
