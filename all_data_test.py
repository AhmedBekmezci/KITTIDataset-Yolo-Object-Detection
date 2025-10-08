from ultralytics import YOLO
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Eğitilen modeli yükle
    model = YOLO(r"C:\Users\Ahmed\Desktop\Visual Studio Code Projects\runs\detect\car_pedestrian_model_full_v8m3\weights\best.pt")

    # Modeli belirtilen test klasöründe değerlendir
    metrics = model.val(
        data=r"C:\Users\Ahmed\Desktop\YOLO_dataset\kitti.yaml",   # sınıf bilgileri için yaml
        imgsz=640,
        batch=4,
        source=r"C:\Users\Ahmed\Desktop\image\testing\image_2"
    )

    class_names = list(model.names.values()) if isinstance(model.names, dict) else model.names
    f1_scores = metrics.box.f1
    mp = metrics.box.p   # precision per class
    mr = metrics.box.r   # recall per class
    ap50 = metrics.box.ap50 if hasattr(metrics.box, 'ap50') else [metrics.box.map50]*len(class_names)

    x = np.arange(len(class_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 1.5*width, mp, width, label='Precision')
    ax.bar(x - 0.5*width, mr, width, label='Recall')
    ax.bar(x + 0.5*width, f1_scores, width, label='F1 Score')
    ax.bar(x + 1.5*width, ap50, width, label='mAP50')

    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Score')
    ax.set_title('Sınıf Bazlı Değerlendirme Metrikleri')
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

