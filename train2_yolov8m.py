from ultralytics import YOLO

def main():
    model = YOLO("yolov8m.pt")
    model.train(
        data="C:/Users/Ahmed/Desktop/YOLO_dataset/kitti.yaml",
        epochs=25,
        imgsz=640,
        batch=4,
        workers=8,  # multiprocess hatası buradan geliyor
        name="car_pedestrian_model_full_v8m",
        show=True,
        val=True
    )

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # Windows için güvenli başlatma
    main()
