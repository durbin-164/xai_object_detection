from ultralytics import YOLO


model = YOLO('yolov8x.pt')

# model = YOLO('/content/last.pt')

# model = YOLO('/content/runs/detect/train/weights/last.pt')


# Train the model
results = model.train(data='/content/yolo_data/data.yaml', epochs=40, imgsz=640)