from ultralytics import YOLO


# Change both model path and data path
model_path = "/home/masud.rana/Documents/Learning_Project/Important/XAI/models/yolo_s/best_epoch_36.pt"
data_path = "/home/masud.rana/Documents/Learning_Project/Important/XAI/data/yolo_data/data.yaml"

model = YOLO(model_path)

metrics = model.val(data=data_path)