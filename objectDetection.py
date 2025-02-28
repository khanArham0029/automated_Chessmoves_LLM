from ultralytics import YOLO

# Load the pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")  # You can use 'yolov8s.pt' for a slightly better model

# Train the model
model.train(data="C:/Users/ASDF/Documents/Personnel/chessWinner/datasets/2d_chess_dataset/data.yaml", epochs=50, imgsz=640, batch=8)