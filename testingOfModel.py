from ultralytics import YOLO

# Load the trained model from the transferred runs folder
model = YOLO(r"runs\runs\detect\train3\weights\best.pt")

# Run inference on a test image
results = model.predict("testingImg2.jpg")

# Show predictions
for r in results:
    r.show()
