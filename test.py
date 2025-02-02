from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load an official detection model
model = YOLO("yolo11n-seg.pt")  # load an official segmentation model
#model = YOLO("path/to/best.pt")  # load a custom model   https://www.youtube.com/watch?v=x1UzbRAdX18

# Track with the model
# results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True)
# results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")

results = model.track(source="https://www.youtube.com/watch?v=TDFcX51yYS8", show=True)
results = model.track(source="https://www.youtube.com/watch?v=TDFcX51yYS8", show=True, tracker="bytetrack.yaml")
