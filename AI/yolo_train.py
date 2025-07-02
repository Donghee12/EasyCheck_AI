from ultralytics import YOLO

model = YOLO("yolo12n.pt")

# Train the model
results = model.train(
    data="C:/Users/user/Desktop/yolo_surgical/YOLODataset/dataset.yaml",
    epochs=600,
    batch=256,
    imgsz=640,
    scale=0.5,  # S:0.9; M:0.9; L:0.9; X:0.9
    mosaic=1.0,
    mixup=0.0,  # S:0.05; M:0.15; L:0.15; X:0.2
    copy_paste=0.1,  # S:0.15; M:0.4; L:0.5; X:0.6
    project="C:/Users/user/Desktop",
    name="train",
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("path/to/image.jpg")
results[0].show()


# from ultralytics import YOLO

# model = YOLO("yolo12n.pt")

# import ultralytics

# print("version ", ultralytics.__version__)

# print("YOLO model info :", model.info())
