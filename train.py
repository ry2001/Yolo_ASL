from ultralytics import YOLO

# load pretrained model
model = YOLO("yolov8s.pt")

# train model using ASL datasets
model.train(data="../datasets/data/data.yaml", epochs=100, resume=True)

# evaluate model performance on the validation set
metrics = model.val()