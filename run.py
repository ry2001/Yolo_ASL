from ultralytics import YOLO
import supervision as sv
import cv2

# open camera
cap = cv2.VideoCapture(0)
# draw box
box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

# load model
model = YOLO('./best.onnx', task='detect') # 100 epochs (s model)

while True:
    # read frames from camera
    ret, frame = cap.read()

    # run model prediction
    result = model.predict(frame, verbose=False, iou=0.3)[0]

    # convert detections to supervision
    detections = sv.Detections.from_ultralytics(result)
    # get label for bounding box
    labels = [
            f"{result.names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _
            in detections
        ]
    # annotate on the frame
    frame = box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        )
    cv2.imshow("yolov8", frame)

    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows()