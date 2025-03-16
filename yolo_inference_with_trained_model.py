from ultralytics import YOLO # type: ignore

model = YOLO("training/runs/detect/train/weights/last.pt")

results = model("input_videos\input_video.mp4", conf=0.2, save=True)

print("Results:" , results)
for box in results[0].boxes:
    with open("Boxes_last_pt.txt", "a+") as f:
        print("Boxes: ")
        print(box.xyxy)
        f.write(str(box.xyxy) + "\n")