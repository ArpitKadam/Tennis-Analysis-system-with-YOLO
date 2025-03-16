from ultralytics import YOLO #type: ignore

model = YOLO("yolov8x")

result = model.track("input_videos/input_video.mp4", save=True)

print(result)

with open("yolov8x_track.txt", "w") as f:
    f.write(str(result))

