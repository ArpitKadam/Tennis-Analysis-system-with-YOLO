import cv2 # type: ignore

def read_video(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Unable to open video file.")
            return None

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()
        return frames
    except Exception as e:
        print(f"Error: {e}")


import os

def save_video(output_video_frames, output_video_path, fps=24):
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, 
                              (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))

        for frame in output_video_frames:
            out.write(frame)

        out.release()
        print(f"Video saved successfully at {output_video_path}")
    except Exception as e:
        print(f"Error: {e}")
