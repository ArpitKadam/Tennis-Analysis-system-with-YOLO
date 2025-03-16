
from utils.video_utils import read_video, save_video
from tracker.player_tracker import PlayerTracker
from tracker.ball_tracker import BallTracker
from court_line_detector.court_line_detector import CourtLineDetector
from minicourt.minicourt import Minicourt
import cv2 # type: ignore

def main():
    ## Reading Video and Generating Frames
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)

    if video_frames is None:
        print("Error: No frames were read from the video. Exiting...")
        return
    
    ## Detecting Players and Balls
    player_tracker = PlayerTracker(model_path="yolov8x.pt")
    ball_tracker = BallTracker(model_path="training\\runs\\detect\\train\\weights\\last.pt")

    player_detections = player_tracker.detect_frames(video_frames, 
                                                     read_from_stubs=True,
                                                     stub_path="tracker_stubs/player_detection.pkl")

    ball_detections = ball_tracker.detect_frames(video_frames,
                                                 read_from_stubs=True,
                                                 stub_path="tracker_stubs/ball_detection.pkl")
    
    ball_detections = ball_tracker.interpolate_ball_position(ball_detections)

    ## Detecting Court Lines
    court_model_path="training\\keypoints_model_2.pth"   ## if you are using keypoints_model.pth then go court_line_detector.py and keep the resnet50, if you are using keypoints_model_2.pth then change it to resnet18
    court_line_detector = CourtLineDetector(model_path=court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    ## Choosing Players
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    ## Mini Court
    mini_court = Minicourt(video_frames[0])


    ## Drawing Bounding Boxes           
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    ## Drawing Court Lines
    ## output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    ## Draw minicourt on video frames
    output_video_frames = mini_court.draw_mini_court(output_video_frames)

    ## Draw frame number on top left corner of the video 
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)




    ## Saving Video
    save_video(output_video_frames, "output_video/output_video.avi")


if __name__ == "__main__":
    main()



