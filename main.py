
from utils.video_utils import read_video, save_video
from tracker.player_tracker import PlayerTracker
from tracker.ball_tracker import BallTracker
from court_line_detector.court_line_detector import CourtLineDetector
from minicourt.minicourt import Minicourt
from utils.bbox_utils import measure_distance
from utils.player_stats_drawing_utils import draw_player_stats
from utils.conversions import convert_meters_distance_to_pixels, convert_pixel_distance_to_meters
import pandas as pd #type: ignore
from constants import *
from copy import deepcopy
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

    ## Convert position to minicourt coordinates
    player_minicourt_detections, ball_mini_court_detections = mini_court.convert_bounding_box_to_minicourt_coordinates(player_detections, ball_detections, court_keypoints)

    ## Detect ball shots
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)
    print("Ball shot frames: ", ball_shot_frames)

    player_stats_data = [{
        'frame_num' : 0,
        'player_1_no_of_shots' : 0,
        'player_1_total_shot_speed' : 0,
        'player_1_last_shot_speed' : 0,
        'player_1_total_player_speed' : 0,
        'player_1_last_player_speed': 0,

        'player_2_no_of_shots' : 0,
        'player_2_total_shot_speed' : 0,
        'player_2_last_shot_speed' : 0,
        'player_2_total_player_speed' : 0,
        'player_2_last_player_speed': 0
    }]

    for ball_shot_index in range(len(ball_shot_frames)-1):
        start_frame = ball_shot_frames[ball_shot_index]
        end_frame =  ball_shot_frames[ball_shot_index+1]
        ball_shot_time_in_seconds = (end_frame - start_frame) / 24

        distance_covered_by_ball_pixels = measure_distance(ball_mini_court_detections[start_frame][1],
                                                           ball_mini_court_detections[end_frame][1])
        
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters(distance_covered_by_ball_pixels,
                                                                            DOUBLE_LINE_WIDTH,
                                                                            mini_court.get_width_of_minicourt())

        ## Speed of the ball
        speed_of_ball_shot = distance_covered_by_ball_meters / ball_shot_time_in_seconds * 3.6

        player_positions = player_minicourt_detections[start_frame]
        player_shot_ball = min(player_positions.keys(), key=lambda player_id: measure_distance(player_positions[player_id],
                                                                                       ball_mini_court_detections[start_frame][1]))
        
        oponent_player_id = 1 if player_shot_ball == 2 else 2

        distance_covered_by_oponent_player_pixels = measure_distance(player_minicourt_detections[start_frame][oponent_player_id],
                                                                    player_minicourt_detections[end_frame][oponent_player_id])
        distance_covered_by_oponent_player_meters = convert_pixel_distance_to_meters(distance_covered_by_oponent_player_pixels,
                                                                                      DOUBLE_LINE_WIDTH,
                                                                                      mini_court.get_width_of_minicourt())

        speed_of_oponent = distance_covered_by_oponent_player_meters / ball_shot_time_in_seconds * 3.6

        current_player_stats = deepcopy(player_stats_data[-1])

        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_shot_ball}_no_of_shots'] += 1
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot

        current_player_stats[f'player_{oponent_player_id}_total_player_speed'] += speed_of_oponent
        current_player_stats[f'player_{oponent_player_id}_last_player_speed'] = speed_of_oponent

        player_stats_data.append(current_player_stats)
    
    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})

    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    player_stats_data_df['player_1_avg_shot_speed'] = player_stats_data_df['player_1_total_shot_speed'] / player_stats_data_df['player_1_no_of_shots']
    player_stats_data_df['player_2_avg_shot_speed'] = player_stats_data_df['player_2_total_shot_speed'] / player_stats_data_df['player_2_no_of_shots']

    player_stats_data_df['player_1_avg_player_speed'] = player_stats_data_df['player_1_total_player_speed'] / player_stats_data_df['player_1_no_of_shots']
    player_stats_data_df['player_2_avg_player_speed'] = player_stats_data_df['player_2_total_player_speed'] / player_stats_data_df['player_2_no_of_shots']

    player_stats_data_df.to_csv('player_stats_data.csv', index=False)

    ## Drawing Bounding Boxes           
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    ## Drawing Court Lines
    ## output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    ## Draw minicourt on video frames
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_minicourt(output_video_frames, player_minicourt_detections)
    output_video_frames = mini_court.draw_points_on_minicourt(output_video_frames, ball_mini_court_detections, color=(0, 255, 255))

    ## Draw Player Stats
    output_video_frames = draw_player_stats(output_video_frames, player_stats_data_df)

    ## Draw frame number on top left corner of the video 
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)




    ## Saving Video
    save_video(output_video_frames, "output_video/output_video.avi")


if __name__ == "__main__":
    main()



