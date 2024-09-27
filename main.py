from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movment_estimator import CameraMovmentEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
import numpy as np
def main():
    video_frames = read_video("./input-videos/08fd33_4.mp4")
    tracker=Tracker("./models/best.pt")
    tracks=tracker.get_object_tracks(video_frames,read_from_stub=True,stub_path="./stubs/track_stub.pkl")

    camera_movment_estimator=CameraMovmentEstimator(video_frames[0])
    camera_movment_per_frame=camera_movment_estimator.get_camera_movment(
        video_frames,read_from_stub=True,stub_path="./stubs/camera_movment_stub.pkl"
    )
    camera_movment_estimator.add_position_to_tracks(tracks)
    camera_movment_estimator.add_adjust_positions_to_tracks(tracks,camera_movment_per_frame)
    view_transformer=ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    


    tracks["ball"]=tracker.interppolate_ball_positions(tracks["ball"])
        # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
    team_assigner=TeamAssigner()
    team_assigner.assign_team_color(video_frames[0],tracks["players"][0])
    for frame_num,player_track in enumerate(tracks["players"]):
        for player_id,track in player_track.items():
            team=team_assigner.get_player_team(video_frames[frame_num],track["bbox"],player_id)
            tracks["players"][frame_num][player_id]["team"]=team
            tracks["players"][frame_num][player_id]["team_color"]=team_assigner.team_colors[team]
    player_ball_assigner=PlayerBallAssigner()
    team_ball_control=[]
    for frame_num,player_track in enumerate(tracks["players"]):
        ball_bbox=tracks["ball"][frame_num][1]["bbox"]
        assigned_player=player_ball_assigner.assign_ball_to_player(player_track,ball_bbox)
        if assigned_player!=-1:
            tracks["players"][frame_num][assigned_player]["has_ball"]=True
            team_ball_control.append(tracks["players"][frame_num][assigned_player]["team"])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control=np.array(team_ball_control)
    output_video_frames=tracker.draw_annotations(video_frames,tracks,team_ball_control)
    output_video_frames=camera_movment_estimator.draw_camera_movment(output_video_frames,camera_movment_per_frame)
        ## Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)
    save_video( output_video_frames, "./output-videos/08fd33_4.mp4")
if __name__ == "__main__":
    main()