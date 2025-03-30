from utils import read_video, save_video
from trackers import Tracker
import cv2

def main():
    # Read Video
    video_frames, fps = read_video('input_videos/spain-portugal-vid.mp4')

    # Initialize Tracker
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/tracks_stubs.pkl')
       

    # Draw output
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # Save Video with original fps
    save_video(output_video_frames, 'output_videos/output_video.mp4', fps=fps)

if __name__ == '__main__':
    main()