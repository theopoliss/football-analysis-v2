from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
import pickle
import os
from utils import get_center_bbox, get_width_bbox

class Tracker():
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 20
        detections = []  # Initialize the detections list
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.18)
            detections.extend(detections_batch) 
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        # Check if saved from pickle already
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names  
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Convert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert goalkeeper (1) to player (2)
            for obj_index, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[obj_index] = cls_names_inv['player']

            # Tracker Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks['players'][frame_num][track_id] = {'bbox': bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks['referees'][frame_num][track_id] = {'bbox': bbox}

        # Save using pickle
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_bbox(bbox)
        width = get_width_bbox(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=3,
            lineType=cv2.LINE_4
        )

        # Rectangle
        rectangle_height = 20
        rectangle_width = 40
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2 - rectangle_height//2) + 15
        y2_rect = (y2 + rectangle_height//2) + 15

        if track_id is not None:
            cv2.rectangle(frame,
                          (x1_rect, y1_rect),
                          (x2_rect, y2_rect),
                          color,
                          cv2.FILLED)
            
            # Text within rectangle
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(frame,
                        f"{track_id}",
                        (x1_text, y1_rect + 15),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.6,
                        (0,0,0),
                        2)

        return frame
    
    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10 ,y+20]
        ])

        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED,)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2)

        return frame


    
    # Code to draw circles instaed of bounding boxes
    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy() 

            players_dict = tracks['players'][frame_num]
            referees_dict = tracks['referees'][frame_num]
            ball_dict = tracks['ball'][frame_num]

            # Draw players
            for track_id, player in players_dict.items():
                self.draw_ellipse(frame, player['bbox'], (0,0,255), track_id)

            # Draw referees
            for track_id, referee in referees_dict.items():
                self.draw_ellipse(frame, referee['bbox'], (0,255,255))

            # Draw ball
            for track_id, ball in ball_dict.items():
                self.draw_triangle(frame, ball['bbox'], (0,255,0))

            output_video_frames.append(frame)

        return output_video_frames 