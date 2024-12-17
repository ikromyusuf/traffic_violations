from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLO model
model = YOLO("yolov8n.pt")  # Replace with your model weights

# Tracking history for object trajectories
trajectories = {}
next_object_id = 0  # ID counter for new objects

# Parameters
MAX_DISTANCE = 50  # Maximum distance to associate a new detection with an existing object
target_class = 2  # Car class (COCO: 2)
yellow = (0, 255, 255)  # Yellow color in BGR

def calculate_centroid(box):
    """Calculate the centroid of a bounding box."""
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return cx, cy

def match_detections_to_tracks(centroids, tracks, max_distance):
    """Match new detections (centroids) to existing tracks."""
    global next_object_id
    matched_tracks = {}

    # Calculate distances between each track and detection
    used_tracks = set()
    used_centroids = set()

    for track_id, track_history in tracks.items():
        last_position = track_history[-1]  # Last known position
        for i, centroid in enumerate(centroids):
            if i in used_centroids:
                continue
            distance = np.linalg.norm(np.array(last_position) - np.array(centroid))
            if distance < max_distance:
                matched_tracks[track_id] = centroid
                used_tracks.add(track_id)
                used_centroids.add(i)
                break

    # Assign new IDs to unmatched centroids
    for i, centroid in enumerate(centroids):
        if i not in used_centroids:
            matched_tracks[next_object_id] = centroid
            next_object_id += 1

    return matched_tracks

# Video input
video_path = "1234.mp4"  # Replace with 0 for webcam
cap = cv2.VideoCapture(video_path)

# Get video properties for saving output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4

# Define the output video writer
output_path = "output_videos/tracked_output1.mp4"
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection
    results = model.predict(frame, conf=0.5, classes=[target_class], verbose=False)
    centroids = []

    # Extract centroids for detected cars
    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        centroids.append(calculate_centroid((x1, y1, x2, y2)))
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

    # Match detections to existing tracks
    matched_tracks = match_detections_to_tracks(centroids, trajectories, MAX_DISTANCE)

    # Update trajectories
    for track_id, centroid in matched_tracks.items():
        if track_id not in trajectories:
            trajectories[track_id] = []
        trajectories[track_id].append(centroid)

        # Draw trajectory
        for i in range(1, len(trajectories[track_id])):
            if trajectories[track_id][i - 1] is None or trajectories[track_id][i] is None:
                continue
            cv2.line(frame, trajectories[track_id][i - 1], trajectories[track_id][i], yellow, 2)

        # Add track ID
        cv2.putText(frame, f"ID {track_id}", (centroid[0] - 10, centroid[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    # Write the frame to the output video
    out.write(frame)

    # Display frame
    cv2.imshow("Tracking Trajectories", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
