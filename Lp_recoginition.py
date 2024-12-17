from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2

def load_models():
    """
    Load YOLO and OCR models.
    """
    ocr_model = PaddleOCR(use_angle_cls=True, lang='en')
    yolo_model = YOLO('best.pt')
    return ocr_model, yolo_model

def draw_bbox_with_text(image, bbox, text, font=cv2.FONT_HERSHEY_SIMPLEX, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    """
    Draw bounding box and overlay text on the image.
    """
    xmin, ymin, xmax, ymax = map(int, bbox)
    
    # Draw bounding box
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 64, 0), 2)
    
    # Calculate text size and position
    font_scale = (xmax - xmin) / 200.0
    thickness = max(1, int((xmax - xmin) / 300))
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = max(0, int((xmin + xmax) / 2 - text_size[0] / 2))
    text_y = ymin - 10 if ymin > 20 else ymax + 20

    # Add text background
    cv2.rectangle(
        image, 
        (text_x, text_y - text_size[1] - 5), 
        (text_x + text_size[0], text_y + 5), 
        bg_color, 
        -1
    )
    # Add text
    cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, thickness)

def extract_text_from_bbox(ocr_model, image, bbox):
    """
    Extract text from the given bounding box using OCR.
    """
    xmin, ymin, xmax, ymax = map(int, bbox)
    cropped_image = image[ymin:ymax, xmin:xmax]
    ocr_results = ocr_model.ocr(cropped_image, cls=True)
    
    if ocr_results and ocr_results[0]:
        return " ".join([line[1][0] for line in ocr_results[0]])
    return ""

def process_video(video_path, output_path, ocr_model, yolo_model):
    """
    Process a video, detect objects and overlay OCR results frame-by-frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Create a VideoWriter object for saving the processed video
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # YOLO detection
        results = yolo_model.predict(frame)

        for detection in results[0].boxes.data.tolist():
            xmin, ymin, xmax, ymax, _, _ = detection
            bbox = (xmin, ymin, xmax, ymax)
            
            # Extract text using OCR
            detected_text = extract_text_from_bbox(ocr_model, frame, bbox)
            
            if detected_text:
                draw_bbox_with_text(frame, bbox, detected_text)
        
        # Write the processed frame to the output video
        out.write(frame)

        # Display the frame (optional)
        cv2.imshow('Processed Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    # Load models
    ocr_model, yolo_model = load_models()
    
    # Input video path
    video_path = 'test.mp4'
    output_path = 'output_video.mp4'
    
    # Process the video
    process_video(video_path, output_path, ocr_model, yolo_model)