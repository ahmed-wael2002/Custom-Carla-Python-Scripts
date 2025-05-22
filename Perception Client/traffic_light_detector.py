import cv2
import numpy as np
from ultralytics import YOLO
from enum import Enum

class TrafficLightState(Enum):
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    UNKNOWN = "unknown"

class TrafficLightDetector:
    def __init__(self, model_path='yolov8m.pt'):
        """
        Initialize the TrafficLightDetector with a YOLOv8 model.
        
        Args:
            model_path (str): Path to the YOLOv8 model weights
        """
        self.model = YOLO(model_path)
        self.color_ranges = {
            'red': ([0, 100, 100], [10, 255, 255]),
            'yellow': ([20, 100, 100], [30, 255, 255]),
            'green': ([40, 100, 100], [80, 255, 255])
        }

    def detect(self, image):
        """
        Detect traffic light state in an image.
        
        Args:
            image (numpy.ndarray): Input image in BGR format
            
        Returns:
            tuple: (TrafficLightState, list of bounding boxes)
        """
        if image is None:
            raise ValueError("Invalid image input")

        # Run inference
        results = self.model(image)
        detected_states = []
        bounding_boxes = []

        # Process results
        for result in results:
            boxes = result.boxes
            if len(boxes) == 0:
                return TrafficLightState.UNKNOWN, []

            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bounding_boxes.append((x1, y1, x2, y2))
                
                # Extract traffic light region
                traffic_light = image[y1:y2, x1:x2]
                
                # Convert to HSV color space
                hsv = cv2.cvtColor(traffic_light, cv2.COLOR_BGR2HSV)
                
                # Check which color is dominant
                max_pixels = 0
                current_color = None
                
                for color, (lower, upper) in self.color_ranges.items():
                    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                    pixel_count = np.sum(mask > 0)
                    
                    if pixel_count > max_pixels:
                        max_pixels = pixel_count
                        current_color = color
                
                # Only consider it a valid detection if we have enough colored pixels
                if max_pixels > 100:  # Threshold to avoid false positives
                    detected_states.append(TrafficLightState(current_color))
                else:
                    detected_states.append(TrafficLightState.UNKNOWN)

        if not detected_states:
            return TrafficLightState.UNKNOWN, []
            
        # Return the most common state if multiple traffic lights are detected
        return max(set(detected_states), key=detected_states.count), bounding_boxes

    def display_detection_result(self, image, state, bounding_boxes):
        """
        Display detection results on the image.
        
        Args:
            image (numpy.ndarray): Input image
            state (TrafficLightState): Detected traffic light state
            bounding_boxes (list): List of bounding boxes (x1, y1, x2, y2)
            
        Returns:
            numpy.ndarray: Image with detection results
        """
        result_image = image.copy()
        
        for x1, y1, x2, y2 in bounding_boxes:
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Add label
            cv2.putText(result_image, f"Traffic Light: {state.value}", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                       (0, 255, 0), 2)
        
        return result_image 

def test_traffic_light_detection():
    """
    Test function to demonstrate traffic light detection functionality.
    """
    # Initialize the detector
    detector = TrafficLightDetector()
    
    # Load a test image (you'll need to provide a test image)
    test_image = cv2.imread('test2.jpg')
    if test_image is None:
        print("Error: Could not load test image")
        return
    
    # Detect traffic light state
    state, bounding_boxes = detector.detect(test_image)
    
    # Display results
    result_image = detector.display_detection_result(test_image, state, bounding_boxes)
    
    # Save the result instead of displaying it
    output_path = 'detection_result.jpg'
    cv2.imwrite(output_path, result_image)
    print(f"Result image saved to: {output_path}")
    print(f"Detected Traffic Light State: {state.value}")

if __name__ == "__main__":
    test_traffic_light_detection() 