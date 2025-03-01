import cv2
import numpy as np
import os
import google.generativeai as genai
from PIL import Image
import io
import time
import mediapipe as mp
import re

GOOGLE_API_KEY = ''

# Configure the Gemini API
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# Set up MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,  # Track only one hand for simplicity
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Function to check if two fingers are up (index and middle)
def two_fingers_up(hand_landmarks):
    if not hand_landmarks:
        return False
    
    landmarks = hand_landmarks.landmark
    
    # Get finger tip and pip (knuckle) positions
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = landmarks[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = landmarks[mp_hands.HandLandmark.PINKY_PIP]
    
    # Check if index and middle fingers are up and others are down
    index_up = index_tip.y < index_pip.y
    middle_up = middle_tip.y < middle_pip.y
    ring_down = ring_tip.y > ring_pip.y
    pinky_down = pinky_tip.y > pinky_pip.y
    
    return index_up and middle_up and ring_down and pinky_down

# Function to send image to Gemini API and get equation solution
def solve_equation_with_gemini(image):
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    try:
        # Generate content using Gemini with the PIL image directly
        response = model.generate_content([
            "The image contains a handwritten math equation. Please extract the equation, solve it, and return both the equation and its solution. Format your response exactly like this without any additional text: 'Equation: [extracted equation] Solution: [solution]'",
            pil_image  # Pass the PIL image directly
        ])
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# Function to display text on canvas with proper wrapping
def put_text_multiline(img, text, org, font_face, font_scale, color, thickness=1, line_spacing=1.5):
    y0, x0 = org
    height, width = img.shape[:2]
    
    # Split the text by newlines first
    for i, line in enumerate(text.split('\n')):
        y = int(y0 + i * font_scale * 40 * line_spacing)
        
        # Calculate text width to implement text wrapping
        (text_width, text_height), _ = cv2.getTextSize(line, font_face, font_scale, thickness)
        
        if text_width > width - 80:  # Leave some margin
            words = line.split()
            current_line = words[0]
            
            for word in words[1:]:
                test_line = current_line + " " + word
                (test_width, _), _ = cv2.getTextSize(test_line, font_face, font_scale, thickness)
                
                if test_width <= width - 80:
                    current_line = test_line
                else:
                    # Draw the current line and start a new one
                    cv2.putText(img, current_line, (x0, y), font_face, font_scale, color, thickness)
                    y += int(font_scale * 40 * line_spacing)
                    current_line = word
            
            # Draw the last line
            cv2.putText(img, current_line, (x0, y), font_face, font_scale, color, thickness)
        else:
            cv2.putText(img, line, (x0, y), font_face, font_scale, color, thickness)

def main():
    # Initialize OpenCV video capture
    cap = cv2.VideoCapture(0)
    
    # Set up canvas for drawing
    canvas = np.zeros((480, 640, 3), dtype=np.uint8) + 255  # White canvas
    last_point = None
    
    # Main drawing mode
    drawing_mode = True
    display_result = False
    result_text = ""
    
    print("Instructions:")
    print("- Hold up 2 fingers (index and middle) to draw")
    print("- Press 'e' to submit the equation for solving")
    print("- Press 'c' to clear the canvas")
    print("- Press 'q' to quit")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Flip frame for more intuitive drawing
        frame = cv2.flip(frame, 1)
        
        # Process hand landmarks
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Show current canvas in a corner of the frame
        small_canvas = cv2.resize(canvas, (213, 160))
        frame[0:160, 0:213] = small_canvas
        
        # If we're in result display mode, show the result on the canvas
        if display_result:
            # Canvas is already cleared when entering this mode
            put_text_multiline(canvas, result_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        # Otherwise, handle drawing mode
        elif drawing_mode:
            # Draw landmarks on frame
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Check if two fingers are up for drawing mode
                    if two_fingers_up(hand_landmarks):
                        # Get index finger tip position
                        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        h, w, c = frame.shape
                        index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
                        
                        # Draw on canvas
                        if last_point is not None:
                            cv2.line(canvas, last_point, (index_x, index_y), (0, 0, 0), 5)
                        
                        last_point = (index_x, index_y)
                        cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), -1)
                    else:
                        last_point = None
        
        # Display the frame
        cv2.imshow("Draw Equation (2 fingers up to draw)", frame)
        cv2.imshow("Canvas", canvas)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        
        # 'e' key to solve equation
        if key == ord('e') and not display_result:
            print("Solving equation...")
            # Add a border to the canvas for better recognition
            bordered_canvas = cv2.copyMakeBorder(canvas, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            
            # Save a copy of the original drawing before clearing
            original_drawing = canvas.copy()
            
            # Clear the canvas to prepare for displaying the result
            canvas = np.zeros((480, 640, 3), dtype=np.uint8) + 255
            
            # Get the solution from Gemini
            result_text = solve_equation_with_gemini(bordered_canvas)
            print("\nResult from Gemini:")
            print(result_text)
            
            # Check if we got a valid result or an error
            if result_text.startswith("Error:"):
                result_text = "Error recognizing equation. Please try again."
                # Restore the original drawing so the user can try again
                canvas = original_drawing
            else:
                # Switch to result display mode
                display_result = True
                
            print("\nPress 'c' to clear and continue, or 'q' to quit")
        
        # 'c' key to clear canvas and return to drawing mode
        elif key == ord('c'):
            canvas = np.zeros((480, 640, 3), dtype=np.uint8) + 255
            display_result = False
            print("Canvas cleared")
        
        # 'q' key to quit
        elif key == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
