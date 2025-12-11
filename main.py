import cv2
from ultralytics import YOLO
import numpy as np
import time # <-- Add this import

# --- CONFIGURATION ---
VIDEO_SOURCE = 0 
MODEL_WEIGHTS = 'yolov8n.pt' 
LEARNING_FRAMES = 150 
NOTIFICATION_COOLDOWN = 5.0 # <-- NEW: Cooldown period in seconds (e.g., 5 seconds)
PROMOTION_TIME_SEC = 10.0
MISSING_FRAME_THRESHOLD = 90
# ---------------------

# --- GLOBAL REFERENCE DICTIONARY ---
STATIC_BACKGROUND_REF = {} 
DYNAMIC_TRACKS_FOR_LEARNING = {}

# --- NEW GLOBAL TIMER ---
# Stores the time (in seconds) when the last notification was printed/sent
LAST_NOTIFICATION_TIME = 0.0
# ------------------------

def learn_static_background(model, cap):
    """
    Identifies static objects by tracking their persistence over LEARNING_FRAMES.
    Objects present for >90% of the frames are deemed static.
    """
    global STATIC_BACKGROUND_REF
    temp_tracks = {}
    LEARNING_THRESHOLD = LEARNING_FRAMES * 0.90 # Must be present in 90% of frames
    
    # We will only process a fixed number of frames to define the static reference
    for i in range(LEARNING_FRAMES):
        success, frame = cap.read()
        if not success:
            break
        
        # Use tracking to get unique IDs
        results = model.track(frame, persist=True, verbose=False)
        boxes = results[0].boxes.cpu().numpy()
        
        if boxes.id is not None:
            for box_xyxy, track_id in zip(boxes.xyxy, boxes.id):
                track_id = int(track_id)
                
                # Calculate the center point of the object (x_center, y_center)
                x_center = int((box_xyxy[0] + box_xyxy[2]) / 2)
                y_center = int((box_xyxy[1] + box_xyxy[3]) / 2)
                
                # Get the class name
                class_id = int(boxes.cls[np.where(boxes.xyxy == box_xyxy)[0][0]])
                class_name = model.names[class_id]
                
                if track_id not in temp_tracks:
                    # Initialize the tracker record
                    temp_tracks[track_id] = {
                        'count': 0,
                        'class_name': class_name,
                        'ref_position_xy': (x_center, y_center)
                    }
                
                # Increment the frame count for this persistent ID
                temp_tracks[track_id]['count'] += 1

        # Display status during learning
        cv2.putText(frame, f"LEARNING: Frame {i}/{LEARNING_FRAMES}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Anomaly Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Finalize the Static Background Reference
    for track_id, data in temp_tracks.items():
        if data['count'] >= LEARNING_THRESHOLD:
            # This object is static and is added to the permanent reference list
            STATIC_BACKGROUND_REF[track_id] = {
                'class_name': data['class_name'], 
                'ref_position_xy': data['ref_position_xy'],
                'current_state': 'PRESENT',
                'missing_frames': 0
            }


def log_anomaly_with_cooldown(message):
    """Prints the anomaly message only if the cooldown has elapsed."""
    global LAST_NOTIFICATION_TIME
    
    current_time = time.time()
    
    if (current_time - LAST_NOTIFICATION_TIME) > NOTIFICATION_COOLDOWN:
        print(message) 
        LAST_NOTIFICATION_TIME = current_time # Reset the timer
        return True # Notification was sent
    
    return False # Notification was blocked by cooldown


def run_anomaly_pipeline():
    """
    Main function to execute the anomaly detection pipeline, handling:
    1. Initial Static Learning.
    2. Real-time Appearance & Disappearance Anomaly Detection.
    3. Adaptive Object Promotion.
    4. Cooldown-controlled logging.
    """
    global STATIC_BACKGROUND_REF, DYNAMIC_TRACKS_FOR_LEARNING
    
    # ... (Initialization code remains the same) ...
    print(f"Loading YOLO model: {MODEL_WEIGHTS}...")
    model = YOLO(MODEL_WEIGHTS) 
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {VIDEO_SOURCE}. Exiting.")
        return

    # ... (Initial Learning Phase code remains the same) ...
    print("\n--- Starting Initial Learning Phase ---")
    learn_static_background(model, cap)

    print("\n--- Initial Learning Complete ---")
    print(f"Normal State Defined with {len(STATIC_BACKGROUND_REF)} Static Objects.")
    print("Static Reference List (IDs and Positions):")
    for tid, data in STATIC_BACKGROUND_REF.items():
        print(f"  ID {tid:<4} | Class: {data['class_name']:<15} | Pos: {data['ref_position_xy']}")
    
    print("--- Starting Monitoring Phase ---")
    
    # --- Main Monitoring Loop ---
    while cap.isOpened():
        success, frame = cap.read()
        
        if success:
            
            current_time = time.time()
            results = model.track(frame, persist=True, verbose=False)
            frame_with_detections = results[0].plot()
            boxes = results[0].boxes.cpu().numpy()
            
            anomaly_messages = [] 
            detected_ids = set() 
            
            # 1. Populate Set of Detected IDs in this Frame
            if boxes.id is not None:
                detected_ids.update(int(tid) for tid in boxes.id)
                # Note: The detection and promotion logic from 3.1 (A, B, C) still runs here
            
            # --- 2. CHECK FOR DISAPPEARANCE ANOMALY (NEW LOGIC) ---
            
            # Iterate through all objects that SHOULD be static
            for track_id, data in STATIC_BACKGROUND_REF.items():
                
                if track_id not in detected_ids:
                    # Object is MISSING in this frame
                    
                    # 2a. Increment the missing frame counter
                    data['missing_frames'] += 1
                    data['current_state'] = 'MISSING'
                    
                    # 2b. Check if the object has exceeded the threshold
                    if data['missing_frames'] >= MISSING_FRAME_THRESHOLD:
                        # ANOMALY TRIGGERED!
                        message = f"🛑 DISAPPEARANCE ANOMALY: {data['class_name']} (ID {track_id}) is missing!"
                        log_anomaly_with_cooldown(message)
                        anomaly_messages.append(message)
                        
                else:
                    # Object IS PRESENT in this frame
                    if data['current_state'] == 'MISSING' and data['missing_frames'] > 0:
                        # Object has returned after being missing (Reappearance Notification)
                        
                        notification = f"✅ REAPPEARANCE: {data['class_name']} (ID {track_id}) has returned."
                        print(notification) # Print immediately as confirmation
                        anomaly_messages.append(notification)

                    # Reset the missing counter and state
                    data['missing_frames'] = 0
                    data['current_state'] = 'PRESENT'

            # --- 3. APPEARANCE/PROMOTION LOGIC (Revised and moved here for cleaner flow) ---
            # This logic now runs AFTER the disappearance check, which is cleaner.
            
            # Process detections for NEW objects and handle LEARNING promotion
            if boxes.id is not None:
                for box_index in range(len(boxes.id)):
                    track_id = int(boxes.id[box_index])
                    
                    # Only process if not a KNOWN STATIC OBJECT (Disappearance handled above)
                    if track_id not in STATIC_BACKGROUND_REF:
                        
                        class_id = int(boxes.cls[box_index])
                        class_name = model.names[class_id]
                        box_xyxy = boxes.xyxy[box_index]
                        x_center = int((box_xyxy[0] + box_xyxy[2]) / 2)
                        y_center = int((box_xyxy[1] + box_xyxy[3]) / 2)
                        current_position = (x_center, y_center)
                        
                        # B. NEW OBJECT (Initialize timer)
                        if track_id not in DYNAMIC_TRACKS_FOR_LEARNING:
                            message = f"🚨 NEW OBJECT: {class_name} (ID {track_id}) - Starting Promotion Timer"
                            log_anomaly_with_cooldown(message)
                            
                            DYNAMIC_TRACKS_FOR_LEARNING[track_id] = {
                                'class_name': class_name,
                                'start_time': current_time,
                                'initial_position': current_position,
                            }
                            anomaly_messages.append(f"NEW: {class_name} (ID {track_id}) - Timer ON")

                        # C. LEARNING OBJECT: Check for Promotion
                        elif track_id in DYNAMIC_TRACKS_FOR_LEARNING:
                            candidate = DYNAMIC_TRACKS_FOR_LEARNING[track_id]
                            time_elapsed = current_time - candidate['start_time']
                            
                            if time_elapsed >= PROMOTION_TIME_SEC:
                                # PROMOTION TRIGGERED! (Same logic as before)
                                STATIC_BACKGROUND_REF[track_id] = {
                                    'class_name': candidate['class_name'], 
                                    'ref_position_xy': candidate['initial_position'],
                                    'current_state': 'PRESENT',
                                    'missing_frames': 0
                                }
                                del DYNAMIC_TRACKS_FOR_LEARNING[track_id]
                                
                                promotion_msg = f"✅ PROMOTION: {candidate['class_name']} (ID {track_id}) added to Normal State."
                                print(promotion_msg)
                                anomaly_messages.append(promotion_msg)
                                
                            else:
                                # Still in learning state
                                countdown = PROMOTION_TIME_SEC - time_elapsed
                                anomaly_messages.append(f"LEARNING: {candidate['class_name']} (ID {track_id}) - {countdown:.1f}s left.")


            # --- 4. Cleanup and Display (Revised) ---
            
            # Remove dynamic tracks that have disappeared (same logic as before)
            tracks_to_delete = []
            for track_id in DYNAMIC_TRACKS_FOR_LEARNING.keys():
                if track_id not in detected_ids:
                    tracks_to_delete.append(track_id)
            
            for track_id in tracks_to_delete:
                 print(f"TRACK LOST: ID {track_id} disappeared before promotion. Deleting.")
                 del DYNAMIC_TRACKS_FOR_LEARNING[track_id]
                
            # Display the anomaly messages on the video frame
            for i, msg in enumerate(anomaly_messages):
                color = (0, 0, 255) 
                cv2.putText(frame_with_detections, msg, (10, 50 + i * 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2) 
            
            cv2.imshow("Anomaly Detector", frame_with_detections)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")

if __name__ == "__main__":
    run_anomaly_pipeline()