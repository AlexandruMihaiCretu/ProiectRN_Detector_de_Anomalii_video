import cv2
from ultralytics import YOLO
import numpy as np
import time
import csv
from datetime import datetime
import os
from pathlib import Path

# --- CONFIGURATION ---
VIDEO_SOURCE = 0 
MODEL_WEIGHTS = 'yolov8n.pt' 
LEARNING_FRAMES = 150 
NOTIFICATION_COOLDOWN = 5.0
PROMOTION_TIME_SEC = 10.0
MISSING_FRAME_THRESHOLD = 90
VETTING_THRESHOLD_FRAMES = 5
VETTING_TRACKS = {}
# ---------------------

# --- LOGGER PREP ---
track_history = {}
frame_count = 0


# --- GLOBAL REFERENCE DICTIONARY ---
STATIC_BACKGROUND_REF = {} 
DYNAMIC_TRACKS_FOR_LEARNING = {}

LAST_NOTIFICATION_TIME = 0.0

SCRIPT_PATH = Path(__file__).resolve()
DATA_DIR_RAW = SCRIPT_PATH.parent.parent / "data" / "raw"
DATA_DIR_RAW.mkdir(parents=True, exist_ok=True)
LOG_FILE = DATA_DIR_RAW / "behavioral_logs.csv"

with open(LOG_FILE, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'timestamp', 'obj_id', 'class_id', 'confidence', 
        'age_frames', 'velocity_px', 'aspect_ratio'
    ])

print(f"Started fresh: {LOG_FILE} has been cleared.")

log_file_handle = open(LOG_FILE, mode='a', newline='')
log_writer = csv.writer(log_file_handle)

# Salvam logurile in CSV cu ID-ul, clasa (ca ID), confidence-ul, varsta detectionului, viteza si aspect ratio-ul
def save_behavioral_log(obj_id, cls, conf, age, velocity, aspect_ratio):
    log_writer.writerow([
        time.time(), obj_id, int(cls), round(float(conf), 4), 
        age, round(velocity, 2), round(aspect_ratio, 2)
    ])  

def process_and_log_behavior(obj_id, cls, conf, box):
    global frame_count, track_history
    # x, y are center coordinates; w, h are width and height
    x, y, w, h = box
    
    # A. Age Calculation
    if obj_id not in track_history:
        track_history[obj_id] = {"last_pos": (x, y), "start_frame": frame_count}
    
    age = frame_count - track_history[obj_id]["start_frame"]
    
    # B. Velocity Calculation (Euclidean distance from last position)
    last_x, last_y = track_history[obj_id]["last_pos"]
    velocity = ((x - last_x)**2 + (y - last_y)**2)**0.5
    
    # Update history for next frame
    track_history[obj_id]["last_pos"] = (x, y)
    
    # C. Aspect Ratio
    aspect_ratio = w / h if h != 0 else 0

    # D. Write to CSV
    save_behavioral_log(obj_id, cls, conf, age, velocity, aspect_ratio)



def learn_static_background(model, cap):
    global STATIC_BACKGROUND_REF, frame_count 
    temp_tracks = {} # This will now be populated
    LEARNING_THRESHOLD = LEARNING_FRAMES * 0.90 
    
    for i in range(LEARNING_FRAMES):
        success, frame = cap.read()
        if not success: break
        
        results = model.track(frame, persist=True)
        frame_count += 1

        if results[0].boxes.id is not None:
            boxes_xywh = results[0].boxes.xywh.cpu().numpy()
            ids = results[0].boxes.id.int().cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            clss = results[0].boxes.cls.int().cpu().numpy()

            # --- FIX: Record the presence of these IDs ---
            for box, obj_id, cls in zip(boxes_xywh, ids, clss):
                if obj_id not in temp_tracks:
                    temp_tracks[obj_id] = {
                        'count': 0, 
                        'class_name': model.names[cls], 
                        'ref_position_xy': (int(box[0]), int(box[1]))
                    }
                temp_tracks[obj_id]['count'] += 1

            if frame_count % 2 == 0:
                for idx in range(len(ids)):
                    process_and_log_behavior(ids[idx], clss[idx], confs[idx], boxes_xywh[idx])
        
        cv2.putText(frame, f"LEARNING: Frame {i+1}/{LEARNING_FRAMES}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Anomaly Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break

    for track_id, data in temp_tracks.items():
        if data['count'] >= LEARNING_THRESHOLD:
            STATIC_BACKGROUND_REF[track_id] = {
                'class_name': data['class_name'], 
                'ref_position_xy': data['ref_position_xy'],
                'current_state': 'PRESENT',
                'missing_frames': 0
            }

def run_anomaly_pipeline():
    """
    Main function to execute the anomaly detection pipeline, handling:
    1. Initial Static Learning.
    2. Real-time Appearance & Disappearance Anomaly Detection.
    3. Adaptive Object Promotion.
    4. Cooldown-controlled logging.
    """
    global STATIC_BACKGROUND_REF, DYNAMIC_TRACKS_FOR_LEARNING
    
    print(f"Loading YOLO model: {MODEL_WEIGHTS}...")
    model = YOLO(MODEL_WEIGHTS) 
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {VIDEO_SOURCE}. Exiting.")
        return

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
        
        clean_frame = frame.copy()

        if success:
            
            current_time = time.time()
            results = model.track(frame, persist=True, verbose=False)
            frame_with_detections = results[0].plot()
            boxes = results[0].boxes.cpu().numpy()
            
            anomaly_messages = [] 
            detected_ids = set() 
            
            if boxes.id is not None:
                detected_ids.update(int(tid) for tid in boxes.id)

            for track_id, data in STATIC_BACKGROUND_REF.items():
                
                if track_id not in detected_ids:
                    # Object is MISSING in this frame
                    data['missing_frames'] += 1
                    
                    # Only check for anomaly if we haven't reported it yet
                    if data['current_state'] != 'REPORTED_MISSING':
                        
                        # Check if the missing duration exceeds the threshold
                        if data['missing_frames'] >= MISSING_FRAME_THRESHOLD:
                            # --- ANOMALY TRIGGERED (FIRST TIME) ---
                            message = f"DISAPPEARANCE ANOMALY: {data['class_name']} (ID {track_id}) is missing!"
                            
                            print(message)

                            # CRITICAL: Update state so we don't report this again
                            data['current_state'] = 'REPORTED_MISSING'
                        
                        else:
                            # Not yet threshold, just mark as generally missing
                            data['current_state'] = 'MISSING'
                    
                    else:
                        pass
                        
                else:
                    # Object IS PRESENT in this frame
                    
                    # If it was previously reported missing, announce its return
                    if data['current_state'] == 'REPORTED_MISSING':
                        notification = f"REAPPEARANCE: {data['class_name']} (ID {track_id}) has returned."
                        print(notification)

                    # Reset counters and state completely
                    data['missing_frames'] = 0
                    data['current_state'] = 'PRESENT'

            # Process detections for NEW objects and handle LEARNING promotion
            if boxes.id is not None:
                for box_index in range(len(boxes.id)):
                    track_id = int(boxes.id[box_index])
                    detected_ids.add(track_id)
                    
                    class_id = int(boxes.cls[box_index])
                    class_name = model.names[class_id]
                    box_xyxy = boxes.xyxy[box_index]
                    x_center = int((box_xyxy[0] + box_xyxy[2]) / 2)
                    y_center = int((box_xyxy[1] + box_xyxy[3]) / 2)
                    current_position = (x_center, y_center)

                    global frame_count
                    frame_count += 1

                    if results[0].boxes.id is not None:
                        m_boxes = results[0].boxes.xywh.cpu().numpy()
                        m_ids = results[0].boxes.id.int().cpu().numpy()
                        m_confs = results[0].boxes.conf.cpu().numpy()
                        m_clss = results[0].boxes.cls.int().cpu().numpy()
                        if frame_count % 2 == 0:
                            for i in range(len(m_ids)):
                                process_and_log_behavior(m_ids[i], m_clss[i], m_confs[i], m_boxes[i])

                    # A. KNOWN STATIC OBJECT
                    if track_id in STATIC_BACKGROUND_REF:
                        STATIC_BACKGROUND_REF[track_id]['current_state'] = 'PRESENT'

                    # B. NEW OBJECT CHECK (With Reappearance and Promotion Prevention)
                    elif track_id not in DYNAMIC_TRACKS_FOR_LEARNING:
                        found_match = False
                        matched_static_id = None

                        # Look for a missing object of the same class
                        for static_id, data in STATIC_BACKGROUND_REF.items():
                            if data['current_state'] in ['MISSING', 'REPORTED_MISSING'] and data['class_name'] == class_name:
                                matched_static_id = static_id
                                found_match = True
                                break
                        
                        if found_match:
                            print(f"REAPPEARANCE: {class_name} is back!")
                            
                            # Move the data to the new current Track ID
                            STATIC_BACKGROUND_REF[track_id] = STATIC_BACKGROUND_REF.pop(matched_static_id)
                            
                            # Update the state and position
                            STATIC_BACKGROUND_REF[track_id]['current_state'] = 'PRESENT'
                            STATIC_BACKGROUND_REF[track_id]['missing_frames'] = 0
                            STATIC_BACKGROUND_REF[track_id]['ref_position_xy'] = current_position
                        
                        if not found_match:
                            VETTING_TRACKS[track_id] = VETTING_TRACKS.get(track_id, 0) + 1

                            if VETTING_TRACKS[track_id] >= VETTING_THRESHOLD_FRAMES:
                                print(f"NEW OBJECT VERIFIED: {class_name} (ID {track_id})")
                                
                                DYNAMIC_TRACKS_FOR_LEARNING[track_id] = {
                                    'class_name': class_name,
                                    'start_time': current_time,
                                    'initial_position': current_position,
                                }

                                del VETTING_TRACKS[track_id]
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
                                msg = f"PROMOTION: {candidate['class_name']} (ID {track_id}) is now Static"
                                print(msg)
                                anomaly_messages.append(msg)
                                
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
                 # Logic: We silently delete the track. No print() needed.
                 del DYNAMIC_TRACKS_FOR_LEARNING[track_id]
            # Display the anomaly messages on the video frame
            for i, msg in enumerate(anomaly_messages):
                color = (0, 0, 255) 
                cv2.putText(frame_with_detections, msg, (10, 50 + i * 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2) 
            
            update_status_dashboard(STATIC_BACKGROUND_REF) 
            #DashBoard

            cv2.imshow("Anomaly Detector", frame_with_detections)
            cv2.imshow("Live Raw Feed", clean_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")

def update_status_dashboard(static_ref):
    # 1. Create a black background (Height depends on number of objects)
    height = max(200, len(static_ref) * 40 + 60)
    dashboard = np.zeros((height, 400, 3), dtype=np.uint8)
    
    # 2. Add Header
    cv2.putText(dashboard, "SYSTEM STATUS: STATIC OBJECTS", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.line(dashboard, (10, 40), (380, 40), (255, 255, 255), 1)
    
    # 3. List the Objects
    y_pos = 70
    for track_id, data in static_ref.items():
        name = data['class_name']
        state = data['current_state']
        
        # Color code: Green for Present, Red for Missing
        color = (0, 255, 0) if state == 'PRESENT' else (0, 0, 255)
        
        status_text = f"ID {track_id}: {name} -> {state}"
        cv2.putText(dashboard, status_text, (20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_pos += 40
        
    # 4. Display the window
    cv2.imshow("Object Status Dashboard", dashboard)

if __name__ == "__main__":
    run_anomaly_pipeline()