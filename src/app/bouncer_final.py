import cv2
from ultralytics import YOLO
import numpy as np
import joblib
import pandas as pd
import time
import collections
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
BRAIN_PATH = os.path.join(project_root, "models", "bouncer_model.pkl")
YOLO_PATH = os.path.join(project_root, "yolov8n.pt")
BOUNCER_BRAIN = joblib.load(BRAIN_PATH)
model = YOLO(YOLO_PATH)

try:
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                                 QHBoxLayout, QWidget, QListWidget, QTableWidget, 
                                 QTableWidgetItem, QHeaderView)
    from PyQt6.QtCore import QThread, pyqtSignal, Qt
    from PyQt6.QtGui import QImage, QPixmap
except ImportError:
    print("PyQt6 not found. Run 'pip install PyQt6'")
    sys.exit()


class SecurityBouncer:
    def __init__(self, model_path="yolov8n.pt", brain_path="models/bouncer_model.pkl"):
        self.model = YOLO(model_path)
        self.bouncer_brain = joblib.load(brain_path)
        
        # --- Config ---
        self.PROMOTION_TIME = 5.0  
        self.MISSING_TIMEOUT = 3.0 
        self.IDLE_CLEANUP = 60.0   
        self.DISTANCE_THRESHOLD = 80
        self.INTRUDER_TIME_LIMIT = 3.0
        self.intruder_notifications = {}
        self.NOTIFICATION_INTERVAL = 60

        # --- State ---
        self.static_inventory = {}   
        self.probation_zone = {}     
        self.track_history = {}      
        self.class_voting = {}       
        self.intruder_tracks = {}    
        self.event_logs = collections.deque(maxlen=8)
        self.active_alert = False

    def log_event(self, message, color=(255, 255, 255)):
        timestamp = time.strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {message}"
        if hasattr(self, 'current_frame_events'):
            self.current_frame_events.append((formatted_msg, color))
        self.event_logs.appendleft((formatted_msg, color))

    def get_consensus_class(self, obj_id, raw_class_name):
        if obj_id not in self.class_voting:
            self.class_voting[obj_id] = collections.deque(maxlen=30)
        self.class_voting[obj_id].append(raw_class_name)
        return max(set(self.class_voting[obj_id]), key=list(self.class_voting[obj_id]).count)

    def analyze_behavior(self, obj_id, cls_id, conf, box):
        x, y, w, h = box
        now = time.time()
        if obj_id not in self.track_history:
            self.track_history[obj_id] = {"last_pos": (x, y), "start_time": now, "last_seen": now}
        
        hist = self.track_history[obj_id]
        velocity = ((x - hist["last_pos"][0])**2 + (y - hist["last_pos"][1])**2)**0.5 
        hist.update({"last_pos": (x, y), "last_seen": now})
        
        features = pd.DataFrame([[
            int(cls_id), float(conf), (now - hist["start_time"]), velocity, (w / h if h != 0 else 0)
        ]], columns=['class_id', 'confidence', 'age_frames', 'velocity_px', 'aspect_ratio'])
        
        return self.bouncer_brain.predict(features)[0] == 1

    def attempt_id_adoption(self, current_id, current_pos, consensus_class):
        for old_id, data in list(self.static_inventory.items()):
            if data['class_name'] == consensus_class and data['current_state'] == 'MISSING':
                self.static_inventory[current_id] = self.static_inventory.pop(old_id)
                
                self.static_inventory[current_id].update({
                    'last_seen_time': time.time()
                })
                
                dist = np.linalg.norm(np.array(current_pos) - np.array(data['ref_pos']))
                
                if dist > self.DISTANCE_THRESHOLD:
                    self.static_inventory[current_id]['current_state'] = 'MOVED'
                    self.log_event(f"RECOVERED: {consensus_class} (Moved from original spot)", (0, 165, 255))
                else:
                    self.static_inventory[current_id]['current_state'] = 'PRESENT'
                    self.log_event(f"RECOVERED: {consensus_class} (Back in place)", (0, 255, 0))
                
                return True
        return False
    def cleanup_memory(self):
        now = time.time()
        for obj_id in list(self.track_history.keys()):
            if now - self.track_history[obj_id]["last_seen"] > self.IDLE_CLEANUP:
                for storage in [self.track_history, self.class_voting, self.probation_zone, self.intruder_tracks]:
                    storage.pop(obj_id, None)

    def process_frame(self, frame):
        now = time.time()
        self.current_frame_events = []
        results = self.model.track(frame, persist=True, verbose=False)
        active_this_frame = []
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            ids = results[0].boxes.id.int().cpu().numpy()
            clss = results[0].boxes.cls.int().cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            for i, obj_id in enumerate(ids):
                active_this_frame.append(obj_id)
                x, y, w, h = boxes[i]
                consensus = self.get_consensus_class(obj_id, self.model.names[clss[i]])
                is_real = self.analyze_behavior(obj_id, clss[i], confs[i], boxes[i])

                if not is_real:
                    cv2.rectangle(frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (100, 100, 100), 1)


                if obj_id not in self.static_inventory and consensus != 'person':
                    self.attempt_id_adoption(obj_id, (x, y), consensus)

                if consensus == 'person':
                            if obj_id not in self.intruder_tracks:
                                self.intruder_tracks[obj_id] = now
                            
                            elapsed_stay = now - self.intruder_tracks[obj_id]
                            if elapsed_stay >= self.INTRUDER_TIME_LIMIT:
                                self.active_alert = True
                                cv2.rectangle(frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (0, 0, 255), 3)

                                last_notify = self.intruder_notifications.get(obj_id, 0)
                                if now - last_notify >= self.NOTIFICATION_INTERVAL:
                                    status = "FIRST DETECTED" if last_notify == 0 else "PERSISTING"
                                    self.log_event(f"INTRUDER {status}: Person ID {obj_id}", (0, 0, 255))
                                    self.intruder_notifications[obj_id] = now

                elif obj_id in self.static_inventory:
                    inv = self.static_inventory[obj_id]
                    dist = np.linalg.norm(np.array((x, y)) - np.array(inv['ref_pos']))
                    
                    if inv['current_state'] == 'MOVED' and dist <= self.DISTANCE_THRESHOLD:
                        inv['current_state'] = 'PRESENT'
                        inv['last_seen_time'] = now
                        self.log_event(f"RESTORED: {inv['class_name']} returned to original spot", (0, 255, 127))
                        color = (0, 255, 0)
                    
                    elif dist > self.DISTANCE_THRESHOLD:
                        if inv['current_state'] != 'MOVED':
                            self.log_event(f"MOVED: {inv['class_name']} left its spot", (0, 165, 255))
                        
                        inv['current_state'] = 'MOVED'
                        color = (0, 165, 255)
                    
                    else:
                        inv['current_state'] = 'PRESENT'
                        inv['last_seen_time'] = now
                        color = (0, 255, 0)

                    cv2.rectangle(frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), color, 2)

                elif is_real:
                    if obj_id not in self.probation_zone:
                        self.probation_zone[obj_id] = {'start_time': now, 'pos': (x,y)}
                    
                    elapsed = now - self.probation_zone[obj_id]['start_time']
                    cv2.rectangle(frame, (int(x-w/2), int(y-h/2-10)), (int(x-w/2 + (w*min(elapsed/self.PROMOTION_TIME, 1))), int(y-h/2-5)), (0, 255, 255), -1)
                    
                    if elapsed >= self.PROMOTION_TIME:
                        self.static_inventory[obj_id] = {
                            'class_name': consensus, 'ref_pos': (int(x), int(y)),
                            'current_state': 'PRESENT', 'last_seen_time': now
                        }
                        self.log_event(f"LEARNED: {consensus}", (255, 255, 0))
                        del self.probation_zone[obj_id]

        for sid, sdata in list(self.static_inventory.items()):
            if sid not in active_this_frame and sdata['current_state'] != 'MISSING':
                if now - sdata['last_seen_time'] > self.MISSING_TIMEOUT:
                    sdata['current_state'] = 'MISSING'
                    self.log_event(f"STOLEN: {sdata['class_name']}", (0, 0, 255))
        
        if not any(id in active_this_frame for id in self.intruder_tracks):
            self.active_alert = False

        self.cleanup_memory()
        return frame, self.current_frame_events

    def draw_dashboard(self):
        dash = np.zeros((700, 400, 3), dtype=np.uint8)
        header_color = (0, 0, 255) if self.active_alert else (0, 255, 0)
        
        cv2.putText(dash, "BOUNCER STATUS", (20, 40), 1, 1.5, header_color, 2)
        cv2.putText(dash, f"STATE: {'INTRUDER' if self.active_alert else 'SECURE'}", (20, 75), 1, 1.2, header_color, 1)
        
        y = 130
        cv2.putText(dash, "INVENTORY:", (20, 115), 1, 1, (150, 150, 150), 1)
        for sid, data in self.static_inventory.items():
            color = (0, 255, 0) if data['current_state'] == "PRESENT" else (0, 0, 255)
            if data['current_state'] == 'MOVED': color = (0, 165, 255)
            cv2.putText(dash, f"{data['class_name'].upper()}: {data['current_state']}", (20, y), 1, 1, color, 1)
            y += 25

        y = 500
        cv2.putText(dash, "EVENT LOG:", (20, 480), 1, 1, (150, 150, 150), 1)
        for msg, color in self.event_logs:
            cv2.putText(dash, msg, (20, y), 1, 0.8, color, 1)
            y += 20
        return dash

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_data_signal = pyqtSignal(dict, list, bool)

    def run(self):
        bouncer = SecurityBouncer()
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                processed_frame, events = bouncer.process_frame(frame)
                self.change_pixmap_signal.emit(processed_frame)
                self.update_data_signal.emit(bouncer.static_inventory, events, bouncer.active_alert)
            time.sleep(0.01)

class BouncerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bouncer AI")
        self.setMinimumSize(1000, 600)
        self.setStyleSheet("background-color: #121212; color: white;")

        layout = QHBoxLayout()
        self.video_label = QLabel("Initializing...")
        self.video_label.setFixedSize(640, 480)
        layout.addWidget(self.video_label)

        side_panel = QVBoxLayout()
        self.status_header = QLabel("SYSTEM SECURE")
        self.status_header.setStyleSheet("font-size: 20px; font-weight: bold; color: green;")
        side_panel.addWidget(self.status_header)

        self.inv_table = QTableWidget(0, 2)
        self.inv_table.setHorizontalHeaderLabels(["Item", "Status"])
        self.inv_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        side_panel.addWidget(self.inv_table)

        self.log_widget = QListWidget()
        side_panel.addWidget(self.log_widget)

        layout.addLayout(side_panel)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_data_signal.connect(self.update_dashboard)
        self.thread.start()

    def update_image(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        q_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def update_dashboard(self, inventory, events, alert):
        self.status_header.setText("⚠️ INTRUDER" if alert else "SYSTEM SECURE")
        self.status_header.setStyleSheet(f"color: {'red' if alert else 'green'}; font-size: 20px;")
        
        self.inv_table.setRowCount(len(inventory))
        for i, (oid, data) in enumerate(inventory.items()):
            self.inv_table.setItem(i, 0, QTableWidgetItem(data['class_name']))
            self.inv_table.setItem(i, 1, QTableWidgetItem(data['current_state']))

        for msg, color in events:
            self.log_widget.insertItem(0, msg)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BouncerApp()
    window.show()
    sys.exit(app.exec())