import cv2
import mss
import os
import time
import numpy as np
from ultralytics import YOLO
import google.generativeai as genai

# Configure Gemini API (Replace with your API key)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load the trained YOLO model
model = YOLO(r"C:\Users\ASDF\Documents\Personnel\chessWinner\other_models\bestKaggle.pt")

# Load the fine-tuned Gemini model
gemini_model = genai.GenerativeModel("tunedModels/chess-tactics-7133")

# Screen capture parameters
monitor = {"top": 155, "left": 170, "width": 650, "height": 610}

# Configuration
CONFIDENCE_THRESHOLD = 0.95
IOU_THRESHOLD = 0.45
FRAME_SKIP = 2  # Process every 3rd frame to reduce CPU load
COOLDOWN_DELAY = 5  # Seconds between API calls

piece_Fen_map = {
    'B': 'B', 'K': 'K', 'N': 'N', 'P': 'P', 'Q': 'Q', 'R': 'R',
    'b': 'b', 'k': 'k', 'n': 'n', 'p': 'p', 'q': 'q', 'r': 'r'
}

class ChessDetector:
    def __init__(self):
        self.sct = mss.mss()
        self.current_move = ""
        self.gemini_model = gemini_model
        self.last_api_call = 0
        
        # Detection control variables
        self.detection_active = False
        self.detection_start_time = 0
        self.trigger_detection = False
        self.aggregated_board = None
        
        # Button parameters
        self.button_pos = (10, 550)  # (x, y)
        self.button_size = (120, 30)  # (width, height)
        
        # Initialize window and mouse callback
        cv2.namedWindow("Chess Detection")
        cv2.setMouseCallback("Chess Detection", self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            btn_x, btn_y = self.button_pos
            btn_w, btn_h = self.button_size
            if (btn_x <= x <= btn_x + btn_w) and (btn_y <= y <= btn_y + btn_h):
                self.trigger_detection = True

    def update_aggregated_board(self, results):
        if not results or not results[0].boxes:
            return

        img_w, img_h = monitor['width'], monitor['height']
        square_w, square_h = img_w/8, img_h/8

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_idx = int(box.cls[0])
                class_name = model.names[cls_idx]

                if class_name not in piece_Fen_map:
                    continue

                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                col = min(7, int(center_x // square_w))
                row = min(7, int(center_y // square_h))

                if conf > self.aggregated_board[row][col]['conf']:
                    self.aggregated_board[row][col] = {
                        'piece': piece_Fen_map[class_name],
                        'conf': conf
                    }

    def get_fen_from_aggregated_board(self):
        fen_rows = []
        for row in self.aggregated_board:
            fen_row = []
            empty_count = 0
            
            for square in row:
                if square['piece']:
                    if empty_count > 0:
                        fen_row.append(str(empty_count))
                        empty_count = 0
                    fen_row.append(square['piece'])
                else:
                    empty_count += 1
                    
            if empty_count > 0:
                fen_row.append(str(empty_count))
            fen_rows.append(''.join(fen_row))

        return "/".join(fen_rows) + " w KQkq - 0 1"

    def get_optimal_move(self, fen_position):
        """Query Gemini model for optimal move"""
        try:
            prompt = f"Given the chess position {fen_position}, what is the optimal next move? Respond only with the move in standard chess notation."
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error getting move from Gemini: {e}")
            return ""

    def draw_button(self, img):
        btn_x, btn_y = self.button_pos
        btn_w, btn_h = self.button_size
        # Draw button background
        cv2.rectangle(img, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), (0, 255, 0), -1)
        # Draw button text
        cv2.putText(img, "Detect Now", (btn_x + 10, btn_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    def run(self):
        try:
            while True:
                # Capture screen
                screenshot = self.sct.grab(monitor)
                img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGRA2BGR)
                display_img = img.copy()

                # Handle detection trigger
                if self.trigger_detection and not self.detection_active:
                    self.detection_active = True
                    self.detection_start_time = time.time()
                    self.aggregated_board = [[{'piece': '', 'conf': 0} for _ in range(8)] for _ in range(8)]
                    self.trigger_detection = False
                    self.current_move = ""  # Clear previous move

                # Process detection if active
                if self.detection_active:
                    current_time = time.time()
                    elapsed = current_time - self.detection_start_time

                    if elapsed < 1.0:  # Detection period
                        # Process every FRAME_SKIP-th frame
                        if int(elapsed * 10) % FRAME_SKIP == 0:
                            results = model(img, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)
                            self.update_aggregated_board(results)
                        # Show countdown
                        cv2.putText(display_img, f"Detecting: {1 - int(elapsed)}s", 
                                   (250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    else:  # Detection period ended
                        self.detection_active = False
                        fen = self.get_fen_from_aggregated_board()
                        print(f"Final FEN: {fen}")
                        
                        if time.time() - self.last_api_call > COOLDOWN_DELAY:
                            self.last_api_call = time.time()
                            self.current_move = self.get_optimal_move(fen.split(' ')[0])
                            print(f"Suggested move: {self.current_move}")

                # Draw interface elements
                self.draw_button(display_img)

                # Display FEN and move if available
                if not self.detection_active and self.aggregated_board:
                    y_start = 30
                    fen_parts = self.get_fen_from_aggregated_board().split(' ')[0].split('/')
                    for i, part in enumerate(fen_parts):
                        cv2.putText(display_img, part, (10, y_start + i*20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    if self.current_move:
                        move_text = f"Suggested Move: {self.current_move}"
                        cv2.putText(display_img, move_text, (10, y_start + len(fen_parts)*20 + 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow("Chess Detection", display_img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            cv2.destroyAllWindows()
            self.sct.close()

if __name__ == "__main__":
    detector = ChessDetector()
    detector.run()