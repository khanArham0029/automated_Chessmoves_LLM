import cv2
import os
from dotenv import load_dotenv
import mss
import numpy as np
import time
import threading
import tkinter as tk
from ultralytics import YOLO
import google.generativeai as genai

# Load the trained YOLO model
model = YOLO(r"C:\Users\ASDF\Documents\Personnel\chessWinner\other_models\bestKaggle.pt")

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Gemini AI Model
genai_model = genai.GenerativeModel("tunedModels/chess-tactics-7133")

# Screen capture parameters
monitor = {"top": 155, "left": 170, "width": 650, "height": 610}

# Configuration
CONFIDENCE_THRESHOLD = 0.92
IOU_THRESHOLD = 0.45
FRAME_SKIP = 2  # Process every 3rd frame to reduce CPU load
API_COOLDOWN = 15  # Seconds between API calls

piece_Fen_map = {
    'B': 'B', 'K': 'K', 'N': 'N', 'P': 'P', 'Q': 'Q', 'R': 'R',
    'b': 'b', 'k': 'k', 'n': 'n', 'p': 'p', 'q': 'q', 'r': 'r'
}

class ChessDetector:
    def __init__(self):
        self.frame_count = 0
        self.sct = mss.mss()
        self.previous_fen = "8/8/8/8/8/8/8/8 w KQkq - 0 1"
        self.last_api_call = 0  # Track last API call time

        # Initialize Tkinter for popups
        self.root = tk.Tk()
        self.root.withdraw()

        # Start UI thread for Tkinter
        self.start_ui_thread()

    def start_ui_thread(self):
        ui_thread = threading.Thread(target=self.root.mainloop, daemon=True)
        ui_thread.start()

    def get_fen_from_detections(self, results):
        if not results or not results[0].boxes:
            return self.previous_fen

        board = [[{'piece': '', 'conf': 0} for _ in range(8)] for _ in range(8)]
        img_w, img_h = monitor['width'], monitor['height']
        square_w, square_h = img_w / 8, img_h / 8

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_idx = int(box.cls[0])
                class_name = model.names[cls_idx]

                if class_name not in piece_Fen_map:
                    continue

                # Calculate center position
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                col = min(7, int(center_x // square_w))
                row = min(7, int(center_y // square_h))

                # Track highest confidence piece per square
                if conf > board[row][col]['conf']:
                    board[row][col] = {
                        'piece': piece_Fen_map[class_name],
                        'conf': conf
                    }

        # Build FEN string
        fen_rows = []
        for row in board:
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

        fen = "/".join(fen_rows) + " w KQkq - 0 1"

        # Only update if FEN has changed
        if fen != self.previous_fen:
            print(f"[DEBUG] FEN updated: {fen}")
            self.previous_fen = fen
            self.get_best_move(fen)  # Call Gemini API

        return fen

    def get_best_move(self, fen):
        """Call Gemini AI to get the best move and show a popup."""
        current_time = time.time()
        if current_time - self.last_api_call < API_COOLDOWN:
            print("[DEBUG] API call skipped due to cooldown.")
            return

        print(f"[DEBUG] Calling Gemini API for FEN: {fen}")
        try:
            response = genai_model.generate_content(f"Given this chess position {fen}, what is the best move?")
            best_move = response.text.strip()
            self.last_api_call = time.time()
            self.show_popup(best_move)
        except Exception as e:
            print(f"[ERROR] Gemini API Error: {e}")

    def show_popup(self, move):
        """Display the best move in a Tkinter popup."""
        popup = tk.Toplevel(self.root)
        popup.title("Best Move Suggested")
        label = tk.Label(popup, text=f"Suggested Move: {move}", font=("Arial", 14, "bold"))
        label.pack(padx=20, pady=20)
        popup.after(5000, popup.destroy)  # Close popup after 5 seconds

    def run(self):
        try:
            while True:
                self.frame_count += 1
                if self.frame_count % FRAME_SKIP != 0:
                    continue

                # Capture screen
                screenshot = self.sct.grab(monitor)
                img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGRA2BGR)

                # Run inference with optimized parameters
                results = model(img, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)

                # Generate FEN
                fen = self.get_fen_from_detections(results)
                #print(f"FEN: {fen}")

                # Visualization
                display_img = img.copy()
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_idx = int(box.cls[0])
                        class_name = model.names[cls_idx]

                        # Color coding: blue for white, red for black
                        color = (255, 0, 0) if class_name.isupper() else (0, 0, 255)
                        cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
                        label = f"{class_name} {float(box.conf[0]):.2f}"
                        cv2.putText(display_img, label, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Display FEN (wrapped for better visibility)
                y_start = 30
                for i, part in enumerate(fen.split(' ')[0].split('/')):
                    cv2.putText(display_img, part, (10, y_start + i * 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.imshow("Chess Detection", display_img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            cv2.destroyAllWindows()
            self.sct.close()

if __name__ == "__main__":
    detector = ChessDetector()
    detector.run()
