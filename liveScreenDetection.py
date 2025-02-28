import cv2
import mss
import os
import time
import numpy as np
import pyttsx3
import chess
import chess.pgn
from ultralytics import YOLO
import google.generativeai as genai

# Configure Gemini API
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
DEBOUNCE_DELAY = 2
COOLDOWN_DELAY = 5

# Piece names mapping
piece_names = {'K': "King", 'Q': "Queen", 'R': "Rook", 'B': "Bishop", 'N': "Knight", 'P': "Pawn"}

class ChessDetector:
    def __init__(self):
        self.sct = mss.mss()
        self.previous_fen = "8/8/8/8/8/8/8/8 w KQkq - 0 1"
        self.last_api_call = 0
        self.current_move = ""
        self.detect_now = False
        self.detection_start_time = None

        # Initialize text-to-speech engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 160)  # Adjust speed (default ~200)
        self.tts_engine.setProperty('volume', 1.0)  # Set volume (0.0 to 1.0)

    def speak_move(self, move_text):
        """Convert text to speech"""
        print(f"Speaking: {move_text}")
        self.tts_engine.say(move_text)
        self.tts_engine.runAndWait()

    def get_optimal_move(self, fen_position):
        """Query Gemini model for optimal move and speak it aloud"""
        try:
            prompt = f"Given the chess position {fen_position}, what is the optimal next move? Respond only with the move in standard chess notation."
            response = gemini_model.generate_content(prompt)
            move = response.text.strip()

            # Convert to human-readable text
            move_text = self.translate_move_to_text(fen_position, move)
            self.speak_move(move_text)  # Speak the move aloud
            return move_text
        except Exception as e:
            print(f"Error getting move from Gemini: {e}")
            return ""

    def translate_move_to_text(self, fen, move):
        """Convert UCI move (e2e4) to human-readable format."""
        try:
            board = chess.Board(fen)
            uci_move = chess.Move.from_uci(move)

            if uci_move not in board.legal_moves:
                print(f"Received move from Gemini: {move}")
                return f"Invalid move: {move}"

            board.push(uci_move)  # Apply move to board

            piece = board.piece_at(uci_move.to_square)
            piece_name = piece_names.get(piece.symbol().upper(), "Piece")
            move_text = f"Move {piece_name} to {chess.square_name(uci_move.to_square)}"

            return move_text
        except Exception as e:
            print(f"Received move from Gemini: {move}")

            return f"Error translating move: {move}"

    def run(self):
        try:
            while True:
                screenshot = self.sct.grab(monitor)
                img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGRA2BGR)

                cv2.putText(img, "Press 'D' to Detect Now", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                if self.detect_now:
                    if self.detection_start_time is None:
                        self.detection_start_time = time.time()

                    elapsed_time = time.time() - self.detection_start_time
                    if elapsed_time > 1:
                        self.detect_now = False
                        self.detection_start_time = None
                    else:
                        # Simulate FEN detection (Replace with actual model inference)
                        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"  # Example FEN
                        print(f"FEN: {fen}")

                        current_time = time.time()
                        if current_time - self.last_api_call > COOLDOWN_DELAY:
                            self.last_api_call = current_time
                            self.current_move = self.get_optimal_move(fen)
                            print(f"Suggested move: {self.current_move}")

                if self.current_move:
                    move_text = f"Suggested Move: {self.current_move}"
                    cv2.putText(img, move_text, (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Chess Detection", img)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('d'):
                    self.detect_now = True
                elif key == ord('q'):
                    break

        finally:
            cv2.destroyAllWindows()
            self.sct.close()

if __name__ == "__main__":
    detector = ChessDetector()
    detector.run()
