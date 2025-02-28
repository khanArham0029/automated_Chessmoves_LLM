import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.environ['GEMINI_API_KEY'])

# Load your fine-tuned model
model_id = "tunedModels/chess-tactics-7133"  # Use your actual tuned model ID
model = genai.GenerativeModel(model_id)

# Test a FEN position
fen_test = "rnbqkb1r/pppppppp/7n/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1"  # Example FEN

# Query the model
response = model.generate_content(f"Given this chess position {fen_test}, what is the best move?")

# Print the response
print(response.text)
