import os
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.environ['GEMINI_API_KEY'])

# Load the fine-tuned model
model_id = "tunedModels/chess-tactics-7133"  # Use your actual tuned model ID
model = genai.GenerativeModel(model_id)

# Load the dataset
df = pd.read_csv("datasets/Tactics/tactic_evals.csv")
df = df.dropna(subset=["FEN", "Move"])  # Remove rows with missing values

# Limit number of samples for faster testing (optional)
num_samples = 50  # Change this as needed
df = df.sample(num_samples, random_state=42)

# Function to query the fine-tuned model
def get_predicted_move(fen):
    response = model.generate_content(f"Given this chess position {fen}, what is the best move?")
    return response.text.strip()  # Clean output

# Evaluate model
correct_predictions = 0
total_samples = len(df)

for index, row in df.iterrows():
    fen = row["FEN"]
    actual_move = row["Move"].strip()  # The correct move from dataset
    
    predicted_move = get_predicted_move(fen)

    print(f"FEN: {fen}")
    print(f"Actual Move: {actual_move}, Predicted Move: {predicted_move}")
    
    if predicted_move == actual_move:
        correct_predictions += 1  # Count correct matches

# Compute accuracy
accuracy = (correct_predictions / total_samples) * 100
print(f"\nModel Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_samples} correct)")
