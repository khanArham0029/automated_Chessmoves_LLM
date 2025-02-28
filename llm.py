import os
import random
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
import time
#from google import genai

# Load API key from .env file
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# List available tuned models (optional)
for i, m in zip(range(5), genai.list_tuned_models()):
    print(m.name)


 #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# # Load the CSV file (Exclude Evaluation column)
# df = pd.read_csv("datasets/Tactics/tactic_evals.csv")
# df = df.dropna(subset=["FEN", "Move"])  # Remove rows with missing values
# df = df[["FEN", "Move"]]  # Keep only relevant columns

# # Convert data into the format required for fine-tuning
# training_data = [
#     {"text_input": row["FEN"], "output": row["Move"]}
#     for _, row in df.iterrows()
# ]

# print(f"Prepared {len(training_data)} training samples.")

# # Select the base model
# base_model = [
#     m for m in genai.list_models()
#     if "createTunedModel" in m.supported_generation_methods and
#     "flash" in m.name
# ][0]

# # Create a unique fine-tuned model name
# name = f'chess-tactics-{random.randint(0, 10000)}'



# # Reduce dataset if necessary
# training_data = training_data[:15000 ] 

# # Fine-tune the model
# operation = genai.create_tuned_model(
#     source_model=base_model.name,  # Use Gemini-1.5 Flash as the base model
#     training_data=training_data,  # Use formatted chess dataset 
#     id=name,  # Unique model ID
#     epoch_count=8,  # Adjust as needed
#     batch_size=8,  # Adjust based on dataset size
#     learning_rate=0.001,  # Default learning rate
# )

#print(f"Fine-tuning started! Model ID: tunedModels/{name}")

# Retrieve fine-tuned model details
#model = genai.get_tuned_model(f'tunedModels/{name}')

# Check model state
#print("Fine-tuning status:", model.state)

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Load your fine-tuned model
# Load your fine-tuned model
# model_id = "tunedModels/chess-tactics-7133"  # Replace with your actual tuned model ID
# model = genai.GenerativeModel(model_name=model_id)

# # Test with a sample FEN position
# fen_test = "r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"  # Initial chess position

# # Create a chess-specific prompt
# prompt = f"""
# Given this chess position in FEN notation:
# {fen_test}

# Please analyze the position and suggest the best move in algebraic notation.
# Explain your reasoning briefly.
# """

# # Generate response
# response = model.generate_content(
#     contents=prompt
# )

# print(response.text)
model_id = "tunedModels/chess-tactics-7133"  # Replace with your actual tuned model ID
model = genai.get_tuned_model('tunedModels/chess-tactics-7133')


print(model)
