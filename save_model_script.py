# save_model_script.py
from sentence_transformers import SentenceTransformer
import os

# Specify the model you want to download
model_name = 'all-MiniLM-L6-v2' 
# Specify the directory where you want to save the model files
# It's good practice to create a subdirectory named after the model.
output_directory = './models' 
save_path = os.path.join(output_directory, model_name)

# Create the output directory if it doesn't exist
os.makedirs(save_path, exist_ok=True)

print(f"Downloading and saving model '{model_name}' to '{save_path}'...")

try:
    # Initialize the model from the Hub (this will trigger download if not cached)
    model = SentenceTransformer(model_name)
    # Save the model to the specified path
    model.save(save_path)
    print(f"Model '{model_name}' successfully saved to '{save_path}'.")
    print("This directory contains all necessary files for the model.")
except Exception as e:
    print(f"An error occurred while downloading or saving the model: {e}")

