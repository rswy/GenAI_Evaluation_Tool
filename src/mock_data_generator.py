# src/mock_data_generator.py (Improved Flat Format - 'contexts' removed)
import json
import random
import copy
import pandas as pd
from pathlib import Path
import sys
import os # Added for potential path operations if needed

"""
Generates mock data in a flat, row-per-evaluation format.
Aims to produce non-zero scores for custom dimensions by explicitly including facts/key points.
'contexts' field has been removed.
"""

# Ensure tasks module can be found if run directly (Add try-except for robustness)
try:
    current_dir = Path(__file__).parent
    project_root_if_direct = current_dir.parent
    if str(project_root_if_direct) not in sys.path:
        sys.path.insert(0, str(project_root_if_direct))
    from tasks.task_registry import RAG_FAQ, SUMMARIZATION, CLASSIFICATION, CHATBOT
except ImportError:
    print("Warning: Could not import task constants from tasks.task_registry. Using string literals.")
    RAG_FAQ = "rag_faq"; SUMMARIZATION = "summarization"; CLASSIFICATION = "classification"; CHATBOT = "chatbot"


def generate_mock_data_flat(num_samples_per_task=3, seed=42):
    """Generates mock data in the flat format with explicit fact/key point inclusion."""
    random.seed(seed) # for reproducibility
    all_data = []
    eval_id_counter = 1

    # --- Common Model Names ---
    MODEL_GOOD = "model_A_good"
    MODEL_PARTIAL = "model_B_partial"
    MODEL_POOR = "model_C_poor_keywords"

    # --- RAG/FAQ Data ---
    rag_cases = [
        {
            "input": {"question": "Capital of France and known for?"}, # 'contexts' removed
            "ref": {"ground_truth": "Paris is the capital, known for Eiffel Tower, Louvre, and fashion.", "ref_facts": "Paris is capital,Eiffel Tower is monument,Louvre is museum", "ref_key_points": "Capital,Known For,Eiffel Tower,Louvre"}
        },
        {
            "input": {"question": "Photosynthesis inputs?"}, # 'contexts' removed
            "ref": {"ground_truth": "The main inputs for photosynthesis are light, water, and carbon dioxide (CO2).", "ref_facts": "Input is light,Input is water,Input is CO2", "ref_key_points": "Inputs,Light,Water,CO2"}
        },
        {
            "input": {"question": "Flu cause and symptoms?"}, # 'contexts' removed
            "ref": {"ground_truth": "The flu is caused by influenza viruses. Common symptoms are fever, cough, and muscle aches.", "ref_facts": "Cause is virus,Symptom is fever,Symptom is cough", "ref_key_points": "Cause,Symptoms,Virus,Fever,Cough"}
        },
    ]

    for i in range(num_samples_per_task):
        case = copy.deepcopy(rag_cases[i % len(rag_cases)])
        input_data = case["input"]
        ref_data = case["ref"]
        facts = ref_data.get("ref_facts", "").split(',') if ref_data.get("ref_facts") else []
        facts = [f.strip() for f in facts if f.strip()]
        key_points = ref_data.get("ref_key_points", "").split(',') if ref_data.get("ref_key_points") else []
        key_points = [kp.strip() for kp in key_points if kp.strip()]

        # --- Generate varied responses with EXPLICIT inclusion ---
        responses = {}
        # GOOD: Include most facts/points
        good_response_parts = [ref_data['ground_truth']]
        # Add some facts/points explicitly to ensure matching
        if len(facts) >= 2: good_response_parts.append(f"Fact check: {facts[0]}. Also: {facts[1]}.")
        elif facts: good_response_parts.append(f"Fact: {facts[0]}.")
        if len(key_points) >= 2 : good_response_parts.append(f"Key points mentioned: {key_points[0]}, {key_points[1]}.")
        elif key_points: good_response_parts.append(f"It covers: {key_points[0]}.")
        responses[MODEL_GOOD] = " ".join(good_response_parts)

        # PARTIAL: Include some fact/point
        partial_response_parts = [ref_data['ground_truth'].split('.')[0] + "."] # Start with first sentence
        if facts: partial_response_parts.append(f"It is true that {random.choice(facts)}.")
        elif key_points: partial_response_parts.append(f"The topic of {random.choice(key_points)} is relevant.")
        responses[MODEL_PARTIAL] = " ".join(partial_response_parts)

        # POOR: Unrelated, maybe keyword, explicitly not mentioning a fact
        responses[MODEL_POOR] = f"An unrelated answer about {random.choice(['weather', 'sports', 'badword'])}. This does not cover {facts[0] if facts else 'the topic'}."

        for model_name, response in responses.items():
            all_data.append({
                # Use different ID prefixes for clarity
                "id": f"rag_{eval_id_counter:03d}", # Padded ID
                "task_type": RAG_FAQ, "model": model_name,
                "question": input_data["question"], 
                # "contexts": input_data.get("contexts"), # 'contexts' removed
                "ground_truth": ref_data["ground_truth"],
                "ref_facts": ref_data.get("ref_facts"), # Keep original comma-sep string for file
                "ref_key_points": ref_data.get("ref_key_points"), # Keep original comma-sep string for file
                "answer": response,
            })
            eval_id_counter += 1

    # --- Summarization Data ---
    sum_cases = [
         {
            "input": {"question": "Summarize: JWST is a large infrared space telescope successor to Hubble, observing the early universe and exoplanets."}, # 'contexts' removed (was None anyway)
            "ref": {"ground_truth": "JWST is a powerful infrared telescope studying the early universe and exoplanets, succeeding Hubble.", "ref_key_points": "JWST,Infrared,Successor,Early universe,Exoplanets"}
        },
        {
            "input": {"question": "Summarize: AI involves intelligent agents perceiving environments and maximizing goal achievement, mimicking human cognition like learning."}, # 'contexts' removed
            "ref": {"ground_truth": "AI uses intelligent agents to perceive surroundings and achieve goals, mimicking human learning.", "ref_key_points": "AI definition,Agents,Goals,Perception,Mimics cognition"}
        },
         {
            "input": {"question": "Summarize: The Great Barrier Reef off Australia is the largest coral system, visible from space, comprised of thousands of reefs and islands."}, # 'contexts' removed
            "ref": {"ground_truth": "The Great Barrier Reef in Australia is the world's biggest coral system, visible from space.", "ref_key_points": "Great Barrier Reef,Location,Largest,Visible from space"}
        },
    ]
    for i in range(num_samples_per_task):
        case = copy.deepcopy(sum_cases[i % len(sum_cases)])
        input_data = case["input"]
        ref_data = case["ref"]
        key_points = ref_data.get("ref_key_points", "").split(',') if ref_data.get("ref_key_points") else []
        key_points = [kp.strip() for kp in key_points if kp.strip()]

        responses = {}
        # GOOD: Include most points
        good_response_parts = [ref_data['ground_truth']]
        if len(key_points) >= 2: good_response_parts.append(f"Main points: {key_points[0]} and {key_points[1]}.")
        elif key_points: good_response_parts.append(f"Key topic: {key_points[0]}.")
        responses[MODEL_GOOD] = " ".join(good_response_parts)

        # PARTIAL: Include one point
        partial_response_parts = [ref_data['ground_truth'].split('.')[0] + "."]
        if key_points: partial_response_parts.append(f"The summary mentions {random.choice(key_points)}.")
        responses[MODEL_PARTIAL] = " ".join(partial_response_parts)

        # POOR: Unrelated
        responses[MODEL_POOR] = f"An unrelated summary about {random.choice(['cooking', 'finance', 'badword'])}."

        for model_name, response in responses.items():
            all_data.append({
                "id": f"sum_{eval_id_counter:03d}",
                "task_type": SUMMARIZATION, "model": model_name,
                "question": input_data["question"], 
                # "contexts": input_data.get("contexts"), # 'contexts' removed
                "ground_truth": ref_data["ground_truth"],
                "ref_facts": ref_data.get("ref_facts"), # Likely None
                "ref_key_points": ref_data.get("ref_key_points"),
                "answer": response,
            })
            eval_id_counter += 1

    # --- Classification Data ---
    cls_cases = [
        {"input": {"question": "This was amazing!"}, "ref": {"ground_truth": "positive"}}, # 'contexts' removed
        {"input": {"question": "Truly awful experience."}, "ref": {"ground_truth": "negative"}}, # 'contexts' removed
        {"input": {"question": "It was adequate."}, "ref": {"ground_truth": "neutral"}}, # 'contexts' removed
        {"input": {"question": "I'm overjoyed!"}, "ref": {"ground_truth": "positive"}}, # 'contexts' removed
        {"input": {"question": "Very disappointing."}, "ref": {"ground_truth": "negative"}}, # 'contexts' removed
        {"input": {"question": "The report is complete."}, "ref": {"ground_truth": "neutral"}}, # 'contexts' removed
    ]
    labels = ["positive", "negative", "neutral"]

    for i in range(num_samples_per_task * 2): # Multiplied to get more classification samples
        case = copy.deepcopy(cls_cases[i % len(cls_cases)])
        input_data = case["input"]
        ref_data = case["ref"]
        true_label = ref_data["ground_truth"]

        other_labels = [l for l in labels if l != true_label]
        responses = {
            MODEL_GOOD: true_label,
            MODEL_PARTIAL: random.choice([true_label, random.choice(other_labels)]) if other_labels else true_label,
            MODEL_POOR: random.choice(other_labels) if other_labels else true_label
        }
        if random.random() < 0.15: responses[MODEL_GOOD] = random.choice(other_labels) if other_labels else true_label # Introduce some errors for MODEL_GOOD

        for model_name, response in responses.items():
            all_data.append({
                "id": f"cls_{eval_id_counter:03d}",
                "test_description": f"Test case for {input_data['question'][:30]}... with {model_name}", # Add description
                "task_type": CLASSIFICATION, "model": model_name,
                "question": input_data["question"], 
                # "contexts": input_data.get("contexts"), # 'contexts' removed
                "ground_truth": ref_data["ground_truth"],
                "ref_facts": None, 
                "ref_key_points": None,
                "answer": response,
            })
            eval_id_counter += 1

    # --- Chatbot Data ---
    chat_cases = [
        {"input": {"question": "Hi there"}, "ref": {"ground_truth": "Hello! How can I help you today?"}}, # 'contexts' removed
        {"input": {"question": "Tell me about your capabilities"}, "ref": {"ground_truth": "I can answer questions, summarize text, and evaluate based on metrics."}}, # 'contexts' removed
        {"input": {"question": "Thanks for the help"}, "ref": {"ground_truth": "You're welcome! Let me know if you need anything else."}}, # 'contexts' removed
    ]

    for i in range(num_samples_per_task):
        case = copy.deepcopy(chat_cases[i % len(chat_cases)])
        input_data = case["input"]
        ref_data = case["ref"]

        responses = {
            MODEL_GOOD: f"{ref_data['ground_truth'][:-6]} How's that?", # Slightly varied good response
            MODEL_PARTIAL: f"{ref_data['ground_truth'].split('!')[0]}. I am here.", # Partial response
            MODEL_POOR: f"Okay. Did you hear about the {random.choice(['game', 'unsafe topic', 'meeting'])}?" # Potentially off-topic/unsafe
        }

        for model_name, response in responses.items():
            all_data.append({
                "id": f"chat_{eval_id_counter:03d}",
                "task_type": CHATBOT, "model": model_name,
                "question": input_data["question"], 
                # "contexts": input_data.get("contexts"), # 'contexts' removed
                "ground_truth": ref_data["ground_truth"],
                "ref_facts": None, 
                "ref_key_points": None,
                "answer": response,
            })
            eval_id_counter += 1

    return all_data


def save_mock_data(data, output_dir="data", base_filename="improved_mock_data"):
    """Saves the generated flat data to JSON and CSV files."""
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Save as JSON
    json_path = output_dir_path / f"{base_filename}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Improved flat mock data generated and saved to {json_path}")

    # Save as CSV
    csv_path = output_dir_path / f"{base_filename}.csv"
    try:
        df = pd.DataFrame(data)
        # Define column order, 'contexts' removed
        cols_order = ['id', 'task_type', 'model', 'test_description', 'question', 'ground_truth', 'answer', 'ref_facts', 'ref_key_points']
        # Add any extra columns found in the data not in the predefined order
        present_cols_in_order = [col for col in cols_order if col in df.columns]
        extra_cols = sorted([col for col in df.columns if col not in present_cols_in_order])
        final_cols = present_cols_in_order + extra_cols
        
        df = df[final_cols]
        # Fill NaN with empty string for CSV
        df.fillna('', inplace=True)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"Improved flat mock data also saved to {csv_path}")
    except ImportError:
        print("Pandas not installed, skipping CSV save.")
    except Exception as e:
        print(f"Error saving CSV: {e}")

# Example of how to run it (e.g., from main.py or directly)
if __name__ == "__main__":
    # Ensure tasks module can be found if run directly (repeated for safety)
    try:
        current_dir = Path(__file__).parent
        project_root_if_direct = current_dir.parent
        if str(project_root_if_direct) not in sys.path:
            sys.path.insert(0, str(project_root_if_direct))
        # from tasks.task_registry import RAG_FAQ, SUMMARIZATION, CLASSIFICATION, CHATBOT # Not strictly needed for running this script directly if constants are redefined
    except ImportError:
        print("Could not re-import task constants when run directly.")

    # Define output directory relative to project root
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"

    mock_data = generate_mock_data_flat(num_samples_per_task=3)
    save_mock_data(mock_data, output_dir=data_dir, base_filename="llm_eval_mock_data_no_contexts") # Use a descriptive name