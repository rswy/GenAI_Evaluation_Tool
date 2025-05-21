# GenAI Evaluation Tool - User Manual

**Version:** 1.1 (Last Updated: May 16, 2025)

## Table of Contents

1.  [Introduction](#1-introduction)
    * [Purpose](#purpose)
    * [Key Features](#key-features)
2.  [Setup and Installation](#2-setup-and-installation)
    * [Prerequisites](#prerequisites)
    * [Creating a Virtual Environment](#creating-a-virtual-environment)
    * [Installing Dependencies](#installing-dependencies)
    * [NLTK Data Download](#nltk-data-download)
    * [Semantic Similarity Model Setup (Important)](#semantic-similarity-model-setup-important)
        * [Online Usage (Automatic Download)](#online-usage-automatic-download)
        * [Offline Usage (Manual Download and Setup)](#offline-usage-manual-download-and-setup)
3.  [Running the Application](#3-running-the-application)
    * [Using the Streamlit Web Interface](#using-the-streamlit-web-interface)
    * [Using the Command-Line Interface (CLI)](#using-the-command-line-interface-cli)
4.  [Input Data Format](#4-input-data-format)
    * [Supported File Types](#supported-file-types)
    * [Required Columns](#required-columns)
    * [Optional Columns for Specific Metrics](#optional-columns-for-specific-metrics)
        * [Understanding `ref_facts` vs. `ref_key_points`](#understanding-ref_facts-vs-ref_key_points)
    * [Other Optional Columns](#other-optional-columns)
    * [Example Data Rows](#example-data-rows)
5.  [Using the Streamlit Web Interface](#5-using-the-streamlit-web-interface)
    * [Sidebar: Input Options](#sidebar-input-options)
    * [Tab: Evaluation & Results](#tab-evaluation--results)
    * [Tab: View/Edit/Add Data](#tab-vieweditadd-data)
    * [Tab: Data Format Guide](#tab-data-format-guide)
    * [Tab: Metrics Tutorial](#tab-metrics-tutorial)
6.  [Metrics and Dimensions Explained](#6-metrics-and-dimensions-explained)
    * [Overview of Dimensions](#overview-of-dimensions)
    * [Detailed Metrics List](#detailed-metrics-list)
        * [Trust & Factuality](#trust--factuality)
        * [Completeness & Coverage](#completeness--coverage)
        * [Fluency & Lexical Similarity](#fluency--lexical-similarity)
        * [Semantic Understanding](#semantic-understanding)
        * [Classification Accuracy](#classification-accuracy)
        * [Conciseness](#conciseness)
        * [Safety (Basic Checks)](#safety-basic-checks)
        * [Privacy/Sensitive Data (Basic Checks)](#privacysensitive-data-basic-checks)
        * [Tone & Professionalism (Placeholder)](#tone--professionalism-placeholder)
        * [Refusal Appropriateness (Placeholder)](#refusal-appropriateness-placeholder)
7.  [Command-Line Usage (`main.py`)](#7-command-line-usage-mainpy)
8.  [Important Considerations & Known Issues](#8-important-considerations--known-issues)
    * [Semantic Similarity Performance](#semantic-similarity-performance)
    * [Rudimentary Nature of Basic Safety/PII Checks](#rudimentary-nature-of-basic-safetypii-checks)
    * [Placeholder Metrics](#placeholder-metrics)
    * [Per-Instance vs. Aggregated Classification Scores](#per-instance-vs-aggregated-classification-scores)
    * [Potential Environment/Runtime Errors](#potential-environmentruntime-errors)
9.  [Extending the Tool (Brief Overview)](#9-extending-the-tool-brief-overview)
10. [Conclusion](#10-conclusion)

---

## 1. Introduction

### Purpose
The GenAI Evaluation Tool is a comprehensive framework designed to assess the performance of Large Language Models (LLMs) across various tasks such as Retrieval Augmented Generation (RAG) for FAQs, text summarization, classification, and general chatbot interactions. It provides both quantitative metrics and qualitative insights to help users understand model behavior, identify strengths, weaknesses, and make informed decisions for model improvement and deployment.

### Key Features
* **Multi-Task Evaluation:** Supports RAG, Summarization, Classification, and Chatbot tasks.
* **Rich Metrics Suite:** Includes metrics for fluency, semantic similarity, factuality, completeness, conciseness, safety, and classification accuracy.
* **Interactive Web UI:** A Streamlit-based interface for easy data loading, editing, evaluation execution, and results visualization.
* **Command-Line Interface:** For batch processing and integration into automated workflows.
* **Customizable Inputs:** Allows users to provide reference facts and key points for more nuanced evaluation of factuality and completeness.
* **Detailed Reporting:** Generates individual and aggregated score reports in CSV and Markdown formats.
* **Interpretability Aids:** Provides explanations for metrics and automated interpretations for individual test case results.
* **Offline Capability:** Supports offline use of semantic similarity models with pre-downloaded assets.

---

## 2. Setup and Installation


### Prerequisites
* Python 3.8 or higher.
* `pip` (Python package installer).

### PIP Approach (with requirements.txt)

#### Creating a Virtual Environment using 
It is highly recommended to use a virtual environment to manage dependencies and avoid conflicts with system-wide packages.

1.  **Create a virtual environment:**
    ```bash
    python -m venv genai-eval-env
    ```
2.  **Activate the virtual environment:**
    * **Windows:**
        ```bash
        genai-eval-env\Scripts\activate
        ```
    * **macOS/Linux:**
        ```bash
        source genai-eval-env/bin/activate
        ```

#### Installing Dependencies
All required Python packages are listed in the `requirements.txt` file.

1.  **Navigate to the project root directory** (where `requirements.txt` is located).
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    This will install Streamlit, pandas, NLTK, scikit-learn, sentence-transformers, and other necessary libraries.


### CONDA approach (with environment.yml)

1. **Create and Activate Environment:**
Open your terminal or Anaconda Prompt, navigate to the directory containing `environment.yml`, and run:
    ```bash
    conda env create -f environment.yml
    ```
    This will create the environment with the specified name (e.g., `llm-eval-env`). After it's created, they activate it:
    ```bash
    conda activate llm-eval
    ```
If it doesn't work, comment out lines 1-35 and uncomment lines 45-63 and rerun the commands above. 

2. **Addressing Slow Package Installations**

Slow installations can be frustrating. Here are strategies to mitigate this:

1.  **Use Mamba (Highly Recommended for Speed):**
    * Mamba is a re-implementation of the Conda package manager in C++ and is significantly faster at dependency solving and downloading.
    * **Installation:** `conda install -n base -c conda-forge mamba` (installs Mamba into your base Conda environment).
    * **Usage:** Users can then replace `conda` with `mamba` for environment creation:
        ```bash
        mamba env create -f environment.yml
        mamba activate llm-eval-env
        ```


### Important: Check if rouge package can be found:**
    ```bash
    1. Activate the virtual environment:  venv/Scripts/activate
    2. type `python`
    3. import rouge_score
    ```

    If rouge_score module cannot be found, do the following: 
    ```bash
    1. pip uninstall rouge_score
    2. pip install the rouge_score whl file in the root folder GENAI_EVALUATION_TOOL_MAIN: 'pip install rouge_score-0.1.2-py3-none-any.whl'
    3. type 'python'
    4. import rouge_score
    ```

    The rouge_score module should now be able to be imported.

### NLTK Data Download
Several metrics rely on data packages from the Natural Language Toolkit (NLTK). These need to be downloaded once.

#### Files not yet downloaded: (*CAN SKIP THIS STEP AND EXTRACT FILES FROM nltk_data.zip*)
1.  Open a Python interpreter within your activated virtual environment:
    ```bash
    python
    ```
2.  In the Python interpreter, run the following commands:
    ```python
    import nltk
    nltk.download('punkt')      # For tokenization
    nltk.download('wordnet')    # For METEOR score and lemmatization
    nltk.download('omw-1.4')    # For WordNet (Open Multilingual Wordnet)
    exit()
    ```
#### Files have been downloaded:
1.  Open a Python interpreter within your activated virtual environment:
    ```bash
    python
    ```
2.  Find the nltk pathway that the system searches nltk data from
    ```python
    import nltk
    print(nltk.data.path)
    ```
3. Copy the extracted nltk_data folder into any path identified in step 2.


### Semantic Similarity Model Setup (Important) *Skip this step unless you want to use another model outside of 'all-MiniLM-L6-v2': Provided in the models folder*
The Semantic Similarity metric uses models from the `sentence-transformers` library. These models need to be available to the tool.

#### Online Usage (Automatic Download) --*Not Needed as the model files are provided in models folder*
If the machine running the tool has internet access, the `sentence-transformers` library will attempt to download the required model (default: `all-MiniLM-L6-v2`) automatically on the first run when the `SemanticSimilarityMetric` is initialized.
* **Note:** This initial download can take some time and requires an internet connection. Subsequent runs will use the cached model.

#### Offline Usage (Manual Download and Setup)
For environments without internet access, you must download the model files beforehand and configure the tool to load them from a local path.

**Step 1: Download the Model (on an Online Machine) --*I have provided the model file for 'all-MiniLM-L6-v2' so feel free to skip this step unless you're intending to use another model***
* Create a Python script (e.g., `save_model.py`) with the following content:
    ```python
    from sentence_transformers import SentenceTransformer
    import os

    model_name = 'all-MiniLM-L6-v2'  # Or any other model you prefer
    # Define a directory relative to your project or an absolute path
    output_base_dir = 'models' # This will create a 'models' folder in the same directory as the script
    save_path = os.path.join(output_base_dir, model_name)

    os.makedirs(save_path, exist_ok=True)
    print(f"Downloading and saving model '{model_name}' to '{save_path}'...")
    try:
        model = SentenceTransformer(model_name)
        model.save(save_path)
        print(f"Model '{model_name}' successfully saved to '{save_path}'.")
    except Exception as e:
        print(f"Error downloading/saving model: {e}")
    ```
* Run this script: `python save_model.py`. This will create a directory like `models/all-MiniLM-L6-v2/` containing the model files.

**Step 2: Transfer Model Files**
* Copy the entire model directory (e.g., `models/all-MiniLM-L6-v2/`) to the offline machine where the GenAI Evaluation Tool will run.
* Place it in a known location. The default configuration in the tool expects it at `./models/all-MiniLM-L6-v2` relative to where the application is run (usually the project root).

**Step 3: Configuration (Code Reference)**
* The `SemanticSimilarityMetric` class (in `src/metrics/fluency_similarity.py`) is configured to first attempt loading from a local path (`DEFAULT_LOCAL_MODEL_PATH_SEMANTIC = "./models/all-MiniLM-L6-v2"`).
* If you place the model in a different location, you might need to adjust the `DEFAULT_LOCAL_MODEL_PATH_SEMANTIC` constant in that file or modify how the path is passed to the `SentenceTransformer` constructor.

---

## 3. Running the Application

### Using the Streamlit Web Interface
The primary way to interact with the tool is through its web interface, powered by Streamlit.

1.  Ensure your virtual environment is activated.
2.  Navigate to the project root directory in your terminal.
3.  Run the command:
    ```bash
    streamlit run streamlit_app.py
    ```
4.  This will typically open the application in your default web browser. If not, the terminal will provide a local URL (e.g., `http://localhost:8501`) to access it.

### Using the Command-Line Interface (CLI)
The tool also provides a CLI for batch processing or integration into automated workflows, primarily through `main.py`.

1.  **To evaluate data from a file:**
    ```bash
    python main.py --input-file path/to/your/datafile.json --output-dir path/to/reports_directory
    ```
    (Replace `.json` with `.csv` or `.xlsx` as appropriate.)

2.  **To generate mock data:**
    ```bash
    python main.py --generate-mock-data --mock-data-output-base data/my_mock_data
    ```
    This will create `data/my_mock_data.json` and `data/my_mock_data.csv`.

---

## 4. Input Data Format
The tool expects input data in a **flat format**, where each row represents a single evaluation instance (one question-answer pair with associated metadata).

### Supported File Types
* JSON (`.json`): A list of dictionaries.
* CSV (`.csv`): Comma-separated values with a header row.
* Excel (`.xlsx`): Data in the first sheet with a header row.

### Required Columns
These columns **must** be present in your input file for each evaluation instance:

* `task_type` (String): The type of task being evaluated. Supported values are:
    * `rag_faq`
    * `summarization`
    * `classification`
    * `chatbot`
* `model` (String): An identifier for the LLM or model configuration being evaluated (e.g., `MyModel_v1.2_temp0.7`).
* `question` (String): The input query, text to summarize/classify, or user utterance.
* `ground_truth` (String): The ideal or reference answer, reference summary, correct label, or reference response.
* `answer` (String): The actual output generated by the LLM for the given `question`.

### Optional Columns for Specific Metrics

#### Understanding `ref_facts` vs. `ref_key_points`
It's crucial to differentiate these two optional inputs as they serve distinct evaluation purposes:

* **`ref_facts` (String, comma-separated):**
    * **Purpose:** Used by the **Fact Presence** score (under "Trust & Factuality").
    * **Content:** Provide a comma-separated list of *specific, discrete, and often verbatim or near-verbatim factual statements* that are expected to be present in the LLM's `answer`. These are critical pieces of information that *must* be stated correctly.
    * **Example:** For a question about the capital of France, if the `ground_truth` is "Paris is the capital of France and was founded in the 3rd century BC.", `ref_facts` might be: `Paris is capital of France,Founded in 3rd century BC`.
    * **Metric Focus:** Verifying the inclusion of non-negotiable factual details. The check is case-insensitive.

* **`ref_key_points` (String, comma-separated):**
    * **Purpose:** Used by the **Key Point Coverage** score (under "Completeness & Coverage").
    * **Content:** Provide a comma-separated list of *broader topics, themes, concepts, or checklist items* that the LLM's `answer` is expected to address or cover. The exact wording is less critical than the coverage of the underlying idea.
    * **Example:** For a question "Summarize the main arguments of the provided article on climate change.", `ref_key_points` might be: `Impact on ecosystems,Economic consequences,Proposed solutions,Role of international cooperation`.
    * **Metric Focus:** Ensuring the answer is comprehensive and addresses all required aspects of a query or instruction. The check is case-insensitive.

If these columns are not provided or are empty for a row, the corresponding `fact_presence_score` or `completeness_score` will be `NaN` (Not a Number) for that row.

### Other Optional Columns
* `id` (String): A unique identifier for the evaluation row. Highly recommended for tracking and analysis. If not provided when adding rows manually in the UI, one will be auto-generated.
* `test_description` (String): A brief description of the test case's purpose or what it's trying to evaluate.
* `contexts` (String): (Primarily for RAG tasks) Relevant context snippets or documents that were provided to the LLM to generate the `answer`. While not directly used by most current core metrics (except potentially for manual review or future context-aware metrics), it's good practice to include it for RAG evaluations.

### Example Data Rows
**(Conceptual CSV format)**

```csv
id,task_type,model,question,ground_truth,answer,ref_facts,ref_key_points,test_description,contexts
rag_001,rag_faq,ModelX_v1,What is the capital of France?,Paris is the capital of France.,The capital city of France is Paris.,Paris is capital of France,Capital city,Verify basic RAG retrieval,"France is a country in Europe. Its capital is Paris."
sum_001,summarization,ModelY_Alpha,Summarize the article about renewable energy.,Renewable energy sources are crucial for combating climate change and offer economic benefits.,Solar and wind power help reduce emissions and create jobs.,Solar reduces emissions,Wind creates jobs,Climate change impact,Economic benefits,Brief summary of renewable energy article,"Full text of the article..."
cls_001,classification,ClassifierZ,Is this review positive or negative?: 'The product is amazing!',positive,positive,,,,Sentiment classification test,
chat_001,chatbot,MyChatBot,Hello there,Hi! How can I help you today?,Hello! I'm here to assist.,,,General greeting test,
```
## 5. Using the Streamlit Web Interface

The Streamlit application provides an interactive way to manage data, run evaluations, and explore results.

### Sidebar: Input Options

Located on the left side of the application:

**Choose data source:**

* **Upload File:** Allows you to upload your evaluation data in `.xlsx`, `.csv`, or `.json` format (following the flat structure described above).
* **Generate Mock Data:** Creates a sample dataset with varied answer quality across different tasks. Useful for quickly testing the tool or understanding the expected data format.
* **Status Messages:** Provides feedback on data loading or mock data generation.

### Tab: Evaluation & Results

This is the main tab for running evaluations and viewing outcomes.

* **Run Evaluation Button:** Click "🚀 Run Evaluation on Data in Editor" to process the data currently loaded in the "View/Edit/Add Data" tab.
    * A spinner will indicate that evaluation is in progress. This can take time, especially with the Semantic Similarity metric or large datasets.
    * A note about the potential one-time download of the semantic similarity model is displayed.

* **Results Display:** After evaluation, results are shown in two sub-tabs:

    * **📄 Individual Scores:**
        * A detailed table showing each test case along with all computed metric scores.
        * Columns include original data, metric scores (with indicators ⬆️ for higher-is-better, ⬇️ for lower-is-better, and "(Placeholder)" tags), and automated interpretations.
        * **Automated Interpretations:**
            * **Observations:** Qualitative remarks based on metric scores (e.g., "Fluency & Lexical Sim.: Strong").
            * **Potential Actions:** Suggestions for improvement based on low scores or detected issues.
            * **Metrics Not Computed or Not Applicable:** Lists metrics that returned NaN due to missing inputs, errors, or placeholder status.
        * **Detailed Interpretation for a Single Test Case:** Select a Test Case ID from the dropdown to view its specific inputs, outputs, and detailed interpretations.
        * **Download:** Option to download the individual scores table (including interpretations) as a CSV file.

    * **📈 Aggregated Results:**
        * **Best Model Summary (Highlights):** Identifies top-performing models for key metrics within each task.
        * **Overall Summary Table:** Displays scores aggregated (typically averaged) by `task_type` and `model`. Non-placeholder metrics with valid scores are prioritized.
        * **Interpreting Your Aggregated Results (Experimental):** An expandable section offering heuristic-based interpretations of aggregated scores for different models and tasks.
        * **Task Specific Metric Table & Chart:** Tabs for each task type present in the results. Within each task tab, further tabs for each evaluation dimension (category) allow you to view tables and bar charts comparing models on specific metrics.
        * **Download:** Options to download aggregated results as CSV and a summary report as Markdown.

### Tab: View/Edit/Add Data

This tab allows you to manage your evaluation dataset interactively.

* **Add New Evaluation Row:**
    * **Input Mode:**
        * **Easy (Required Fields Only):** Shows only essential fields plus `test_description`.
        * **Custom (Select Additional Fields):** Shows all fields, including `ref_facts`, `ref_key_points`, and `contexts`.
    * A form to manually input data for a new evaluation case. Required fields are marked with \*.
    * Click "➕ Add Evaluation Row to Editor" to add the new row to the data table below.

* **Data Editor:**
    * Displays the current dataset (loaded from file, generated, or manually added) in an editable table.
    * You can modify cell values, add new rows (using the + button at the bottom of the editor), or delete rows.
    * Changes made here are used when you click "Run Evaluation".

* **Download:** Option to download the currently edited data as a CSV file.

### Tab: Data Format Guide

Provides a quick reference for the expected input data format, including required and optional columns with brief explanations. This is especially helpful for understanding the distinction between `ref_facts` and `ref_key_points`.

### Tab: Metrics Tutorial

Offers detailed explanations for each evaluation dimension and the metrics within them.

For each metric, it describes:

* Its name and underlying key.
* The evaluation dimension (category) it belongs to.
* Whether a higher or lower score is better.
* Its use case and how to interpret its score.
* Common tasks it's used for.
* Any specific input data fields it relies on (e.g., `ref_facts`).
* Its status (e.g., fully implemented or placeholder).

## 6. Metrics and Dimensions Explained

The tool organizes metrics into several dimensions (categories) to provide a structured view of LLM performance.

### Overview of Dimensions

* **Trust & Factuality:** Reliability and factual correctness.
* **Completeness & Coverage:** How comprehensively the answer addresses the query.
* **Fluency & Lexical Similarity:** Linguistic quality and word/phrase overlap with reference.
* **Semantic Understanding:** Similarity in meaning to the reference, beyond just words.
* **Classification Accuracy:** Performance on classification tasks.
* **Conciseness:** Brevity and focus of the response.
* **Safety (Basic Checks):** Rudimentary checks for unsafe content.
* **Privacy/Sensitive Data (Basic Checks):** Rudimentary checks for PII.
* **Tone & Professionalism (Placeholder):** Intended for tonal assessment.
* **Refusal Appropriateness (Placeholder):** Intended for evaluating how well the model refuses inappropriate queries.

### Detailed Metrics List

**Trust & Factuality**

* **Fact Presence (`fact_presence_score`)**
    * **Purpose:** Checks if specific, predefined factual statements are explicitly mentioned in the answer.
    * **Input:** Requires comma-separated factual statements in the `ref_facts` column.
    * **Score:** 0-1 (fraction of provided facts found). Higher is better.
    * **Interpretation:** A high score indicates the LLM included the expected factual details.
* **NLI Entailment Score (`nli_entailment_score`) - (Placeholder)**
    * **Status:** Not implemented. Returns NaN.
    * **Intended Purpose:** Use Natural Language Inference to verify if the answer entails or contradicts known facts.
* **LLM Judge Factuality (`llm_judge_factuality`) - (Placeholder)**
    * **Status:** Not implemented. Returns NaN.
    * **Intended Purpose:** Use another LLM to assess the factuality of the current LLM's answer.

**Completeness & Coverage**

* **Key Point Coverage (`completeness_score`)**
    * **Purpose:** Assesses if the answer covers a predefined list of broader key topics, concepts, or checklist items.
    * **Input:** Requires comma-separated key points/topics in the `ref_key_points` column.
    * **Score:** 0-1 (fraction of key points covered). Higher is better.
    * **Interpretation:** A high score suggests the answer is comprehensive regarding the expected topics.

**Fluency & Lexical Similarity**

These metrics primarily measure similarity based on word and phrase overlap.

* **BLEU (`bleu`)**
    * **Score:** 0-1. Higher is better (more n-gram overlap with reference).
* **ROUGE-1, ROUGE-2, ROUGE-L (`rouge_1`, `rouge_2`, `rouge_l`)**
    * **Score:** 0-1. Higher is better (more unigram, bigram, or Longest Common Subsequence overlap respectively).
* **METEOR (`meteor`)**
    * **Score:** 0-1. Higher is better (considers synonyms and stemming for unigram matching).

**Semantic Understanding**

* **Semantic Similarity (`semantic_similarity_score`)**
    * **Purpose:** Measures similarity in meaning between the answer and ground truth using sentence embeddings.
    * **Score:** Typically -1 to 1 (or 0 to 1 if normalized and positive). Higher is better (more semantically similar).
    * **Note:** Requires `sentence-transformers` library. May download model files on first run if online and model not cached. Supports offline use with pre-downloaded models.
    * **Interpretation:** High scores indicate the answer's meaning is close to the reference, even if wording differs. Low scores suggest a divergence in meaning.

**Classification Accuracy**

Used for classification tasks.

* **Accuracy (`accuracy`)**
    * **Per-instance:** 1.0 if predicted label matches ground truth, 0.0 otherwise.
    * **Aggregated:** Overall percentage of correct predictions. Higher is better.
* **Precision (`precision`), Recall (`recall`), F1-Score (`f1_score`)**
    * **Per-instance:** Simplified as 1.0/0.0 based on correctness for the specific class pair.
    * **Aggregated:** Standard macro-averaged Precision, Recall, and F1-score. Higher is better.

**Conciseness**

* **Length Ratio (`length_ratio`)**
    * **Score:** Ratio of answer word count to ground truth word count.
    * **Interpretation:** Closer to 1.0 is often desired. Values significantly >1 suggest verbosity; <1 suggest brevity. "Higher is better" is set to False, meaning the UI will color lower values (closer to 0, if reference is long) or values closer to 1 as "better" depending on context.

**Safety (Basic Checks)**

* **Safety Keyword Score (`safety_keyword_score`)**
    * **Score:** 1.0 if NO predefined unsafe keywords are found; 0.0 if ANY are found. Higher is better (safer).
    * **Note:** This is a rudimentary check based on a simple keyword list. It is not an exhaustive safety evaluation. Observations are only shown in the UI if an issue is detected (score < 1.0).

**Privacy/Sensitive Data (Basic Checks)**

* **PII Detection Score (`pii_detection_score`)**
    * **Score:** 1.0 if NO common PII patterns (e.g., email, basic phone regex) are found; 0.0 if ANY are found. Higher is better (less PII).
    * **Note:** This is a basic regex check and NOT a comprehensive PII scan. Observations are only shown in the UI if an issue is detected (score < 1.0).

**Tone & Professionalism (Placeholder)**

* **Professional Tone (`professional_tone_score`) - (Placeholder)**
    * **Status:** Not implemented. Returns NaN.
    * **Intended Purpose:** Assess if the response adheres to a professional tone.

**Refusal Appropriateness (Placeholder)**

* **Refusal Quality (`refusal_quality_score`) - (Placeholder)**
    * **Status:** Not implemented. Returns NaN.
    * **Intended Purpose:** Evaluate if the model appropriately and politely refuses to answer out-of-scope, harmful, or sensitive queries.

## 7. Command-Line Usage (main.py)

The `main.py` script allows for non-interactive evaluation.

**Arguments:**

* `--input-file <path>`: Path to the input data file (JSON, XLSX, CSV).
* `--output-dir <directory>`: Directory to save evaluation reports (default: `reports`).
* `--generate-mock-data`: Action flag to generate mock data.
* `--mock-data-output-base <base_path_and_filename>`: Base path for generated mock data (default: `data/llm_eval_mock_data`).

**Example:**

```bash
python main.py --input-file data/my_eval_data.csv --output-dir custom_reports
```


**Output:**

* Individual scores CSV: `evaluation_report_individual_scores_<timestamp>.csv`
* Aggregated summary CSV: `evaluation_report_aggregated_summary_<timestamp>.csv`
* Aggregated summary Markdown: `evaluation_report_aggregated_summary_<timestamp>.md`

## 8. Important Considerations & Known Issues

**Semantic Similarity Performance**

* The Semantic Similarity metric is computationally more intensive than lexical metrics due to the loading of sentence embedding models and vector computations.
* Evaluation times will increase when this metric is active, especially for large datasets.
* The `evaluator.py` script initializes metric instances (including the sentence transformer model) once per evaluation run to optimize performance.

**Rudimentary Nature of Basic Safety/PII Checks**

* The `safety_keyword_score` and `pii_detection_score` are basic checks.
* They rely on simple keyword lists and regular expressions, respectively.
* They are not exhaustive and should not be solely relied upon for critical safety or PII compliance. They serve as a first-pass indicator.
* The UI has been updated to only show observations for these metrics if a potential issue is detected (score \< 1.0), avoiding a false sense of security from "passed" messages.

**Placeholder Metrics**

* Metrics tagged as (Placeholder) (e.g., NLI Entailment, LLM Judge Factuality, Professional Tone, Refusal Quality) are not fully implemented and will return NaN scores.
* Their inclusion signifies areas for future development. Full implementation often requires dedicated models, external APIs, or complex logic.

**Per-Instance vs. Aggregated Classification Scores**

* For classification tasks, the per-instance scores for Accuracy, Precision, Recall, and F1-Score displayed in the "Individual Scores" table typically represent binary correctness (1.0 if correct for that specific instance's label pair, 0.0 if incorrect). This is a simplified view for individual case analysis.
* The aggregated scores for these metrics (shown in the "Aggregated Results" tab) are the standard, more statistically meaningful measures (e.g., overall accuracy, macro-averaged F1-score across all samples for a model/task).

**Troubleshooting Potential Environment/Runtime Errors**

* **PyTorch/asyncio/Streamlit Errors:** Users might encounter `RuntimeError` related to asyncio event loops or PyTorch internal classes (like `__path__._path`) when running Streamlit. These often stem from:
    * Library version incompatibilities (Streamlit, PyTorch, `sentence-transformers`).
    * Corrupted PyTorch installations.
    * Interactions with Streamlit's file watcher.

**The Most Effective Solution for This Specific Error:**

The most common and effective way to resolve this particular RuntimeError: Tried to instantiate class '__path__._path' when using Streamlit with PyTorch/Sentence-Transformers is to change Streamlit's file watcher type.

* How to apply the fix:
    1. Create a `.streamlit` directory in the root of your project (i.e., in GenAI_Evaluation_Tool-main/, at the same level as your streamlit_app.py).
    2. Inside the `.streamlit` directory, create a file named config.toml.

    3. Add the following content to config.toml:

    ```
    Ini, TOML

    [server]
    # This tells Streamlit to use a polling mechanism for file changes,
    # which is often more stable with complex libraries than the default.
    # Options are: "watchdog", "poll", "none"
    fileWatcherType = "poll"

    # If "poll" doesn't work, you can try "none" to disable the watcher entirely.
    # This means Streamlit won't automatically reload when you save file changes;
    # you'll need to manually stop and restart the Streamlit server.
    # This is a good diagnostic step to confirm the watcher is the issue.
    # fileWatcherType = "none"
    ```

    Explanation:

    Streamlit's default file watcher (watchdog) can sometimes be too aggressive or interact in unexpected ways with how certain libraries (especially those with C++ extensions like PyTorch) expose their internal module structures. This seems to be what's happening with torch._classes and __path__._path.
    Setting fileWatcherType = "poll" uses a different, often more compatible, method to check for file changes.
    Setting fileWatcherType = "none" disables the watcher. If the error disappears with "none", it definitively points to the file watcher interaction as the root cause. You can then decide if "poll" is an acceptable long-term solution or if you prefer to work with the watcher disabled during development involving these libraries.

    **Other Checks (Secondary, but good practice)**
    1. Clean Environment: If you haven't already, ensure you are working in a clean virtual environment with dependencies freshly installed from your requirements.txt.
    2. Reinstall PyTorch correctly from [pytorch.org](https://pytorch.org/), ensuring compatibility with your OS and CUDA (if applicable).
        * ``` pip uninstall torch torchvision torchaudio ```
        * Install the correct version from pytorch.org. For CPU: ```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu```
    3. Streamlit Version: Ensure your Streamlit version is relatively recent. `pip install --upgrade streamlit`

    4. Isolate problematic imports: Temporarily comment out imports related to `sentence-transformers` to see if the error persists.
    
## 9. Extending the Tool (Brief Overview)

The tool is designed to be extensible:

**Adding New Metrics:**

1.  Create a new Python class that inherits from `src.metrics.base_metric.BaseMetric`.
2.  Implement the `compute(self, references, predictions, **kwargs)` method. This method should process a single instance and return a dictionary of score(s).
3.  Register the new metric class in `src.metrics.__init__.py` (in `METRIC_CLASS_REGISTRY` and `get_metric_instances` function).
4.  Add the metric key(s) to `tasks.task_registry.py` under the relevant `TASK_METRIC_MAP`.
5.  Update `streamlit_app.py` (`METRIC_INFO`, `DIMENSION_DESCRIPTIONS`, interpretation logic) and `reporter.py` (`METRIC_INFO_REPORTER`) to include information and handling for the new metric.

**Implementing Placeholder Metrics:** Follow the same process as adding a new metric, replacing the placeholder logic with actual computation.

## 10. Conclusion

The GenAI Evaluation Tool offers a robust and user-friendly platform for assessing Large Language Models. By understanding its diverse features, input data requirements, and the nuances of its evaluation metrics, users can derive significant insights into their model's performance. This tool is intended to be a starting point; users are encouraged to adapt and extend it to meet their specific evaluation needs.
