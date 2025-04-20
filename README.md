# README: AI Text Paraphrasing and Detection Pipeline (RAFT + SICO)

## Overview

This project implements a pipeline within a Google Colab environment to:
1.  **Prepare datasets:** Format data from various sources (Hugging Face datasets, custom TSV files) for use with the SICO (Style Imitation and Comparison) toolkit.
2.  **Train SICO components:** The scripts prepare data and train paraphrasing prompts using `SICO_train.py`.
3.  **Generate SICO paraphrases:** Use `SICO_test_gen.py` to generate paraphrased text based on a trained SICO setup.
4.  **Paraphrase AI-generated text:** Using the RAFT (Robustness Assessment for Fairness and Trustworthiness) toolkit, specifically its synonym replacement capabilities.
5.  **Detect AI-generated text:** Employ SICO's detection capabilities (`run_test_detection.py`) with various detectors (e.g., `chatdetect`, `logrank`, `detectgpt`) on both original AI text and text modified by RAFT.
6.  **Evaluate detector performance:** Aggregate detection scores and calculate performance metrics (AUC, F1-score) to compare detectability between original and RAFT-modified texts.

## Prerequisites and Setup

* **Environment:** Google Colab is the intended execution environment.
* **Google Drive:** The code heavily relies on Google Drive for storing code, datasets, and results. Ensure your Drive is mounted at `/content/drive`.
* **Directory Structure:** The scripts expect a specific directory structure within Google Drive:
    * `/content/drive/MyDrive/raft/`: Contains the RAFT code and datasets.
    * `/content/drive/MyDrive/SICO/`: Contains the SICO code, environment, datasets, and outputs.
    * `/content/drive/MyDrive/data/`: Contains source data (original AI text, RAFT modifications) and final aggregated results.
        * `Oryginal/`: Holds original AI text files (e.g., `abstracts_long.tsv`).
        * `RAFT_modification/`: Holds RAFT output files (e.g., `abstracts_long.csv`).
        * `Final_results/`: Destination for final detection score files.
* **Python Packages (RAFT):** Install required packages for RAFT using `pip`:
    ```bash
    pip install -U transformers huggingface-hub
    pip install -U git+[https://github.com/huggingface/transformers.git](https://github.com/huggingface/transformers.git)
    # Assuming requirements.txt is in the raft directory
    pip install -r /content/drive/MyDrive/raft/requirements.txt
    pip install language-tool-python
    pip install bitsandbytes
    ```
* **Miniconda and SICO Environment:** SICO requires a specific Conda environment.
    ```bash
    # Install Miniconda in Colab
    bash miniconda.sh -bfp /usr/local
    # Add conda to sys.path (may vary slightly depending on Python version)
    # import sys
    # sys.path.append('/usr/local/lib/python3.8/site-packages/') # Adjust python version if needed

    # Create SICO environment from file
    cd /content/drive/MyDrive/SICO
    conda env create -f environment.yml

    # Activate environment (conceptually - scripts use direct path)
    # The scripts call python directly: /usr/local/envs/chatgpt/bin/python

    # Download NLTK data for the SICO environment
    /usr/local/envs/chatgpt/bin/python -m nltk.downloader all
    ```
* **API Keys:** The SICO scripts (`SICO_train.py`, `SICO_test_gen.py`) require an `OPENAI_API_KEY`. A placeholder (`sk-`) is used in the example commands, which might be sufficient if only using local models or certain SICO features, but full functionality (especially with `llm=chatgpt`) typically requires a valid key.
* **CUDA:** GPU acceleration (CUDA) is assumed for both RAFT and potentially SICO components. Environment variables like `PYTORCH_CUDA_ALLOC_CONF` might be needed for memory management. Locale settings (`LC_ALL`, `LANG`) might need to be exported before running SICO detection scripts to avoid encoding errors.

## Workflow Steps

### 1. SICO - Environment and Data Setup

* **Environment:** Set up the Conda environment as described in Prerequisites.
* **Training Data Formatting (Example):**
    * Loads a dataset (e.g., `Hello-SimpleAI/HC3`) using `datasets`.
    * Cleans and processes questions, human answers, and AI answers.
    * Creates triplets (input, human, ai).
    * Splits data into `incontext`, `eval`, and `test` sets.
    * Saves these sets as TSV files (`incontext.tsv`, `eval.tsv`, `test.tsv`) in the SICO dataset directory (e.g., `/content/drive/MyDrive/SICO/datasets/hc3_wiki/`).
* **SICO Training (Implicit/Optional):**
    * Run `SICO_train.py`. This likely trains prompts or helper models for paraphrasing or detection based on the formatted training data. Requires specifying dataset, LLM, detector, task, and sizes.
* **SICO Paraphrase Generation (Optional Example):**
    * Run `SICO_test_gen.py` to apply a trained SICO paraphrasing method to a test set. Outputs generated text to `outputs/test_results/.../generated_text.tsv`.


### 2. RAFT - Synonym Replacement

* **Input:** A JSONL file containing text to be paraphrased (e.g., `abstracts_long.jsonl`), placed in `/content/drive/MyDrive/raft/datasets/custom_input/test.jsonl`.
* **Process:** The `experiment.py` script is run with specific configurations:
    * `--dataset custom_input`: Uses the custom input file.
    * `--mask_pct 0.1`: Percentage of words to potentially replace.
    * `--top_k 10`: Number of synonym candidates to consider.
    * `--proxy_model`, `--detector`, `--candidate_generation`, `--local_llm_model_id`: Specify the models used in the RAFT process.
    * `--output_path`: Directory for results.
* **Output:** A JSON file (e.g., `result_0.json`) inside a subdirectory within `./experiments/` containing the original text, the RAFT-sampled (paraphrased) text, and indices of changed words. A subsequent cell processes this JSON to show changes. *Note:* Later scripts assume RAFT results are converted to a CSV format (e.g., `abstracts_long.csv`) and stored in `/content/drive/MyDrive/data/RAFT_modification/`.


### 3. Script 1: Processing RAFT Results + SICO Detection

* **Purpose:** To evaluate how well SICO detectors identify text that has been paraphrased by RAFT.
* **Inputs:**
    * Original AI text TSV file (e.g., `/content/drive/MyDrive/data/Oryginal/abstracts_long.tsv`).
    * RAFT paraphrased text CSV file (e.g., `/content/drive/MyDrive/data/RAFT_modification/abstracts_long.csv`).
    * Configuration: `SOURCE_BASENAME`, `SICO_DATASET_NAME`, `DETECTOR_TO_TEST`.
* **Process:**
    1.  Creates a results directory structure within SICO's outputs for this specific method (e.g., `SICO/outputs/test_results/hc3_wiki/RAFT-Paraphrased-abstracts_long`).
    2.  Reads the original AI text and the RAFT-modified text.
    3.  Pairs the original AI text (as 'input') with the corresponding RAFT-modified text (as 'SICO-output').
    4.  Saves these pairs into the required `generated_text.tsv` format within the method's directory.
    5.  Executes `run_test_detection.py` using the specified detector (`DETECTOR_TO_TEST`) on the prepared `generated_text.tsv`. This script calculates detection scores for each RAFT-modified text.
    6.  Copies the resulting score file (e.g., `detectgpt_score.tsv`) to the `Final_results` directory, renaming it descriptively (e.g., `abstracts_long_RAFT_detectgpt_scores.tsv`).
* **Output:** Detection scores for RAFT-modified text stored in `/content/drive/MyDrive/data/Final_results/`.

### 4. Script 2: Processing Original AI Text + SICO Detection

* **Purpose:** To establish a baseline by evaluating how well SICO detectors identify the *original*, unmodified AI text.
* **Inputs:**
    * Original AI text TSV file (e.g., `/content/drive/MyDrive/data/Oryginal/abstracts_long.tsv`).
    * Configuration: `SOURCE_BASENAME`, `SICO_DATASET_NAME`, `DETECTOR_TO_TEST`.
* **Process:**
    1.  Creates a results directory structure within SICO's outputs for this specific method (e.g., `SICO/outputs/test_results/hc3_wiki/Original-AI-abstracts_long`).
    2.  Reads the original AI text.
    3.  Formats it into the required `generated_text.tsv` (using placeholders for 'input' and the AI text as 'SICO-output').
    4.  Executes `run_test_detection.py` using the specified detector (`DETECTOR_TO_TEST`) on the prepared `generated_text.tsv`. This calculates detection scores for each original AI text.
    5.  Copies the resulting score file (e.g., `detectgpt_score.tsv`) to the `Final_results` directory, renaming it descriptively (e.g., `abstracts_long_Original_AI_detectgpt_scores.tsv`).
* **Output:** Detection scores for original AI text stored in `/content/drive/MyDrive/data/Final_results/`.

### 5. Results Aggregation and Evaluation

* **Purpose:** To compare detector performance on original vs. RAFT-modified text.
* **Input:** Assumes all relevant `*_scores.tsv` files (from Script 1 and Script 2 for various datasets and detectors) are collected, potentially from a ZIP archive or directly from the `Final_results` directory.
* **Process:**
    1.  Finds all `*_scores.tsv` files.
    2.  Parses filenames to extract metadata (domain, length, variant - Original_AI/RAFT, detector).
    3.  Reads the scores from each file.
    4.  Assigns ground truth labels (Original_AI = 1, RAFT = 0, assuming RAFT aims to make text look *less* like AI, i.e., more human). *Note: The script assumes RAFT output should be labeled 0.*
    5.  Defines classification thresholds for each detector (`THRESHOLDS` dictionary).
    6.  Calculates predictions based on scores and thresholds.
    7.  Groups data by domain, length, and detector.
    8.  Calculates aggregate metrics for each group: mean score, AUC (Area Under the ROC Curve), and F1-score.
* **Output:**
    * Prints a summary table to the console.
    * Saves the summary metrics to `AUC_F1_summary.csv`.

## Configuration Highlights

Key parameters that can be modified in the scripts include:

* **RAFT:** `input_filename`, `--mask_pct`, `--top_k`, model IDs (`--proxy_model`, `--detector`, `--local_llm_model_id`).
* **SICO Data Formatting:** `hf_dataset_name`, `split_name`, `output_dataset_name`, `incontext_size`, `eval_size`, `test_size`.
* **SICO Training/Generation:** `--dataset`, `--llm`, `--detector`, `--task`, `--train-iter`, `TEST_SAMPLE_SIZE`.
* **Processing Scripts (1 & 2):** `SOURCE_BASENAME`, `SICO_DATASET_NAME` (used for directory structure, assumed consistent), `ORIGINAL_AI_TSV_PATH`, `ORIGINAL_AI_COL_INDEX`, `RAFT_RESULTS_CSV_PATH`, `RAFT_SAMPLED_COL_NAME`, `DETECTOR_TO_TEST`.
* **Evaluation:** `THRESHOLDS` dictionary for classification cutoffs.

## Data Formats

* **Original AI Text:** Expected in TSV format, often with multiple columns. The relevant AI text column index is specified (`ORIGINAL_AI_COL_INDEX`). Header might be present or absent depending on the script section. Script 1 and 2 read it assuming a header row (`header=0`).
* * **SICO Data:** TSV files (`incontext.tsv`, `eval.tsv`, `test.tsv`) with columns 'input', 'human', 'ai'.
* **SICO `generated_text.tsv`:** TSV file with columns 'input', 'SICO-output' used as input for detection.
* **SICO `*_score.tsv`:** Single-column TSV file (no header) containing raw detection scores.
* **RAFT Input:** JSONL format, where each line is a JSON object containing text.
* **RAFT Output (Raw):** JSON file with 'original', 'sampled', 'replacement_keys'.
* **RAFT Output (Processed for Script 1):** CSV format with at least a column named `sampled_text` (or as configured by `RAFT_SAMPLED_COL_NAME`). Header assumed.


## Notes

* The pipeline relies heavily on specific file paths within Google Drive. Ensure these paths match your setup.
* Error handling is present but might need enhancement for robustness in different scenarios.
* The SICO environment setup (Conda within Colab) can sometimes be fragile.
* Ensure sufficient compute resources (RAM, GPU VRAM) are available, especially for larger models used in RAFT and SICO.
