import shared_dir
# Upewnij się, że data_utils jest poprawnie zaimportowane
from .data_utils import load_list_from_tsv
import sys # Dodano dla obsługi błędów

# --- ORYGINALNA LOGIKA SICO (OCZEKUJE 3 KOLUMN W TSV) ---

def load_eval_data(dataset_name, task_name, eval_size=32):
    """Loads data for evaluating prompt utility during training."""
    file_path = shared_dir.dataset_dir + dataset_name + '/eval.tsv'
    print(f"[dataloader] Loading eval data from: {file_path}", flush=True)
    try:
        # Oryginalny kod zakładał 3 kolumny: input, human, ai
        raw_data_list = load_list_from_tsv(file_path, skip_header=True)
        print(f"[dataloader] Raw eval data rows loaded: {len(raw_data_list)}", flush=True)

        if task_name == 'paraphrase':
            # Dla parafrazy, wejściem jest oryginalny tekst AI (kolumna 2)
            final_data = [d[2] for d in raw_data_list if len(d) >= 3]
        else:
            # Dla innych zadań, wejściem jest prompt/input (kolumna 0)
            final_data = [d[0] for d in raw_data_list if len(d) >= 1]

        if not final_data:
             print(f"WARNING: No valid data loaded for eval (task: {task_name}). Check {file_path} format (expected 3 columns).", flush=True)
        elif len(final_data) < eval_size:
            print(f'WARNING: Eval size requested ({eval_size}) > available data ({len(final_data)}). Using all available.', flush=True)

        return final_data[:eval_size]

    except FileNotFoundError:
        print(f"ERROR: Eval file not found: {file_path}", flush=True)
        raise
    except IndexError as e:
        print(f"ERROR: IndexError while reading {file_path}. Expected 3 columns. Error: {e}", flush=True)
        raise
    except Exception as e:
        print(f"ERROR: Unexpected error loading eval data from {file_path}: {e}", flush=True)
        raise


def load_init_data_list(dataset_name, task_name, ic_size):
    """Loads data for initializing in-context examples."""
    file_path = shared_dir.dataset_dir + dataset_name + '/incontext.tsv'
    print(f"[dataloader] Loading in-context data from: {file_path}", flush=True)
    try:
        # Oryginalny kod zakładał 3 kolumny: input, human, ai
        raw_data_list = load_list_from_tsv(file_path, skip_header=True)
        print(f"[dataloader] Raw in-context data rows loaded: {len(raw_data_list)}", flush=True)

        if task_name == 'paraphrase':
            # Dla parafrazy, triplet to (AI_input, Human_ref, AI_input)
            # -> (kolumna 2, kolumna 1, kolumna 2)
            final_data = [(d[2], d[1], d[2]) for d in raw_data_list if len(d) >= 3]
        else:
            # Dla innych zadań, triplet to (Input, Human_ref, AI_output)
            # -> (kolumna 0, kolumna 1, kolumna 2)
            final_data = [(d[0], d[1], d[2]) for d in raw_data_list if len(d) >= 3]

        if not final_data:
             print(f"WARNING: No valid data loaded for in-context (task: {task_name}). Check {file_path} format (expected 3 columns).", flush=True)
        elif len(final_data) < ic_size:
            print(f'WARNING: In-context size requested ({ic_size}) > available data ({len(final_data)}). Using all available.', flush=True)

        return final_data[:ic_size]

    except FileNotFoundError:
        print(f"ERROR: In-context file not found: {file_path}", flush=True)
        raise
    except IndexError as e:
        print(f"ERROR: IndexError while reading {file_path}. Expected 3 columns. Error: {e}", flush=True)
        raise
    except Exception as e:
        print(f"ERROR: Unexpected error loading in-context data from {file_path}: {e}", flush=True)
        raise


def load_test_input(dataset_name, task_name):
    """Loads input data for testing generation."""
    file_path = shared_dir.dataset_dir + dataset_name + '/test.tsv'
    print(f"[dataloader] Loading test input data from: {file_path}", flush=True)
    try:
        # Oryginalny kod zakładał 3 kolumny
        raw_data_list = load_list_from_tsv(file_path, skip_header=True)
        print(f"[dataloader] Raw test data rows loaded: {len(raw_data_list)}", flush=True)

        if task_name == 'paraphrase':
            # Dla parafrazy, wejściem jest oryginalny tekst AI (kolumna 2)
            final_data = [d[2] for d in raw_data_list if len(d) >= 3]
        else:
            # Dla innych zadań, wejściem jest prompt/input (kolumna 0)
            final_data = [d[0] for d in raw_data_list if len(d) >= 1]

        if not final_data:
             print(f"WARNING: No valid data loaded for test input (task: {task_name}). Check {file_path} format (expected 3 columns).", flush=True)

        return final_data

    except FileNotFoundError:
        print(f"ERROR: Test file not found: {file_path}", flush=True)
        # Zwracamy pustą listę, żeby generowanie mogło obsłużyć brak danych
        return []
    except IndexError as e:
        print(f"ERROR: IndexError while reading {file_path}. Expected 3 columns for task '{task_name}'. Error: {e}", flush=True)
        return [] # Zwracamy pustą listę
    except Exception as e:
        print(f"ERROR: Unexpected error loading test input data from {file_path}: {e}", flush=True)
        return [] # Zwracamy pustą listę


def load_test_output_human(dataset_name):
    """Loads reference human outputs from test set."""
    file_path = shared_dir.dataset_dir + dataset_name + '/test.tsv'
    print(f"[dataloader] Loading reference human outputs from: {file_path}", flush=True)
    y_human_list = [] # Domyślnie pusta lista
    try:
        raw_data_list = load_list_from_tsv(file_path, skip_header=True)
        # Oryginalny kod oczekuje tekstu ludzkiego w kolumnie 1 (index 1)
        y_human_list = [d[1] for d in raw_data_list if len(d) >= 2]
        if not y_human_list and raw_data_list: # Jeśli wczytano wiersze, ale nie udało się pobrać kolumny 1
             print(f"WARNING: No valid data found in column 1 (human answers) in {file_path}.", flush=True)

    except FileNotFoundError:
        print(f"WARNING: Test file not found for human outputs: {file_path}", flush=True)
    except IndexError as e:
        print(f"WARNING: IndexError reading human outputs from {file_path}. Expected >=2 columns. Error: {e}", flush=True)
    except Exception as e:
        print(f"ERROR: Unexpected error loading human outputs from {file_path}: {e}", flush=True)
    return y_human_list


def load_test_output_ai(dataset_name):
    """Loads reference original AI outputs from test set."""
    file_path = shared_dir.dataset_dir + dataset_name + '/test.tsv'
    print(f"[dataloader] Loading reference AI outputs from: {file_path}", flush=True)
    y_ai_list = [] # Domyślnie pusta lista
    try:
        raw_data_list = load_list_from_tsv(file_path, skip_header=True)
        # Oryginalny kod oczekuje tekstu AI w kolumnie 2 (index 2)
        y_ai_list = [d[2] for d in raw_data_list if len(d) >= 3]
        if not y_ai_list and raw_data_list: # Jeśli wczytano wiersze, ale nie udało się pobrać kolumny 2
             print(f"WARNING: No valid data found in column 2 (AI answers) in {file_path}.", flush=True)

    except FileNotFoundError:
        print(f"WARNING: Test file not found for AI outputs: {file_path}", flush=True)
    except IndexError as e:
        print(f"WARNING: IndexError reading AI outputs from {file_path}. Expected >=3 columns. Error: {e}", flush=True)
    except Exception as e:
        print(f"ERROR: Unexpected error loading AI outputs from {file_path}: {e}", flush=True)
    return y_ai_list