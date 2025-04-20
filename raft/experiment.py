import os
import re
import json
import time
import uuid
import datetime
import argparse
import numpy as np
from tqdm import tqdm

import torch
import glob
# === NLTK WordNet for Antonym Check ===
import nltk
try:
    from nltk.corpus import wordnet
    # Download necessary NLTK data (run once)
    nltk.download('punkt', quiet=True) # Needed for tokenization by some NLTK functions if not already covered
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True) # Open Multilingual Wordnet
except ImportError:
    print("BŁĄD: Biblioteka NLTK lub jej korpus wordnet nie jest zainstalowany.")
    print("Uruchom: pip install nltk")
    print("Następnie w Pythonie: import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')")
    wordnet = None
# === Koniec NLTK ===

# === LanguageTool for Grammar Check ===
try:
    import language_tool_python
except ImportError:
    print("Biblioteka language_tool_python nie jest zainstalowana. Uruchom: pip install language_tool_python")
    print("Pamiętaj, że wymaga ona zainstalowanej Javy.")
    language_tool_python = None
# === Koniec LanguageTool ===

import logging
import pandas as pd
# === OpenAI API Client ===
try:
    from openai import OpenAI
except ImportError:
    print("Biblioteka openai nie jest zainstalowana. Uruchom: pip install openai")
    OpenAI = None # Set to None if import fails
# === Koniec OpenAI ===

from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from gensim.models import KeyedVectors

# Import detectors (assuming they are in the 'detectors' subdirectory)
from detectors.baselines import Baselines
from detectors.ghostbuster import Ghostbuster
from detectors.detect_gpt import Detect_GPT
from detectors.fast_detect_gpt import Fast_Detect_GPT
from detectors.roberta_gpt2_detector import GPT2RobertaDetector

import difflib

# Helper Functions
def word_similarity(word1, word2):
    """Oblicza podobieństwo między dwoma słowami (0 – brak, 1 – identyczne)."""
    return difflib.SequenceMatcher(None, word1, word2).ratio()

def is_antonym(word1, word2):
    """Sprawdza, czy word2 jest antonimem word1 używając WordNet."""
    if not wordnet: return False
    word1, word2 = word1.lower(), word2.lower() # Porównujemy małe litery
    if word1 == word2: return False # Słowo nie jest swoim własnym antonimem

    synsets1 = wordnet.synsets(word1)
    synsets2 = wordnet.synsets(word2)
    if not synsets1 or not synsets2: return False

    for syn1 in synsets1:
        for lemma1 in syn1.lemmas():
            if lemma1.antonyms():
                for antonym in lemma1.antonyms():
                    # Sprawdzamy, czy nazwa antonimu pasuje do któregokolwiek lemma drugiego słowa
                    for syn2 in synsets2:
                         if antonym.name() in [l.name() for l in syn2.lemmas()]:
                            return True
    return False

# --- Experiment Class ---
class Experiment:
    def __init__(self, args):
        self.dataset = args.dataset
        self.data_generator_llm = args.data_generator_llm
        self.proxy_model_str = args.proxy_model
        self.detector = args.detector
        self.output_path = args.output_path
        self.proxy_model_device = args.proxy_model_device
        self.target_detector_device = args.target_detector_device
        self.mask_pct = args.mask_pct
        self.top_k = args.top_k
        self.candidate_generation = args.candidate_generation
        self.dataset_dir = args.dataset_dir
        self.word2vec_model_path = args.word2vec_model_path
        self.local_llm_model_id = args.local_llm_model_id

        self.proxy_model_tokenizer = None
        self.proxy_model = None
        self.word_vectors = None
        self.logger = None
        self.config = None
        self.experiment_name = None
        self.experiment_path = None
        self.data = None
        self.candidate_llm_model = None
        self.candidate_llm_tokenizer = None
        self.language_tool = None
        self.openai_client = None # Inicjalizacja atrybutu

        # Inicjalizacja LanguageTool
        if language_tool_python:
            try:
                self.language_tool = language_tool_python.LanguageTool('en-US')
                print("LanguageTool initialized successfully.")
            except Exception as e:
                print(f"BŁĄD: Nie udało się zainicjalizować LanguageTool: {e}")
                self.language_tool = None
        else:
             print("OSTRZEŻENIE: language_tool_python nie jest zainstalowany. Filtr gramatyczny nie będzie działał.")

        # Inicjalizacja OpenAI Client (jeśli wybrano)
        if self.candidate_generation == "gpt-3.5-turbo":
            if not OpenAI:
                 print("BŁĄD KRYTYCZNY: Nie można zaimportować biblioteki openai. Zainstaluj ją: pip install openai")
                 raise ImportError("Biblioteka OpenAI nie jest dostępna.")
            try:
                # Klient automatycznie użyje zmiennej środowiskowej OPENAI_API_KEY
                self.openai_client = OpenAI()
                # Opcjonalnie: Sprawdź połączenie (może generować koszty/błędy przy starcie)
                # self.openai_client.models.list()
                print("OpenAI client initialized successfully.")
            except Exception as e:
                print(f"BŁĄD: Nie udało się zainicjalizować klienta OpenAI: {e}")
                print("Upewnij się, że zmienna środowiskowa OPENAI_API_KEY jest poprawnie ustawiona.")
                self.openai_client = None # Ustaw na None, jeśli inicjalizacja zawiedzie
                # Można zdecydować o przerwaniu działania, jeśli OpenAI jest wymagane
                # raise ConnectionError("Nie udało się zainicjalizować klienta OpenAI.") from e

        # Konfiguracja pozostałych elementów
        self.proxy_model_type = ("detection" if self.proxy_model_str in ["roberta-base-detector", "roberta-large-detector"] else "next-token-generation")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        # Ustawienie progów semantycznych (można dostosować) - użyjemy tych z przedostatniej próby
        self.similarity_threshold = 0.85
        self.word_semantic_threshold = 0.65
        self.word_similarity_threshold = 0.80 # Ten próg nie jest obecnie używany w logice

        # Ładowanie modeli do generowania kandydatów (jeśli nie OpenAI)
        if self.candidate_generation == "word2vec":
            w2v_path = self.word2vec_model_path if self.word2vec_model_path else "./assets/GoogleNews-vectors-negative300.bin"
            if not os.path.exists(w2v_path): raise FileNotFoundError(f"Nie znaleziono pliku Word2Vec: {w2v_path}")
            self.word_vectors = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
            print("Word2Vec model loaded.")
        elif self.candidate_generation == "local_llm":
            self.load_local_candidate_llm(self.local_llm_model_id)
            print(f"Local LLM for candidates ({self.local_llm_model_id}) loaded.")

    def filter_punctuation(self, string):
        pattern = r"^[\W_]+|[\W_]+$"
        left_punctuation = re.findall(r"^[\W_]*?", string)
        right_punctuation = re.findall(r"[\W_]*?$", string)
        clean_string = re.sub(pattern, "", string)
        return "".join(left_punctuation), "".join(right_punctuation), clean_string

    def get_top_similar_words(self, word, n=15):
        if not self.word_vectors: return []
        try: similar_words = self.word_vectors.most_similar(word.lower(), topn=n); return similar_words
        except (KeyError, AttributeError):
            msg = f"INFO: Słowo '{word}' nie występuje w modelu Word2Vec."
            if hasattr(self, 'logger') and self.logger: self.logger.info(msg)
            return []
        except Exception as e:
            msg = f"BŁĄD Word2Vec dla '{word}': {e}";
            if hasattr(self, 'logger') and self.logger: self.logger.error(msg, exc_info=True)
            else: print(msg)
            return []

    def get_candidate_words_word2vec(self, word, top_k=15):
        left_p, right_p, clean_word = self.filter_punctuation(word)
        if not clean_word: return []
        similar = self.get_top_similar_words(clean_word, n=top_k * 5)
        candidates = []
        for candidate_word_str, score in similar:
            if candidate_word_str.isalpha():
                candidates.append(candidate_word_str)
            if len(candidates) >= top_k: break
        if hasattr(self, 'logger') and self.logger:
             self.logger.debug(f"Word2Vec kandydaci dla '{clean_word}': {candidates}")
        return candidates

    def load_local_candidate_llm(self, model_id):
        model_id_path = os.path.join("/content/drive/MyDrive/raft/local_models", model_id.replace("/", "_"))
        if not os.path.isdir(model_id_path): raise FileNotFoundError(f"Nie znaleziono katalogu modelu lokalnego: {model_id_path}.")
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        if not torch.cuda.is_available(): pass
        try:
            self.candidate_llm_tokenizer = AutoTokenizer.from_pretrained(model_id_path, trust_remote_code=True)
            self.candidate_llm_model = AutoModelForCausalLM.from_pretrained(
                model_id_path, device_map="auto", quantization_config=quantization_config, trust_remote_code=True
            )
            if self.candidate_llm_tokenizer.pad_token_id is None:
                if self.candidate_llm_tokenizer.eos_token_id is not None:
                    self.candidate_llm_tokenizer.pad_token_id = self.candidate_llm_tokenizer.eos_token_id; self.candidate_llm_model.config.pad_token_id = self.candidate_llm_model.config.eos_token_id
                else:
                    self.candidate_llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'}); self.candidate_llm_model.resize_token_embeddings(len(self.candidate_llm_tokenizer))
                    if self.candidate_llm_model.config.pad_token_id is None: self.candidate_llm_model.config.pad_token_id = self.candidate_llm_tokenizer.pad_token_id
        except ImportError: print("KRYTYCZNY BŁĄD: Biblioteka 'bitsandbytes' jest wymagana do kwantyzacji 4-bit. Zainstaluj ją."); raise
        except Exception as e: print(f"KRYTYCZNY BŁĄD: Nie udało się załadować lokalnego LLM '{model_id}' z '{model_id_path}': {e}"); raise

    def get_candidate_words_local_llm(self, paragraph_context, word_to_replace, top_k):
        if not self.candidate_llm_model or not self.candidate_llm_tokenizer: return []
        # Uproszczony prompt bez wymogu POS
        prompt = f"""Given the context and the word "[{word_to_replace}]", provide up to {top_k} single-word semantically similar replacements that could fit in the context.
        Output ONLY a comma-separated list of the replacement words. Do not include numbers, explanations, or the original word.
        Context: "{paragraph_context.replace(f'[{word_to_replace}]', '[WORD_TO_REPLACE]', 1)}"
        Replacement words for [{word_to_replace}]:"""
        inputs = self.candidate_llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.candidate_llm_model.device)
        candidates = []
        try:
            self.candidate_llm_model.eval()
            with torch.no_grad():
                output_sequences = self.candidate_llm_model.generate(
                    **inputs, max_new_tokens=60, num_return_sequences=1, do_sample=True, temperature=0.7, top_k=50, top_p=0.9,
                    pad_token_id=self.candidate_llm_tokenizer.pad_token_id or self.candidate_llm_tokenizer.eos_token_id,
                    eos_token_id=self.candidate_llm_tokenizer.eos_token_id
                )
            generated_ids = output_sequences[0][inputs['input_ids'].shape[-1]:]; generated_text = self.candidate_llm_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            if hasattr(self, 'logger') and self.logger: self.logger.debug(f"LLM Raw Output for '{word_to_replace}': '{generated_text}'")
            generated_text = re.sub(r"Replacement words for.*?:\s*", "", generated_text, flags=re.IGNORECASE); generated_text = re.sub(r"^\W+|\W+$", "", generated_text)
            raw_candidates = [c.strip() for c in generated_text.split(',') if c.strip()]
            undesired_words = {"candidate", "word", "replacement", "alternative", word_to_replace.lower()}
            valid_word_pattern = r"^[A-Za-z]+(?:-[A-Za-z]+)*$"
            for cand in raw_candidates:
                if re.match(valid_word_pattern, cand) and cand.lower() not in undesired_words:
                     candidates.append(cand)
                if len(candidates) >= top_k: break
            if not candidates:
                 if hasattr(self, 'logger') and self.logger: self.logger.debug(f"Brak kandydatów po parsowaniu dla '{word_to_replace}'.")
            else:
                 if hasattr(self, 'logger') and self.logger: self.logger.debug(f"LLM kandydaci dla '{word_to_replace}': {candidates}")
        except Exception as e:
             if hasattr(self, 'logger') and self.logger: self.logger.error(f"Błąd generowania kandydatów LLM dla '{word_to_replace}': {e}", exc_info=True)
             else: print(f"BŁĄD generowania kandydatów LLM dla '{word_to_replace}': {e}")
             return []
        return candidates[:top_k]

    # === NOWA FUNKCJA: Generowanie kandydatów przez OpenAI ===
    def get_candidate_words_openai(self, paragraph_context, word_to_replace, top_k):
        """Generuje kandydatów używając API GPT-3.5 Turbo."""
        if not self.openai_client:
            msg = "Klient OpenAI nie został zainicjalizowany."
            if hasattr(self, 'logger') and self.logger: self.logger.error(msg)
            else: print(f"BŁĄD: {msg}")
            return []

        prompt = f"""Given the sentence context and the target word "[{word_to_replace}]", provide up to {top_k} single-word synonyms or very close semantic replacements that could fit naturally in the context.
        Focus on maintaining the original meaning and grammatical correctness.
        Output ONLY a comma-separated list of the replacement words. Do not include the original word, numbers, or any explanations.
        Context: "{paragraph_context.replace(f'[{word_to_replace}]', '[WORD_TO_REPLACE]', 1)}"
        Replacement words for [{word_to_replace}]:"""

        candidates = []
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that suggests word replacements."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=60,
                n=1,
                stop=None
            )
            generated_text = response.choices[0].message.content.strip()
            if hasattr(self, 'logger') and self.logger: self.logger.debug(f"OpenAI Raw Output for '{word_to_replace}': '{generated_text}'")

            generated_text = re.sub(r"Replacement words for.*?:\s*", "", generated_text, flags=re.IGNORECASE)
            generated_text = re.sub(r"^\W+|\W+$", "", generated_text)
            raw_candidates = [c.strip() for c in generated_text.split(',') if c.strip()]
            undesired_words = {"candidate", "word", "replacement", "alternative", word_to_replace.lower()}
            valid_word_pattern = r"^[A-Za-z]+(?:-[A-Za-z]+)*$"

            for cand in raw_candidates:
                if re.match(valid_word_pattern, cand) and cand.lower() not in undesired_words:
                     candidates.append(cand)
                if len(candidates) >= top_k: break

            if not candidates:
                 if hasattr(self, 'logger') and self.logger: self.logger.debug(f"Brak kandydatów po parsowaniu z OpenAI dla '{word_to_replace}'.")
            else:
                 if hasattr(self, 'logger') and self.logger: self.logger.debug(f"OpenAI kandydaci dla '{word_to_replace}': {candidates}")

        except Exception as e:
            msg = f"Błąd podczas wywołania API OpenAI dla '{word_to_replace}': {e}"
            if hasattr(self, 'logger') and self.logger: self.logger.error(msg, exc_info=True)
            else: print(f"BŁĄD: {msg}")
            print("Sprawdź swój klucz API OpenAI, limity użycia lub połączenie sieciowe.")
            return []

        return candidates[:top_k]
    # === KONIEC NOWEJ FUNKCJI ===

    def semantic_similarity(self, text1, text2):
        emb1 = self.embedding_model.encode(text1, convert_to_tensor=True); emb2 = self.embedding_model.encode(text2, convert_to_tensor=True)
        sim = util.cos_sim(emb1, emb2); return sim.item()

    def word_semantic_similarity(self, word1, word2):
        emb1 = self.embedding_model.encode(word1, convert_to_tensor=True); emb2 = self.embedding_model.encode(word2, convert_to_tensor=True)
        sim = util.cos_sim(emb1, emb2); return sim.item()

    def load_data(self, file_path):
        if os.path.exists(file_path):
            with open(file_path, "r", encoding='utf-8') as file: data = []; [data.append(json.loads(line.strip())) for line in file if line.strip()]; return data
        else: raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    def flatten(self, lst):
        flattened_list = []; [flattened_list.extend(self.flatten(item)) if isinstance(item, list) else flattened_list.append(item) for item in lst]; return flattened_list

    def load_dataset(self) -> None:
        if self.dataset not in ["xsum", "squad", "abstract", "custom_input"]: raise ValueError("Selected Dataset is invalid.")
        if self.dataset == "custom_input":
            custom_path = os.path.join(self.dataset_dir, "custom_input/test.jsonl")
            if not os.path.exists(custom_path):
                try: available_files = "\n".join(os.listdir(self.dataset_dir))
                except FileNotFoundError: available_files = "Nie można wylistować."
                raise ValueError(f"Custom dataset not found at {custom_path}\nAvailable in {self.dataset_dir}:\n{available_files}")
            try:
                loaded_data = self.load_data(custom_path)
                if isinstance(loaded_data, list) and len(loaded_data) > 0 and isinstance(loaded_data[0], dict) and 'text' in loaded_data[0]: self.data = {"sampled": [item['text'] for item in loaded_data if 'text' in item]}
                elif isinstance(loaded_data, dict) and 'text' in loaded_data: self.data = {"sampled": [loaded_data['text']]}
                else:
                     with open(custom_path, "r", encoding='utf-8') as file: lines = file.readlines()
                     self.data = {"sampled": [json.loads(line.strip())['text'] for line in lines if line.strip() and 'text' in json.loads(line.strip())]}
                if not self.data["sampled"]: raise ValueError("Plik test.jsonl pusty, brak klucza 'text' lub niepoprawny format JSON.")
                print(f"Loaded {len(self.data['sampled'])} samples from custom input.")
                return
            except Exception as e: raise ValueError(f"Error loading or parsing custom dataset from {custom_path}: {str(e)}")
        else:
            # ... (logika ładowania innych datasetów - bez zmian) ...
            pass # Dodaj tutaj logikę dla innych datasetów, jeśli potrzebne

    def load_proxy_model(self) -> None:
        # ... (bez zmian) ...
        proxy_model_map = {"roberta-base-detector": "roberta-base", "roberta-large-detector": "roberta-large"}
        proxy_model_checkpoint_map = {"roberta-base-detector": "./assets/detector-base.pt", "roberta-large-detector": "./assets/detector-large.pt"}
        device_handled_internally = False; model_identifier_for_print = self.proxy_model_str
        if self.proxy_model_str in proxy_model_map.keys():
            checkpoint_path = proxy_model_checkpoint_map[self.proxy_model_str]
            if not os.path.exists(checkpoint_path): raise FileNotFoundError(f"Nie znaleziono pliku checkpoint: {checkpoint_path}")
            try:
                self.proxy_model = GPT2RobertaDetector(proxy_model_map[self.proxy_model_str], self.proxy_model_device, checkpoint_path)
                device_handled_internally = True
                print(f"Proxy model (detector type) '{self.proxy_model_str}' loaded successfully.")
            except Exception as e:
                print(f"BŁĄD ładowania proxy model (detector type) '{self.proxy_model_str}': {e}")
                raise
        elif self.proxy_model_str == "gpt2":
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            try:
                self.proxy_model_tokenizer = GPT2Tokenizer.from_pretrained(self.proxy_model_str)
                self.proxy_model = GPT2LMHeadModel.from_pretrained(self.proxy_model_str)
                print(f"Proxy model '{self.proxy_model_str}' loaded successfully.")
            except Exception as e:
                print(f"BŁĄD ładowania proxy model '{self.proxy_model_str}': {e}")
                raise
        else:
            model_identifier_hf = self.proxy_model_str
            if self.proxy_model_str == "opt-2.7b": model_identifier_hf = "facebook/opt-2.7b"
            elif self.proxy_model_str == "neo-2.7b": model_identifier_hf = "EleutherAI/gpt-neo-2.7B"
            elif self.proxy_model_str == "gpt-j-6b": model_identifier_hf = "EleutherAI/gpt-j-6b"
            model_identifier_for_print = model_identifier_hf
            model_identifier_local = os.path.join("/content/drive/MyDrive/raft/local_models", model_identifier_hf.replace("/", "_"))
            if not os.path.isdir(model_identifier_local): raise FileNotFoundError(f"Nie znaleziono katalogu modelu proxy: {model_identifier_local}.")
            from transformers import AutoTokenizer, AutoModelForCausalLM
            try:
                dtype = torch.float16 if self.proxy_model_device == "cuda" else torch.float32
                self.proxy_model_tokenizer = AutoTokenizer.from_pretrained(model_identifier_local, torch_dtype=dtype)
                self.proxy_model = AutoModelForCausalLM.from_pretrained(model_identifier_local, torch_dtype=dtype)
                print(f"Proxy model '{model_identifier_for_print}' loaded from '{model_identifier_local}'.")
            except Exception as e:
                print(f"BŁĄD ładowania proxy model '{model_identifier_for_print}' z '{model_identifier_local}': {e}")
                raise
        if not device_handled_internally and self.proxy_model_device != "cpu" and hasattr(self.proxy_model, 'to'):
            try:
                self.proxy_model.to(self.proxy_model_device)
                print(f"Proxy model moved to device: {self.proxy_model_device}")
            except Exception as e:
                print(f"OSTRZEŻENIE: Nie udało się przenieść proxy model na urządzenie '{self.proxy_model_device}': {e}")
                pass

    def load_detector(self) -> None:
        # ... (bez zmian) ...
        raft_base_dir = os.getcwd()
        config_path = os.path.join(raft_base_dir, "detectors/*sampling_discrepancy.json")
        detector_base_path = os.path.join(raft_base_dir, "assets/detector-base.pt")
        detector_large_path = os.path.join(raft_base_dir, "assets/detector-large.pt")
        if self.detector == "dgpt":
            if not glob.glob(config_path): pass
            self.detector_model = Detect_GPT(config_path, 0.3, 1.0, 2, 10, "gpt2-xl", "t5-3b", device0=self.target_detector_device, device1=self.target_detector_device)
        elif self.detector == "fdgpt":
            if not glob.glob(config_path): pass
            self.detector_model = Fast_Detect_GPT("gpt2-xl", "gpt2-xl", "xsum", config_path, self.target_detector_device)
        elif self.detector == "ghostbuster": self.detector_model = Ghostbuster()
        elif self.detector == "logrank": self.detector_model = Baselines("logrank", "gpt-neo-2.7B", device=self.target_detector_device)
        elif self.detector == "logprob": self.detector_model = Baselines("likelihood", "gpt-neo-2.7B", device=self.target_detector_device)
        elif self.detector == "roberta-base":
            if not os.path.exists(detector_base_path): raise FileNotFoundError(f"Nie znaleziono: {detector_base_path}")
            self.detector_model = GPT2RobertaDetector("roberta-base", self.target_detector_device, detector_base_path)
        elif self.detector == "roberta-large":
            if not os.path.exists(detector_large_path): raise FileNotFoundError(f"Nie znaleziono: {detector_large_path}")
            self.detector_model = GPT2RobertaDetector("roberta-large", self.target_detector_device, detector_large_path)
        else: raise ValueError(f"Nieznany detektor: {self.detector}")
        print(f"Detector '{self.detector}' loaded successfully.")

    def create_experiment(self) -> None:
        current_date = datetime.datetime.now(); formatted_date = current_date.strftime("%Y-%m-%d_%H-%M-%S"); uid = str(uuid.uuid4())
        proxy_model_name_sanitized = self.proxy_model_str.replace('/', '_')
        self.experiment_name = f"{self.dataset}_{self.data_generator_llm}_{proxy_model_name_sanitized}_{self.detector}_{formatted_date}_{uid.split('-')[0]}"
        self.experiment_path = os.path.join(self.output_path, self.experiment_name); os.makedirs(self.experiment_path, exist_ok=True)
        self.config = {
            "dataset": self.dataset, "data_generator_llm": self.data_generator_llm, "proxy_model": self.proxy_model_str, "proxy_type": self.proxy_model_type,
            "mask_pct": self.mask_pct, "detector": self.detector, "output_path": self.output_path, "dataset_dir": self.dataset_dir,
            "timestamp_created": str(current_date), "candidate_generation": self.candidate_generation,
            "local_llm_model_id": self.local_llm_model_id if self.candidate_generation == "local_llm" else None,
            "sentence_similarity_threshold": self.similarity_threshold,
            "word_semantic_threshold": self.word_semantic_threshold,
            "experiment_name": self.experiment_name, "experiment_path": self.experiment_path,
            "grammar_tool": "language_tool_python" if self.language_tool else "None",
            "antonym_check": True if wordnet else False
        }
        config_filepath = os.path.join(self.experiment_path, "config.json");
        try:
            with open(config_filepath, "w") as f: json.dump(self.config, f, indent=4)
        except Exception as e: pass
        log_filepath = os.path.join(self.experiment_path, "experiment.log");
        # Zamknij istniejące handlery, aby uniknąć wielokrotnego logowania
        for handler in logging.root.handlers[:]: logging.root.removeHandler(handler); handler.close()
        logging.basicConfig(filename=log_filepath, filemode="w", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.DEBUG)
        self.logger = logging.getLogger(self.experiment_name); self.logger.info(f"Creating experiment {self.experiment_name}")
        check_path_pattern = os.path.join(self.output_path, f"{self.dataset}_{self.data_generator_llm}_{proxy_model_name_sanitized}_{self.detector}_*")
        existing_experiments = [d for d in glob.glob(check_path_pattern) if os.path.isdir(d) and os.path.basename(d).split('_')[:-2] == self.experiment_name.split('_')[:-2]]
        if len(existing_experiments) > 1: self.logger.warning(f"Duplicated experiment base name detected: {len(existing_experiments)} matching runs found.")

    def raft(self) -> None:
        def log_info(msg):
            if hasattr(self, 'logger') and self.logger: self.logger.info(msg)
        def log_warning(msg):
            if hasattr(self, 'logger') and self.logger: self.logger.warning(msg)
        def log_error(msg, exc_info=False):
            if hasattr(self, 'logger') and self.logger: self.logger.error(msg, exc_info=exc_info)
            else: print(f"BŁĄD: {msg}")
        def log_debug(msg):
            if hasattr(self, 'logger') and self.logger: self.logger.debug(msg)

        data = self.data; originals_likelihood, results_likelihood = [], []; original_crits_list, result_crits_list = [], []
        original_texts, result_texts = [], []
        if not data or "sampled" not in data or not data["sampled"]: log_error("Brak danych 'sampled' do przetworzenia."); return
        n_samples = len(data["sampled"]); log_info(f"Rozpoczynanie przetwarzania RAFT dla {n_samples} tekstów...")

        try:
            for index, paragraph in enumerate(tqdm(data["sampled"], desc="Przetwarzanie RAFT")):
                if not isinstance(paragraph, str) or not paragraph.strip():
                     log_warning(f"Pominięto niepoprawną lub pustą próbkę {index}.")
                     original_texts.append(paragraph); result_texts.append(paragraph); originals_likelihood.append(np.nan); results_likelihood.append(np.nan); original_crits_list.append(np.nan); result_crits_list.append(np.nan); continue
                words = paragraph.split(); len_paragraph = len(words)
                if len_paragraph == 0:
                    log_warning(f"Pominięto pustą próbkę {index} po podziale na słowa."); original_texts.append(paragraph); result_texts.append(paragraph); originals_likelihood.append(np.nan); results_likelihood.append(np.nan); original_crits_list.append(np.nan); result_crits_list.append(np.nan); continue

                words_to_replace_indices = []
                # ... (logika rankingu słów - bez zmian) ...
                if self.proxy_model_type == "next-token-generation":
                    try:
                        encoding = self.proxy_model_tokenizer(paragraph, add_special_tokens=True, max_length=1024, truncation=True, return_offsets_mapping=True, return_tensors="pt").to(self.proxy_model_device)
                        model_inputs = {k: v for k, v in encoding.items() if k != 'offset_mapping'}
                        tokens_id = encoding["input_ids"][0]; offsets = encoding["offset_mapping"][0]
                        word_boundaries = [(m.start(), m.end()) for m in re.finditer(r'\S+', paragraph)]; word_scores = [0.0] * len(word_boundaries)
                        if len(tokens_id) <= 1: log_warning(f"Próbka {index} jest zbyt krótka do rankingu ({len(tokens_id)} tokenów)."); words_to_replace_indices = []
                        else:
                             with torch.no_grad(): outputs = self.proxy_model(**model_inputs); logits = outputs.logits
                             log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1); token_importances = []
                             for i in range(len(tokens_id)):
                                  if i == 0: token_importances.append(0.0); continue
                                  current_token_log_prob = log_probs[i-1, tokens_id[i]].item(); token_importances.append(-current_token_log_prob)
                             for i in range(len(tokens_id)):
                                 token_offset = offsets[i];
                                 if token_offset[0] == 0 and token_offset[1] == 0: continue
                                 for word_idx, (w_start, w_end) in enumerate(word_boundaries):
                                     if max(token_offset[0], w_start) < min(token_offset[1], w_end): word_scores[word_idx] = max(word_scores[word_idx], token_importances[i]); break
                             sorted_word_indices = sorted(range(len(word_boundaries)), key=lambda i: word_scores[i], reverse=True)
                             num_masks = int(len(word_boundaries) * self.mask_pct); words_to_replace_indices = sorted_word_indices[:num_masks]
                             log_debug(f"Próbka {index}: Ranking słów (indeks, słowo, max_token_importance): {[(idx, words[idx], word_scores[idx]) for idx in words_to_replace_indices]}")
                    except Exception as proxy_rank_e: log_error(f"Próbka {index}: Błąd rankingu słów przez proxy: {proxy_rank_e}", exc_info=True); words_to_replace_indices = []
                else: # Tryb detection
                    # ... (logika rankingu dla detektorów - bez zmian) ...
                    pass

                current_words = list(words); mask_keys = []; num_replaced = 0
                for key in tqdm(sorted(words_to_replace_indices), desc=f"Zamiana słów (próbka {index+1})", leave=False):
                    try:
                        if key >= len(current_words): continue
                        word_to_replace = current_words[key]
                        left_punctuation, right_punctuation, clean_word_to_replace = self.filter_punctuation(word_to_replace)
                        if not clean_word_to_replace: log_debug(f"Próbka {index}: Pomijanie słowa '{word_to_replace}' (indeks {key}, puste po oczyszczeniu)."); continue

                        current_paragraph_context = " ".join(current_words)
                        # Używamy oryginalnego słowa w kontekście, aby uniknąć problemów z tokenizacją [WORD_TO_REPLACE]
                        context_for_llm = current_paragraph_context

                        predicted_words = []
                        try:
                            if self.candidate_generation == "word2vec":
                                predicted_words = self.get_candidate_words_word2vec(clean_word_to_replace, self.top_k)
                            elif self.candidate_generation == "local_llm":
                                predicted_words = self.get_candidate_words_local_llm(context_for_llm, clean_word_to_replace, self.top_k)
                            # === ZMIANA: Dodano wywołanie dla OpenAI ===
                            elif self.candidate_generation == "gpt-3.5-turbo":
                                if not self.openai_client: log_warning("Próba użycia OpenAI, ale klient nie został zainicjalizowany."); continue
                                predicted_words = self.get_candidate_words_openai(context_for_llm, clean_word_to_replace, self.top_k)
                            # === KONIEC ZMIANY ===
                        except Exception as gen_e: log_error(f"Błąd generowania kandydatów dla '{clean_word_to_replace}': {gen_e}", exc_info=True); predicted_words = []

                        if not predicted_words: log_debug(f"Próbka {index}: Brak kandydatów dla '{clean_word_to_replace}' (indeks {key})."); continue

                        valid_candidates = predicted_words

                        min_score = float("inf"); word_best = word_to_replace; candidate_found_better = False
                        current_paragraph_for_eval = " ".join(current_words)
                        try:
                            score_before_replacement = self.detector_model.crit(current_paragraph_for_eval); min_score = score_before_replacement
                            log_debug(f"Próbka {index}, słowo {key} ('{word_to_replace}'): Score przed zamianą = {score_before_replacement:.4f}")
                        except Exception as score_e: log_warning(f"Nie można uzyskać wyniku detektora przed zamianą słowa {key} w próbce {index}: {score_e}"); continue

                        log_debug(f"--- Kandydaci dla '{clean_word_to_replace}' (indeks {key}) ---")
                        for predicted_word_clean in valid_candidates:
                            log_debug(f"  Sprawdzanie kandydata: '{predicted_word_clean}'")

                            # --- Filtry jakościowe ---
                            # 1. Filtr długości (odrzuca pojedyncze litery, jeśli oryginał był dłuższy)
                            if len(predicted_word_clean) == 1 and len(clean_word_to_replace) > 1:
                                log_debug(f"    -> ODRZUCONY (Filtr długości)")
                                continue
                            # 2. Filtr dla 'a'/'i' (odrzuca inne pojedyncze litery jako zamienniki dla 'a' lub 'i')
                            if clean_word_to_replace.lower() in ['a', 'i'] and len(predicted_word_clean) == 1 and predicted_word_clean.lower() != clean_word_to_replace.lower():
                                log_debug(f"    -> ODRZUCONY (Filtr 'a'/'i')")
                                continue

                            # Dopasowanie kapitalizacji i interpunkcji
                            if word_to_replace.istitle(): candidate_adjusted = predicted_word_clean.capitalize()
                            elif word_to_replace.isupper(): candidate_adjusted = predicted_word_clean.upper()
                            elif word_to_replace and word_to_replace[0].isupper() and len(predicted_word_clean) > 0 : candidate_adjusted = predicted_word_clean[0].upper() + predicted_word_clean[1:]
                            else: candidate_adjusted = predicted_word_clean.lower()
                            predicted_word_full = left_punctuation + candidate_adjusted + right_punctuation

                            paragraph_new = " ".join(current_words[:key] + [predicted_word_full] + current_words[key+1:])

                            # 3. Filtr gramatyczny language_tool
                            if self.language_tool:
                                try:
                                    matches = self.language_tool.check(paragraph_new)
                                    if len(matches) > 0:
                                        error_rules = [m.ruleId + '(' + m.message + ')' for m in matches]
                                        log_debug(f"    -> ODRZUCONY (LanguageTool Found Errors: {error_rules})")
                                        continue
                                    log_debug(f"    LanguageTool Check: Passed")
                                except Exception as lt_e:
                                    log_warning(f"Błąd podczas sprawdzania gramatyki przez LanguageTool dla '{paragraph_new[:50]}...': {lt_e}")
                                    log_debug(f"    -> ODRZUCONY (LanguageTool Error)")
                                    continue
                            else:
                                log_debug(f"    LanguageTool Check: SKIPPED (not initialized)")

                            # 4. Filtr antonimów
                            if wordnet and is_antonym(clean_word_to_replace.lower(), predicted_word_clean.lower()):
                                log_debug(f"    -> ODRZUCONY (WordNet Antonym Found)")
                                continue
                            log_debug(f"    Antonym Check: Passed (or WordNet unavailable)")

                            # 5. Filtry semantyczne (z zaostrzonymi progami)
                            word_sem_sim = self.word_semantic_similarity(clean_word_to_replace, predicted_word_clean)
                            log_debug(f"    Word Sem Sim: {word_sem_sim:.4f} (Threshold: {self.word_semantic_threshold})")
                            if word_sem_sim < self.word_semantic_threshold: log_debug(f"    -> ODRZUCONY (Word Sem Sim)"); continue

                            sent_sim = self.semantic_similarity(current_paragraph_for_eval, paragraph_new)
                            log_debug(f"    Sentence Sim: {sent_sim:.4f} (Threshold: {self.similarity_threshold})")
                            if sent_sim < self.similarity_threshold: log_debug(f"    -> ODRZUCONY (Sentence Sim)"); continue
                            # --- Koniec filtrów jakościowych ---

                            # Ocena detektora
                            try:
                                score = self.detector_model.crit(paragraph_new)
                                log_debug(f"    Detector Score: {score:.4f} (Current Min: {min_score:.4f})")
                                if score < min_score:
                                    log_debug(f"    -> AKCEPTOWANY (Lepszy niż min_score)")
                                    word_best = predicted_word_full; min_score = score; candidate_found_better = True
                                else: log_debug(f"    -> ODRZUCONY (Score nie lepszy)")
                            except Exception as eval_e: log_warning(f"Próbka {index}: Błąd oceny kandydata '{predicted_word_full}' przez detektor: {eval_e}"); log_debug(f"    -> ODRZUCONY (Błąd detektora)"); pass
                        log_debug(f"--- Koniec kandydatów dla '{clean_word_to_replace}' ---")

                        if candidate_found_better:
                            log_info(f"Próbka {index}: Dokonano zamiany '{word_to_replace}' -> '{word_best}' (indeks {key}), nowy score: {min_score:.4f} (stary: {score_before_replacement:.4f})")
                            current_words[key] = word_best; mask_keys.append(key); num_replaced += 1
                        else: log_debug(f"Próbka {index}: Brak lepszych kandydatów spełniających kryteria dla '{word_to_replace}' (indeks {key}) – pozostawiono oryginał.")
                    except Exception as word_proc_e: log_error(f"Próbka {index}: Nieoczekiwany błąd podczas przetwarzania słowa {key} ('{word_to_replace}'): {word_proc_e}", exc_info=True); continue

                paragraph_final = " ".join(current_words); original_texts.append(paragraph); result_texts.append(paragraph_final)
                original_ll, original_crit = np.nan, np.nan; result_ll, result_crit = np.nan, np.nan
                log_info(f"Próbka {index}: Zakończono przetwarzanie, dokonano {num_replaced} zamian.")
                try:
                    original_crit_val, result_crit_val = np.nan, np.nan; original_ll_val, result_ll_val = np.nan, np.nan
                    try: original_crit_val = self.detector_model.crit(paragraph)
                    except Exception as e: log_warning(f"Próbka {index}: Błąd oceny crit dla oryginału: {e}")
                    try: result_crit_val = self.detector_model.crit(paragraph_final)
                    except Exception as e: log_warning(f"Próbka {index}: Błąd oceny crit dla wyniku: {e}")
                    if self.detector in ["roberta-base", "roberta-large", "ghostbuster"]: original_ll_val = original_crit_val; result_ll_val = result_crit_val
                    else:
                        try: original_ll_val, _, _ = self.detector_model.run(paragraph)
                        except Exception as e: log_warning(f"Próbka {index}: Błąd oceny run (likelihood) dla oryginału: {e}")
                        try: result_ll_val, _, _ = self.detector_model.run(paragraph_final)
                        except Exception as e: log_warning(f"Próbka {index}: Błąd oceny run (likelihood) dla wyniku: {e}")
                    original_crit = original_crit_val; result_crit = result_crit_val; original_ll = original_ll_val; result_ll = result_ll_val
                except Exception as final_score_e: log_error(f"Błąd podczas końcowej oceny detektora dla próbki {index}: {final_score_e}", exc_info=True)

                originals_likelihood.append(original_ll); results_likelihood.append(result_ll)
                original_crits_list.append(original_crit); result_crits_list.append(result_crit)
                output_json = {
                    "original": paragraph, "sampled": paragraph_final, "replacement_keys": sorted(mask_keys),
                    "original_crit": float(original_crit) if not np.isnan(original_crit) else None, "sampled_crit": float(result_crit) if not np.isnan(result_crit) else None,
                    "original_llm_likelihood": float(original_ll) if not np.isnan(original_ll) else None, "sampled_llm_likelihood": float(result_ll) if not np.isnan(result_ll) else None,
                }
                try:
                    individual_filepath = os.path.join(self.experiment_path, f"result_{index}.json")
                    with open(individual_filepath, "w", encoding='utf-8') as output_file: json.dump(output_json, output_file, indent=4, ensure_ascii=False)
                except Exception as e: log_error(f"Błąd zapisu wyniku indywidualnego {index}: {e}", exc_info=True)
                if (index + 1) % 50 == 0 or (index + 1) == n_samples :
                    log_info(f"Zapisywanie wyników pośrednich/końcowych po próbce {index+1}...")
                    self._save_aggregated_results(originals_likelihood, results_likelihood, original_crits_list, result_crits_list, original_texts, result_texts)

            # Zapis CSV po zakończeniu pętli
            try:
                df = pd.DataFrame({
                    "original_text": original_texts, "sampled_text": result_texts,
                    "original_crits": [float(x) if x is not None and not np.isnan(x) else None for x in original_crits_list],
                    "sampled_crits": [float(x) if x is not None and not np.isnan(x) else None for x in result_crits_list],
                    "original_llm_likelihood": [float(x) if x is not None and not np.isnan(x) else None for x in originals_likelihood],
                    "sampled_llm_likelihood": [float(x) if x is not None and not np.isnan(x) else None for x in results_likelihood],
                })
                df.to_csv(os.path.join(self.experiment_path, "results_summary.csv"), index=False)
                log_info("Zapisano końcowe podsumowanie CSV.")
            except Exception as csv_e: log_error(f"Błąd zapisu końcowego podsumowania CSV: {csv_e}", exc_info=True)

        finally:
             # Zamknięcie language_tool
             if hasattr(self, 'language_tool') and self.language_tool:
                 try:
                     self.language_tool.close()
                     print("LanguageTool closed.")
                 except Exception as lt_close_e:
                     msg = f"Błąd podczas zamykania LanguageTool: {lt_close_e}"
                     if hasattr(self, 'logger') and self.logger: self.logger.warning(msg)
                     else: print(msg)

             # Zamknięcie loggera
             if hasattr(self, 'logger') and self.logger:
                for handler in self.logger.handlers[:]:
                    try: handler.flush(); handler.close(); self.logger.removeHandler(handler)
                    except: pass
                logging.shutdown()
                print("Logger closed.")


    def _save_aggregated_results(self, originals_ll, results_ll, original_crits, result_crits, original_txts, result_txts):
        def safe_float_list(data): return [float(x) if x is not None and not np.isnan(x) else None for x in data]
        valid_originals_ll = [x for x in originals_ll if x is not None and not np.isnan(x)]
        valid_results_ll = [x for x in results_ll if x is not None and not np.isnan(x)]
        mean_original_ll = np.mean(valid_originals_ll) if valid_originals_ll else np.nan
        mean_result_ll = np.mean(valid_results_ll) if valid_results_ll else np.nan
        result_summary_json = {
            "mean_original_llm_likelihood": float(mean_original_ll) if not np.isnan(mean_original_ll) else None,
            "mean_sampled_llm_likelihood": float(mean_result_ll) if not np.isnan(mean_result_ll) else None,
            "originals_llm_likelihood": safe_float_list(originals_ll), "sampled_llm_likelihood": safe_float_list(results_ll),
            "original_crits": safe_float_list(original_crits), "sampled_crits": safe_float_list(result_crits),
        }
        try:
            summary_filepath = os.path.join(self.experiment_path, "results_summary.json")
            with open(summary_filepath, "w", encoding='utf-8') as result_file: json.dump(result_summary_json, result_file, indent=4, ensure_ascii=False)
        except Exception as e: pass

    def run(self) -> None:
        try:
            self.load_dataset(); self.load_proxy_model(); self.load_detector(); self.create_experiment(); self.raft()
        except Exception as run_e:
            msg = f"KRYTYCZNY BŁĄD podczas Experiment.run(): {run_e}"
            if hasattr(self, 'logger') and self.logger: self.logger.critical(msg, exc_info=True)
            else: print(msg)
            import traceback; traceback.print_exc()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["xsum", "squad", "abstract", "custom_input"], default="custom_input")
    parser.add_argument("--mask_pct", default=0.1, type=float)
    parser.add_argument("--top_k", default=15, type=int)
    parser.add_argument("--data_generator_llm", choices=["gpt-3.5-turbo", "mixtral-8x7B-Instruct", "llama-3-70b-chat"], default="gpt-3.5-turbo", help="Original LLM (less relevant for custom_input)")
    parser.add_argument("--proxy_model", choices=["roberta-base-detector", "roberta-large-detector", "gpt2", "opt-2.7b", "neo-2.7b", "gpt-j-6b"], default="gpt2")
    parser.add_argument("--detector", choices=["logprob", "logrank", "dgpt", "fdgpt", "ghostbuster", "roberta-base", "roberta-large"], default="roberta-base")
    parser.add_argument("--output_path", default="./experiments/")
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--proxy_model_device", default=default_device)
    parser.add_argument("--target_detector_device", default=default_device)
    parser.add_argument("--candidate_generation", choices=["gpt-3.5-turbo", "word2vec", "local_llm"], default="word2vec")
    parser.add_argument("--local_llm_model_id", type=str, default="microsoft/phi-2", help="HF model ID for local candidate LLM (if candidate_generation='local_llm')")
    parser.add_argument("--dataset_dir", default="./datasets/")
    parser.add_argument("--word2vec_model_path", type=str, default="./assets/GoogleNews-vectors-negative300.bin")
    return parser.parse_args()

print("Done Loading!")

if __name__ == "__main__":
    args = get_args()
    print("Running test...")
    print("Initializing")
    experiment = Experiment(args=args)
    experiment.run()