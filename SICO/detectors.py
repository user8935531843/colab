# Plik: /content/drive/MyDrive/SICO/detectors.py
# Wersja z dodaną obsługą skracania (truncation) dla długich tekstów

import os
import time
import re
import torch
import numpy as np
import openai
import requests
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, GPT2LMHeadModel, GPT2TokenizerFast, T5Tokenizer, RobertaForSequenceClassification, RobertaTokenizer, AutoModelForSeq2SeqLM
import functools # Dodano import functools

# Upewnij się, że data_utils i model_path są importowalne
try:
    from my_utils.data_utils import load_from_pickle, save_to_pickle
    from my_utils.model_path import get_model_path
except ImportError:
    print("ERROR: Could not import helper modules (data_utils, model_path). Make sure they are in the correct directory and PYTHONPATH if needed.")
    # Definicje zastępcze, aby uniknąć błędów importu, ale funkcjonalność będzie ograniczona
    def get_model_path(name): print(f"Warning: Using fallback get_model_path for {name}"); return name
    def load_from_pickle(path): print(f"Warning: Using fallback load_from_pickle for {path}"); return {}
    def save_to_pickle(obj, path): print(f"Warning: Using fallback save_to_pickle for {path}")

# --- Klasa Bazowa ---
class AIDetector:
    def __call__(self, text_list, disable_tqdm=True):
        raise NotImplementedError
    def get_threshold(self):
        raise NotImplementedError
    def save_cache(self):
        return

# --- Detektory RoBERTa (chatdetect, gpt2detect) ---
class RoBERTaAIDetector(AIDetector):
    def __init__(self, device, name='chatdetect', batch_size=64):
        print(f"[Detector Init] Initializing {name}...", flush=True)
        if name == 'chatdetect':
            model_name = get_model_path("Hello-SimpleAI/chatgpt-detector-roberta")
            self.ai_label = 1
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            # Spróbuj pobrać max_length, ustaw domyślny jeśli brakuje
            self.model_max_length = getattr(self.tokenizer, 'model_max_length', 512)
        elif name == 'gpt2detect':
            model_name = get_model_path('roberta-base-openai-detector')
            self.ai_label = 0
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            self.model = RobertaForSequenceClassification.from_pretrained(model_name)
            # Jawne ustawienie limitu dla RoBERTa
            self.model_max_length = 512
            # Ustawienie atrybutu w tokenizerze dla spójności, jeśli go nie ma
            if not hasattr(self.tokenizer, 'model_max_length'):
                 self.tokenizer.model_max_length = self.model_max_length
        else:
            raise ValueError(f'Wrong RoBERTa detector name: {name}')

        self.device = device
        self.model.to(device)
        self.model.eval()
        self.batch_size = batch_size
        print(f"[Detector Init] {name} initialized on {device}. Max length: {self.model_max_length}", flush=True)

    def __call__(self, text_list, disable_tqdm=True):
        if not text_list: return [], []
        num_examples = len(text_list)
        num_batches = (num_examples + self.batch_size - 1) // self.batch_size
        all_logits = []
        all_labels = []
        print(f"[Detector Call] Processing {num_examples} texts in {num_batches} batches (batch size: {self.batch_size}).", flush=True)

        with torch.no_grad():
            batch_iterator = range(num_batches)
            if not disable_tqdm:
                batch_iterator = tqdm(batch_iterator, desc=f"Detector {self.model.name_or_path.split('/')[-1]}")

            for i in batch_iterator:
                start_index = i * self.batch_size
                end_index = min((i + 1) * self.batch_size, num_examples)
                batch_texts = text_list[start_index:end_index]
                if not batch_texts: continue

                try:
                    # --- POPRAWKA: Zapewnienie max_length ---
                    batch_inputs = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,                   # Włącz skracanie
                        max_length=self.model_max_length,  # Użyj limitu modelu/tokenizera
                        return_tensors="pt"
                    )
                    # --- KONIEC POPRAWKI ---
                    batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
                    batch_outputs = self.model(**batch_inputs)
                    batch_predicted_labels = torch.argmax(batch_outputs.logits, dim=1)
                    all_logits.append(batch_outputs.logits.cpu())
                    all_labels.append(batch_predicted_labels.cpu())
                except Exception as e:
                    print(f"ERROR during RoBERTa batch {i} processing: {e}", flush=True)
                    # Wypełnij domyślnymi wartościami dla tego batcha
                    batch_len = len(batch_texts)
                    dummy_logits = torch.zeros((batch_len, self.model.config.num_labels)) # Zakładamy 2 etykiety
                    dummy_labels = torch.zeros(batch_len, dtype=torch.long)
                    all_logits.append(dummy_logits)
                    all_labels.append(dummy_labels)

        if not all_logits:
             print("ERROR: No results obtained from RoBERTa detector.", flush=True)
             return [0.5] * num_examples, [0] * num_examples

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)

        # Sprawdź czy liczba wyników pasuje do liczby wejść
        if len(logits) != num_examples:
             print(f"WARNING: Length mismatch after processing batches ({len(logits)} vs {num_examples}). Results might be inaccurate.", flush=True)
             # Dopasuj długość, jeśli to możliwe (choć to ukrywa błąd)
             if len(logits) < num_examples:
                 padding_needed = num_examples - len(logits)
                 logits = torch.cat([logits, torch.zeros((padding_needed, logits.shape[1]))], dim=0)
                 labels = torch.cat([labels, torch.zeros(padding_needed, dtype=torch.long)], dim=0)
             else: # Za dużo wyników? Obetnij
                 logits = logits[:num_examples]
                 labels = labels[:num_examples]


        probs = torch.nn.functional.softmax(logits, dim=-1)
        if self.ai_label >= probs.shape[1]:
             print(f"ERROR: Invalid ai_label ({self.ai_label}) for model output shape {probs.shape}", flush=True)
             ai_score_list = [0.5] * len(logits)
             detect_preds = [0] * len(labels)
        else:
             ai_score_list = probs[:, self.ai_label].numpy().tolist()
             detect_preds = labels.numpy()
             if self.ai_label == 0:
                 detect_preds = 1 - detect_preds

        return ai_score_list, detect_preds.tolist()

    def get_threshold(self): return 0.5

# --- Detektor DetectGPT ---
class DetectGPT(AIDetector):
    def __init__(self, threshold=0.9, sample_num=50, mask_device='cuda:0', base_device='cuda:1', cache_path=None, use_cache=True):
        print("[Detector Init] Initializing DetectGPT...", flush=True)
        gpt_model_path = get_model_path("gpt2-medium")
        self.base_device = base_device
        print(f"[Detector Init] Loading base model {gpt_model_path}...", flush=True)
        self.base_model = GPT2LMHeadModel.from_pretrained(gpt_model_path).to(base_device)
        self.base_tokenizer = GPT2TokenizerFast.from_pretrained(gpt_model_path)
        self.base_model.eval()
        self.base_model_max_length = self.base_model.config.n_positions # Powinno być 1024
        # Ustawienie w tokenizerze dla pewności
        self.base_tokenizer.model_max_length = self.base_model_max_length
        print(f"[Detector Init] Base model loaded on {self.base_model.device}. Max length: {self.base_model_max_length}", flush=True)

        t5_model_path = get_model_path('t5-large')
        self.mask_device = mask_device
        print(f"[Detector Init] Loading mask model {t5_model_path}...", flush=True)
        try:
            self.mask_model = AutoModelForSeq2SeqLM.from_pretrained(t5_model_path, torch_dtype=torch.float16).to(mask_device)
        except:
            print("Warning: float16 not supported, loading T5 in float32.", flush=True)
            self.mask_model = AutoModelForSeq2SeqLM.from_pretrained(t5_model_path).to(mask_device)
        self.mask_tokenizer = T5Tokenizer.from_pretrained(t5_model_path)
        # Upewnij się, że mask_tokenizer ma ustawioną max_length (T5 ma różne warianty)
        self.mask_model_max_length = getattr(self.mask_tokenizer, 'model_max_length', 512)
        if not self.mask_model_max_length: self.mask_model_max_length = 512 # Ustaw domyślny, jeśli brak
        self.mask_tokenizer.model_max_length = self.mask_model_max_length

        self.mask_model.eval()
        print(f"[Detector Init] Mask model loaded on {self.mask_model.device}. Max length: {self.mask_model_max_length}", flush=True)

        self.stride = 51
        self.mask_rate = 0.3
        self.span_length = 2
        self.perturb_pct = 0.3
        self.chunk_size = 100
        self.sample_num = sample_num
        self.threshold = threshold
        self.pattern = re.compile(r"<extra_id_\d+>")

        self.cache_path = cache_path
        self.use_cache = use_cache
        self.cache_change = False
        if self.cache_path and os.path.exists(self.cache_path) and self.use_cache:
            try:
                 self.cache = load_from_pickle(self.cache_path)
                 print(f"[Detector Init] Loaded cache from {self.cache_path} ({len(self.cache)} entries).", flush=True)
            except Exception as e:
                 print(f"Warning: Failed to load cache from {self.cache_path}: {e}", flush=True)
                 self.cache = {}
        else:
            self.cache = {}
            print(f"[Detector Init] Initialized empty cache (path: {self.cache_path}, use_cache: {self.use_cache}).", flush=True)

    @staticmethod
    def count_masks(texts):
        return [len(re.findall(r"<extra_id_\d+>", text)) for text in texts]

    def __call__(self, text_list, disable_tqdm=True):
        # (reszta metody __call__ bez zmian - używa get_ai_score)
        if not text_list: return [], []
        ai_prob_list = []
        label_list = []
        iterator = range(len(text_list))
        if not disable_tqdm: iterator = tqdm(iterator, desc="DetectGPT Processing")

        print(f"[Detector Call] Processing {len(text_list)} texts with DetectGPT.", flush=True)
        for i in iterator:
            text = text_list[i]
            if self.use_cache and text in self.cache:
                ai_score = self.cache[text]
            else:
                try:
                    ai_score = self.get_ai_score(text)
                    if np.isnan(ai_score) or np.isinf(ai_score):
                         print(f"Warning: Got NaN/Inf score for text {i}. Setting score to 0.", flush=True)
                         ai_score = 0.0
                    self.cache[text] = ai_score
                    self.cache_change = True
                except Exception as e:
                    print(f"ERROR calculating DetectGPT score for text {i}: {e}", flush=True)
                    ai_score = 0.0

            cur_label = 1 if ai_score > self.threshold else 0
            ai_prob_list.append(ai_score)
            label_list.append(cur_label)

            if hasattr(iterator, 'set_description'):
                 iterator.set_description(f'DetectGPT Score: {np.mean(ai_prob_list):.4f}')

        print(f"[Detector Call] Finished DetectGPT processing.", flush=True)
        return ai_prob_list, label_list

    def get_threshold(self): return self.threshold
    def save_cache(self):
        if self.cache_path and self.cache_change:
            print(f"[Detector SaveCache] Saving cache ({len(self.cache)} entries) to {self.cache_path}", flush=True)
            try:
                save_to_pickle(self.cache, self.cache_path)
                self.cache_change = False
            except Exception as e:
                 print(f"ERROR saving cache to {self.cache_path}: {e}", flush=True)

    def get_ai_score(self, text):
        # (Kod tej metody bez zmian - wywołuje perturb_texts i get_lls/get_ll)
        try:
            p_sampled_text = self.perturb_texts([text] * self.sample_num)
            if not p_sampled_text or all(s == '' for s in p_sampled_text):
                 print(f"Warning: Perturbation failed for text '{text[:50]}...'. Returning score 0.", flush=True)
                 return 0.0
        except Exception as e:
             print(f"ERROR during perturb_texts for '{text[:50]}...': {e}", flush=True)
             return 0.0

        try:
             p_sequence_ll = self.get_lls(p_sampled_text)
             original_ll = self.get_ll(text)
             valid_lls = [ll for ll in p_sequence_ll if ll is not None and not np.isnan(ll) and not np.isinf(ll)]
             if not valid_lls:
                 print(f"Warning: No valid perturbed log-likelihoods for '{text[:50]}...'. Returning score 0.", flush=True)
                 return 0.0
        except Exception as e:
             print(f"ERROR during get_lls/get_ll for '{text[:50]}...': {e}", flush=True)
             return 0.0

        mean_perturbed_ll = np.mean(valid_lls)
        std_perturbed_ll = np.std(valid_lls)
        if std_perturbed_ll == 0: std_perturbed_ll = 1

        if original_ll is None or np.isnan(original_ll) or np.isinf(original_ll):
             print(f"Warning: Invalid original log-likelihood ({original_ll}) for '{text[:50]}...'. Returning score 0.", flush=True)
             return 0.0

        z_score = (original_ll - mean_perturbed_ll) / std_perturbed_ll
        return z_score

    def perturb_texts(self, texts):
        # (Kod tej metody bez zmian)
        outputs = []
        try:
            for i in range(0, len(texts), self.chunk_size):
                chunk = texts[i:i + self.chunk_size]
                if not chunk: continue
                outputs.extend(self._perturb_texts(chunk))
        except Exception as e:
             print(f"ERROR in perturb_texts chunk processing: {e}", flush=True)
             outputs.extend([''] * len(chunk))
        return outputs

    def _perturb_texts(self, texts):
        # (Kod tej metody bez zmian)
        masked_texts = [self.tokenize_and_mask(x) for x in texts]
        raw_fills = self.replace_masks(masked_texts)
        extracted_fills = self.extract_fills(raw_fills)
        perturbed_texts = self.apply_extracted_fills(masked_texts, extracted_fills)
        attempts = 1
        while '' in perturbed_texts and attempts <= 10:
            idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
            print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].', flush=True)
            texts_to_retry = [texts[idx] for idx in idxs]
            if not texts_to_retry: break
            masked_texts_retry = [self.tokenize_and_mask(x) for x in texts_to_retry]
            try:
                raw_fills_retry = self.replace_masks(masked_texts_retry)
                extracted_fills_retry = self.extract_fills(raw_fills_retry)
                new_perturbed_texts = self.apply_extracted_fills(masked_texts_retry, extracted_fills_retry)
                retry_map = {original_idx: new_text for original_idx, new_text in zip(idxs, new_perturbed_texts)}
                for original_idx in idxs:
                     if retry_map.get(original_idx):
                         perturbed_texts[original_idx] = retry_map[original_idx]
            except Exception as e:
                 print(f"ERROR during perturbation retry (attempt {attempts}): {e}", flush=True)
            attempts += 1
            if attempts > 10: print(f"WARNING: Max perturbation retry attempts exceeded.", flush=True); break
        return perturbed_texts

    def replace_masks(self, texts):
        n_expected = self.count_masks(texts)
        stop_id = self.mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0] if n_expected and max(n_expected) > 0 else self.mask_tokenizer.eos_token_id
        # --- POPRAWKA: Dodano truncation i max_length ---
        tokens = self.mask_tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=self.mask_model_max_length # Użyj limitu T5
        ).to(self.mask_device)
        # --- KONIEC POPRAWKI ---
        try:
            outputs = self.mask_model.generate(**tokens, max_length=150, do_sample=True, top_p=1.0, num_return_sequences=1, eos_token_id=stop_id)
            return self.mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)
        except Exception as e:
             print(f"ERROR in replace_masks (T5 generate): {e}", flush=True)
             return [""] * len(texts)

    def extract_fills(self, texts):
        # (Kod tej metody bez zmian)
        texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]
        extracted_fills = []
        for x in texts:
             parts = self.pattern.split(x)
             if len(parts) > 1: extracted_fills.append([y.strip() for y in parts[1:-1]])
             else: extracted_fills.append([])
        return extracted_fills

    def apply_extracted_fills(self, masked_texts, extracted_fills):
        # (Kod tej metody bez zmian)
        tokens = [x.split(' ') for x in masked_texts]
        n_expected = self.count_masks(masked_texts)
        final_texts = []
        for idx, (text_tokens, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
            if len(fills) < n: final_texts.append(''); continue
            current_fill_idx = 0; result_tokens = []
            try:
                for token in text_tokens:
                    if token.startswith('<extra_id_') and token.endswith('>'):
                         if current_fill_idx < len(fills): result_tokens.append(fills[current_fill_idx]); current_fill_idx += 1
                         else: result_tokens.append(token)
                    else: result_tokens.append(token)
                final_texts.append(" ".join(result_tokens))
            except Exception as e: print(f"Error applying fills for example {idx}: {e}", flush=True); final_texts.append('')
        return final_texts

    def tokenize_and_mask(self, text):
        # (Kod tej metody bez zmian)
        tokens = text.split(' '); mask_string = '<<<mask>>>'
        n_spans = self.perturb_pct * len(tokens) / (self.span_length + 2); n_spans = int(n_spans)
        if n_spans == 0 and len(tokens)>0 : n_spans = 1
        n_masks = 0; attempts = 0; max_attempts = 10 * n_spans + 5; masked_indices = set()
        while n_masks < n_spans and attempts < max_attempts:
            start = np.random.randint(0, max(1, len(tokens) - self.span_length))
            end = start + self.span_length; valid_span = True
            for i in range(start, end):
                if i in masked_indices: valid_span = False; break
            if not valid_span: attempts += 1; continue
            search_start = max(0, start - 1); search_end = min(len(tokens), end + 1)
            neighbor_tokens = tokens[search_start:start] + tokens[end:search_end]
            if mask_string not in neighbor_tokens:
                tokens[start:end] = [mask_string]; shift = 1 - self.span_length
                new_masked_indices = set();
                for idx in masked_indices:
                     if idx > start : new_masked_indices.add(idx + shift)
                     else: new_masked_indices.add(idx)
                new_masked_indices.add(start); masked_indices = new_masked_indices
                n_masks += 1
            attempts += 1
        if n_masks == 0 and len(tokens)>0 :
             if tokens: tokens[0] = mask_string; n_masks = 1
        num_filled = 0; final_tokens = []
        for token in tokens:
            if token == mask_string: final_tokens.append(f'<extra_id_{num_filled}>'); num_filled += 1
            else: final_tokens.append(token)
        text = ' '.join(final_tokens)
        return text

    def get_ll(self, text):
        if not text: return -float('inf')
        try:
            with torch.no_grad():
                # --- POPRAWKA: Dodano truncation i max_length ---
                tokenized = self.base_tokenizer(
                    text, return_tensors="pt",
                    truncation=True, max_length=self.base_model_max_length # Użyj limitu GPT-2
                ).to(self.base_device)
                # --- KONIEC POPRAWKI ---
                if tokenized.input_ids.shape[1] == 0: return -float('inf')
                labels = tokenized.input_ids
                loss = self.base_model(**tokenized, labels=labels).loss
                return -loss.item()
        except Exception as e:
             print(f"ERROR in get_ll for text '{text[:50]}...': {e}", flush=True)
             return -float('inf')

    def get_lls(self, texts):
        results = []
        for text in texts: results.append(self.get_ll(text))
        return results

# --- Detektor OpenAI (Prawdopodobnie nie działa) ---
# (Kod klasy OpenAIDetector bez zmian w stosunku do ostatniej poprawionej wersji)
class OpenAIDetector(AIDetector):
    def __init__(self, cache_path=None):
        print("[Detector Init] Initializing OpenAIDetector (Note: API likely deprecated)...", flush=True)
        try:
            OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
            if not OPENAI_API_KEY: raise ValueError("OPENAI_API_KEY environment variable not set.")
            print(f'OPENAI KEY: {OPENAI_API_KEY[:6]}...', flush=True)
            import openai
            openai.api_key = OPENAI_API_KEY
            self.model_engine = "model-detect-v2" # Ten silnik już nie istnieje
        except ImportError: print("ERROR: openai library not installed...", flush=True); raise
        except Exception as e: print(f"ERROR initializing OpenAI: {e}", flush=True); self.model_engine = None
        self.official_classes = ['very unlikely', 'unlikely', 'unclear if it is', 'possibly', 'likely']; self.binary_classes = {'very unlikely':0, 'unlikely':0, 'unclear if it is':0, 'possibly':1, 'likely':1}; self.class_range = [0.1, 0.45, 0.9, 0.98]; self.official_threshold = 0.1
        self.cache_path = cache_path; self.use_cache = True; self.cache_change = False
        if self.cache_path and os.path.exists(self.cache_path) and self.use_cache:
            try: self.cache = load_from_pickle(self.cache_path); print(f"[Detector Init] Loaded OpenAI cache ({len(self.cache)} entries).", flush=True)
            except Exception as e: print(f"Warning: Failed to load OpenAI cache: {e}", flush=True); self.cache = {}
        else: self.cache = {}; print(f"[Detector Init] Initialized empty OpenAI cache.", flush=True)
    def get_threshold(self): return self.official_threshold
    def __call__(self, text_list, disable_tqdm=True):
        if not self.model_engine: print("ERROR: OpenAIDetector not initialized...", flush=True); return [0.5]*len(text_list), [0]*len(text_list)
        if not text_list: return [], []
        ai_prob_list=[]; label_list=[]; iterator=range(len(text_list))
        if not disable_tqdm: iterator=tqdm(iterator, desc="OpenAI Detector")
        for i in iterator:
            text=text_list[i]; prompt=f"Is the following text AI-generated?\n\nText: \"{text}\"\n\nClassification:"; cache_key=text
            if self.use_cache and cache_key in self.cache: ai_prob, binary_label=self.cache[cache_key]
            else:
                ai_prob, binary_label=0.5, 0
                for attempt in range(3):
                    try:
                        response=openai.Completion.create(engine=self.model_engine, prompt=prompt, max_tokens=1, temperature=0, logprobs=5)
                        top_logprobs=response["choices"][0].get("logprobs",{}).get("top_logprobs",[{}])[0]
                        if top_logprobs:
                           ai_related_prob=0.0
                           for token, logprob in top_logprobs.items():
                               if token.strip().lower() in ['likely','possibly','ai']: ai_related_prob=max(ai_related_prob, np.exp(logprob))
                           ai_prob=ai_related_prob
                        else:
                           completion_text=response["choices"][0].get("text","").strip().lower()
                           if completion_text in ['likely','possibly']: ai_prob=0.95
                           elif completion_text in ['unclear']: ai_prob=0.5
                           else: ai_prob=0.05
                        official_label_str=self.get_official_label(ai_prob); binary_label=self.binary_classes.get(official_label_str, 0)
                        self.cache[cache_key]=(ai_prob, binary_label); self.cache_change=True; break
                    except openai.error.OpenAIError as e: print(f"OpenAI API error: {e}", flush=True); time.sleep(5) if attempt<2 else print("Max retries exceeded...", flush=True)
                    except Exception as e: print(f"Unexpected error during OpenAI call: {e}", flush=True); break
            ai_prob_list.append(ai_prob); label_list.append(binary_label)
            if hasattr(iterator,'set_description'): iterator.set_description(f'OpenAI Prob.: {np.mean(ai_prob_list):.4f}')
        return ai_prob_list, label_list
    def get_official_label(self, prob): class_index=next((i for i,x in enumerate(self.class_range) if x>prob), len(self.class_range)); class_label=self.official_classes[class_index]; return class_label
    def save_cache(self):
        if self.cache_path and self.cache_change:
             print(f"[Detector SaveCache] Saving OpenAI cache ({len(self.cache)} entries) to {self.cache_path}", flush=True)
             try: save_to_pickle(self.cache, self.cache_path); self.cache_change=False
             except Exception as e: print(f"ERROR saving OpenAI cache: {e}", flush=True)

# --- Detektor GPTZero ---
# (Kod klasy GPTZeroDetector bez zmian w stosunku do ostatniej poprawionej wersji)
class GPTZeroDetector(AIDetector):
    def __init__(self, cache_path=None):
        print("[Detector Init] Initializing GPTZeroDetector...", flush=True); self.base_url='https://api.gptzero.me/v2/predict'; self.official_threshold=0.85; self.api_key=os.getenv("GPTZERO_API_KEY")
        if not self.api_key: print("WARNING: GPTZERO_API_KEY not set...", flush=True)
        else: print(f'GPTZero KEY: {self.api_key[:6]}...', flush=True)
        self.cache_path = cache_path; self.use_cache = True; self.cache_change = False
        if self.cache_path and os.path.exists(self.cache_path) and self.use_cache:
            try: self.cache = load_from_pickle(self.cache_path); print(f"[Detector Init] Loaded GPTZero cache ({len(self.cache)} entries).", flush=True)
            except Exception as e: print(f"Warning: Failed to load GPTZero cache: {e}", flush=True); self.cache = {}
        else: self.cache = {}; print(f"[Detector Init] Initialized empty GPTZero cache.", flush=True)
    def get_threshold(self): return self.official_threshold
    def __call__(self, text_list, disable_tqdm=True):
        if not self.api_key: print("ERROR: GPTZERO_API_KEY not set...", flush=True); return [0.5]*len(text_list), [0]*len(text_list)
        if not text_list: return [], []
        prob_list=[]; label_list=[]; iterator=range(len(text_list))
        if not disable_tqdm: iterator=tqdm(iterator, desc="GPTZero Detector")
        for i in iterator:
            text=text_list[i]
            if self.use_cache and text in self.cache: prob_score=self.cache[text]
            else:
                prob_score=0.5
                for attempt in range(3):
                    try:
                        url=f'{self.base_url}/text'; headers={'accept':'application/json','X-Api-Key':self.api_key,'Content-Type':'application/json'}; data={'document':text}
                        response=requests.post(url, headers=headers, json=data); response.raise_for_status()
                        response_json=response.json(); prob_score=response_json['documents'][0]['completely_generated_prob']
                        self.cache[text]=prob_score; self.cache_change=True; break
                    except requests.exceptions.RequestException as e: print(f"GPTZero API error: {e}", flush=True); time.sleep(5) if attempt<2 else print("Max retries exceeded...", flush=True)
                    except Exception as e: print(f"Unexpected error during GPTZero call: {e}", flush=True); break
            prob_list.append(prob_score); label_list.append(1 if prob_score > self.official_threshold else 0)
            if hasattr(iterator,'set_description'): iterator.set_description(f'GPTZero Prob.: {np.mean(prob_list):.4f}')
        return prob_list, label_list
    def save_cache(self):
        if self.cache_path and self.cache_change:
             print(f"[Detector SaveCache] Saving GPTZero cache ({len(self.cache)} entries)...", flush=True)
             try: save_to_pickle(self.cache, self.cache_path); self.cache_change = False
             except Exception as e: print(f"ERROR saving GPTZero cache: {e}", flush=True)

# --- Detektor RankDetector (logrank) ---
class RankDetector(AIDetector):
    def __init__(self, base_device, log_rank=True):
        print("[Detector Init] Initializing RankDetector (logrank)...", flush=True)
        gpt_model_path = get_model_path("gpt2-medium")
        self.base_device = base_device
        print(f"[Detector Init] Loading base model {gpt_model_path}...", flush=True)
        self.base_model = GPT2LMHeadModel.from_pretrained(gpt_model_path).to(base_device)
        self.base_tokenizer = GPT2TokenizerFast.from_pretrained(gpt_model_path)
        self.base_model.eval()
        self.base_model_max_length = self.base_model.config.n_positions
        # Ustawienie w tokenizerze dla pewności
        self.base_tokenizer.model_max_length = self.base_model_max_length
        print(f"[Detector Init] RankDetector initialized on {self.base_device}. Max length: {self.base_model_max_length}", flush=True)

        self.log_rank = log_rank
        self.threshold = -1.40

    def get_rank_onetext(self, text):
        if not text: return float('inf')
        try:
            with torch.no_grad():
                # --- POPRAWKA: Dodano truncation i max_length ---
                tokenized = self.base_tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,                     # Włącz skracanie
                    max_length=self.base_model_max_length # Użyj limitu GPT-2
                ).to(self.base_device)
                # --- KONIEC POPRAWKI ---

                if tokenized.input_ids.shape[1] <= 1: return float('inf')
                logits = self.base_model(**tokenized).logits[:, :-1]
                labels = tokenized.input_ids[:, 1:]
                seq_len = min(logits.shape[1], labels.shape[1])
                if seq_len == 0: return float('inf')
                logits = logits[:, :seq_len]; labels = labels[:, :seq_len]

                matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero(as_tuple=False)
                if matches.shape[0] == 0: return float('inf')

                ranks_tensor = torch.full_like(labels[0], logits.shape[-1], dtype=torch.float, device=self.base_device)
                # Poprawka: Użyj indeksów z `matches` do aktualizacji `ranks_tensor`
                # matches[:, 1] to indeksy czasowe, matches[:, 2] to rangi (0-based)
                # Potrzebujemy zmapować indeksy czasowe z `matches` na indeksy w `ranks_tensor`
                time_indices = matches[:, 1]
                rank_values = matches[:, 2].float() + 1.0 # Ranga 1-based
                # Użyj zaawansowanego indeksowania, aby przypisać rangi we właściwych miejscach
                # Zakładając, że pierwszy wymiar `matches` odpowiada batchowi (tutaj 1)
                # a drugi wymiar odpowiada krokowi czasowemu
                if ranks_tensor.shape[0] == time_indices.shape[0]: # Sprawdź, czy pasują (powinny dla batch_size=1)
                    ranks_tensor[time_indices] = rank_values
                else:
                    # Jeśli nie pasują (co nie powinno się zdarzyć przy batch_size=1 i poprawnych matches)
                    # Zastosuj prostsze podejście, ale może być mniej dokładne jeśli są duplikaty w time_indices
                    print(f"Warning: Mismatch in rank calculation shapes. Applying ranks sequentially.")
                    valid_ranks = rank_values[time_indices < ranks_tensor.shape[0]]
                    ranks_tensor[:len(valid_ranks)] = valid_ranks # To może być złe, jeśli brakuje kroków czasowych w matches

                ranks = ranks_tensor
                if self.log_rank: ranks = torch.log(ranks)
                return -ranks.mean().item()

        except Exception as e:
             print(f"ERROR in get_rank_onetext for text '{text[:50]}...': {e}", flush=True)
             return float('inf')

    def __call__(self, text_list, disable_tqdm=True):
        # (Kod __call__ bez zmian - używa get_rank_onetext)
        if not text_list: return [], []
        ai_score_list = []; label_list = []
        iterator = range(len(text_list))
        if not disable_tqdm: iterator = tqdm(iterator, desc="LogRank Detector")
        print(f"[Detector Call] Processing {len(text_list)} texts with LogRank.", flush=True)
        for i in iterator:
            text = text_list[i]
            cur_score = self.get_rank_onetext(text)
            if np.isinf(cur_score): print(f"Warning: Got Inf score for text {i}. Setting score to 0.", flush=True); cur_score = 0.0
            cur_label = 1 if cur_score > self.threshold else 0
            ai_score_list.append(cur_score)
            label_list.append(cur_label)
            if hasattr(iterator, 'set_description'): iterator.set_description(f'LogRank Score: {np.mean(ai_score_list):.4f}')
        print(f"[Detector Call] Finished LogRank processing.", flush=True)
        return ai_score_list, label_list

    def get_threshold(self): return self.threshold
    def save_cache(self): return