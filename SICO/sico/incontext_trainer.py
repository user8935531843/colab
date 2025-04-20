import logging
from pathlib import Path
import copy
import pickle
import time
import datetime # Added datetime import

import torch
import functools
from tqdm import tqdm
import numpy as np
import wandb
from transformers import set_seed
import shared_dir
from my_utils.my_logger import MyLogger

# Assuming detectors are correctly imported or defined in detectors.py
from detectors import RoBERTaAIDetector, OpenAIDetector, RankDetector, DetectGPT, GPTZeroDetector

from my_utils.data_utils import save_list_to_tsv, save_to_pickle, load_from_pickle
from sico.cand_generator import WordNetCandGenerator, ParaLLMGenerator
from sico.LLM_api import get_llm_api
from sico.prompt_constructor import PromptConstructor
from my_utils.text_utils import replace_changeline
from sico.context_optimizer import context_text_optimization
from my_utils.my_dataloader import load_eval_data


class SICOTrainer:

    def __init__(self, dataset, llm_name, detector_name, task_type, eval_size=32, ic_num=8, max_train_iter=6,
                 gen_kwargs=(),
                 para_num=8, save_dir='./', tag='', seed=5050, disable_wandb=True
                 ):
        # ----- Added Print: Constructor Start -----
        print(f"[{datetime.datetime.now()}] Initializing SICOTrainer...", flush=True)

        set_seed(seed)
        self.time_stamp = datetime.datetime.now().strftime('%m%d_%H%M%S')

        # logger
        log_filename = 'train_{}_{}_{}_{}_{}.log'.format(dataset, task_type, llm_name, detector_name, self.time_stamp)
        log_dir = shared_dir.log_folder_dir + log_filename
        # Use INFO level for file logs unless DEBUG is specifically needed
        self.logger = MyLogger(log_dir, level=logging.INFO)
        self.logger.info("SICOTrainer initialization started.")

        self.eval_size = eval_size
        self.incontext_example_num = ic_num
        self.max_train_iter = max_train_iter
        self.max_edit_iter = 1
        self.max_word_change_rate = 1
        self.max_sent_change_rate = 1
        self.cand_type = 'Sub-Para/Word'
        self.gen_kwargs = gen_kwargs
        self.init_feature_num = 5
        self.dataset = dataset
        self.llm_name = llm_name
        self.detector_name = detector_name
        self.task_type = task_type
        # --- Added missing assignment from args ---
        self.para_num = para_num
        # --- End Added ---

        train_config = {
            'Dataset': self.dataset, 'LLM': self.llm_name, 'Detector for training': self.detector_name,
            'Task type': self.task_type, 'Generation Args:': self.gen_kwargs, 'Eval size': self.eval_size,
            'ICE number': self.incontext_example_num,
            # Corrected f-string syntax if para_num was intended here
            'Edit': f'{self.cand_type}:[Word:{self.max_word_change_rate}, Sent-{self.para_num}:{self.max_sent_change_rate}]',
            'seed': seed
        }
        params_info = '\nHyper-params\n'
        for k in train_config: params_info += f'{k}={train_config[k]}\n'
        self.logger.info(params_info)
        print(params_info, flush=True) # Added flush=True

        # init wandb
        # ----- Added Print: Wandb Init -----
        print(f"[{datetime.datetime.now()}] Initializing wandb (disabled={disable_wandb})...", flush=True)
        proj_name = f'{dataset}_{llm_name}_{detector_name}_{task_type}'
        run_name = f'Opt={self.cand_type},EvalSize={self.eval_size},#ICE={self.incontext_example_num},T={self.time_stamp}'
        mode = 'disabled' if disable_wandb else None
        try:
             wandb.init(project=proj_name, config=train_config, name=run_name, mode=mode)
             print(f"[{datetime.datetime.now()}] wandb initialized successfully (mode: {mode}).", flush=True)
        except Exception as wandb_err:
             # Already prints a warning if it fails, but we add one too
             print(f"[{datetime.datetime.now()}] WARNING: Failed to initialize wandb: {wandb_err}. Continuing without wandb.", flush=True)
             self.logger.warning(f"Failed to initialize wandb (mode: {mode}): {wandb_err}")


        # init base models
        # ----- Added Print: GPU Init -----
        print(f"[{datetime.datetime.now()}] Initializing GPU devices...", flush=True)
        available_gpus = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
        # --- Using the corrected GPU logic ---
        if not available_gpus:
            self.logger.warning("No CUDA GPUs found! Using CPU. This might be very slow.")
            print(f"[{datetime.datetime.now()}] WARNING: No CUDA GPUs found! Using CPU.", flush=True)
            self.input_device = 'cpu'; self.detector_device = 'cpu'; generator_device = 'cpu'
        else:
            self.input_device = available_gpus[0]; self.detector_device = available_gpus[0]; generator_device = available_gpus[0]
            self.logger.info(f"Found {len(available_gpus)} GPU(s). Using {self.input_device} for all tasks.")
            print(f"[{datetime.datetime.now()}] Found {len(available_gpus)} GPU(s). Using {self.input_device} for all tasks.", flush=True)
        # --- End Corrected GPU logic ---

        # proxy detector
        # ----- Added Print: Detector Init -----
        print(f"[{datetime.datetime.now()}] Initializing detector: {detector_name} on {self.detector_device}...", flush=True)
        try:
            if detector_name == 'chatdetect':
                detector = RoBERTaAIDetector(self.detector_device)
            elif detector_name == 'openai':
                detector = OpenAIDetector(shared_dir.cache_folder_dir + f'train_{dataset}_{detector_name}.cache')
            elif detector_name == 'logrank':
                detector = RankDetector(self.detector_device, log_rank=True)
            elif detector_name == 'detectgpt':
                detector = DetectGPT(threshold=0.5, sample_num=100, mask_device=self.detector_device,
                                     base_device=self.detector_device,
                                     cache_path=shared_dir.cache_folder_dir + f'train_{dataset}_{detector_name}.cache',
                                     use_cache=True)
            elif detector_name == 'gptzero':
                detector = GPTZeroDetector(cache_path=shared_dir.cache_folder_dir + f'train_{dataset}_{detector_name}.cache')
            else: raise ValueError(f"Unknown detector: {detector_name}") # Use specific exception
            self.detector = detector
            print(f"[{datetime.datetime.now()}] Detector {detector_name} initialized successfully.", flush=True)
        except Exception as e:
             self.logger.error(f"CRITICAL ERROR during detector initialization ({detector_name}): {e}", exc_info=True)
             print(f"[{datetime.datetime.now()}] CRITICAL ERROR: Failed to initialize detector {detector_name}: {e}. Aborting.", flush=True)
             raise e # Abort if detector fails

        # LLM
        # ----- Added Print: LLM Init -----
        print(f"[{datetime.datetime.now()}] Initializing LLM: {llm_name} on {self.input_device}...", flush=True)
        try:
             self.llm_api = get_llm_api(self.llm_name, self.input_device)
             print(f"[{datetime.datetime.now()}] LLM {llm_name} initialized successfully.", flush=True)
        except Exception as e:
             self.logger.error(f"CRITICAL ERROR during LLM initialization ({llm_name}): {e}")
             print(f"[{datetime.datetime.now()}] CRITICAL ERROR: Failed to initialize LLM {llm_name}: {e}. Aborting.", flush=True)
             raise e # Abort if LLM fails

        # candidate generator for optimization
        # ----- Added Print: Cand Gen Init -----
        print(f"[{datetime.datetime.now()}] Initializing candidate generators...", flush=True)
        try:
            if task_type == 'paraphrase': threshold_ = 1e-4
            else: threshold_ = 1e-5
            # Ensure generator_device is defined from GPU logic above
            self.word_cand_generator = WordNetCandGenerator(mlm_conf_threshold=threshold_, device=generator_device)
            self.para_cand_generator = ParaLLMGenerator(self.llm_api, gen_kwargs, para_num=para_num)
            print(f"[{datetime.datetime.now()}] Candidate generators initialized successfully.", flush=True)
        except Exception as e:
             self.logger.error(f"CRITICAL ERROR during candidate generator initialization: {e}", exc_info=True)
             print(f"[{datetime.datetime.now()}] CRITICAL ERROR: Failed to initialize candidate generators: {e}. Aborting.", flush=True)
             raise e

        # prompt constructor
        self.prompt_constructor = PromptConstructor(self.task_type)

        # load dataset
        # ----- Added Print: Eval Data Load -----
        print(f"[{datetime.datetime.now()}] Loading evaluation data (eval_data_list)...", flush=True)
        try:
             self.eval_data_list = load_eval_data(dataset_name=self.dataset, task_name=self.task_type, eval_size=self.eval_size)
             print(f"[{datetime.datetime.now()}] Loaded {len(self.eval_data_list)} evaluation examples.", flush=True)
             if not self.eval_data_list:
                 # Changed from raising error to warning, maybe can proceed without eval? Check SICO logic.
                 # For now, let's warn and potentially fail later if needed.
                 print(f"[{datetime.datetime.now()}] WARNING: Evaluation data list is empty!", flush=True)
                 self.logger.warning("Evaluation data list is empty!")
                 # raise ValueError("Evaluation data list is empty!")
        except Exception as e:
             self.logger.error(f"CRITICAL ERROR during evaluation data loading: {e}", exc_info=True)
             print(f"[{datetime.datetime.now()}] CRITICAL ERROR: Failed to load evaluation data: {e}. Aborting.", flush=True)
             raise e

        # training record init
        self.best_incontext_examples = None; self.feature = None; self.best_prompt_text = '';
        self.best_score = -9999; self.best_acc = 100; self.best_dev_acc = 100; self.init_sum_num = 5
        self.total_iter = max_train_iter; self.ai_label = 1; self.human_label = 0

        # save results path setup
        self.tag = tag
        if save_dir:
            self.save_path = Path(save_dir, tag)
            self.save_path.mkdir(parents=True, exist_ok=True)
            # ----- Added Print: Save Path -----
            print(f"[{datetime.datetime.now()}] Results will be saved to: {self.save_path}", flush=True)

        self.icd_pickle_filename = '{}_feature_ice.pkl'; self.prompt_text_filename = '{}_final_prompt.txt'
        self.query_num_list = []; self.prompt_history_list = []
        # ----- Added Print: Constructor End -----
        print(f"[{datetime.datetime.now()}] SICOTrainer initialization finished.", flush=True)
        self.logger.info("SICOTrainer initialization finished.")


    def extract_feature(self, human_task_outputs, ai_task_outputs):
        # ----- Added Print: Step Start -----
        print(f"\n[{datetime.datetime.now()}] Starting Step 1: Feature Extraction (extract_feature)...", flush=True)
        self.logger.info("Starting Step 1: Feature Extraction")
        start_time = time.time()

        print(f"[{datetime.datetime.now()}] Constructing feature extraction prompt...", flush=True)
        extract_feature_prompt = self.prompt_constructor.prompt_extract_feature(human_task_outputs=human_task_outputs, ai_task_outputs=ai_task_outputs)

        # --- Using the corrected prompt truncation and max_new_tokens logic ---
        try: max_input_length = self.llm_api.max_length - 1024
        except AttributeError: max_input_length = 3000
        if max_input_length <= 0: max_input_length = 1024
        estimated_tokens = len(extract_feature_prompt.split()); chars_per_token_approx = 4
        max_chars = max_input_length * chars_per_token_approx
        prompt_truncated = False # Flag to check if truncation happened
        if len(extract_feature_prompt) > max_chars:
             self.logger.warning(f"Feature extraction prompt too long ({len(extract_feature_prompt)} chars). Truncating to ~{max_chars}.")
             print(f"[{datetime.datetime.now()}] WARNING: Feature extraction prompt too long. Truncating.", flush=True)
             extract_feature_prompt = extract_feature_prompt[:max_chars]
             prompt_truncated = True
        approx_prompt_len = len(extract_feature_prompt.split())
        # Adjust calculation if prompt was truncated, maybe allow more tokens?
        # Using the previously corrected logic here
        max_new_tokens_for_call = int(approx_prompt_len * 0.5) + 50
        if max_new_tokens_for_call <= 0: max_new_tokens_for_call = 100
        self.logger.info(f"Feature extraction prompt length (approx): {approx_prompt_len}. Max new tokens: {max_new_tokens_for_call}")
        print(f"[{datetime.datetime.now()}] Feature prompt length ~{approx_prompt_len} tokens. Max new tokens: {max_new_tokens_for_call}", flush=True)
        # --- End truncation logic ---

        print(f"[{datetime.datetime.now()}] Calling LLM API for {self.init_feature_num} feature candidates...", flush=True)
        try:
             feature_list = self.llm_api(extract_feature_prompt, max_new_tokens_for_call, self.init_feature_num, {'temperature': 0.9})
             if not feature_list or all(not f for f in feature_list): raise ValueError("LLM API returned empty feature list.")
             print(f"[{datetime.datetime.now()}] LLM API returned {len(feature_list)} feature candidates.", flush=True)
        except Exception as e:
             self.logger.error(f"Error during LLM call in extract_feature: {e}", exc_info=True)
             print(f"[{datetime.datetime.now()}] ERROR in LLM API during feature extraction: {e}", flush=True)
             raise e

        # evaluate the features
        print(f"[{datetime.datetime.now()}] Evaluating {len(feature_list)} extracted features...", flush=True)
        utility_score_list = []; final_prompt_list = []
        for i, cur_feature in enumerate(feature_list):
            # ----- Added Print: Feature Eval -----
            print(f"[{datetime.datetime.now()}]  - Evaluating feature {i+1}/{len(feature_list)}...", flush=True)
            cur_final_prompt = self.prompt_constructor.get_final_prompt(cur_feature, [])
            try:
                 # evaluate_prompt now has internal prints
                 cur_utility_score, _acc = self.evaluate_prompt(cur_final_prompt, disable_tqdm=True)
                 utility_score_list.append(cur_utility_score)
                 final_prompt_list.append(cur_final_prompt)
                 print(f"[{datetime.datetime.now()}]  - Feature {i+1} evaluated. U={cur_utility_score:.4f}, Acc={_acc:.2%}", flush=True)
            except Exception as e:
                 self.logger.error(f"Error evaluating feature {i}: {e}", exc_info=True)
                 print(f"[{datetime.datetime.now()}] ERROR evaluating feature {i}: {e}. Skipping.", flush=True)
                 utility_score_list.append(-9999) # Assign low score on error
                 final_prompt_list.append("FEATURE_EVAL_ERROR")

        if not utility_score_list or all(s == -9999 for s in utility_score_list):
             self.logger.error("Failed to evaluate any extracted features.")
             print(f"[{datetime.datetime.now()}] CRITICAL ERROR: Failed to evaluate features. Aborting.", flush=True)
             raise ValueError("Feature evaluation failed for all candidates.")

        best_idx = np.argmax(utility_score_list)
        best_final_prompt = final_prompt_list[best_idx]
        best_utility_score = utility_score_list[best_idx]
        best_feature = feature_list[best_idx]

        msg_ = f'\n[{datetime.datetime.now()}] ================= Init Feature Result ==================\n' \
               f'Best U-Score: {best_utility_score:.4f}\n' \
               f'Best Feature:\n{best_feature}\n' \
                '========================================================='
        self.logger.info(msg_) # Log includes timestamp via formatter
        print(msg_, flush=True)
        # ----- Added Print: Step End -----
        print(f"[{datetime.datetime.now()}] Finished Step 1: Feature Extraction. Duration: {time.time() - start_time:.2f}s", flush=True)
        self.logger.info(f"Finished Step 1. Duration: {time.time() - start_time:.2f}s")
        return best_feature


    def construct_incontext_outputs(self, feature_text, ai_task_outputs):
         # ----- Added Print: Step Start -----
        print(f"\n[{datetime.datetime.now()}] Starting Step 2: Construct initial y_ic (construct_incontext_outputs)...", flush=True)
        self.logger.info("Starting Step 2: Constructing initial in-context outputs y_ic")
        start_time = time.time()

        paraphrase_final_prompt = self.prompt_constructor.get_final_prompt_paraphrase(feature_text, [])
        llm_input_list = [paraphrase_final_prompt.format(ai_output) for ai_output in ai_task_outputs]

        task_outputs_ic = []
        print(f"[{datetime.datetime.now()}] Generating {len(llm_input_list)} initial y_ic examples via LLM API...", flush=True)
        for i, llm_input in enumerate(tqdm(llm_input_list, desc="Construct y_ic")):
            try:
                # Simple max_new_tokens calculation (same as before)
                original_ai_len = len(ai_task_outputs[i].split()); max_new_tokens_for_call = int(original_ai_len * 1.5)
                if max_new_tokens_for_call <= 0: max_new_tokens_for_call = 100
                try: max_model_len = self.llm_api.max_length - 50
                except: max_model_len = 2000
                max_new_tokens_for_call = min(max_new_tokens_for_call, max_model_len)

                # ----- Added Print: LLM Call -----
                # print(f"[{datetime.datetime.now()}]   - Generating y_ic for example {i}...", flush=True) # Can be too verbose
                generated_t_raw = self.llm_api(llm_input, max_new_tokens_for_call, 1, self.gen_kwargs)

                if generated_t_raw and generated_t_raw[0]:
                     generated_t = replace_changeline(generated_t_raw[0])
                     if generated_t.strip():
                         task_outputs_ic.append(generated_t)
                         # self.logger.debug(f'y_ic {i} generated: {generated_t[:50]}...') # Use logger for debug
                     else:
                         self.logger.warning(f"y_ic generation for example {i} resulted in empty string after cleaning. Using original y_ai.")
                         print(f"[{datetime.datetime.now()}] WARNING: y_ic generation {i} resulted in empty string. Using original.", flush=True)
                         task_outputs_ic.append(replace_changeline(ai_task_outputs[i]))
                else:
                     self.logger.warning(f"y_ic generation for example {i} failed or returned empty. Using original y_ai.")
                     print(f"[{datetime.datetime.now()}] WARNING: y_ic generation {i} failed or empty. Using original.", flush=True)
                     task_outputs_ic.append(replace_changeline(ai_task_outputs[i]))
            except Exception as e:
                 self.logger.error(f"Error generating y_ic for example {i}: {e}", exc_info=True)
                 print(f"[{datetime.datetime.now()}] ERROR generating y_ic for example {i}: {e}. Using original.", flush=True)
                 task_outputs_ic.append(replace_changeline(ai_task_outputs[i]))

        # ----- Added Print: Step End -----
        print(f"[{datetime.datetime.now()}] Finished Step 2: Constructing y_ic. Duration: {time.time() - start_time:.2f}s", flush=True)
        self.logger.info(f"Finished Step 2. Duration: {time.time() - start_time:.2f}s")
        return task_outputs_ic


    def evaluate_prompt(self, final_prompt, return_text=False, disable_tqdm=False):
        # ----- Added Print: Step Start -----
        # Note: This is called multiple times, e.g., during feature eval and main loop eval
        print(f"[{datetime.datetime.now()}] -> Starting evaluate_prompt for {len(self.eval_data_list)} examples...", flush=True)
        self.logger.info(f"Starting evaluate_prompt for {len(self.eval_data_list)} examples.")
        start_time = time.time()

        llm_input_list = [final_prompt.format(task_input_eval) for task_input_eval in self.eval_data_list]
        eval_task_outputs = []
        valid_indices = []
        print(f"[{datetime.datetime.now()}] -> Generating eval outputs via LLM ({len(llm_input_list)} examples)...", flush=True)
        for i, llm_input in enumerate(tqdm(llm_input_list, disable=disable_tqdm, desc="Evaluate Prompt LLM Gen")):
            try:
                max_new_tokens_for_call = 1024 # Use fixed value from previous fix
                # ----- Added Print: LLM Call -----
                # print(f"[{datetime.datetime.now()}]   - Generating eval output {i}...", flush=True) # Too verbose
                eval_task_output_raw = self.llm_api(llm_input, max_new_tokens_for_call, 1, self.gen_kwargs)

                # --- Using the corrected empty string check logic ---
                if eval_task_output_raw is None or not eval_task_output_raw:
                    self.logger.warning(f"LLM API returned empty result for eval input {i}. Skipping.")
                    print(f"[{datetime.datetime.now()}] -> WARNING: LLM returned empty result for eval {i}.", flush=True)
                    continue
                eval_task_output = eval_task_output_raw[0]
                eval_task_output = replace_changeline(eval_task_output)
                if not eval_task_output.strip():
                     self.logger.warning(f"LLM API result for eval input {i} is empty after cleaning. Skipping.")
                     print(f"[{datetime.datetime.now()}] -> WARNING: LLM result empty after cleaning for eval {i}.", flush=True)
                     continue
                # --- End empty string check ---

                eval_task_outputs.append(eval_task_output)
                valid_indices.append(i)
                # self.logger.debug(f'Eval LLM Output {i}: {eval_task_output[:50]}...') # Use logger
            except Exception as e:
                self.logger.error(f"Error during LLM generation for eval input {i}: {e}", exc_info=True)
                print(f"[{datetime.datetime.now()}] -> ERROR during LLM generation for eval {i}: {e}", flush=True)
                continue

        if not eval_task_outputs:
            self.logger.warning("Evaluation task outputs list is empty. Cannot calculate score.")
            print(f"[{datetime.datetime.now()}] -> WARNING: No valid LLM outputs for evaluation.", flush=True)
            if return_text: return -9999, 0.0, []
            else: return -9999, 0.0

        # ----- Added Print: Detector Call -----
        print(f"[{datetime.datetime.now()}] -> Calling detector for {len(eval_task_outputs)} generated texts...", flush=True)
        if not hasattr(self, 'detector') or self.detector is None: raise RuntimeError("Detector not initialized.")
        try:
            ai_score_list, label_list = self.detector(eval_task_outputs)
            print(f"[{datetime.datetime.now()}] -> Detector finished.", flush=True)
        except Exception as e:
             # Use logger.exception for traceback
             self.logger.exception(f"Error during detector call: {e}")
             print(f"[{datetime.datetime.now()}] -> ERROR during detector call: {e}", flush=True)
             if return_text: return -9999, 0.0, eval_task_outputs
             else: return -9999, 0.0

        if not ai_score_list or len(ai_score_list) != len(eval_task_outputs):
             self.logger.warning(f"Detector results empty or length mismatch ({len(ai_score_list)} vs {len(eval_task_outputs)}).")
             print(f"[{datetime.datetime.now()}] -> WARNING: Problem with detector results.", flush=True)
             U_score = 1 - np.mean(ai_score_list) if ai_score_list else -9999
             acc = 0.0
             if return_text: return U_score, acc, eval_task_outputs
             else: return U_score, acc

        gt_label = np.ones(len(eval_task_outputs), dtype=int)
        if len(label_list) != len(gt_label):
             self.logger.error(f"Detector label list length mismatch ({len(label_list)} vs {len(gt_label)})")
             print(f"[{datetime.datetime.now()}] -> ERROR: Detector label list length mismatch.", flush=True)
             acc = 0.0
        else:
             acc = (np.array(label_list) == gt_label).sum() / len(eval_task_outputs)

        U_score = 1 - np.mean(ai_score_list)
        # ----- Added Print: Step End -----
        print(f"[{datetime.datetime.now()}] -> Finished evaluate_prompt. U={U_score:.4f}, Acc={acc:.2%}. Duration: {time.time() - start_time:.2f}s", flush=True)
        self.logger.info(f"Finished evaluate_prompt. U_score: {U_score:.4f}, Acc: {acc:.2%}")

        if return_text: return U_score, acc, eval_task_outputs
        else: return U_score, acc


    def train(self, init_data_list):
        # ----- Added Print: Method Start -----
        print(f"\n[{datetime.datetime.now()}] Starting main train() method...", flush=True)
        self.logger.info("Starting main train() method.")

        if not init_data_list:
             print(f"[{datetime.datetime.now()}] CRITICAL ERROR: init_data_list is empty. Aborting training.", flush=True)
             self.logger.error("init_data_list is empty. Cannot start training.")
             return

        try:
            task_inputs, task_outputs_human, task_outputs_ai = list(zip(*init_data_list))
        except ValueError:
            self.logger.exception("Failed to unpack init_data_list. Check data format.")
            print(f"[{datetime.datetime.now()}] CRITICAL ERROR: Failed to unpack init_data_list. Check format in incontext.tsv.", flush=True)
            return


        # step 1: get feature (has internal prints)
        try:
             self.feature = self.extract_feature(task_outputs_human, task_outputs_ai)
        except Exception as e:
             self.logger.exception(f"Feature extraction failed: {e}")
             print(f"[{datetime.datetime.now()}] CRITICAL ERROR: Feature extraction failed. Aborting.", flush=True)
             return

        # step 2: construct in-context outputs y_ic (has internal prints)
        try:
             task_outputs_ic = self.construct_incontext_outputs(self.feature, task_outputs_ai)
             if len(task_outputs_ic) != len(task_inputs):
                 self.logger.warning(f"Length mismatch y_ic ({len(task_outputs_ic)}) vs x_ic ({len(task_inputs)}).")
                 print(f"[{datetime.datetime.now()}] WARNING: Length mismatch y_ic vs x_ic.", flush=True)
                 if len(task_outputs_ic) > len(task_inputs): task_outputs_ic = task_outputs_ic[:len(task_inputs)]
                 else: raise ValueError(f"Failed to construct enough y_ic ({len(task_outputs_ic)}/{len(task_inputs)}).")
        except Exception as e:
             self.logger.exception(f"Constructing in-context outputs failed: {e}")
             print(f"[{datetime.datetime.now()}] CRITICAL ERROR: Constructing in-context outputs failed. Aborting.", flush=True)
             return

        # init in-context examples
        self.best_incontext_examples = copy.deepcopy(list(zip(task_inputs, task_outputs_ic)))
        self.best_prompt_text = self.prompt_constructor.get_final_prompt(self.feature, self.best_incontext_examples)

        # evaluate initial prompt
        print(f"[{datetime.datetime.now()}] Evaluating initial prompt...", flush=True)
        try:
            # Use disable_tqdm=True for initial eval as it's just one step
            self.best_score, self.best_acc = self.evaluate_prompt(self.best_prompt_text, disable_tqdm=True)
        except Exception as e:
             self.logger.exception(f"Initial prompt evaluation failed: {e}")
             print(f"[{datetime.datetime.now()}] CRITICAL ERROR: Initial prompt evaluation failed. Aborting.", flush=True)
             return

        # ----- Added Print: Initial Score -----
        print(f"[{datetime.datetime.now()}] Initial prompt evaluation complete. Score (1-P_AI): {self.best_score:.4f}, Detection Acc: {self.best_acc:.2%}", flush=True)

        self.save_incontext_data((self.feature, self.best_incontext_examples), self.best_prompt_text, tag_=0)

        self.logger.info('================ Starting Training Loop =======================')
        self.logger.info(f'Init Context Score (1-P_AI): {self.best_score:.4f}, Acc_vs_AI: {self.best_acc:.2%}')
        print(f"\n[{datetime.datetime.now()}] === Starting Main Training Loop ({self.total_iter} iterations planned) ===", flush=True)
        print(f"[{datetime.datetime.now()}] Initial Best Score (1-P_AI): {self.best_score:.4f}, Initial Detection Acc: {self.best_acc:.2%}", flush=True)

        # Log initial score to wandb (if enabled)
        try:
             if wandb.run is not None:
                 wandb.log({'Utility Score': self.best_score, 'Accuracy': self.best_acc}, step=0)
                 wandb.log({'Best Utility Score': self.best_score, 'Best Accuracy': self.best_acc}, step=0)
        except Exception as wandb_err:
             self.logger.warning(f"Error logging initial scores to wandb: {wandb_err}")
             print(f"[{datetime.datetime.now()}] WARNING: Error logging initial scores to wandb.", flush=True)


        # step 3: Substitution-based in-context optimization
        train_i = 1
        # Use TQDM for the main loop as well
        # with tqdm(initial=train_i, total=self.total_iter, desc="Main Training Loop") as pbar: # Original was here
        pbar = tqdm(range(train_i, self.total_iter + 1), desc="Main Training Loop") # Use range for better control
        for train_i in pbar: # Iterate using the pbar range
            iter_start_time = time.time()
            # ----- Added Print: Iteration Start -----
            print(f"\n[{datetime.datetime.now()}] === Starting Training Iteration {train_i}/{self.total_iter} ===", flush=True)
            self.logger.info(f"Starting Training Iteration {train_i}/{self.total_iter}")

            # Alternate optimization type
            opt_type = 'sent' if train_i % 2 != 0 else 'word'
            print(f"[{datetime.datetime.now()}] Iter {train_i}: Optimizing type: {opt_type}", flush=True)

            # Optimize in-context examples
            try:
                # ----- Added Print: Optimization Start -----
                print(f"[{datetime.datetime.now()}] Iter {train_i}: -> Starting example optimization (_optimize_ic_outputs)...", flush=True)
                opt_start_time = time.time()
                new_ic_examples, new_ic_score = self._optimize_ic_outputs(self.best_incontext_examples, opt_type)
                # ----- Added Print: Optimization End -----
                print(f"[{datetime.datetime.now()}] Iter {train_i}: -> Finished example optimization. Avg human-score: {new_ic_score:.4f}. Duration: {time.time() - opt_start_time:.2f}s", flush=True)

                # Log to wandb (if enabled)
                try:
                     if wandb.run is not None: wandb.log({'ICD Human Score': new_ic_score}, step=train_i)
                except Exception as wandb_err:
                     self.logger.warning(f"Error logging ICD Human Score to wandb (step {train_i}): {wandb_err}")
                     print(f"[{datetime.datetime.now()}] WARNING: Error logging ICD Human Score (step {train_i}).", flush=True)

            except Exception as e:
                 self.logger.exception(f"Error during _optimize_ic_outputs in iteration {train_i} (type: {opt_type}): {e}")
                 print(f"[{datetime.datetime.now()}] ERROR in _optimize_ic_outputs (iter {train_i}, type: {opt_type}): {e}. Skipping iteration.", flush=True)
                 # pbar.update(1) # Removed as we use range now
                 continue # Skip to next iteration on optimization error

            # Evaluate and save the new prompt
            try:
                 # ----- Added Print: Eval Start -----
                 print(f"[{datetime.datetime.now()}] Iter {train_i}: -> Evaluating and saving new prompt (eval_and_save)...", flush=True)
                 eval_start_time = time.time()
                 # eval_and_save has internal prints
                 update_made = self.eval_and_save(new_ic_examples, train_i)
                 # ----- Added Print: Eval End -----
                 print(f"[{datetime.datetime.now()}] Iter {train_i}: -> Finished evaluation and saving. Best score updated: {update_made}. Duration: {time.time() - eval_start_time:.2f}s", flush=True)
            except Exception as e:
                 self.logger.exception(f"Error during eval_and_save in iteration {train_i}: {e}")
                 print(f"[{datetime.datetime.now()}] ERROR in eval_and_save (iter {train_i}): {e}. Skipping iteration.", flush=True)
                 # pbar.update(1) # Removed as we use range now
                 continue # Skip to next iteration on evaluation error

            # ----- Added Print: Iteration End -----
            print(f"[{datetime.datetime.now()}] === Finished Training Iteration {train_i}. Iteration Duration: {time.time() - iter_start_time:.2f}s ===", flush=True)
            self.logger.info(f"Finished Training Iteration {train_i}. Duration: {time.time() - iter_start_time:.2f}s")

            # pbar.update(1) # Removed as we use range now
            # No need to check train_i > self.total_iter because we use range

        # ----- Added Print: Loop End -----
        print(f"\n[{datetime.datetime.now()}] Finished training loop after {self.total_iter} iterations.", flush=True)
        self.logger.info(f"Finished training loop after {self.total_iter} iterations.")


    # --- eval_and_save, save_incontext_data, save_data_list, _optimize_ic_outputs ---
    # Add similar print(..., flush=True) statements inside these methods as well,
    # especially at the start/end and around potentially long operations (detector calls in _human_score,
    # candidate generation in _optimize_ic_outputs, saving files).
    # Example for eval_and_save (already partially done above):

    def eval_and_save(self, new_ic_examples, step):
         # ----- Added Print: Method Start -----
         print(f"[{datetime.datetime.now()}] --> Starting eval_and_save for step {step}...", flush=True)
         self.logger.info(f"Starting eval_and_save for step {step}.")
         start_time = time.time()

         cur_final_prompt = self.prompt_constructor.get_final_prompt(self.feature, new_ic_examples)

         # evaluate_prompt has internal prints now
         new_score, new_acc, new_texts = self.evaluate_prompt(cur_final_prompt, return_text=True, disable_tqdm=True)

         # Save generated texts for this step
         if new_texts:
              # ----- Added Print: Saving Texts -----
              print(f"[{datetime.datetime.now()}] --> Saving generated texts for step {step}...", flush=True)
              # Make sure eval_data_list has enough elements if some were skipped during generation
              eval_inputs_for_saving = self.eval_data_list[:len(new_texts)]
              self.save_data_list(list(zip(eval_inputs_for_saving, new_texts)), tag=step)
         else:
              print(f"[{datetime.datetime.now()}] --> WARNING: No texts generated in step {step} to save.", flush=True)
              self.logger.warning(f"No texts generated to save in step {step}.")

         # Log to wandb (if enabled)
         try:
              if wandb.run is not None: wandb.log({'Utility Score': new_score, 'Accuracy': new_acc}, step=step)
         except Exception as wandb_err:
              self.logger.warning(f"Error logging Utility/Accuracy to wandb (step {step}): {wandb_err}")
              print(f"[{datetime.datetime.now()}] WARNING: Error logging Utility/Accuracy to wandb (step {step}).", flush=True)

         is_update = False

         # Save prompt and examples for this iteration
         # ----- Added Print: Saving Iter Data -----
         print(f"[{datetime.datetime.now()}] --> Saving prompt/examples for step {step}...", flush=True)
         self.save_incontext_data((self.feature, new_ic_examples), cur_final_prompt, tag_=step)
         self.logger.info(f'Iter {step} Prompt (start):\n{cur_final_prompt[:200]}...') # Log only start

         if new_score > self.best_score:
            self.best_score = new_score
            self.best_acc = new_acc
            self.logger.info(f'-- Iter {step} FOUND BETTER SCORE: {self.best_score:.4f}, Acc: {self.best_acc:.2%}')
            print(f"[{datetime.datetime.now()}] --> !!! NEW BEST SCORE !!! Step: {step}, Score: {self.best_score:.4f}, Acc: {self.best_acc:.2%}", flush=True)

            self.best_incontext_examples = new_ic_examples
            self.best_prompt_text = cur_final_prompt
            # Save best prompt/examples (overwriting 'best' files)
            # ----- Added Print: Saving Best Data -----
            print(f"[{datetime.datetime.now()}] --> Saving new best prompt/examples...", flush=True)
            self.save_incontext_data((self.feature, self.best_incontext_examples), self.best_prompt_text, tag_='best')
            is_update = True
         else:
             # ----- Added Print: No Improvement -----
             print(f"[{datetime.datetime.now()}] --> Score in step {step} ({new_score:.4f}) not better than best ({self.best_score:.4f}).", flush=True)


         # Log best score to wandb (if enabled)
         try:
              if wandb.run is not None: wandb.log({'Best Utility Score': self.best_score, 'Best Accuracy': self.best_acc}, step=step)
         except Exception as wandb_err:
              self.logger.warning(f"Error logging Best Score/Acc to wandb (step {step}): {wandb_err}")
              print(f"[{datetime.datetime.now()}] WARNING: Error logging Best Score/Acc to wandb (step {step}).", flush=True)

         # ----- Added Print: Method End -----
         print(f"[{datetime.datetime.now()}] --> Finished eval_and_save for step {step}. Duration: {time.time() - start_time:.2f}s", flush=True)
         self.logger.info(f"Finished eval_and_save for step {step}. Duration: {time.time() - start_time:.2f}s")
         return is_update


    def save_incontext_data(self, feature_incontext_examples, final_prompt, tag_='best'):
        if not hasattr(self, 'save_path') or self.save_path is None:
            print(f"[{datetime.datetime.now()}] ---> WARNING: save_path not set in save_incontext_data. Cannot save.", flush=True)
            return

        save_file_pkl = self.save_path.joinpath(self.icd_pickle_filename.format(tag_))
        save_file_txt = self.save_path.joinpath(self.prompt_text_filename.format(tag_))
        # ----- Added Print: Saving -----
        print(f"[{datetime.datetime.now()}] ---> Saving step '{tag_}' data to {self.save_path}...", flush=True)
        self.logger.info(f'Saving step {tag_} data to {self.save_path}')
        try:
            if feature_incontext_examples: save_to_pickle(feature_incontext_examples, save_file_pkl)
            if final_prompt:
                with open(save_file_txt, 'w', encoding='utf-8') as f: f.write(final_prompt)
            print(f"[{datetime.datetime.now()}] ---> Step '{tag_}' data saved successfully.", flush=True)
        except Exception as e:
             self.logger.error(f"Error saving step {tag_} data: {e}", exc_info=True)
             print(f"[{datetime.datetime.now()}] ---> ERROR saving step '{tag_}' data: {e}", flush=True)


    def save_data_list(self, text_data_list, tag='normal'):
        if not hasattr(self, 'save_path') or self.save_path is None:
            print(f"[{datetime.datetime.now()}] ---> WARNING: save_path not set in save_data_list. Cannot save.", flush=True)
            return

        cur_filename = f'text_{tag}.tsv'; save_dir = self.save_path.joinpath(cur_filename)
        # ----- Added Print: Saving -----
        print(f"[{datetime.datetime.now()}] ---> Saving text list '{tag}' to {save_dir}...", flush=True)
        try:
             save_list_to_tsv(text_data_list, save_dir)
             # Print removed from original as it was redundant with the one above
             # print(f'Save to {save_dir}')
        except Exception as e:
             self.logger.error(f"Error saving text list {tag}: {e}", exc_info=True)
             print(f"[{datetime.datetime.now()}] ---> ERROR saving text list {tag}: {e}", flush=True)


    def _optimize_ic_outputs(self, ic_examples, edit_type):
        # ----- Added Print: Method Start -----
        print(f"[{datetime.datetime.now()}] ----> Starting example optimization (_optimize_ic_outputs, type: '{edit_type}')...", flush=True)
        self.logger.info(f"Starting _optimize_ic_outputs for type '{edit_type}'.")
        start_time = time.time()

        def _human_score(text_list):
            try:
                 # self.logger.debug(f"Detector (_human_score) evaluating {len(text_list)} texts.")
                 if not text_list: return []
                 # ----- Added Print: Detector Call -----
                 # print(f"[{datetime.datetime.now()}] ------> Calling detector (_human_score)...", flush=True) # Potentially too verbose
                 ai_score_list, _ = self.detector(text_list)
                 if not ai_score_list: return [0.0] * len(text_list)
                 human_score_list = [1 - d if d is not None else 0.0 for d in ai_score_list]
                 return human_score_list
            except Exception as e:
                 self.logger.error(f"Error in _human_score (detector call): {e}", exc_info=True)
                 print(f"[{datetime.datetime.now()}] ------> ERROR in detector during _human_score: {e}", flush=True)
                 return [0.0] * len(text_list)

        new_ic_examples = []; human_score_list = []; total_query_num = 0

        print(f"[{datetime.datetime.now()}] ----> Optimizing {len(ic_examples)} examples one by one...", flush=True)
        for i in range(len(ic_examples)):
            # ----- Added Print: Example Loop -----
            print(f"[{datetime.datetime.now()}] ------> Optimizing example {i+1}/{len(ic_examples)} (type: {edit_type})...", flush=True)
            ex_opt_start_time = time.time()
            cur_data = ic_examples[i]; cur_x_ic = cur_data[0]; cur_y_ic = cur_data[1]

            try:
                # ----- Added Print: Candidate Gen -----
                # print(f"[{datetime.datetime.now()}] ---------> Generating candidates...", flush=True) # Too verbose
                if edit_type == 'sent':
                    paraphrase_template = self.prompt_constructor.get_final_prompt_paraphrase(self.feature, [])
                    y_part_list, y_idx_part_list, y_cand_dict = self.para_cand_generator.generate_para_dict(cur_y_ic, paraphrase_template)
                elif edit_type == 'word':
                    y_part_list, y_idx_part_list, y_cand_dict = self.word_cand_generator.generate_cand_dict(cur_y_ic)
                else: raise ValueError(f'Wrong edit type: {edit_type}')

                if not y_cand_dict:
                     print(f"[{datetime.datetime.now()}] ---------> WARNING: No substitution candidates found for example {i}. Using original.", flush=True)
                     self.logger.warning(f"No substitution candidates found for example {i}, type {edit_type}.")
                     new_y_ic = cur_y_ic; new_human_score = np.mean(_human_score([cur_y_ic])) if cur_y_ic else 0.0; query_num = 0
                else:
                     # ----- Added Print: Context Opt -----
                     # print(f"[{datetime.datetime.now()}] ---------> Running context_text_optimization...", flush=True) # Too verbose
                     new_y_ic, new_human_score, query_num, edit_num = context_text_optimization(
                         start_text=cur_y_ic, start_part_list=y_part_list,
                         idx_part_list=y_idx_part_list, cand_dict=y_cand_dict,
                         eval_f=_human_score, max_iter=self.max_edit_iter, change_rate=1)
                     # ----- Added Print: Context Opt Result -----
                     print(f"[{datetime.datetime.now()}] ---------> Example {i+1} optimized. Human-score: {new_human_score:.4f}. Detector queries: {query_num}", flush=True)

            except Exception as e:
                 self.logger.exception(f"Error generating/optimizing candidates for example {i} (type: {edit_type}): {e}")
                 print(f"[{datetime.datetime.now()}] ------> ERROR generating/optimizing candidates for example {i}: {e}. Using original.", flush=True)
                 new_y_ic = cur_y_ic; new_human_score = np.mean(_human_score([cur_y_ic])) if cur_y_ic else 0.0; query_num = 0

            new_ic_examples.append((cur_x_ic, new_y_ic))
            self.query_num_list.append(query_num)
            total_query_num += query_num
            if isinstance(new_human_score, (int, float)) and not np.isnan(new_human_score): human_score_list.append(new_human_score)
            else:
                 self.logger.warning(f"Invalid human_score ({new_human_score}) for example {i}. Setting to 0.")
                 print(f"[{datetime.datetime.now()}] ------> WARNING: Invalid human_score for example {i}: {new_human_score}. Setting to 0.", flush=True)
                 human_score_list.append(0.0)
            # ----- Added Print: Example Loop End -----
            print(f"[{datetime.datetime.now()}] ------> Finished optimizing example {i+1}. Duration: {time.time() - ex_opt_start_time:.2f}s", flush=True)


        avg_human_score = np.mean(human_score_list) if human_score_list else 0.0
        # ----- Added Print: Method End -----
        print(f"[{datetime.datetime.now()}] ----> Finished optimization type '{edit_type}'. Avg human-score: {avg_human_score:.4f}. Total detector queries: {total_query_num}. Duration: {time.time() - start_time:.2f}s", flush=True)
        self.logger.info(f"Finished _optimize_ic_outputs type '{edit_type}'. Avg score: {avg_human_score:.4f}. Queries: {total_query_num}. Duration: {time.time() - start_time:.2f}s")
        return new_ic_examples, avg_human_score