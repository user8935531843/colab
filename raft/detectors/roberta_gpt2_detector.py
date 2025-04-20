from .detector import Detector

import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer


class GPT2RobertaDetector(Detector):
    def __init__(
        self,
        model_name="roberta-large",
        device="cpu",
        checkpoint="./assets/detector-large.pt",
    ):
        # Ładowanie wag z checkpointu
        checkpoint_weights = torch.load(checkpoint, map_location=device)

        self.model = RobertaForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.device = device

        # Ładowanie stanu modelu – kluczowe dla poprawnej detekcji
        self.model.load_state_dict(checkpoint_weights["model_state_dict"], strict=False)
        self.model.to(self.device)
        self.model.eval()
        print("%s detector model loaded on %s" % (model_name, device))

    def get_tokens(self, query: str):
        # Tokenizacja z ograniczeniem do maksymalnej długości i dodaniem tokenów BOS/EOS
        tokens_id = self.tokenizer.encode(query)
        tokens_id = tokens_id[: self.tokenizer.model_max_length - 2]
        tokens_id = [self.tokenizer.bos_token_id] + tokens_id + [self.tokenizer.eos_token_id]
        tokens = self.tokenizer.convert_ids_to_tokens(tokens_id)
        return tokens

    def _calculate_likelihood(self, query: str, indexes: any):
        # Tokenizacja wejściowego tekstu
        tokens = self.tokenizer.encode(query)
        tokens = tokens[: self.tokenizer.model_max_length - 2]
        tokens = torch.tensor(
            [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        ).unsqueeze(0)

        # Jeśli podano indeksy, tworzymy niestandardową maskę
        if indexes is not None:
            mask = torch.zeros_like(tokens, dtype=torch.float32)
            for i in range(len(mask[0])):
                # Ustawiamy wartość 1 dla tokenów wskazanych przez indexes, a 0.5 dla pozostałych.
                mask[0][i] = 1.0 if i in indexes else 0.5
        else:
            mask = torch.ones_like(tokens, dtype=torch.float32)

        with torch.no_grad():
            logits = self.model(tokens.to(self.device), attention_mask=mask.to(self.device))[0]
            probs = logits.softmax(dim=-1)

        # model zwraca 2 klasy:
        # pierwsza – prawdopodobieństwo, że tekst jest generowany przez LLM,
        # druga – prawdopodobieństwo, że tekst jest napisany przez człowieka.
        probs_list = probs.detach().cpu().flatten().numpy().tolist()
        # Spodziewamy się, że lista ma 2 elementy
        machine_prob, human_prob = probs_list[0], probs_list[1]
        return machine_prob, human_prob

    def llm_likelihood(self, query: str, indexes=None):
        # Zwracamy prawdopodobieństwo, że tekst jest generowany przez LLM.
        return self._calculate_likelihood(query, indexes)[0]

    def human_likelihood(self, query: str, indexes=None):
        return self._calculate_likelihood(query, indexes)[1]

    def crit(self, query):
        # Metoda krytyczna – na potrzeby ataku zwracamy wartość odpowiadającą prawdopodobieństwu generatywności.
        return self.llm_likelihood(query)
