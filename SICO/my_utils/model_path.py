
local_flag = True

model_path_dict = {
    "Hello-SimpleAI/chatgpt-detector-roberta": "/data/data/hf_model_hub/HC3-chatgpt-detector-roberta",
    "roberta-base-openai-detector": "/data/data/hf_model_hub/roberta-base-openai-detector",
    "gpt2-medium": "/data/data/hf_model_hub/gpt2/medium",
    "t5-large": "/data/data/hf_model_hub/t5/large",
    "distilroberta-base": "/data/data/hf_model_hub/distillroberta"
}

def get_model_path(model_name):
    """
    Zwraca identyfikator modelu dla Hugging Face Transformers.
    Dla znanych modeli z Hub (które są kluczami w model_path_dict),
    zwraca samą nazwę/identyfikator (klucz), ignorując błędną ścieżkę w wartości.
    """
    known_simple_hub_names = ["gpt2-medium", "t5-large", "distilroberta-base"] # Dodaj inne proste nazwy, jeśli są używane

    # Sprawdź, czy nazwa modelu jest w słowniku LUB jest jedną ze znanych prostych nazw
    if model_name in model_path_dict or model_name in known_simple_hub_names:
        # Zawsze zwracaj samą nazwę/identyfikator dla tych modeli,
        # aby biblioteka transformers mogła je poprawnie obsłużyć (cache/download)
        # print(f"DEBUG: Zwracam identyfikator Huba: '{model_name}'") # Opcjonalny wydruk do debugowania
        return model_name
    else:
        # Jeśli model_name nie jest znany, zgłoś błąd lub zwróć nazwę bez zmian
        # (zakładając, że może to być poprawna ścieżka lokalna do innego modelu)
        print(f"OSTRZEŻENIE: Model '{model_name}' nie rozpoznany jako standardowy model Huba w get_model_path. Zwracam bez zmian.")
        return model_name



