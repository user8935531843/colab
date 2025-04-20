# Początek pliku /content/drive/MyDrive/SICO/my_utils/my_logger.py
import logging
import datetime # Zachowaj import datetime, jeśli go używasz gdzieś indziej

class MyLogger:
    def __init__(self, filename, level=logging.INFO): # Użyj INFO jako domyślnego
        self.filename = filename
        self.level = level
        self.logger = logging.getLogger(filename)
        self.logger.setLevel(level)

        # Sprawdź, czy handler już istnieje, aby uniknąć duplikatów
        if not self.logger.hasHandlers():
            self.file_handler = logging.FileHandler(filename, encoding='utf-8') # Dodaj encoding
            self.file_handler.setLevel(level)
            # Użyj standardowego formatu, aby uniknąć problemów
            self.file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            self.file_handler.setFormatter(self.file_formatter)
            self.logger.addHandler(self.file_handler)
        # Usuwamy zbędną metodę log() i bezpośrednio używamy metod loggera
        # To upraszcza klasę i eliminuje potencjalne błędy

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        # Ta metoda automatycznie dodaje informacje o wyjątku
        self.logger.exception(msg, *args, **kwargs)
# Koniec pliku /content/drive/MyDrive/SICO/my_utils/my_logger.py