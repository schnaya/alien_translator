import json
from tokenizers import Tokenizer, models, trainers, normalizers, pre_tokenizers, decoders, processors
import os

class CustomTokenizer:
    def __init__(self, vocab_size=1500000, special_tokens=None):
        if special_tokens is None:
            special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]

        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

        # Инициализация токенизатора
        self.tokenizer = Tokenizer(models.WordPiece())

        # Установка нормализации текста
        self.tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFD(),
            normalizers.Lowercase(),
            normalizers.StripAccents()
        ])

        # Установка претокенизации
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        # Установка декодера
        self.tokenizer.decoder = decoders.WordPiece()

    def train(self, corpus):
        """
        Обучает токенизатор на предоставленном корпусе.
        :param corpus: Список текстов для тренировки.
        """
        trainer = trainers.WordPieceTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2,
            special_tokens=self.special_tokens
        )
        self.tokenizer.train_from_iterator(corpus, trainer)

        # Установка постобработки
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[("[CLS]", 1), ("[SEP]", 2), ("[PAD]", 3), ("[MASK]", 4)]
        )

    def encode(self, text):
        """
        Кодирует текст в токены.
        :param text: Исходный текст.
        :return: Закодированный объект.
        """
        return self.tokenizer.encode(text)

    def decode(self, token_ids):
        """
        Декодирует список токенов обратно в текст.
        :param token_ids: Список идентификаторов токенов.
        :return: Декодированный текст.
        """
        return self.tokenizer.decode(token_ids)

    def save(self, path):
        """
        Сохраняет токенизатор в указанный путь.
        :param path: Путь для сохранения.
        """
        self.tokenizer.save(path)
        print(f"Токенизатор сохранен по пути: {path}")

    @classmethod
    def load(cls, path):
        """
        Загружает токенизатор из файла. Если файл не найден, создаёт новый с возможностью тренировки и сохранения.
        :param path: Путь к файлу токенизатора.
        :return: Экземпляр класса CustomTokenizer.
        """
        if not os.path.exists(path):
            create_new = input(f"Файл не найден: {path}. Создать новый токенизатор? (да/нет): ")
            if create_new.lower() == 'да':
                instance = cls()
                print("Создан новый токенизатор.")

                # Предложить обучить новый токенизатор
                train_now = input("Хотите обучить новый токенизатор сейчас? (да/нет): ")
                if train_now.lower() == 'да':
                    corpus_path = input("Введите путь к текстовому файлу для обучения: ")
                    if os.path.exists(corpus_path):
                        with open(corpus_path, 'r', encoding='utf-8') as f:
                            corpus = f.readlines()
                        instance.train(corpus)
                        print("Токенизатор успешно обучен.")
                    else:
                        print("Файл для обучения не найден. Создан токенизатор без обучения.")

                # Предложить сохранить токенизатор
                save_now = input("Хотите сохранить новый токенизатор? (да/нет): ")
                if save_now.lower() == 'да':
                    instance.save(path)
                return instance,instance.tokenizer.get_vocab_size()
            else:
                raise FileNotFoundError(f"Файл не найден и создание нового токенизатора отклонено: {path}")
        else:
            instance = cls()
            instance.tokenizer = Tokenizer.from_file(path)

            # Получение размера словаря
            vocab_size = instance.tokenizer.get_vocab_size()

            # Проверка совпадения размера словаря
            if vocab_size != instance.vocab_size:
                print(f"Размер словаря токенизатора ({vocab_size}) отличается от ожидаемого ({instance.vocab_size}).")
                create_new = input(f"Создать новый токенизатор? (да/нет): ")
                if create_new.lower() == 'да':
                    instance = cls()
                    print("Создан новый токенизатор.")
                    train_now = input("Хотите обучить новый токенизатор сейчас? (да/нет): ")
                    if train_now.lower() == 'да':
                        corpus_path = input("Введите путь к текстовому файлу для обучения: ")
                        if os.path.exists(corpus_path):
                            with open(corpus_path, 'r', encoding='utf-8') as f:
                                corpus = f.readlines()
                            instance.train(corpus)
                            print("Токенизатор успешно обучен.")
                        else:
                            print("Файл для обучения не найден. Создан токенизатор без обучения.")

                    # Предложить сохранить токенизатор
                    save_now = input("Хотите сохранить новый токенизатор? (да/нет): ")
                    if save_now.lower() == 'да':
                        instance.save(path)
                    return instance,instance.tokenizer.get_vocab_size()
                else:
                    print("Продолжаем использование загруженного токенизатора с текущим размером словаря.")

            print(f"Токенизатор загружен из файла: {path} с размером словаря {vocab_size}.")
            return instance,instance.tokenizer.get_vocab_size()
