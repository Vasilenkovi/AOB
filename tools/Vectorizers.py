import pandas as pd
from abc import ABC, abstractmethod
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# Абстрактный базовый класс для векторизатора
class TextVectorizer(ABC):
    @abstractmethod
    def fit(self, sentences):
        """Обучить модель на наборе предложений."""
        pass

    @abstractmethod
    def transform(self, term):
        """Получить вектор для термина."""
        pass


# Реализация для Word2Vec
class Word2VecVectorizer(TextVectorizer):
    def __init__(self, vector_size=100, window=5, min_count=1, workers=4):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None

    def fit(self, sentences):
        from gensim.models import Word2Vec
        self.model = Word2Vec(sentences, vector_size=self.vector_size, window=self.window, min_count=self.min_count, workers=self.workers)
        return self

    def transform(self, term):
        if term in self.model.wv:
            return self.model.wv[term]
        return None


# Реализация для BERT (используем готовую модель)
class BertVectorizer(TextVectorizer):
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def fit(self, sentences):
        # BERT не требует обучения, но мы можем загрузить модель здесь
        return self

    def transform(self, term):
        return self.model.encode(term)