import pandas as pd
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


class TaxoGen:
    def __init__(self, df=None, max_level=3, n_clusters=5, threshold=0.25, opt_iter=3):
        """
        Инициализация класса TaxoGen.

        :param df: DataFrame с столбцами id, key_words, named_entities.
                   key_words и named_entities должны содержать предобработанные токены.
        :param max_level: Максимальный уровень вложенности таксономии.
        :param n_clusters: Количество кластеров на каждом уровне.
        :param threshold: Порог представительности термина для кластера.
        """
        self.df = df
        self.max_level = max_level
        self.n_clusters = n_clusters
        self.threshold = threshold
        self.opt_iter = opt_iter

    def set_vectorizer(self, vectorizer):
        self.vectorizer = vectorizer
    
    def train_local_embeddings(self, terms, vectorizer):
        """Обучение локальных встраиваний на наборе терминов."""
        # Для Word2Vec создаем "предложения" из одного термина
        sentences = [[term] for term in terms]
        vectorizer.fit(sentences)
        return self.vectorizer

    def calculate_representativeness(self, term, cluster_terms):
        """Вычисление представительности термина для кластера."""
        # Для простоты используем косинусное сходство с центром кластера
        cluster_terms_in_vocab = [t for t in cluster_terms if self.vectorizer.transform(t) is not None]
        if not cluster_terms_in_vocab:
            return 0

        cluster_center = np.mean([self.vectorizer.transform(t) for t in cluster_terms_in_vocab], axis=0)
        term_vec = self.vectorizer.transform(term)
        if term_vec is None:
            return 0

        con = cosine_similarity([term_vec], [cluster_center])[0][0]
        return con

    def adaptive_clustering(self, terms, vectorizer):
        """Адаптивная сферическая кластеризация."""
        # Обучение локальных встраиваний
        self.vectorizer = self.train_local_embeddings(terms, vectorizer=vectorizer)

        # Фильтрация терминов, для которых есть векторы
        valid_terms = [term for term in terms if self.vectorizer.transform(term) is not None]
        if not valid_terms:
            return []

        embeddings = np.array([self.vectorizer.transform(term) for term in valid_terms])

        # Кластеризация
        kmeans = KMeans(n_clusters=min(self.n_clusters, len(valid_terms)), random_state=42)
        clusters = kmeans.fit_predict(embeddings)

        # Итеративное уточнение кластеров
        for _ in range(self.opt_iter):
            new_clusters = []
            general_terms = set()

            for cluster_id in range(min(self.n_clusters, len(valid_terms))):
                cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
                cluster_terms = [valid_terms[i] for i in cluster_indices]
                if not cluster_terms:
                    continue

                # Вычисление представительности каждого термина в кластере
                representativeness = {}
                for term in cluster_terms:
                    rep = self.calculate_representativeness(term, cluster_terms)
                    representativeness[term] = rep

                # Фильтрация общих терминов
                general_terms.update([term for term, rep in representativeness.items() if rep < self.threshold])
                specific_terms = [term for term in cluster_terms if term not in general_terms]
                if specific_terms:
                    new_clusters.append(specific_terms)

            # Обновление кластеров
            valid_terms = [term for term in valid_terms if term not in general_terms]
            if not valid_terms:
                break

            # Повторная кластеризация
            embeddings = np.array([self.vectorizer.transform(term) for term in valid_terms])
            clusters = kmeans.fit_predict(embeddings)

        return new_clusters

    def build_taxonomy(self, vectorizer):
        """Построение таксономии."""
        # Инициализация корневого узла - используем уже предобработанные токены
        all_terms = set()
        for keywords in self.df['key_words']:
            all_terms.update(keywords)
        for entities in self.df['named_entities']:
            all_terms.update(entities)
        all_terms = list(all_terms)

        # Рекурсивное построение таксономии
        taxonomy = self.recursive_build(all_terms, vectorizer, level=0)
        return taxonomy

    def recursive_build(self, terms, level, vectorizer):
        """Рекурсивное построение таксономии."""
        if level >= self.max_level or len(terms) <= 1:
            return terms

        # Адаптивная кластеризация
        clusters = self.adaptive_clustering(terms, vectorizer)

        # Рекурсивное построение для каждого кластера
        taxonomy = {}
        for cluster in clusters:
            if not cluster:
                continue
            taxonomy[cluster[0]] = self.recursive_build(cluster, level + 1)

        return taxonomy
