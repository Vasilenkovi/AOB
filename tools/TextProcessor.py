import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import spacy


# Проверка загрузки модели spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Модель en_core_web_sm не найдена.")
    raise


def process_text(text, chunk_size=500000):
    # Приведение к нижнему регистру и удаление знаков препинания
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)

    # Токенизация и удаление стоп-слов
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Разбиваем текст на части для лемматизации
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    lemmatized_words = []

    # Лемматизация с использованием spaCy
    for chunk in chunks:
        doc = nlp(chunk)
        lemmatized_words.extend([token.lemma_ for token in doc])

    return lemmatized_words
