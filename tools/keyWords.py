from keybert import KeyBERT
from typing import List
from sentence_transformers import SentenceTransformer
import yake

embedding_model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2",
    device="cuda"
)
kw_model = KeyBERT(embedding_model)


def extract_keywords_with_bert(texts, top_n=5):
    keywords_list = []
    text_str = ' '.join(texts)
    keywords = kw_model.extract_keywords(text_str, top_n=top_n)
    keywords_list = [kw[0] for kw in keywords]
    return keywords_list


def extract_keywords_with_yake(texts: List[str], top_n: int = 5, language: str = "en", max_ngram_size: int = 1) -> List[List[str]]:
    """
    Извлекает ключевые слова из списка текстов с использованием YAKE.

    Args:
        texts: Список текстов.
        top_n: Количество ключевых слов для извлечения из каждого текста.
        language: Язык текста (по умолчанию "en" для английского).
        max_ngram_size: Максимальный размер n-грамм для извлечения ключевых фраз.

    Returns:
        Список списков ключевых слов для каждого текста.
    """
    keywords_list = []
    text_str = ' '.join(texts)
    # Инициализация извлекателя ключевых слов YAKE
    keyword_extractor = yake.KeywordExtractor(
            lan=language,
            n=max_ngram_size,
            dedupLim=0.9,
            dedupFunc='seqm',
            windowsSize=1,
            top=top_n,
            features=None
    )

        # Извлечение ключевых слов
    keywords_list = keyword_extractor.extract_keywords(text_str)

    return keywords_list
