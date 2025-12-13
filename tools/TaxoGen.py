import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, SKOS
import hashlib
from tqdm import tqdm


class TaxonomyBuilder:
    """
    Построение таксономии ключевых слов и экспорт в RDF (SKOS)
    """

    def __init__(
        self,
        vectorizer,
        df: pd.DataFrame,
        keywords_column: str,
        base_uri: str = "http://localhost/taxonomy/",
        distance_threshold: float = 0.6
    ):
        self.vectorizer = vectorizer
        self.df = df
        self.keywords_column = keywords_column
        self.base_uri = Namespace(base_uri)
        self.distance_threshold = distance_threshold

        self.graph = Graph()
        self.graph.bind("skos", SKOS)
        self.graph.bind("taxo", self.base_uri)

        self.keywords = self._collect_keywords()
        self.embeddings = self._vectorize_keywords()

    def _collect_keywords(self):
        keywords = set()
        for cell in self.df[self.keywords_column]:
            if isinstance(cell, list):
                for kw in cell:
                    if isinstance(kw, str) and kw.strip():
                        keywords.add(kw.lower())
        return sorted(keywords)

    def _vectorize_keywords(self):
        vectors = {}
        for kw in tqdm(self.keywords, desc="Vectorizing"):
            vec = self.vectorizer.transform(kw)
            if vec is not None:
                vectors[kw] = np.asarray(vec)
        return vectors

    def _term_uri(self, term: str) -> URIRef:
        digest = hashlib.md5(term.encode("utf-8")).hexdigest()
        return URIRef(f"{self.base_uri}{digest}")

    def build_taxonomy(self):
        terms = list(self.embeddings.keys())
        X = np.vstack([self.embeddings[t] for t in terms])

        print("Hierarchical clustering...")
        Z = linkage(X, method="average", metric="cosine")

        cluster_ids = fcluster(
            Z,
            t=self.distance_threshold,
            criterion="distance"
        )

        clusters = {}
        for term, cid in zip(terms, cluster_ids):
            clusters.setdefault(cid, []).append(term)

        print("Building RDF graph...")
        for cluster_terms in tqdm(
            clusters.values(),
            desc="Adding clusters"
        ):
            self._add_cluster(cluster_terms)

    def _add_cluster(self, terms):
        """
        Первый термин кластера — родитель,
        остальные — дочерние
        """

        parent = terms[0]
        parent_uri = self._term_uri(parent)

        self.graph.add((parent_uri, RDF.type, SKOS.Concept))
        self.graph.add((parent_uri, SKOS.prefLabel, Literal(parent)))

        for term in terms[1:]:
            term_uri = self._term_uri(term)

            self.graph.add((term_uri, RDF.type, SKOS.Concept))
            self.graph.add((term_uri, SKOS.prefLabel, Literal(term)))

            self.graph.add((term_uri, SKOS.broader, parent_uri))
            self.graph.add((parent_uri, SKOS.narrower, term_uri))

    def save_rdf(self, filepath: str, format: str = "turtle"):
        """
        Форматы:
        - turtle
        - xml
        - nt
        - json-ld
        """
        self.graph.serialize(destination=filepath, format=format)
