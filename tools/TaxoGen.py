import numpy as np
from sklearn.cluster import KMeans
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, SKOS
from tqdm import tqdm
import hashlib

class TaxoGenBuilder:
    def __init__(
        self,
        vectorizer,
        df,
        keywords_column,
        base_uri="http://localhost/taxonomy/",
        max_depth=4,
        n_clusters=5,
        rep_threshold=0.5,
        min_terms=5
    ):
        self.vectorizer = vectorizer
        self.df = df
        self.col = keywords_column
        self.base = Namespace(base_uri)

        self.max_depth = max_depth
        self.n_clusters = n_clusters
        self.rep_threshold = rep_threshold
        self.min_terms = min_terms

        self.graph = Graph()
        self.graph.bind("skos", SKOS)
        self.graph.bind("taxo", self.base)

        self.terms = self._collect_terms()
        self.emb = self._vectorize()

    def _collect_terms(self):
        s = set()
        for cell in self.df[self.col]:
            if isinstance(cell, list):
                for t in cell:
                    s.add(t.lower())
        return sorted(s)

    def _vectorize(self):
        d = {}
        for t in tqdm(self.terms, desc="Vectorizing"):
            d[t] = self.vectorizer.transform(t)
        return d

    def _uri(self, terms):
        # Формируем ключ с разделителем '_'
        key = "_".join(sorted(terms))
        h = hashlib.md5(key.encode()).hexdigest()
        return URIRef(f"{self.base}{h}")

    def _representative_terms(self, cluster):
        X = np.vstack([self.emb[t] for t in cluster])
        centroid = X.mean(axis=0)

        sims = np.dot(X, centroid) / (
            np.linalg.norm(X, axis=1) * np.linalg.norm(centroid)
        )

        max_sim = sims.max()
        reps = [
            t for t, s in zip(cluster, sims)
            if s >= self.rep_threshold * max_sim
        ]
        return reps

    def _cluster(self, terms):
        X = np.vstack([self.emb[t] for t in terms])
        X /= np.linalg.norm(X, axis=1, keepdims=True)

        k = min(self.n_clusters, len(terms))
        if k < 2:
            return [terms]

        labels = KMeans(k, n_init=10, random_state=0).fit_predict(X)

        clusters = {}
        for t, l in zip(terms, labels):
            clusters.setdefault(l, []).append(t)

        return list(clusters.values())

    def _build(self, terms, depth, parent_uri=None):
        if depth > self.max_depth or len(terms) < self.min_terms:
            return

        clusters = self._cluster(terms)

        for cluster in clusters:
            reps = self._representative_terms(cluster)
            residual = list(set(cluster) - set(reps))

            # Формируем метку узла с разделителем '_'
            node_label = "_".join(sorted(reps))
            node_uri = self._uri(reps)

            self.graph.add((node_uri, RDF.type, SKOS.Concept))
            self.graph.add((node_uri, SKOS.prefLabel, Literal(node_label)))

            for t in cluster:
                self.graph.add((node_uri, SKOS.altLabel, Literal(t)))

            if parent_uri:
                self.graph.add((node_uri, SKOS.broader, parent_uri))
                self.graph.add((parent_uri, SKOS.narrower, node_uri))

            if len(residual) >= self.min_terms:
                self._build(residual, depth + 1, node_uri)

    def build_taxonomy(self):
        self._build(self.terms, depth=1)

    def save(self, path="taxonomy.ttl"):
        self.graph.serialize(path, format="turtle")
