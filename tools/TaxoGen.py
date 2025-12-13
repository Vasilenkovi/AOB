from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from .Vectorizers import TextVectorizer
from tqdm import tqdm

class TaxonomyBuilder:
    def __init__(self, vectorizer: TextVectorizer):
        self.vectorizer = vectorizer

    def spherical_kmeans(self, vectors, n_clusters):
        normalized_vectors = normalize(vectors)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(normalized_vectors)
        return kmeans.labels_

    def compute_representativeness(self, term, cluster, clusters, terms):
        cluster_terms = [t for t, c in zip(terms, clusters) if c == cluster]
        if not cluster_terms:
            return 0
        pop = cluster_terms.count(term) / len(cluster_terms)

        other_clusters = [c for c in set(clusters) if c != cluster]
        if not other_clusters:
            return pop
        con = 0
        for other_cluster in other_clusters:
            other_terms = [t for t, c in zip(terms, clusters) if c == other_cluster]
            if not other_terms:
                continue
            other_pop = other_terms.count(term) / len(other_terms)
            con += (pop - other_pop)
        con = con / len(other_clusters)

        return np.sqrt(pop * con)

    def adaptive_clustering(self, terms, vectors, n_clusters, delta=0.2):
        clusters = self.spherical_kmeans(vectors, n_clusters)
        changed = True
        while changed:
            changed = False
            new_clusters = clusters.copy()
            for idx, (term,) in tqdm(enumerate(zip(terms)), total=len(terms), desc="Adaptive Clustering"):
                cluster = clusters[idx]
                if cluster == -1:
                    continue
                rep = self.compute_representativeness(term, cluster, clusters, terms)
                if rep < delta:
                    new_clusters[idx] = -1
                    changed = True
            clusters = new_clusters
        return clusters

    def remove_duplicates(self, taxonomy):
        if isinstance(taxonomy, list):
            return list(set(taxonomy))
        elif isinstance(taxonomy, dict):
            unique_taxonomy = {}
            for key, value in taxonomy.items():
                if isinstance(value, list):
                    unique_taxonomy[key] = list(set(value))
                elif isinstance(value, dict):
                    unique_taxonomy[key] = self.remove_duplicates(value)
            return unique_taxonomy
        return taxonomy

    def create_taxonomy(self, df, keywords_column, output_file=None, max_levels=2, num_clusters=5, delta=0.2):
        
        keywords = df[keywords_column].tolist()
        if not isinstance(keywords[0], list):
            keywords = [[kw] for kw in keywords]
        print(keywords, type(keywords))
        vectors = []
        valid_keywords = []
        for kw_list in tqdm(keywords, desc="Vectorizing"):
            for kw in kw_list:
                vec = self.vectorizer.transform(kw)
                if vec is not None:
                    vectors.append(vec)
                    valid_keywords.append(kw)

        X = np.array(vectors)

        def build_taxonomy(keywords, vectors, level):
            if level >= max_levels or len(keywords) <= 1:
                return keywords

            clusters = self.adaptive_clustering(keywords, vectors, num_clusters, delta)

            taxonomy = {}
            parent_terms = []
            for kw, label in tqdm(zip(keywords, clusters), total=len(keywords), desc=f"Building Level {level}"):
                if label == -1:
                    parent_terms.append(kw)
                else:
                    if label not in taxonomy:
                        taxonomy[label] = []
                    taxonomy[label].append(kw)

            for label in tqdm(taxonomy, desc=f"Processing Clusters at Level {level}"):
                cluster_keywords = taxonomy[label]
                cluster_vectors = []
                for kw in cluster_keywords:
                    idx = keywords.index(kw)
                    cluster_vectors.append(vectors[idx])
                taxonomy[label] = build_taxonomy(cluster_keywords, cluster_vectors, level + 1)

            if parent_terms:
                taxonomy['parent'] = parent_terms

            return taxonomy

        taxonomy = build_taxonomy(valid_keywords, X, 0)
        taxonomy = self.remove_duplicates(taxonomy)

        if output_file:
            self.save_taxonomy_to_rdf(taxonomy, output_file)

        return taxonomy

    def save_taxonomy_to_rdf(self, taxonomy, output_file):
        g = Graph()

        ns = Namespace("http://example.org/taxonomy#")

        def add_to_graph(taxonomy_node, parent_uri=None):
            if isinstance(taxonomy_node, list):
                for term in tqdm(taxonomy_node, desc="Adding Terms"):
                    term_uri = URIRef(ns[term.replace(" ", "_")])
                    g.add((term_uri, RDF.type, ns.Term))
                    g.add((term_uri, RDFS.label, Literal(term)))
                    if parent_uri:
                        g.add((parent_uri, ns.hasChild, term_uri))
            elif isinstance(taxonomy_node, dict):
                for key, value in tqdm(taxonomy_node.items(), desc="Adding Clusters"):
                    if key == 'parent':
                        for term in value:
                            term_uri = URIRef(ns[term.replace(" ", "_")])
                            g.add((term_uri, RDF.type, ns.Term))
                            g.add((term_uri, RDFS.label, Literal(term)))
                            if parent_uri:
                                g.add((parent_uri, ns.hasChild, term_uri))
                    else:
                        key_uri = URIRef(ns[f"cluster_{key}"])
                        g.add((key_uri, RDF.type, ns.Cluster))
                        if parent_uri:
                            g.add((parent_uri, ns.hasChild, key_uri))
                        add_to_graph(value, key_uri)

        add_to_graph(taxonomy)

        g.serialize(destination=output_file, format='turtle')
