"""
Corpus management for the Encyclopédie.

Handles loading articles, managing the search index, and extracting metadata.
Uses d'AlemBERT for historical French semantic search with FAISS indexing.
"""

import logging
import pickle
import re
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional imports for semantic search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available - semantic search will be limited")

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available - semantic search will be limited")


class DAlemBERTEncoder:
    """
    Encoder using d'AlemBERT for historical French text.

    d'AlemBERT is a RoBERTa model trained on 16th-20th century French,
    ideal for understanding 18th century encyclopedia text.
    """

    def __init__(self, model_name: str = "pjox/dalembert", device: Optional[str] = None):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers library required for d'AlemBERT encoder")

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading d'AlemBERT model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        logger.info(f"d'AlemBERT loaded on {self.device}")

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """Encode texts to embeddings using mean pooling."""
        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)

                outputs = self.model(**encoded)

                # Mean pooling over non-padding tokens
                attention_mask = encoded['attention_mask']
                token_embeddings = outputs.last_hidden_state
                mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                mean_embeddings = sum_embeddings / sum_mask

                all_embeddings.append(mean_embeddings.cpu().numpy())

        embeddings = np.vstack(all_embeddings).astype('float32')

        # L2 normalize for cosine similarity
        if FAISS_AVAILABLE:
            faiss.normalize_L2(embeddings)
        else:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-9)

        return embeddings


class EncyclopediaCorpus:
    """
    Manages the Encyclopédie corpus with semantic search capabilities.

    Supports:
    - Semantic search via d'AlemBERT embeddings and FAISS
    - Article lookup by ID/vedette
    - Author filtering
    - Domain filtering
    - Cross-reference (renvois) extraction
    """

    def __init__(
        self,
        articles_path: Path | str,
        model_path: Optional[Path | str] = None,
        index_path: Optional[Path | str] = None,
        blocks_path: Optional[Path | str] = None,
        chunks_path: Optional[Path | str] = None,
    ):
        """
        Initialize the corpus.

        Args:
            articles_path: Path to the full articles CSV
            model_path: Path to the embeddings model (for d'AlemBERT or sentence-transformers)
            index_path: Path to the FAISS index (.faiss file)
            blocks_path: Path to the blocks CSV (legacy, for backward compatibility)
            chunks_path: Path to the chunks metadata (.pkl file from d'AlemBERT indexing)
        """
        self.articles_path = Path(articles_path)
        self.model_path = Path(model_path) if model_path else None
        self.index_path = Path(index_path) if index_path else None
        self.blocks_path = Path(blocks_path) if blocks_path else None
        self.chunks_path = Path(chunks_path) if chunks_path else None

        # Auto-detect chunks path from index path
        if self.index_path and not self.chunks_path:
            potential_chunks = self.index_path.parent / self.index_path.name.replace('.faiss', '_chunks.pkl')
            if potential_chunks.exists():
                self.chunks_path = potential_chunks

        # Lazy-loaded components
        self._articles_df: Optional[pd.DataFrame] = None
        self._blocks_df: Optional[pd.DataFrame] = None
        self._chunks: Optional[list[dict]] = None
        self._encoder: Optional[DAlemBERTEncoder] = None
        self._faiss_index: Optional[Any] = None  # faiss.Index

        # Caches
        self._author_articles: dict[str, list[str]] = {}
        self._domain_articles: dict[str, list[str]] = {}
        self._renvois_graph: dict[str, list[str]] = {}
        self._renvois_incoming: dict[str, list[str]] = {}
        self._entree_to_article: dict[str, str] = {}
        self._article_to_entrees: dict[str, list[str]] = {}

    @property
    def articles_df(self) -> pd.DataFrame:
        """Load articles DataFrame lazily."""
        if self._articles_df is None:
            if not self.articles_path.exists():
                raise FileNotFoundError(f"Articles file not found: {self.articles_path}")
            self._articles_df = pd.read_csv(self.articles_path)
            if 'auteurs' in self._articles_df.columns:
                self._articles_df['auteurs'] = (
                    self._articles_df['auteurs']
                    .fillna('Anonyme')
                    .astype(str)
                    .str.strip()
                )
            self._build_caches()
        return self._articles_df

    @property
    def blocks_df(self) -> Optional[pd.DataFrame]:
        """Load blocks DataFrame lazily (legacy support)."""
        if self._blocks_df is None and self.blocks_path and self.blocks_path.exists():
            self._blocks_df = pd.read_csv(self.blocks_path)
            if 'auteur' in self._blocks_df.columns:
                self._blocks_df['auteur'] = (
                    self._blocks_df['auteur']
                    .fillna('Anonyme')
                    .astype(str)
                    .str.strip()
                )
        return self._blocks_df

    @property
    def chunks(self) -> Optional[list[dict]]:
        """Load chunks metadata lazily."""
        if self._chunks is None and self.chunks_path and self.chunks_path.exists():
            with open(self.chunks_path, 'rb') as f:
                self._chunks = pickle.load(f)
            logger.info(f"Loaded {len(self._chunks)} chunks from {self.chunks_path}")
        return self._chunks

    @property
    def encoder(self) -> Optional[DAlemBERTEncoder]:
        """Load d'AlemBERT encoder lazily."""
        if self._encoder is None:
            if not TRANSFORMERS_AVAILABLE:
                return None

            # When using FAISS index, always use d'AlemBERT from HuggingFace
            # (the index was built with d'AlemBERT embeddings)
            if self.index_path and self.index_path.suffix == '.faiss':
                self._encoder = DAlemBERTEncoder("pjox/dalembert")
            elif self.model_path and self.model_path.exists():
                # Use local model if available (legacy support)
                self._encoder = DAlemBERTEncoder(str(self.model_path))
            else:
                # Default to d'AlemBERT from HuggingFace
                self._encoder = DAlemBERTEncoder("pjox/dalembert")
        return self._encoder

    @property
    def faiss_index(self) -> Optional[Any]:
        """Load FAISS index lazily."""
        if self._faiss_index is None and self.index_path and self.index_path.exists():
            if not FAISS_AVAILABLE:
                raise RuntimeError("FAISS not available - install with: pip install faiss-cpu")

            index_file = self.index_path
            # Handle both .faiss and .pkl extensions
            if index_file.suffix == '.pkl':
                # Try to find corresponding .faiss file
                faiss_file = index_file.with_suffix('.faiss')
                if faiss_file.exists():
                    index_file = faiss_file
                else:
                    # Legacy: try to load from pkl (sklearn NearestNeighbors)
                    return None

            if index_file.suffix == '.faiss':
                self._faiss_index = faiss.read_index(str(index_file))
                logger.info(f"Loaded FAISS index with {self._faiss_index.ntotal} vectors")
        return self._faiss_index

    def _build_caches(self) -> None:
        """Build lookup caches for authors, domains, and renvois."""
        df = self._articles_df

        # Author -> articles cache
        if 'auteurs' in df.columns:
            for _, row in df.iterrows():
                author = row.get('auteurs', 'Anonyme')
                article_id = row.get('articleID', '')
                if author and article_id:
                    for auth in str(author).split(','):
                        auth = auth.strip()
                        if auth:
                            if auth not in self._author_articles:
                                self._author_articles[auth] = []
                            self._author_articles[auth].append(article_id)

        # Domain -> articles cache
        if 'domainesEnccre' in df.columns:
            for _, row in df.iterrows():
                domain = row.get('domainesEnccre', '')
                article_id = row.get('articleID', '')
                if domain and article_id:
                    for dom in re.split(r'[,|]', str(domain)):
                        dom = dom.strip()
                        if dom:
                            if dom not in self._domain_articles:
                                self._domain_articles[dom] = []
                            self._domain_articles[dom].append(article_id)

        # Build entreeID <-> articleID mappings
        for _, row in df.iterrows():
            entree_id = row.get('entreeID', '')
            article_id = row.get('articleID', '')
            if entree_id and article_id:
                self._entree_to_article[entree_id] = article_id
                if article_id not in self._article_to_entrees:
                    self._article_to_entrees[article_id] = []
                self._article_to_entrees[article_id].append(entree_id)

        # Renvois graph cache
        if 'idRenvois_art' in df.columns:
            for _, row in df.iterrows():
                entree_id = row.get('entreeID', '')
                renvois = row.get('idRenvois_art', '')
                if entree_id and renvois and pd.notna(renvois):
                    renvois_list = [r.strip() for r in str(renvois).split(';') if r.strip()]
                    if renvois_list:
                        self._renvois_graph[entree_id] = renvois_list
                        for target in renvois_list:
                            if target not in self._renvois_incoming:
                                self._renvois_incoming[target] = []
                            self._renvois_incoming[target].append(entree_id)

    def search_semantic(
        self,
        query: str,
        limit: int = 10,
        author: Optional[str] = None,
        domain: Optional[str] = None,
        threshold: float = 0.3,
    ) -> list[dict[str, Any]]:
        """
        Semantic search using d'AlemBERT and FAISS.

        Args:
            query: Search query (in French, can be modern or historical)
            limit: Maximum number of results
            author: Filter by author name (optional)
            domain: Filter by domain (optional)
            threshold: Minimum similarity threshold (0-1, default 0.3)

        Returns:
            List of matching articles with scores
        """
        # Try FAISS-based search first (new d'AlemBERT index)
        if self.faiss_index is not None and self.chunks is not None and self.encoder is not None:
            return self._search_faiss(query, limit, author, domain, threshold)

        # Fall back to legacy blocks-based search
        if self.blocks_df is not None:
            return self._search_legacy(query, limit, author, domain, threshold)

        raise RuntimeError("Semantic search requires FAISS index with chunks, or legacy blocks data")

    def _search_faiss(
        self,
        query: str,
        limit: int,
        author: Optional[str],
        domain: Optional[str],
        threshold: float,
    ) -> list[dict[str, Any]]:
        """Search using FAISS index with d'AlemBERT embeddings."""
        # Encode query
        query_embedding = self.encoder.encode([query])

        # Search - get more results to allow for filtering
        k = min(limit * 10, self.faiss_index.ntotal)
        scores, indices = self.faiss_index.search(query_embedding, k)

        results = []
        seen_articles = set()

        for score, idx in zip(scores[0], indices[0]):
            if len(results) >= limit:
                break

            if idx < 0 or idx >= len(self.chunks):
                continue

            score = float(score)
            if score < threshold:
                continue

            chunk = self.chunks[idx]
            article_id = chunk.get('article_id', '')

            # Skip duplicates (same article from different chunks)
            if article_id in seen_articles:
                continue

            # Apply filters
            chunk_author = str(chunk.get('auteur', ''))
            chunk_domain = str(chunk.get('domaine', ''))

            if author and author.lower() not in chunk_author.lower():
                continue
            if domain and domain.lower() not in chunk_domain.lower():
                continue

            seen_articles.add(article_id)

            # Get full article content length
            full_article = self.articles_df[self.articles_df['articleID'] == article_id]
            content_length = 0
            if not full_article.empty:
                content = full_article.iloc[0].get('contenu_clean', '')
                content_length = len(content) if content else 0

            results.append({
                'score': score,
                'articleID': article_id,
                'vedette': chunk.get('vedette', 'Inconnu'),
                'auteur': chunk_author or 'Anonyme',
                'domaine': chunk_domain,
                'text': chunk.get('text', '')[:500],  # Preview
                'content_length': content_length,
                'chunk_info': f"Chunk {chunk.get('chunk_index', 0)+1}/{chunk.get('total_chunks', 1)}",
                'url': f"https://enccre.academie-sciences.fr/encyclopedie/article/{article_id}/",
            })

        return results

    def _search_legacy(
        self,
        query: str,
        limit: int,
        author: Optional[str],
        domain: Optional[str],
        threshold: float,
    ) -> list[dict[str, Any]]:
        """Legacy search using blocks DataFrame (backward compatibility)."""
        # This is a simplified text-based search for when FAISS is not available
        df = self.blocks_df
        query_terms = query.lower().split()

        # Filter by author/domain first
        filtered = df.copy()
        if author:
            author_col = 'auteur' if 'auteur' in filtered.columns else 'auteurs'
            filtered = filtered[filtered[author_col].str.lower().str.contains(author.lower(), na=False)]
        if domain:
            domain_col = 'domaine' if 'domaine' in filtered.columns else 'domainesEnccre'
            filtered = filtered[filtered[domain_col].str.lower().str.contains(domain.lower(), na=False)]

        # Score by term matching
        text_col = 'block_text' if 'block_text' in filtered.columns else 'text'

        def score_text(text):
            if not isinstance(text, str):
                return 0
            text_lower = text.lower()
            return sum(1 for term in query_terms if term in text_lower) / max(len(query_terms), 1)

        filtered['_score'] = filtered[text_col].apply(score_text)
        matches = filtered[filtered['_score'] > 0].nlargest(limit, '_score')

        results = []
        seen_articles = set()

        for _, row in matches.iterrows():
            article_id = row.get('articleID', '')
            if article_id in seen_articles:
                continue
            seen_articles.add(article_id)

            results.append({
                'score': float(row['_score']),
                'articleID': article_id,
                'vedette': row.get('vedette', row.get('article_head', 'Inconnu')),
                'auteur': row.get('auteur', row.get('auteurs', 'Anonyme')),
                'domaine': row.get('domaine', row.get('domainesEnccre', '')),
                'text': str(row.get(text_col, ''))[:500],
                'content_length': 0,
                'url': f"https://enccre.academie-sciences.fr/encyclopedie/article/{article_id}/",
            })

        return results

    def get_article(
        self,
        article_id: str,
        max_chars: Optional[int] = None,
        offset: int = 0,
    ) -> Optional[dict[str, Any]]:
        """Get article by ID with optional content chunking."""
        df = self.articles_df

        # Try entreeID first, then articleID
        matches = df[df['entreeID'] == article_id]
        if matches.empty:
            matches = df[df['articleID'] == article_id]
        if matches.empty:
            return None

        row = matches.iloc[0].to_dict()
        full_content = row.get('contenu_clean', '')
        content_length = len(full_content)

        if max_chars is not None and max_chars > 0:
            content = full_content[offset:offset + max_chars]
            has_more = (offset + max_chars) < content_length
            next_offset = offset + max_chars if has_more else None
        else:
            content = full_content
            has_more = False
            next_offset = None

        return {
            'entreeID': row.get('entreeID', ''),
            'articleID': row.get('articleID', ''),
            'vedette': row.get('vedette', ''),
            'auteurs': row.get('auteurs', 'Anonyme'),
            'domaines': row.get('domainesEnccre', ''),
            'designants': row.get('designants', ''),
            'contenu': content,
            'renvois': row.get('idRenvois_art', ''),
            'url': f"https://enccre.academie-sciences.fr/encyclopedie/article/{row.get('articleID', '')}/",
            'content_length': content_length,
            'offset': offset,
            'has_more': has_more,
            'next_offset': next_offset,
        }

    def search_by_vedette(self, vedette: str, exact: bool = False) -> list[dict[str, Any]]:
        """Search articles by vedette (title)."""
        df = self.articles_df
        if exact:
            matches = df[df['vedette'].str.lower() == vedette.lower()]
        else:
            matches = df[df['vedette'].str.lower().str.contains(vedette.lower(), na=False)]

        results = []
        for _, row in matches.iterrows():
            content = row.get('contenu_clean', '')
            results.append({
                'articleID': row.get('articleID', ''),
                'vedette': row.get('vedette', ''),
                'auteurs': row.get('auteurs', 'Anonyme'),
                'domaines': row.get('domainesEnccre', ''),
                'content_length': len(content) if content else 0,
                'url': f"https://enccre.academie-sciences.fr/encyclopedie/article/{row.get('articleID', '')}/",
            })
        return results

    def get_authors(self) -> list[str]:
        """Get list of all authors."""
        _ = self.articles_df
        return sorted(self._author_articles.keys())

    def get_domains(self) -> list[str]:
        """Get list of all domains."""
        _ = self.articles_df
        return sorted(self._domain_articles.keys())

    def get_articles_by_author(self, author: str) -> list[dict[str, Any]]:
        """Get all articles by a specific author."""
        _ = self.articles_df
        matching_authors = [a for a in self._author_articles.keys() if author.lower() in a.lower()]

        article_ids = set()
        for auth in matching_authors:
            article_ids.update(self._author_articles[auth])

        results = []
        for article_id in article_ids:
            article = self.get_article(article_id)
            if article:
                results.append(article)

        return results

    def get_articles_by_domain(self, domain: str) -> list[dict[str, Any]]:
        """Get all articles in a specific domain."""
        _ = self.articles_df
        matching_domains = [d for d in self._domain_articles.keys() if domain.lower() in d.lower()]

        article_ids = set()
        for dom in matching_domains:
            article_ids.update(self._domain_articles[dom])

        results = []
        for article_id in list(article_ids)[:100]:
            article = self.get_article(article_id)
            if article:
                results.append({
                    'articleID': article['articleID'],
                    'vedette': article['vedette'],
                    'auteurs': article['auteurs'],
                    'domaines': article['domaines'],
                    'content_length': article.get('content_length', 0),
                    'url': article['url'],
                })

        return results

    def get_renvois(self, article_id: str) -> dict[str, Any]:
        """Get cross-references (renvois) for an article."""
        _ = self.articles_df

        entree_ids = []
        if article_id in self._entree_to_article:
            entree_ids = [article_id]
        elif article_id in self._article_to_entrees:
            entree_ids = self._article_to_entrees[article_id]

        outgoing = set()
        for eid in entree_ids:
            outgoing.update(self._renvois_graph.get(eid, []))

        incoming = set()
        for eid in entree_ids:
            incoming.update(self._renvois_incoming.get(eid, []))
        incoming.update(self._renvois_incoming.get(article_id, []))

        return {
            'articleID': article_id,
            'entree_ids': entree_ids,
            'outgoing_renvois': list(outgoing),
            'incoming_renvois': list(incoming),
            'outgoing_count': len(outgoing),
            'incoming_count': len(incoming),
        }

    def extract_definition(self, article_id: str) -> Optional[dict[str, Any]]:
        """Extract the definition from an article."""
        article = self.get_article(article_id)
        if not article:
            return None

        text = article.get('contenu', '')
        vedette = article.get('vedette', '')

        sentences = re.split(r'[.;]', text)
        definition = ''

        for sent in sentences[:3]:
            sent = sent.strip()
            if len(sent) > 20:
                definition = sent + '.'
                break

        return {
            'articleID': article_id,
            'vedette': vedette,
            'auteurs': article.get('auteurs', 'Anonyme'),
            'definition': definition,
            'domaines': article.get('domaines', ''),
            'url': article['url'],
        }

    def compare_term_usage(self, term: str, limit: int = 10) -> dict[str, Any]:
        """Compare how different authors use a term across the Encyclopédie."""
        df = self.articles_df
        term_lower = term.lower()
        matches = df[df['contenu_clean'].str.lower().str.contains(term_lower, na=False)]

        usages_by_author: dict[str, list[dict]] = {}

        for _, row in matches.head(limit * 3).iterrows():
            author = row.get('auteurs', 'Anonyme')
            if pd.isna(author) or author == '':
                author = 'Anonyme'

            article_id = row.get('articleID', '')
            vedette = row.get('vedette', '')
            text = row.get('contenu_clean', '')

            text_lower = text.lower()
            idx = text_lower.find(term_lower)
            if idx >= 0:
                start = max(0, idx - 100)
                end = min(len(text), idx + len(term) + 100)
                context = '...' + text[start:end] + '...'
            else:
                context = text[:200] + '...'

            if author not in usages_by_author:
                usages_by_author[author] = []

            if len(usages_by_author[author]) < 3:
                usages_by_author[author].append({
                    'articleID': article_id,
                    'vedette': vedette,
                    'context': context,
                    'url': f"https://enccre.academie-sciences.fr/encyclopedie/article/{article_id}/",
                })

        return {
            'term': term,
            'total_occurrences': len(matches),
            'authors_count': len(usages_by_author),
            'usages_by_author': usages_by_author,
        }

    def get_author_network(self, author: str) -> dict[str, Any]:
        """Get the network of authors connected to a given author.

        Optimized version using cached data structures instead of repeated lookups.
        """
        df = self.articles_df
        author_lower = author.lower()

        # Find all article IDs for this author using cache
        matching_authors = [a for a in self._author_articles.keys() if author_lower in a.lower()]
        article_ids = set()
        for auth in matching_authors:
            article_ids.update(self._author_articles[auth])

        if not article_ids:
            return {
                'author': author,
                'articles_count': 0,
                'domains': [],
                'connected_through_renvois': {},
                'shared_domain_authors': {},
            }

        # Get author's domains directly from DataFrame (fast)
        author_df = df[df['articleID'].isin(article_ids)]
        author_domains = set()
        for domains_str in author_df['domainesEnccre'].dropna():
            for d in re.split(r'[,|]', str(domains_str)):
                if d.strip():
                    author_domains.add(d.strip())

        # Get entree IDs for this author's articles
        author_entree_ids = set()
        for aid in article_ids:
            if aid in self._article_to_entrees:
                author_entree_ids.update(self._article_to_entrees[aid])

        # Count connections via renvois using cached graph (fast)
        connected_through_renvois: dict[str, int] = {}

        # Create a quick lookup: entreeID -> author
        entree_to_author = {}
        for _, row in df.iterrows():
            eid = row.get('entreeID', '')
            auth = row.get('auteurs', 'Anonyme')
            if eid and auth:
                entree_to_author[eid] = auth

        # Outgoing renvois
        for eid in author_entree_ids:
            for target_eid in self._renvois_graph.get(eid, []):
                target_author = entree_to_author.get(target_eid, '')
                if target_author and target_author.lower() != author_lower:
                    connected_through_renvois[target_author] = connected_through_renvois.get(target_author, 0) + 1

        # Incoming renvois
        for eid in author_entree_ids:
            for source_eid in self._renvois_incoming.get(eid, []):
                source_author = entree_to_author.get(source_eid, '')
                if source_author and source_author.lower() != author_lower:
                    connected_through_renvois[source_author] = connected_through_renvois.get(source_author, 0) + 1

        # Shared domain authors using cache (fast)
        shared_domain_authors: dict[str, list[str]] = {}
        for domain in list(author_domains)[:15]:  # Limit domains to check
            if domain in self._domain_articles:
                # Sample max 100 articles per domain for speed
                domain_article_ids = self._domain_articles[domain][:100]
                domain_df = df[df['articleID'].isin(domain_article_ids)]
                for other_author in domain_df['auteurs'].dropna().unique():
                    if other_author and other_author.lower() != author_lower:
                        if other_author not in shared_domain_authors:
                            shared_domain_authors[other_author] = []
                        if domain not in shared_domain_authors[other_author]:
                            shared_domain_authors[other_author].append(domain)

        return {
            'author': author,
            'articles_count': len(article_ids),
            'domains': list(author_domains)[:20],
            'connected_through_renvois': dict(sorted(
                connected_through_renvois.items(),
                key=lambda x: x[1],
                reverse=True
            )[:20]),
            'shared_domain_authors': dict(sorted(
                shared_domain_authors.items(),
                key=lambda x: len(x[1]),
                reverse=True
            )[:15]),
        }
