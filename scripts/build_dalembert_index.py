"""
Build semantic search index using d'AlemBERT for the Encyclopédie.

d'AlemBERT (pjox/dalembert) is a RoBERTa model trained on 16th-20th century French,
making it ideal for 18th century encyclopedia text.

This script:
1. Loads articles from CSV
2. Chunks long articles intelligently (preserving short articles as-is)
3. Encodes with d'AlemBERT
4. Builds a FAISS index with L2 normalization for cosine similarity
5. Saves everything for use by the MCP server
"""

import os
import sys
import pickle
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import faiss
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# Configuration
ARTICLES_CSV = "data/articles.csv"
OUTPUT_DIR = "data"
INDEX_NAME = "dalembert_index"

# Chunking parameters
MAX_WORDS_PER_CHUNK = 400  # d'AlemBERT max is 512 tokens, ~400 words is safe
CHUNK_SIZE = 350  # Words per chunk for long articles
CHUNK_OVERLAP = 50  # Overlap between chunks

# Model parameters
MODEL_NAME = "pjox/dalembert"
BATCH_SIZE = 16  # Adjust based on GPU memory (RTX 3090 can handle more)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DAlemBERTEncoder:
    """Encoder using d'AlemBERT for historical French text."""

    def __init__(self, model_name: str = MODEL_NAME, device: str = DEVICE):
        logger.info(f"Loading d'AlemBERT model: {model_name}")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        logger.info(f"Model loaded on {device}")

    def encode(self, texts: list[str], batch_size: int = BATCH_SIZE,
               show_progress: bool = True) -> np.ndarray:
        """Encode texts to embeddings using mean pooling."""
        all_embeddings = []

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding", total=len(texts) // batch_size + 1)

        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i:i + batch_size]

                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)

                # Forward pass
                outputs = self.model(**encoded)

                # Mean pooling over non-padding tokens
                attention_mask = encoded['attention_mask']
                token_embeddings = outputs.last_hidden_state

                # Expand attention mask for broadcasting
                mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

                # Sum embeddings and divide by number of tokens
                sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                mean_embeddings = sum_embeddings / sum_mask

                all_embeddings.append(mean_embeddings.cpu().numpy())

        embeddings = np.vstack(all_embeddings).astype('float32')

        # L2 normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        return embeddings


def chunk_article(text: str, article_id: str, vedette: str, metadata: dict,
                  max_words: int = MAX_WORDS_PER_CHUNK,
                  chunk_size: int = CHUNK_SIZE,
                  overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Chunk an article intelligently.

    - Short articles (<= max_words) are kept as single chunks
    - Long articles are split with overlap to maintain context
    """
    words = text.split()
    word_count = len(words)

    chunks = []

    if word_count <= max_words:
        # Keep as single chunk
        chunks.append({
            'chunk_id': f"{article_id}_0",
            'article_id': article_id,
            'vedette': vedette,
            'text': text,
            'chunk_index': 0,
            'total_chunks': 1,
            'word_count': word_count,
            **metadata
        })
    else:
        # Split into overlapping chunks
        chunk_index = 0
        start = 0

        while start < word_count:
            end = min(start + chunk_size, word_count)
            chunk_text = ' '.join(words[start:end])

            chunks.append({
                'chunk_id': f"{article_id}_{chunk_index}",
                'article_id': article_id,
                'vedette': vedette,
                'text': chunk_text,
                'chunk_index': chunk_index,
                'total_chunks': -1,  # Will be filled later
                'word_count': end - start,
                **metadata
            })

            chunk_index += 1
            start = end - overlap if end < word_count else word_count

        # Update total_chunks
        for chunk in chunks:
            chunk['total_chunks'] = chunk_index

    return chunks


def build_chunks_from_articles(articles_df: pd.DataFrame) -> tuple[list[dict], list[str]]:
    """Build chunks from all articles."""
    all_chunks = []
    all_texts = []

    logger.info(f"Processing {len(articles_df)} articles...")

    for _, row in tqdm(articles_df.iterrows(), total=len(articles_df), desc="Chunking"):
        article_id = row.get('articleID', '')
        vedette = row.get('vedette', '')
        text = str(row.get('contenu_clean', ''))

        if not text or len(text) < 10:
            continue

        metadata = {
            'auteur': row.get('auteurs', 'Anonyme'),
            'domaine': row.get('domainesEnccre', ''),
            'entree_id': row.get('entreeID', ''),
            'designants': row.get('designants', ''),
        }

        chunks = chunk_article(text, article_id, vedette, metadata)

        for chunk in chunks:
            all_chunks.append(chunk)
            all_texts.append(chunk['text'])

    logger.info(f"Created {len(all_chunks)} chunks from {len(articles_df)} articles")

    # Stats
    single_chunks = sum(1 for c in all_chunks if c['total_chunks'] == 1)
    multi_chunks = len(all_chunks) - single_chunks
    logger.info(f"  - Single-chunk articles: {single_chunks}")
    logger.info(f"  - Multi-chunk articles: {multi_chunks}")

    return all_chunks, all_texts


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build FAISS index with Inner Product (cosine similarity when normalized)."""
    dimension = embeddings.shape[1]
    logger.info(f"Building FAISS index (dimension: {dimension}, vectors: {len(embeddings)})")

    # IndexFlatIP = Inner Product = Cosine similarity when vectors are L2 normalized
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    logger.info(f"Index built with {index.ntotal} vectors")
    return index


def verify_index(index: faiss.Index, embeddings: np.ndarray, chunks: list[dict],
                 encoder: DAlemBERTEncoder) -> None:
    """Verify the index works correctly."""
    logger.info("Verifying index...")

    # Test 1: Random vector should not return sequential indices
    random_vec = np.random.randn(1, embeddings.shape[1]).astype('float32')
    faiss.normalize_L2(random_vec)
    scores, indices = index.search(random_vec, 10)

    is_sequential = all(indices[0][i] == i for i in range(10))
    if is_sequential:
        logger.error("WARNING: Index returning sequential results - may be corrupted!")
    else:
        logger.info("  - Random vector test: PASSED (non-sequential indices)")

    # Test 2: Search for "philosophie"
    test_query = "philosophie naturelle et sciences"
    query_embedding = encoder.encode([test_query], show_progress=False)
    scores, indices = index.search(query_embedding, 5)

    logger.info(f"  - Test query '{test_query}':")
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        chunk = chunks[idx]
        logger.info(f"    {i+1}. Score: {score:.3f} | {chunk['vedette'][:50]} | {chunk['text'][:80]}...")


def main(articles_path: Optional[str] = None, output_dir: Optional[str] = None):
    """Main function to build the index."""

    # Paths
    if articles_path is None:
        script_dir = Path(__file__).parent.parent
        articles_path = script_dir / ARTICLES_CSV
    else:
        articles_path = Path(articles_path)

    if output_dir is None:
        output_dir = articles_path.parent
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Articles: {articles_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Device: {DEVICE}")

    # Load articles
    logger.info("Loading articles...")
    articles_df = pd.read_csv(articles_path)
    logger.info(f"Loaded {len(articles_df)} articles")

    # Build chunks
    chunks, texts = build_chunks_from_articles(articles_df)

    # Initialize encoder
    encoder = DAlemBERTEncoder()

    # Encode chunks
    logger.info("Encoding chunks with d'AlemBERT...")
    embeddings = encoder.encode(texts, batch_size=BATCH_SIZE)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Build FAISS index
    index = build_faiss_index(embeddings)

    # Verify index
    verify_index(index, embeddings, chunks, encoder)

    # Save everything
    logger.info("Saving index and metadata...")

    # Save FAISS index
    faiss_path = output_dir / f"{INDEX_NAME}.faiss"
    faiss.write_index(index, str(faiss_path))
    logger.info(f"  - FAISS index: {faiss_path}")

    # Save chunks metadata
    metadata_path = output_dir / f"{INDEX_NAME}_chunks.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(chunks, f)
    logger.info(f"  - Chunks metadata: {metadata_path}")

    # Save embeddings (for potential re-indexing)
    embeddings_path = output_dir / f"{INDEX_NAME}_embeddings.npy"
    np.save(embeddings_path, embeddings)
    logger.info(f"  - Embeddings: {embeddings_path}")

    # Save config for reference
    config = {
        'model_name': MODEL_NAME,
        'max_words_per_chunk': MAX_WORDS_PER_CHUNK,
        'chunk_size': CHUNK_SIZE,
        'chunk_overlap': CHUNK_OVERLAP,
        'total_chunks': len(chunks),
        'embedding_dimension': embeddings.shape[1],
        'articles_count': len(articles_df),
    }
    config_path = output_dir / f"{INDEX_NAME}_config.pkl"
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    logger.info(f"  - Config: {config_path}")

    logger.info("Done!")
    logger.info(f"Total chunks: {len(chunks)}")
    logger.info(f"Index size: {faiss_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build d'AlemBERT index for Encyclopédie")
    parser.add_argument("--articles", type=str, help="Path to articles.csv")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for encoding")
    args = parser.parse_args()

    if args.batch_size:
        BATCH_SIZE = args.batch_size

    main(args.articles, args.output)
