import re
from typing import List, Dict

from sklearn.datasets import fetch_20newsgroups

from logger.logging_config import get_logger

logger = get_logger(__name__)

# Compiled once at module load — used in every document clean pass
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_URL_RE = re.compile(r"http[s]?://\S+|www\.\S+")
_WHITESPACE_RE = re.compile(r"\s+")

# Minimum post length after cleaning.
# Posts shorter than this are invariably noise ("Me too", "Thanks", single words)
# and contribute nothing to semantic cluster structure.
_MIN_WORD_COUNT = 20


def load_and_clean() -> List[Dict]:
    """
    Load the 20 Newsgroups corpus and apply the full preprocessing pipeline.

    Design decisions documented here (per assignment requirement):

    remove=('headers', 'footers', 'quotes'):
        - Headers (From:, Subject:, Organization:) are metadata, not topical content.
          Including them would cause the embedding model to cluster by sender domain
          rather than by topic.
        - Footers are email signatures — entirely irrelevant to semantic content.
        - Quoted replies repeat prior posts verbatim, artificially inflating lexical
          similarity between responses and the original. Removing them ensures each
          document represents a single, coherent semantic contribution.

    Email/URL removal (strip entirely, not replace with token):
        - Embedding models treat unknown tokens as noise anyway.
        - University domains (e.g., mit.edu, stanford.edu) in email addresses would
          create spurious similarity between posts from the same institution regardless
          of topic. Removing them eliminates this signal pollution.

    Min word count filter (< 20 words):
        - The 20 Newsgroups dataset contains hundreds of posts that are one-line
          reactions: "I agree", "Me too", "Thanks for the info". These carry no
          semantic signal and create degenerate vectors that destabilize clustering.

    No stopword removal or stemming:
        - SentenceTransformers (MiniLM) are trained on natural sentences.
          Removing stopwords breaks grammatical structure and degrades embedding
          quality. The model already down-weights function words internally.
    """
    logger.info("Loading 20 Newsgroups dataset with headers/footers/quotes removed")

    dataset = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
        shuffle=False,
    )

    total_raw = len(dataset.data)
    documents = []
    skipped = 0

    for idx, (raw_text, target_idx) in enumerate(zip(dataset.data, dataset.target)):
        cleaned = _clean_text(raw_text)
        word_count = len(cleaned.split())

        if word_count < _MIN_WORD_COUNT:
            skipped += 1
            continue

        documents.append(
            {
                "original_index": idx,
                "text": cleaned,
                "category": dataset.target_names[target_idx],
                "word_count": word_count,
            }
        )

    logger.info(
        "Preprocessing complete | raw=%d | retained=%d | skipped_short=%d",
        total_raw,
        len(documents),
        skipped,
    )
    return documents


def _clean_text(text: str) -> str:
    text = _EMAIL_RE.sub("", text)
    text = _URL_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()
