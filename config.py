"""
Configuration file for Vector Database and Embedding Module
"""

import os

# Pinecone Configuration
PINECONE_API_KEY = "pcsk_6VnPun_3BW3eoFL2X3wXssTxzcVqTEZTpgZJtvmsoBfA5WFsr8F6e5DiZHzkwK5xKXQyTU"
PINECONE_ENVIRONMENT = "gcp-starter"
PINECONE_INDEX_NAME = "sri-lankan-legal-docs"
PINECONE_INDEX_NAME_ACTS_DENSE = "law-acts-dense"
PINECONE_INDEX_NAME_CASES_DENSE = "law-cases-dense"
PINECONE_INDEX_NAME_ACTS_SPARSE = "law-acts-sparse"
PINECONE_INDEX_NAME_CASES_SPARSE = "law-cases-sparse"

# Model Configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Default model
LEGAL_BERT_MODEL = "nlpaueb/legal-bert-base-uncased"
SENTENCE_BERT_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Document Processing Configuration
CHUNK_SIZE = 512  # Maximum tokens per chunk
CHUNK_OVERLAP = 50  # Overlap between chunks
MAX_SEQUENCE_LENGTH = 512

# Classification Configuration
CLASSIFICATION_MODEL = "bert-base-uncased"
NUM_LABELS = 50  # Approximate number of legal categories

# Data Paths
ACTS_DATA_PATH = "acts_2024.json"
CASES_DATA_PATH = "cases_2024.json"
TAGS_EXCEL_PATH = "tags for classifcation and metadata-vector embeddings.xlsx"

# Evaluation Configuration
EVALUATION_METRICS = ["precision", "recall", "f1", "mrr", "ndcg"]
TOP_K_VALUES = [1, 3, 5, 10]

# Languages
SUPPORTED_LANGUAGES = ["english", "sinhala", "tamil"]

# Vector Database Configuration
VECTOR_DIMENSION = 384  # For all-MiniLM-L6-v2
METRIC = "cosine"
PODS = 1
REPLICAS = 1

# Batch Processing
BATCH_SIZE = 32
MAX_WORKERS = 4
