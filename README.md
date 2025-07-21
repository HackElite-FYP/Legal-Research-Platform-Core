# Sri Lankan Legal Document Vector Database and Embedding Module

A comprehensive system for processing, classifying, embedding, and retrieving Sri Lankan legal documents using state-of-the-art NLP techniques and vector databases.

## Features

### Core Functionality
- **Document Processing**: Intelligent chunking of legal Acts and Cases with sequence numbering
- **Multi-language Support**: Handles Sinhala, Tamil, and English legal texts
- **BERT-based Classification**: Automated categorization using legal domain tags
- **Advanced Embeddings**: Support for Legal-BERT, Sentence-BERT, and other transformer models
- **Vector Database**: Pinecone integration for efficient similarity search
- **Comprehensive Evaluation**: Standard IR metrics and expert validation framework

### Key Components

1. **Document Processor** (`document_processor.py`)
   - Chunks legal documents into manageable pieces
   - Extracts metadata and sequence numbers
   - Handles multilingual content detection

2. **Legal Classifier** (`classifier.py`)
   - BERT-based multi-label classification
   - Uses predefined legal category tags
   - Supports fine-tuning on legal domain data

3. **Embedding Generator** (`embedding_generator.py`)
   - Multiple embedding model support
   - Batch processing for efficiency
   - Model comparison capabilities

4. **Vector Database** (`vector_database.py`)
   - Pinecone integration
   - Metadata filtering
   - Hybrid search capabilities

5. **Evaluation Framework** (`evaluation.py`)
   - Standard IR metrics (Precision@k, Recall@k, MRR, NDCG)
   - Expert validation system
   - Model comparison tools

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster processing)
- Pinecone account and API key

### Setup Instructions

1. **Clone the repository and install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Configure Pinecone API key:**
   - Update `config.py` with your Pinecone API key
   - Or set environment variable: `export PINECONE_API_KEY="your-api-key"`

3. **Download required NLTK data:**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Usage

### Quick Start

```python
from main import VectorDBEmbeddingSystem

# Initialize the system
system = VectorDBEmbeddingSystem()

# Run complete pipeline
results = system.run_complete_pipeline(
    train_classifier=True,
    evaluate_system=True
)

# Search for documents
search_results = system.search_documents(
    "property ownership rights",
    top_k=10
)
```

### Step-by-Step Usage

#### 1. Document Processing
```python
from document_processor import DocumentProcessor

processor = DocumentProcessor()
processed_acts, processed_cases = processor.process_all_documents()
```

#### 2. Classification
```python
from classifier import LegalClassificationSystem

classifier = LegalClassificationSystem()
classifier.train_classifier(all_processed_docs)
classified_docs = classifier.classify_documents(all_processed_docs)
```

#### 3. Embedding Generation
```python
from embedding_generator import EmbeddingGenerator

generator = EmbeddingGenerator("sentence-transformers/all-MiniLM-L6-v2")
embedded_docs = generator.embed_documents(classified_docs)
```

#### 4. Vector Database Operations
```python
from vector_database import VectorDatabaseManager

db_manager = VectorDatabaseManager()
db_manager.index_documents(embedded_docs)

# Search
results = db_manager.search_legal_documents(
    "marriage and divorce laws",
    generator,
    top_k=5
)
```

#### 5. Evaluation
```python
from evaluation import ComprehensiveEvaluator

evaluator = ComprehensiveEvaluator()
evaluation_results = evaluator.evaluate_system(
    db_manager,
    generator,
    test_queries
)
```

## Configuration

Key configuration options in `config.py`:

```python
# Model Configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LEGAL_BERT_MODEL = "nlpaueb/legal-bert-base-uncased"

# Document Processing
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Vector Database
VECTOR_DIMENSION = 384
PINECONE_INDEX_NAME = "sri-lankan-legal-docs"

# Evaluation
TOP_K_VALUES = [1, 3, 5, 10]
```

## Data Format

### Input Data Structure
```json
{
  "id": "unique-document-id",
  "filename": "Document Title",
  "primaryLang": "English",
  "text": "Full document text...",
  "wordCount": 2500
}
```

### Classification Tags
The system uses predefined legal categories from the Excel file:
- **Family Law**: Divorce, Marriage, Custody, etc.
- **Property Law**: Ownership, Transfer, Boundaries, etc.
- **Commercial Law**: Contracts, Torts, IP Rights, etc.
- **Labour Law**: Employment, Wages, Termination, etc.

## Evaluation Metrics

### Automatic Metrics
- **Precision@k**: Relevance of top-k results
- **Recall@k**: Coverage of relevant documents
- **F1@k**: Harmonic mean of precision and recall
- **MRR**: Mean Reciprocal Rank
- **NDCG@k**: Normalized Discounted Cumulative Gain

### Expert Validation
- Inter-expert agreement (Spearman correlation)
- Fleiss' Kappa for consistency
- Qualitative feedback analysis

## Model Comparison

Compare different embedding models:

```python
system = VectorDBEmbeddingSystem()
comparison = system.compare_embedding_models([
    "sentence-transformers/all-MiniLM-L6-v2",
    "nlpaueb/legal-bert-base-uncased",
    "sentence-transformers/all-mpnet-base-v2"
])
```

## Advanced Features

### Filtered Search
```python
results = db_manager.search_legal_documents(
    query="property disputes",
    embedding_generator=generator,
    filters={
        'document_type': 'case',
        'language': 'english',
        'category': 'Property'
    }
)
```

### Hybrid Search
```python
results = db_manager.db.hybrid_search(
    query_embedding=embedding,
    query_text="contract breach remedies",
    alpha=0.7  # Weight for vector vs text search
)
```

## File Structure

```
├── main.py                 # Main integration module
├── config.py              # Configuration settings
├── utils.py               # Utility functions
├── document_processor.py  # Document processing
├── classifier.py          # BERT-based classification
├── embedding_generator.py # Embedding generation
├── vector_database.py     # Pinecone integration
├── evaluation.py          # Evaluation framework
├── requirements.txt       # Dependencies
├── README.md             # This file
├── acts_2024.json        # Legal Acts data
├── cases_2024.json       # Legal Cases data
└── tags for classifcation and metadata-vector embeddings.xlsx
```

## Troubleshooting

### Common Issues

1. **Pinecone Connection Error**
   - Verify API key and environment
   - Check internet connection
   - Ensure index name is unique

2. **Memory Issues**
   - Reduce batch size in config
   - Use smaller embedding models
   - Process documents in smaller chunks

3. **Model Loading Errors**
   - Check model name spelling
   - Ensure sufficient disk space
   - Verify internet connection for model download

## Performance Optimization

### For Large Datasets
- Use GPU acceleration
- Increase batch sizes
- Implement parallel processing
- Use model quantization

### For Better Accuracy
- Fine-tune embedding models on legal data
- Expand classification tags
- Implement query expansion
- Use ensemble methods

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## License

This project is licensed under the MIT License.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{sri_lankan_legal_vector_db,
  title={Sri Lankan Legal Document Vector Database and Embedding Module},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/sri-lankan-legal-vector-db}
}
```

## Support

For questions and support:
- Create an issue on GitHub
- Contact: your-email@example.com

## Acknowledgments

- Legal-BERT model by Chalkidis et al.
- Sentence-BERT by Reimers & Gurevych
- Pinecone vector database
- Sri Lankan legal document providers
