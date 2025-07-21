"""
Embedding Generation Module for Legal Documents
Supports Legal-BERT, Sentence-BERT, and other transformer models
"""

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import logging
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import config
import utils

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generate embeddings for legal documents using various models"""
    
    def __init__(self, model_name: str = config.EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        
        self._load_model()
        logger.info(f"EmbeddingGenerator initialized with {model_name} on {self.device}")
    
    def _load_model(self):
        """Load the embedding model"""
        try:
            if "sentence-transformers" in self.model_name or self.model_name in [
                "all-MiniLM-L6-v2", "all-mpnet-base-v2", "multi-qa-mpnet-base-dot-v1"
            ]:
                # Use SentenceTransformers for these models
                self.model = SentenceTransformer(self.model_name, device=self.device)
                self.is_sentence_transformer = True
                logger.info(f"Loaded SentenceTransformer model: {self.model_name}")
                
            else:
                # Use Hugging Face transformers for other models (like Legal-BERT)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.is_sentence_transformer = False
                logger.info(f"Loaded Transformer model: {self.model_name}")
                
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {str(e)}")
            # Fallback to default model
            self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.is_sentence_transformer = True
            logger.info(f"Fallback to default model: {self.model_name}")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return np.zeros(self.get_embedding_dimension())
        
        try:
            if self.is_sentence_transformer:
                embedding = self.model.encode(text, convert_to_numpy=True)
            else:
                embedding = self._generate_transformer_embedding(text)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return np.zeros(self.get_embedding_dimension())
    
    def _generate_transformer_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using Hugging Face transformer model"""
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=config.MAX_SEQUENCE_LENGTH
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling of last hidden states
            embeddings = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            
            # Mean pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = sum_embeddings / sum_mask
        
        return embedding.cpu().numpy()[0]
    
    def generate_batch_embeddings(self, texts: List[str], batch_size: int = config.BATCH_SIZE) -> List[np.ndarray]:
        """Generate embeddings for a batch of texts"""
        if not texts:
            return []
        
        embeddings = []
        
        if self.is_sentence_transformer:
            # SentenceTransformers can handle batching efficiently
            for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
                embeddings.extend(batch_embeddings)
        else:
            # Process individually for transformer models
            for text in tqdm(texts, desc="Generating embeddings"):
                embedding = self.generate_embedding(text)
                embeddings.append(embedding)
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the model"""
        if self.is_sentence_transformer:
            return self.model.get_sentence_embedding_dimension()
        else:
            # For transformer models, use config
            return self.model.config.hidden_size
    
    def embed_documents(self, processed_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for processed documents"""
        logger.info(f"Generating embeddings for {len(processed_docs)} documents...")
        
        # Extract texts
        texts = [doc['text'] for doc in processed_docs]
        
        # Generate embeddings
        embeddings = self.generate_batch_embeddings(texts)
        
        # Add embeddings to documents
        embedded_docs = []
        for doc, embedding in zip(processed_docs, embeddings):
            doc['embedding'] = embedding.tolist()  # Convert to list for JSON serialization
            doc['metadata']['embedding_model'] = self.model_name
            doc['metadata']['embedding_dimension'] = len(embedding)
            embedded_docs.append(doc)
        
        logger.info("Embedding generation completed!")
        return embedded_docs

class MultiModelEmbeddingGenerator:
    """Generate embeddings using multiple models for comparison"""
    
    def __init__(self, model_names: List[str] = None):
        if model_names is None:
            model_names = [
                config.EMBEDDING_MODEL_NAME,
                config.LEGAL_BERT_MODEL,
                config.SENTENCE_BERT_MODEL
            ]
        
        self.generators = {}
        for model_name in model_names:
            try:
                self.generators[model_name] = EmbeddingGenerator(model_name)
                logger.info(f"Loaded model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {str(e)}")
        
        logger.info(f"MultiModelEmbeddingGenerator initialized with {len(self.generators)} models")
    
    def generate_multi_embeddings(self, processed_docs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Generate embeddings using all loaded models"""
        results = {}
        
        for model_name, generator in self.generators.items():
            logger.info(f"Generating embeddings with {model_name}...")
            embedded_docs = generator.embed_documents(processed_docs.copy())
            results[model_name] = embedded_docs
        
        return results
    
    def compare_models(self, sample_texts: List[str]) -> Dict[str, Dict[str, Any]]:
        """Compare different embedding models on sample texts"""
        comparison = {}
        
        for model_name, generator in self.generators.items():
            embeddings = generator.generate_batch_embeddings(sample_texts)
            
            comparison[model_name] = {
                'dimension': generator.get_embedding_dimension(),
                'sample_embeddings': [emb[:5].tolist() for emb in embeddings[:3]],  # First 5 dims of first 3 embeddings
                'embedding_norms': [float(np.linalg.norm(emb)) for emb in embeddings]
            }
        
        return comparison

def test_embedding_models():
    """Test different embedding models"""
    sample_texts = [
        "This Act may be cited as the Saweera Foundation (Incorporation) Act, No. 29 of 2024.",
        "The Court of Appeal has jurisdiction to hear appeals from the High Court.",
        "Property rights are fundamental to the legal system of Sri Lanka."
    ]
    
    # Test single model
    generator = EmbeddingGenerator()
    embeddings = generator.generate_batch_embeddings(sample_texts)
    
    print(f"Model: {generator.model_name}")
    print(f"Embedding dimension: {generator.get_embedding_dimension()}")
    print(f"Generated {len(embeddings)} embeddings")
    
    # Test multiple models
    multi_generator = MultiModelEmbeddingGenerator()
    comparison = multi_generator.compare_models(sample_texts)
    
    print("\nModel Comparison:")
    for model_name, stats in comparison.items():
        print(f"{model_name}: dimension={stats['dimension']}, avg_norm={np.mean(stats['embedding_norms']):.4f}")

if __name__ == "__main__":
    test_embedding_models()
