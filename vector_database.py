"""
Vector Database Integration Module
Handles Pinecone vector database operations for legal document storage and retrieval
"""

from pinecone import Pinecone, ServerlessSpec
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import time
import config

logger = logging.getLogger(__name__)

class VectorDatabase:
    """Pinecone vector database interface for legal documents"""

    def __init__(self, api_key: str = config.PINECONE_API_KEY,
                 environment: str = config.PINECONE_ENVIRONMENT,
                 index_name: str = config.PINECONE_INDEX_NAME):

        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.pc = None
        self.index = None

        self._initialize_pinecone()
        logger.info(f"VectorDatabase initialized with index: {index_name}")

    def _initialize_pinecone(self):
        """Initialize Pinecone connection"""
        try:
            # Initialize Pinecone client
            self.pc = Pinecone(api_key=self.api_key)

            # Check if index exists, create if not
            existing_indexes = self.pc.list_indexes().names()
            if self.index_name not in existing_indexes:
                logger.info(f"Creating new index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=config.VECTOR_DIMENSION,
                    metric=config.METRIC,
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                # Wait for index to be ready
                logger.info("Waiting for index to be ready...")
                time.sleep(30)

            # Connect to index
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")

        except Exception as e:
            logger.error(f"Error initializing Pinecone: {str(e)}")
            raise
    
    def upsert_documents(self, embedded_docs: List[Dict[str, Any]], batch_size: int = 100):
        """Upload embedded documents to Pinecone"""
        if not self.index:
            logger.error("Pinecone index not initialized")
            return
        
        logger.info(f"Upserting {len(embedded_docs)} documents to Pinecone...")
        
        # Prepare vectors for upsert
        vectors = []
        for doc in embedded_docs:
            if 'embedding' not in doc or not doc['embedding']:
                logger.warning(f"No embedding found for document {doc['id']}")
                continue
            
            # Prepare metadata (Pinecone has limitations on metadata size)
            metadata = self._prepare_metadata(doc['metadata'])
            
            vector = {
                'id': doc['id'],
                'values': doc['embedding'],
                'metadata': metadata
            }
            vectors.append(vector)
        
        # Upsert in batches
        for i in tqdm(range(0, len(vectors), batch_size), desc="Upserting to Pinecone"):
            batch = vectors[i:i + batch_size]
            try:
                self.index.upsert(vectors=batch)
            except Exception as e:
                logger.error(f"Error upserting batch {i//batch_size + 1}: {str(e)}")
                continue
        
        logger.info(f"Successfully upserted {len(vectors)} vectors to Pinecone")
    
    def _prepare_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare metadata for Pinecone (handle size limitations and data types)"""
        # Pinecone metadata limitations: max 40KB, specific data types
        prepared_metadata = {}
        
        # Include essential fields
        essential_fields = [
            'document_type', 'original_id', 'filename', 'chunk_index', 
            'sequence_number', 'primary_language', 'detected_language',
            'word_count', 'primary_category'
        ]
        
        for field in essential_fields:
            if field in metadata:
                value = metadata[field]
                # Convert to appropriate type
                if isinstance(value, (int, float, str, bool)):
                    prepared_metadata[field] = value
                elif isinstance(value, list) and len(value) > 0:
                    # Convert list to string if it's not too long
                    str_value = str(value)
                    if len(str_value) < 1000:
                        prepared_metadata[field] = str_value
        
        # Add classification tags (limited)
        if 'classification_tags' in metadata:
            tags = metadata['classification_tags']
            if isinstance(tags, list) and tags:
                # Take first 5 tags to avoid size limits
                prepared_metadata['classification_tags'] = ','.join(tags[:5])
        
        # Add text preview
        if 'chunk_text_preview' in metadata:
            preview = metadata['chunk_text_preview']
            if len(preview) < 500:  # Limit preview size
                prepared_metadata['text_preview'] = preview
        
        return prepared_metadata
    
    def search(self, query_embedding: List[float], top_k: int = 10, 
               filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if not self.index:
            logger.error("Pinecone index not initialized")
            return []
        
        try:
            # Perform search
            search_results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            # Format results
            results = []
            for match in search_results['matches']:
                result = {
                    'id': match['id'],
                    'score': match['score'],
                    'metadata': match.get('metadata', {})
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching Pinecone: {str(e)}")
            return []
    
    def search_with_filters(self, query_embedding: List[float], 
                           document_type: Optional[str] = None,
                           language: Optional[str] = None,
                           category: Optional[str] = None,
                           top_k: int = 10) -> List[Dict[str, Any]]:
        """Search with specific filters"""
        filter_dict = {}
        
        if document_type:
            filter_dict['document_type'] = document_type
        if language:
            filter_dict['detected_language'] = language
        if category:
            filter_dict['primary_category'] = category
        
        return self.search(query_embedding, top_k, filter_dict)
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the index"""
        if not self.index:
            return {}
        
        try:
            stats = self.index.describe_index_stats()
            return {
                'total_vector_count': stats.get('total_vector_count', 0),
                'dimension': stats.get('dimension', 0),
                'index_fullness': stats.get('index_fullness', 0),
                'namespaces': stats.get('namespaces', {})
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return {}
    
    def delete_all_vectors(self):
        """Delete all vectors from the index (use with caution)"""
        if not self.index:
            logger.error("Pinecone index not initialized")
            return
        
        try:
            self.index.delete(delete_all=True)
            logger.info("All vectors deleted from index")
        except Exception as e:
            logger.error(f"Error deleting vectors: {str(e)}")
    
    def hybrid_search(self, query_embedding: List[float], 
                     query_text: str, 
                     top_k: int = 10,
                     alpha: float = 0.7) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector similarity and text matching
        alpha: weight for vector search (1-alpha for text search)
        """
        # Get vector search results
        vector_results = self.search(query_embedding, top_k * 2)  # Get more for reranking
        
        # Simple text matching score (can be improved with BM25)
        text_scores = {}
        query_words = set(query_text.lower().split())
        
        for result in vector_results:
            text_preview = result['metadata'].get('text_preview', '')
            if text_preview:
                text_words = set(text_preview.lower().split())
                # Jaccard similarity
                intersection = len(query_words.intersection(text_words))
                union = len(query_words.union(text_words))
                text_score = intersection / union if union > 0 else 0
                text_scores[result['id']] = text_score
        
        # Combine scores
        for result in vector_results:
            vector_score = result['score']
            text_score = text_scores.get(result['id'], 0)
            combined_score = alpha * vector_score + (1 - alpha) * text_score
            result['combined_score'] = combined_score
            result['text_score'] = text_score
        
        # Sort by combined score and return top_k
        hybrid_results = sorted(vector_results, key=lambda x: x['combined_score'], reverse=True)
        return hybrid_results[:top_k]

class VectorDatabaseManager:
    """High-level manager for vector database operations"""
    
    def __init__(self):
        self.db = VectorDatabase()
        logger.info("VectorDatabaseManager initialized")
    
    def index_documents(self, embedded_docs: List[Dict[str, Any]]):
        """Index embedded documents in the vector database"""
        logger.info(f"Indexing {len(embedded_docs)} documents...")
        
        # Validate embeddings
        valid_docs = []
        for doc in embedded_docs:
            if 'embedding' in doc and doc['embedding']:
                valid_docs.append(doc)
            else:
                logger.warning(f"Skipping document {doc['id']} - no embedding")
        
        if valid_docs:
            self.db.upsert_documents(valid_docs)
            logger.info(f"Successfully indexed {len(valid_docs)} documents")
        else:
            logger.error("No valid documents to index")
    
    def search_legal_documents(self, query: str, embedding_generator, 
                              filters: Optional[Dict[str, str]] = None,
                              top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for legal documents using natural language query"""
        # Generate query embedding
        query_embedding = embedding_generator.generate_embedding(query)
        
        # Search with filters
        if filters:
            results = self.db.search_with_filters(
                query_embedding.tolist(),
                document_type=filters.get('document_type'),
                language=filters.get('language'),
                category=filters.get('category'),
                top_k=top_k
            )
        else:
            results = self.db.search(query_embedding.tolist(), top_k)
        
        return results
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the vector database"""
        stats = self.db.get_index_stats()
        return {
            'index_name': self.db.index_name,
            'total_documents': stats.get('total_vector_count', 0),
            'dimension': stats.get('dimension', 0),
            'index_fullness': stats.get('index_fullness', 0)
        }

if __name__ == "__main__":
    # Test vector database operations
    db_manager = VectorDatabaseManager()
    
    # Get database info
    info = db_manager.get_database_info()
    print("Database Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test search (requires indexed documents)
    from embedding_generator import EmbeddingGenerator
    
    generator = EmbeddingGenerator()
    results = db_manager.search_legal_documents(
        "property rights and ownership",
        generator,
        top_k=5
    )
    
    print(f"\nSearch Results: {len(results)} documents found")
    for i, result in enumerate(results[:3]):
        print(f"{i+1}. ID: {result['id']}, Score: {result['score']:.4f}")
        print(f"   Type: {result['metadata'].get('document_type', 'unknown')}")
        print(f"   Category: {result['metadata'].get('primary_category', 'unknown')}")
        print("---")
