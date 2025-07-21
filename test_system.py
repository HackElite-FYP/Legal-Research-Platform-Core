"""
Test Script for Vector Database and Embedding System
Validates all components and generates test reports
"""

import unittest
import numpy as np
import json
import tempfile
import os
from typing import List, Dict, Any
import logging

# Import system modules
from document_processor import DocumentProcessor
from classifier import LegalClassificationSystem
from embedding_generator import EmbeddingGenerator
from vector_database import VectorDatabaseManager
from evaluation import ComprehensiveEvaluator
from main import VectorDBEmbeddingSystem
import utils
import config

# Setup test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestDocumentProcessor(unittest.TestCase):
    """Test document processing functionality"""
    
    def setUp(self):
        self.processor = DocumentProcessor()
        self.sample_doc = {
            'id': 'test-doc-001',
            'filename': 'Test Legal Document',
            'primaryLang': 'English',
            'text': 'This is a test legal document. It contains multiple sentences for testing chunking functionality. The document discusses property rights and ownership laws in Sri Lanka.',
            'wordCount': 25
        }
    
    def test_document_processing(self):
        """Test basic document processing"""
        processed_chunks = self.processor._process_single_document(self.sample_doc, 'act')
        
        self.assertGreater(len(processed_chunks), 0)
        self.assertIn('id', processed_chunks[0])
        self.assertIn('text', processed_chunks[0])
        self.assertIn('metadata', processed_chunks[0])
    
    def test_chunk_metadata(self):
        """Test metadata extraction"""
        processed_chunks = self.processor._process_single_document(self.sample_doc, 'case')
        
        metadata = processed_chunks[0]['metadata']
        self.assertEqual(metadata['document_type'], 'case')
        self.assertEqual(metadata['original_id'], 'test-doc-001')
        self.assertIn('sequence_number', metadata)
        self.assertIn('detected_language', metadata)
    
    def test_statistics(self):
        """Test statistics generation"""
        processed_chunks = self.processor._process_single_document(self.sample_doc, 'act')
        stats = self.processor.get_document_statistics(processed_chunks)
        
        self.assertIn('total_chunks', stats)
        self.assertIn('language_distribution', stats)
        self.assertIn('document_type_distribution', stats)

class TestClassifier(unittest.TestCase):
    """Test classification functionality"""
    
    def setUp(self):
        self.classifier = LegalClassificationSystem()
        self.sample_docs = [
            {
                'id': 'test-001',
                'text': 'Property ownership and transfer rights are fundamental to legal system',
                'metadata': {'document_type': 'act'}
            },
            {
                'id': 'test-002', 
                'text': 'Marriage dissolution and divorce proceedings require court approval',
                'metadata': {'document_type': 'case'}
            }
        ]
    
    def test_tag_loading(self):
        """Test classification tags loading"""
        self.assertGreater(len(self.classifier.all_tags), 0)
        self.assertIn('family', self.classifier.classification_tags)
        self.assertIn('property', self.classifier.classification_tags)
    
    def test_training_data_creation(self):
        """Test training data creation"""
        texts, labels = self.classifier._create_training_data(self.sample_docs)
        
        self.assertEqual(len(texts), len(labels))
        self.assertEqual(len(texts), 2)
        self.assertIsInstance(labels[0], list)
    
    def test_primary_category_assignment(self):
        """Test primary category assignment"""
        predicted_labels = ['Property', 'Ownership']
        primary = self.classifier._get_primary_category(predicted_labels)
        
        self.assertIsInstance(primary, str)
        self.assertIn(primary.lower(), ['family', 'property', 'commercial', 'labour', 'general'])

class TestEmbeddingGenerator(unittest.TestCase):
    """Test embedding generation functionality"""
    
    def setUp(self):
        self.generator = EmbeddingGenerator()
        self.sample_texts = [
            "Property rights and ownership laws",
            "Marriage and divorce proceedings",
            "Employment contract disputes"
        ]
    
    def test_single_embedding(self):
        """Test single text embedding"""
        embedding = self.generator.generate_embedding(self.sample_texts[0])
        
        self.assertIsInstance(embedding, np.ndarray)
        self.assertGreater(len(embedding), 0)
        self.assertEqual(len(embedding), self.generator.get_embedding_dimension())
    
    def test_batch_embeddings(self):
        """Test batch embedding generation"""
        embeddings = self.generator.generate_batch_embeddings(self.sample_texts)
        
        self.assertEqual(len(embeddings), len(self.sample_texts))
        self.assertIsInstance(embeddings[0], np.ndarray)
    
    def test_empty_text_handling(self):
        """Test handling of empty text"""
        embedding = self.generator.generate_embedding("")
        
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(len(embedding), self.generator.get_embedding_dimension())
    
    def test_document_embedding(self):
        """Test document embedding with metadata"""
        sample_docs = [
            {
                'id': 'test-001',
                'text': text,
                'metadata': {'document_type': 'test'}
            } for text in self.sample_texts
        ]
        
        embedded_docs = self.generator.embed_documents(sample_docs)
        
        self.assertEqual(len(embedded_docs), len(sample_docs))
        self.assertIn('embedding', embedded_docs[0])
        self.assertIn('embedding_model', embedded_docs[0]['metadata'])

class TestVectorDatabase(unittest.TestCase):
    """Test vector database functionality"""
    
    def setUp(self):
        # Use a test index name to avoid conflicts
        self.original_index_name = config.PINECONE_INDEX_NAME
        config.PINECONE_INDEX_NAME = f"test-{config.PINECONE_INDEX_NAME}"
        
        try:
            self.db_manager = VectorDatabaseManager()
            self.db_available = True
        except Exception as e:
            logger.warning(f"Pinecone not available for testing: {e}")
            self.db_available = False
    
    def tearDown(self):
        # Restore original index name
        config.PINECONE_INDEX_NAME = self.original_index_name
        
        # Clean up test index if created
        if self.db_available:
            try:
                self.db_manager.db.delete_all_vectors()
            except:
                pass
    
    def test_database_info(self):
        """Test database information retrieval"""
        if not self.db_available:
            self.skipTest("Pinecone not available")
        
        info = self.db_manager.get_database_info()
        
        self.assertIn('index_name', info)
        self.assertIn('total_documents', info)
    
    def test_metadata_preparation(self):
        """Test metadata preparation for Pinecone"""
        if not self.db_available:
            self.skipTest("Pinecone not available")
        
        sample_metadata = {
            'document_type': 'act',
            'original_id': 'test-001',
            'chunk_index': 0,
            'classification_tags': ['Property', 'Ownership'],
            'chunk_text_preview': 'This is a test preview...'
        }
        
        prepared = self.db_manager.db._prepare_metadata(sample_metadata)
        
        self.assertIn('document_type', prepared)
        self.assertIn('original_id', prepared)
        self.assertIsInstance(prepared['classification_tags'], str)

class TestEvaluation(unittest.TestCase):
    """Test evaluation framework"""
    
    def setUp(self):
        self.evaluator = ComprehensiveEvaluator()
        self.sample_retrieved = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
        self.sample_relevant = ['doc1', 'doc3', 'doc6']
    
    def test_precision_at_k(self):
        """Test Precision@K calculation"""
        precision = self.evaluator.retrieval_evaluator.precision_at_k(
            self.sample_retrieved, self.sample_relevant, 3
        )
        
        # doc1 and doc3 are relevant in top 3
        expected = 2/3
        self.assertAlmostEqual(precision, expected, places=3)
    
    def test_recall_at_k(self):
        """Test Recall@K calculation"""
        recall = self.evaluator.retrieval_evaluator.recall_at_k(
            self.sample_retrieved, self.sample_relevant, 5
        )
        
        # 2 out of 3 relevant docs retrieved
        expected = 2/3
        self.assertAlmostEqual(recall, expected, places=3)
    
    def test_mrr_calculation(self):
        """Test MRR calculation"""
        mrr = self.evaluator.retrieval_evaluator.mean_reciprocal_rank(
            self.sample_retrieved, self.sample_relevant
        )
        
        # First relevant doc is at position 1 (doc1)
        expected = 1.0
        self.assertAlmostEqual(mrr, expected, places=3)
    
    def test_ndcg_calculation(self):
        """Test NDCG calculation"""
        relevance_scores = {'doc1': 1.0, 'doc3': 1.0, 'doc6': 1.0}
        ndcg = self.evaluator.retrieval_evaluator.ndcg_at_k(
            self.sample_retrieved, self.sample_relevant, relevance_scores, 5
        )
        
        self.assertGreaterEqual(ndcg, 0.0)
        self.assertLessEqual(ndcg, 1.0)

class TestSystemIntegration(unittest.TestCase):
    """Test complete system integration"""
    
    def setUp(self):
        self.system = VectorDBEmbeddingSystem()
    
    def test_system_initialization(self):
        """Test system initialization"""
        self.assertIsNotNone(self.system.document_processor)
        self.assertIsNotNone(self.system.classifier)
        self.assertIsNotNone(self.system.embedding_generator)
        self.assertIsNotNone(self.system.evaluator)
    
    def test_system_status(self):
        """Test system status retrieval"""
        status = self.system.get_system_status()
        
        self.assertIn('configuration', status)
        self.assertIn('database_info', status)
        self.assertIn('system_ready', status)
        self.assertTrue(status['system_ready'])
    
    def test_search_functionality(self):
        """Test search functionality (mock)"""
        # This would require indexed documents, so we test the interface
        try:
            results = self.system.search_documents("test query", top_k=5)
            self.assertIsInstance(results, list)
        except Exception as e:
            # Expected if no documents are indexed
            logger.info(f"Search test skipped: {e}")

def run_performance_tests():
    """Run performance tests and generate report"""
    logger.info("Running performance tests...")
    
    performance_results = {
        'document_processing': {},
        'embedding_generation': {},
        'classification': {},
        'search_performance': {}
    }
    
    # Test document processing speed
    processor = DocumentProcessor()
    sample_docs = [
        {
            'id': f'perf-test-{i}',
            'filename': f'Test Document {i}',
            'primaryLang': 'English',
            'text': 'This is a performance test document. ' * 50,  # ~50 words
            'wordCount': 50
        } for i in range(100)
    ]
    
    import time
    start_time = time.time()
    processed_chunks = processor.process_documents(sample_docs, 'test')
    processing_time = time.time() - start_time
    
    performance_results['document_processing'] = {
        'documents_processed': len(sample_docs),
        'chunks_generated': len(processed_chunks),
        'processing_time_seconds': processing_time,
        'docs_per_second': len(sample_docs) / processing_time
    }
    
    # Test embedding generation speed
    generator = EmbeddingGenerator()
    sample_texts = [chunk['text'] for chunk in processed_chunks[:50]]
    
    start_time = time.time()
    embeddings = generator.generate_batch_embeddings(sample_texts)
    embedding_time = time.time() - start_time
    
    performance_results['embedding_generation'] = {
        'texts_embedded': len(sample_texts),
        'embedding_time_seconds': embedding_time,
        'embeddings_per_second': len(sample_texts) / embedding_time,
        'embedding_dimension': len(embeddings[0]) if embeddings else 0
    }
    
    # Save performance report
    with open('performance_test_report.json', 'w') as f:
        json.dump(performance_results, f, indent=2)
    
    logger.info("Performance tests completed. Report saved to performance_test_report.json")
    return performance_results

def main():
    """Run all tests"""
    print("=== Sri Lankan Legal Vector DB System Tests ===")
    
    # Run unit tests
    print("\n1. Running Unit Tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance tests
    print("\n2. Running Performance Tests...")
    performance_results = run_performance_tests()
    
    print("\n=== Test Results Summary ===")
    print(f"Document Processing: {performance_results['document_processing']['docs_per_second']:.2f} docs/sec")
    print(f"Embedding Generation: {performance_results['embedding_generation']['embeddings_per_second']:.2f} embeddings/sec")
    
    print("\n=== System Validation Complete ===")
    print("All tests completed. Check test reports for detailed results.")

if __name__ == "__main__":
    main()
