"""
Main Integration Module for Vector Database and Embedding System
Orchestrates the complete pipeline from document processing to evaluation
"""

import logging
import json
import time
from typing import List, Dict, Any, Optional
import config
from document_processor import DocumentProcessor
from classifier import LegalClassificationSystem
from embedding_generator import EmbeddingGenerator, MultiModelEmbeddingGenerator
from vector_database import VectorDatabaseManager
from evaluation import ComprehensiveEvaluator
import utils

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vector_db_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VectorDBEmbeddingSystem:
    """Complete Vector Database and Embedding System for Legal Documents"""
    
    def __init__(self, embedding_model: str = config.EMBEDDING_MODEL_NAME):
        self.embedding_model = embedding_model
        self.document_processor = DocumentProcessor()
        self.classifier = LegalClassificationSystem()
        self.embedding_generator = EmbeddingGenerator(embedding_model)
        self.vector_db_manager = VectorDatabaseManager()
        self.evaluator = ComprehensiveEvaluator()
        
        logger.info(f"VectorDBEmbeddingSystem initialized with model: {embedding_model}")
    
    def run_complete_pipeline(self, train_classifier: bool = True, 
                             evaluate_system: bool = True) -> Dict[str, Any]:
        """Run the complete pipeline from document processing to evaluation"""
        logger.info("Starting complete pipeline execution...")
        
        pipeline_results = {
            'processing': {},
            'classification': {},
            'embedding': {},
            'indexing': {},
            'evaluation': {}
        }
        
        try:
            # Step 1: Document Processing
            logger.info("Step 1: Processing documents...")
            processed_acts, processed_cases = self.document_processor.process_all_documents()
            all_processed_docs = processed_acts + processed_cases
            
            pipeline_results['processing'] = {
                'total_documents': len(all_processed_docs),
                'acts_chunks': len(processed_acts),
                'cases_chunks': len(processed_cases),
                'statistics': self.document_processor.get_document_statistics(all_processed_docs)
            }
            
            # Step 2: Classification
            logger.info("Step 2: Classifying documents...")
            if train_classifier:
                self.classifier.train_classifier(all_processed_docs)
                self.classifier.save_model('legal_classifier.pth')
            
            classified_docs = self.classifier.classify_documents(all_processed_docs)
            
            pipeline_results['classification'] = {
                'classified_documents': len(classified_docs),
                'categories_found': len(set(doc['metadata'].get('primary_category', 'Unknown') 
                                          for doc in classified_docs))
            }
            
            # Step 3: Embedding Generation
            logger.info("Step 3: Generating embeddings...")
            embedded_docs = self.embedding_generator.embed_documents(classified_docs)
            
            pipeline_results['embedding'] = {
                'embedded_documents': len(embedded_docs),
                'embedding_model': self.embedding_model,
                'embedding_dimension': self.embedding_generator.get_embedding_dimension()
            }
            
            # Step 4: Vector Database Indexing
            logger.info("Step 4: Indexing in vector database...")
            self.vector_db_manager.index_documents(embedded_docs)
            
            db_info = self.vector_db_manager.get_database_info()
            pipeline_results['indexing'] = db_info
            
            # Step 5: Evaluation (if requested)
            if evaluate_system:
                logger.info("Step 5: Evaluating system...")
                test_queries = self._create_test_queries()
                evaluation_results = self.evaluator.evaluate_system(
                    self.vector_db_manager,
                    self.embedding_generator,
                    test_queries
                )
                pipeline_results['evaluation'] = evaluation_results
                
                # Generate evaluation report
                self.evaluator.generate_evaluation_report(
                    evaluation_results,
                    f'evaluation_report_{int(time.time())}.json'
                )
            
            logger.info("Pipeline execution completed successfully!")
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            pipeline_results['error'] = str(e)
        
        return pipeline_results
    
    def _create_test_queries(self) -> List[Dict[str, Any]]:
        """Create test queries for evaluation"""
        test_queries = [
            {
                'query': 'property ownership and transfer rights',
                'relevant_docs': [],  # Would be populated with ground truth
                'category': 'property'
            },
            {
                'query': 'marriage dissolution and divorce proceedings',
                'relevant_docs': [],
                'category': 'family'
            },
            {
                'query': 'commercial contract disputes and remedies',
                'relevant_docs': [],
                'category': 'commercial'
            },
            {
                'query': 'employment termination and severance pay',
                'relevant_docs': [],
                'category': 'labour'
            },
            {
                'query': 'intellectual property rights and copyright',
                'relevant_docs': [],
                'category': 'commercial'
            },
            {
                'query': 'child custody and maintenance obligations',
                'relevant_docs': [],
                'category': 'family'
            },
            {
                'query': 'land acquisition and compensation',
                'relevant_docs': [],
                'category': 'property'
            },
            {
                'query': 'workplace safety and employee rights',
                'relevant_docs': [],
                'category': 'labour'
            }
        ]
        
        return test_queries
    
    def search_documents(self, query: str, filters: Optional[Dict[str, str]] = None,
                        top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for documents using natural language query"""
        logger.info(f"Searching for: '{query}'")
        
        results = self.vector_db_manager.search_legal_documents(
            query, 
            self.embedding_generator,
            filters=filters,
            top_k=top_k
        )
        
        logger.info(f"Found {len(results)} results")
        return results
    
    def compare_embedding_models(self, model_names: List[str] = None) -> Dict[str, Any]:
        """Compare different embedding models"""
        if model_names is None:
            model_names = [
                config.EMBEDDING_MODEL_NAME,
                config.LEGAL_BERT_MODEL,
                config.SENTENCE_BERT_MODEL
            ]
        
        logger.info(f"Comparing embedding models: {model_names}")
        
        # Load processed documents
        processed_acts = utils.load_processed_data('processed_acts.json')
        processed_cases = utils.load_processed_data('processed_cases.json')
        all_docs = processed_acts + processed_cases
        
        if not all_docs:
            logger.error("No processed documents found. Run document processing first.")
            return {}
        
        # Use subset for comparison
        sample_docs = all_docs[:100]  # Use first 100 documents
        
        multi_generator = MultiModelEmbeddingGenerator(model_names)
        model_results = {}
        
        for model_name in model_names:
            try:
                logger.info(f"Testing model: {model_name}")
                
                # Generate embeddings
                generator = EmbeddingGenerator(model_name)
                embedded_docs = generator.embed_documents(sample_docs.copy())
                
                # Create temporary vector database for this model
                temp_db = VectorDatabaseManager()
                temp_db.db.index_name = f"temp-{model_name.replace('/', '-')}"
                temp_db.index_documents(embedded_docs)
                
                # Evaluate
                test_queries = self._create_test_queries()
                evaluation_results = self.evaluator.evaluate_system(
                    temp_db,
                    generator,
                    test_queries
                )
                
                model_results[model_name] = evaluation_results
                
            except Exception as e:
                logger.error(f"Error testing model {model_name}: {str(e)}")
                continue
        
        # Compare results
        comparison = self.evaluator.compare_models(model_results)
        
        # Save comparison report
        with open(f'model_comparison_{int(time.time())}.json', 'w') as f:
            json.dump({
                'model_results': model_results,
                'comparison': comparison
            }, f, indent=2)
        
        return comparison
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and statistics"""
        status = {
            'timestamp': time.time(),
            'configuration': {
                'embedding_model': self.embedding_model,
                'vector_dimension': config.VECTOR_DIMENSION,
                'chunk_size': config.CHUNK_SIZE,
                'supported_languages': config.SUPPORTED_LANGUAGES
            },
            'database_info': self.vector_db_manager.get_database_info(),
            'classification_tags': len(self.classifier.all_tags),
            'system_ready': True
        }
        
        return status

def main():
    """Main execution function"""
    print("=== Sri Lankan Legal Document Vector Database System ===")
    print("Initializing system...")
    
    # Initialize system
    system = VectorDBEmbeddingSystem()
    
    # Get system status
    status = system.get_system_status()
    print(f"System Status: {status['system_ready']}")
    print(f"Database Documents: {status['database_info']['total_documents']}")
    
    # Run complete pipeline
    print("\nRunning complete pipeline...")
    results = system.run_complete_pipeline(
        train_classifier=True,
        evaluate_system=True
    )
    
    # Print results summary
    print("\n=== Pipeline Results Summary ===")
    for step, result in results.items():
        if isinstance(result, dict) and 'error' not in result:
            print(f"{step.capitalize()}: ✓")
            if step == 'processing':
                print(f"  - Total documents processed: {result.get('total_documents', 0)}")
            elif step == 'embedding':
                print(f"  - Embedding dimension: {result.get('embedding_dimension', 0)}")
            elif step == 'evaluation':
                auto_metrics = result.get('automatic_metrics', {})
                print(f"  - Precision@5: {auto_metrics.get('precision@5', 0):.3f}")
                print(f"  - Recall@5: {auto_metrics.get('recall@5', 0):.3f}")
                print(f"  - MRR: {auto_metrics.get('mrr', 0):.3f}")
        else:
            print(f"{step.capitalize()}: ✗")
    
    # Test search functionality
    print("\n=== Testing Search Functionality ===")
    test_queries = [
        "property ownership rights",
        "marriage and divorce laws",
        "employment contract disputes"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        search_results = system.search_documents(query, top_k=3)
        
        for i, result in enumerate(search_results):
            print(f"  {i+1}. Score: {result['score']:.3f}")
            print(f"     Type: {result['metadata'].get('document_type', 'unknown')}")
            print(f"     Category: {result['metadata'].get('primary_category', 'unknown')}")
    
    print("\n=== System Ready for Use ===")
    print("You can now:")
    print("1. Search documents using natural language queries")
    print("2. Filter by document type, language, or category")
    print("3. Evaluate system performance with expert validation")
    print("4. Compare different embedding models")

if __name__ == "__main__":
    main()
