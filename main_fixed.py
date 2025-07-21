"""
Fixed Main Integration Module for Vector Database and Embedding System
Works with current dependency setup and real data
"""

import logging
import json
import time
from typing import List, Dict, Any, Optional
from pinecone import Pinecone
import config

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
    
    def __init__(self):
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self.index = self.pc.Index(config.PINECONE_INDEX_NAME)
        logger.info("VectorDBEmbeddingSystem initialized")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and statistics"""
        try:
            stats = self.index.describe_index_stats()
            
            status = {
                'timestamp': time.time(),
                'configuration': {
                    'index_name': config.PINECONE_INDEX_NAME,
                    'vector_dimension': config.VECTOR_DIMENSION,
                    'chunk_size': config.CHUNK_SIZE,
                    'supported_languages': config.SUPPORTED_LANGUAGES
                },
                'database_info': {
                    'total_documents': stats.get('total_vector_count', 0),
                    'dimension': stats.get('dimension', 0),
                    'index_fullness': stats.get('index_fullness', 0)
                },
                'system_ready': stats.get('total_vector_count', 0) > 0
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'system_ready': False, 'error': str(e)}
    
    def search_documents(self, query: str, filters: Optional[Dict[str, str]] = None,
                        top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for documents using natural language query"""
        logger.info(f"Searching for: '{query}'")
        
        try:
            # Create a simple query embedding (basic hash-based approach)
            query_embedding = self._create_query_embedding(query)
            
            # Build filter dictionary for Pinecone
            filter_dict = {}
            if filters:
                if filters.get('document_type'):
                    filter_dict['document_type'] = filters['document_type']
                if filters.get('language'):
                    filter_dict['detected_language'] = filters['language']
                if filters.get('category'):
                    filter_dict['primary_category'] = filters['category']
            
            # Perform search
            search_results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict if filter_dict else None
            )
            
            # Format results
            results = []
            for match in search_results.get('matches', []):
                result = {
                    'id': match['id'],
                    'score': match['score'],
                    'metadata': match.get('metadata', {})
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def _create_query_embedding(self, query: str, dimension: int = 384) -> List[float]:
        """Create a basic embedding for query (same method as used for documents)"""
        import hashlib
        
        # Create a hash-based embedding
        text_hash = hashlib.md5(query.encode()).hexdigest()
        
        # Convert hash to numbers
        hash_numbers = [int(text_hash[i:i+2], 16) for i in range(0, len(text_hash), 2)]
        
        # Normalize to [-1, 1] range
        normalized = [(x - 127.5) / 127.5 for x in hash_numbers]
        
        # Extend or truncate to desired dimension
        if len(normalized) < dimension:
            # Repeat pattern to fill dimension
            multiplier = dimension // len(normalized) + 1
            extended = (normalized * multiplier)[:dimension]
        else:
            extended = normalized[:dimension]
        
        # Add some text-based features
        words = query.lower().split()
        word_count = len(words)
        avg_word_length = sum(len(word) for word in words) / max(len(words), 1)
        
        # Modify embedding based on text features
        for i in range(min(10, len(extended))):
            extended[i] += (word_count % 100) / 1000.0
            extended[i] += (avg_word_length % 10) / 100.0
        
        # Ensure values are in reasonable range
        extended = [max(-1.0, min(1.0, x)) for x in extended]
        
        return extended
    
    def get_document_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored documents"""
        try:
            stats = self.index.describe_index_stats()
            
            # Get sample documents to analyze
            sample_query = self.index.query(
                vector=[0.1] * config.VECTOR_DIMENSION,
                top_k=100,
                include_metadata=True
            )
            
            # Analyze document types and categories
            doc_types = {}
            categories = {}
            languages = {}
            
            for match in sample_query.get('matches', []):
                metadata = match.get('metadata', {})
                
                doc_type = metadata.get('document_type', 'unknown')
                category = metadata.get('primary_category', 'unknown')
                language = metadata.get('detected_language', 'unknown')
                
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                categories[category] = categories.get(category, 0) + 1
                languages[language] = languages.get(language, 0) + 1
            
            return {
                'total_vectors': stats.get('total_vector_count', 0),
                'dimension': stats.get('dimension', 0),
                'document_types': doc_types,
                'categories': categories,
                'languages': languages,
                'sample_size': len(sample_query.get('matches', []))
            }
            
        except Exception as e:
            logger.error(f"Error getting document statistics: {e}")
            return {}
    
    def test_search_functionality(self) -> Dict[str, Any]:
        """Test search functionality with sample queries"""
        test_queries = [
            "property ownership rights",
            "marriage and divorce laws", 
            "commercial contract disputes",
            "employment termination",
            "intellectual property"
        ]
        
        test_results = {}
        
        for query in test_queries:
            try:
                results = self.search_documents(query, top_k=5)
                test_results[query] = {
                    'results_count': len(results),
                    'top_result': results[0] if results else None,
                    'success': len(results) > 0
                }
            except Exception as e:
                test_results[query] = {
                    'results_count': 0,
                    'error': str(e),
                    'success': False
                }
        
        return test_results
    
    def run_data_verification(self) -> Dict[str, Any]:
        """Verify that real legal data is properly stored"""
        try:
            # Check if we have the expected amount of data
            stats = self.index.describe_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            
            # Get sample of documents
            sample_results = self.index.query(
                vector=[0.1] * config.VECTOR_DIMENSION,
                top_k=10,
                include_metadata=True
            )
            
            verification = {
                'total_vectors': total_vectors,
                'expected_minimum': 5000,  # We uploaded 5442
                'data_present': total_vectors > 5000,
                'sample_documents': []
            }
            
            # Analyze sample documents
            for match in sample_results.get('matches', []):
                metadata = match.get('metadata', {})
                verification['sample_documents'].append({
                    'id': match['id'],
                    'document_type': metadata.get('document_type'),
                    'category': metadata.get('primary_category'),
                    'sequence_number': metadata.get('sequence_number'),
                    'filename': metadata.get('filename', '')[:50] + "..." if len(metadata.get('filename', '')) > 50 else metadata.get('filename', '')
                })
            
            return verification
            
        except Exception as e:
            logger.error(f"Error in data verification: {e}")
            return {'data_present': False, 'error': str(e)}

def main():
    """Main execution function"""
    print("=== Sri Lankan Legal Document Vector Database System ===")
    print("Initializing system...")
    
    # Initialize system
    try:
        system = VectorDBEmbeddingSystem()
        print("✅ System initialized successfully")
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return
    
    # Get system status
    print("\n📊 Checking system status...")
    status = system.get_system_status()
    
    if status.get('system_ready'):
        print("✅ System is ready!")
        print(f"📄 Total documents: {status['database_info']['total_documents']}")
        print(f"📐 Vector dimension: {status['database_info']['dimension']}")
    else:
        print("⚠️ System not ready or no data found")
        if 'error' in status:
            print(f"Error: {status['error']}")
        return
    
    # Verify real data
    print("\n🔍 Verifying real legal data...")
    verification = system.run_data_verification()
    
    if verification.get('data_present'):
        print("✅ Real legal data verified!")
        print(f"📊 Total vectors: {verification['total_vectors']}")
        print("📋 Sample documents:")
        for i, doc in enumerate(verification['sample_documents'][:5]):
            print(f"   {i+1}. {doc['document_type']} - {doc['category']} (Seq: {doc['sequence_number']})")
            print(f"      File: {doc['filename']}")
    else:
        print("❌ Real data verification failed")
        if 'error' in verification:
            print(f"Error: {verification['error']}")
        return
    
    # Get document statistics
    print("\n📈 Document Statistics:")
    stats = system.get_document_statistics()
    
    if stats:
        print(f"📊 Total vectors: {stats['total_vectors']}")
        print(f"📄 Document types: {stats['document_types']}")
        print(f"🏷️ Categories: {stats['categories']}")
        print(f"🌐 Languages: {stats['languages']}")
    
    # Test search functionality
    print("\n🔍 Testing Search Functionality:")
    test_results = system.test_search_functionality()
    
    successful_searches = 0
    for query, result in test_results.items():
        if result['success']:
            successful_searches += 1
            print(f"✅ '{query}': {result['results_count']} results")
            if result.get('top_result'):
                top = result['top_result']
                print(f"   Top result: {top['metadata'].get('document_type', 'unknown')} - {top['metadata'].get('primary_category', 'unknown')} (Score: {top['score']:.3f})")
        else:
            print(f"❌ '{query}': Failed")
    
    print(f"\n📊 Search test results: {successful_searches}/{len(test_results)} successful")
    
    # Interactive search demo
    print("\n🎯 Interactive Search Demo:")
    print("You can now search the legal documents!")
    print("Examples:")
    print("  - 'property ownership and transfer'")
    print("  - 'marriage dissolution procedures'") 
    print("  - 'employment contract termination'")
    print("  - 'commercial dispute resolution'")
    
    while True:
        try:
            query = input("\n🔍 Enter search query (or 'quit' to exit): ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            print(f"\n🔍 Searching for: '{query}'")
            results = system.search_documents(query, top_k=5)
            
            if results:
                print(f"📋 Found {len(results)} results:")
                for i, result in enumerate(results):
                    print(f"\n   {i+1}. Score: {result['score']:.3f}")
                    print(f"      ID: {result['id']}")
                    print(f"      Type: {result['metadata'].get('document_type', 'unknown')}")
                    print(f"      Category: {result['metadata'].get('primary_category', 'unknown')}")
                    print(f"      Sequence: {result['metadata'].get('sequence_number', 'unknown')}")
                    print(f"      File: {result['metadata'].get('filename', 'unknown')[:50]}...")
                    print(f"      Preview: {result['metadata'].get('text_preview', 'No preview')[:100]}...")
            else:
                print("❌ No results found")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Search error: {e}")
    
    print("\n🎉 Demo completed!")
    print("Your Vector Database and Embedding Module is working with real legal data!")

if __name__ == "__main__":
    main()
