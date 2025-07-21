"""
Document Processing Module for Legal Documents
Handles chunking, preprocessing, and metadata extraction
"""

import logging
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import config
import utils

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Process legal documents for vector database storage"""
    
    def __init__(self):
        self.classification_tags = utils.load_classification_tags()
        logger.info("DocumentProcessor initialized")
    
    def process_documents(self, documents: List[Dict[str, Any]], doc_type: str) -> List[Dict[str, Any]]:
        """
        Process documents into chunks with metadata
        
        Args:
            documents: List of document dictionaries
            doc_type: Type of document ('act' or 'case')
            
        Returns:
            List of processed document chunks
        """
        processed_chunks = []
        
        logger.info(f"Processing {len(documents)} {doc_type} documents...")
        
        for doc in tqdm(documents, desc=f"Processing {doc_type}s"):
            try:
                chunks = self._process_single_document(doc, doc_type)
                processed_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error processing document {doc.get('id', 'unknown')}: {str(e)}")
                continue
        
        logger.info(f"Generated {len(processed_chunks)} chunks from {len(documents)} documents")
        return processed_chunks
    
    def _process_single_document(self, document: Dict[str, Any], doc_type: str) -> List[Dict[str, Any]]:
        """Process a single document into chunks"""
        text = document.get('text', '')
        if not text:
            logger.warning(f"Empty text in document {document.get('id', 'unknown')}")
            return []
        
        # Clean the text
        cleaned_text = utils.clean_text(text)
        
        # Split into chunks
        chunks = utils.chunk_text(cleaned_text)
        
        processed_chunks = []
        for i, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                continue
                
            # Create unique ID for this chunk
            chunk_id = utils.create_document_id(doc_type, document.get('id', ''), i)
            
            # Extract metadata
            metadata = utils.extract_metadata(document, i, chunk_text)
            
            # Add classification tags (will be populated by classifier)
            metadata['classification_tags'] = []
            metadata['predicted_categories'] = []
            
            processed_chunk = {
                'id': chunk_id,
                'text': chunk_text,
                'metadata': metadata,
                'embedding': None  # Will be populated by embedding module
            }
            
            processed_chunks.append(processed_chunk)
        
        return processed_chunks
    
    def process_all_documents(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Process all acts and cases documents"""
        # Load raw data
        acts_data = utils.load_json_data(config.ACTS_DATA_PATH)
        cases_data = utils.load_json_data(config.CASES_DATA_PATH)
        
        # Process documents
        processed_acts = self.process_documents(acts_data, 'act')
        processed_cases = self.process_documents(cases_data, 'case')
        
        # Save processed data
        utils.save_processed_data(processed_acts, 'processed_acts.json')
        utils.save_processed_data(processed_cases, 'processed_cases.json')
        
        return processed_acts, processed_cases
    
    def get_document_statistics(self, processed_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about processed documents"""
        if not processed_docs:
            return {}
        
        total_chunks = len(processed_docs)
        total_words = sum(doc['metadata']['word_count'] for doc in processed_docs)
        
        # Language distribution
        languages = {}
        doc_types = {}
        
        for doc in processed_docs:
            lang = doc['metadata']['detected_language']
            doc_type = doc['metadata']['document_type']
            
            languages[lang] = languages.get(lang, 0) + 1
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        # Chunk size distribution
        chunk_sizes = [doc['metadata']['word_count'] for doc in processed_docs]
        avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes)
        
        stats = {
            'total_chunks': total_chunks,
            'total_words': total_words,
            'average_chunk_size': avg_chunk_size,
            'language_distribution': languages,
            'document_type_distribution': doc_types,
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes)
        }
        
        return stats

if __name__ == "__main__":
    # Test the document processor
    processor = DocumentProcessor()
    processed_acts, processed_cases = processor.process_all_documents()
    
    # Print statistics
    acts_stats = processor.get_document_statistics(processed_acts)
    cases_stats = processor.get_document_statistics(processed_cases)
    
    print("Acts Statistics:")
    for key, value in acts_stats.items():
        print(f"  {key}: {value}")
    
    print("\nCases Statistics:")
    for key, value in cases_stats.items():
        print(f"  {key}: {value}")
