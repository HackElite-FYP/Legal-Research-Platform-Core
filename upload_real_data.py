"""
Upload Real Legal Documents (Acts and Cases) with Proper Processing
- Document chunking with sequence numbers
- Real embeddings generation
- Classification and metadata
- Store all data in Pinecone
"""

from pinecone import Pinecone
import json
import time
import logging
from typing import List, Dict, Any
import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_existing_data():
    """Clear existing test data from Pinecone index"""
    print("🧹 Clearing existing test data...")
    
    try:
        pc = Pinecone(api_key=config.PINECONE_API_KEY)
        index = pc.Index(config.PINECONE_INDEX_NAME)
        
        # Delete all existing vectors
        index.delete(delete_all=True)
        
        print("✅ Existing data cleared")
        time.sleep(5)  # Wait for deletion to complete
        
        return True
        
    except Exception as e:
        print(f"❌ Error clearing data: {e}")
        return False

def process_and_upload_real_data():
    """Process and upload real legal documents with proper chunking and embeddings"""
    print("🚀 Processing and uploading real legal documents...")
    
    try:
        # Import required modules
        from document_processor import DocumentProcessor
        from classifier import LegalClassificationSystem
        from embedding_generator import EmbeddingGenerator
        from vector_database import VectorDatabaseManager
        
        print("📋 Initializing processing modules...")
        
        # Initialize components
        processor = DocumentProcessor()
        classifier = LegalClassificationSystem()
        embedding_generator = EmbeddingGenerator()
        db_manager = VectorDatabaseManager()
        
        print("✅ All modules initialized successfully")
        
        # Step 1: Process all documents
        print("\n📄 Step 1: Processing documents...")
        processed_acts, processed_cases = processor.process_all_documents()
        all_processed_docs = processed_acts + processed_cases
        
        print(f"✅ Processed {len(processed_acts)} act chunks and {len(processed_cases)} case chunks")
        print(f"📊 Total chunks: {len(all_processed_docs)}")
        
        # Step 2: Classify documents
        print("\n🏷️ Step 2: Classifying documents...")
        print("⏳ Training classifier on legal documents...")
        classifier.train_classifier(all_processed_docs[:100])  # Train on subset for speed
        
        print("🔍 Classifying all documents...")
        classified_docs = classifier.classify_documents(all_processed_docs)
        
        print(f"✅ Classified {len(classified_docs)} document chunks")
        
        # Step 3: Generate embeddings
        print("\n🧠 Step 3: Generating embeddings...")
        embedded_docs = embedding_generator.embed_documents(classified_docs)
        
        print(f"✅ Generated embeddings for {len(embedded_docs)} documents")
        print(f"📐 Embedding dimension: {embedding_generator.get_embedding_dimension()}")
        
        # Step 4: Upload to Pinecone
        print("\n📤 Step 4: Uploading to Pinecone...")
        db_manager.index_documents(embedded_docs)
        
        print("✅ All real legal documents uploaded successfully!")
        
        # Step 5: Verify upload
        print("\n🔍 Step 5: Verifying upload...")
        time.sleep(10)  # Wait for indexing
        
        db_info = db_manager.get_database_info()
        print(f"📊 Final database statistics:")
        print(f"   Total documents: {db_info['total_documents']}")
        print(f"   Index name: {db_info['index_name']}")
        print(f"   Dimension: {db_info['dimension']}")
        
        # Test search functionality
        print("\n🔍 Testing search with real data...")
        test_queries = [
            "property ownership rights",
            "marriage and divorce laws",
            "commercial contract disputes"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            results = db_manager.search_legal_documents(query, embedding_generator, top_k=3)
            
            for i, result in enumerate(results):
                print(f"   {i+1}. Score: {result['score']:.3f}")
                print(f"      ID: {result['id']}")
                print(f"      Type: {result['metadata'].get('document_type', 'unknown')}")
                print(f"      Category: {result['metadata'].get('primary_category', 'unknown')}")
                print(f"      Sequence: {result['metadata'].get('sequence_number', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing real data: {e}")
        logger.error(f"Error details: {str(e)}", exc_info=True)
        return False

def show_document_statistics():
    """Show statistics about the processed documents"""
    print("\n📊 Document Statistics:")
    
    try:
        # Load raw data
        with open('acts_2024.json', 'r', encoding='utf-8') as f:
            acts_data = json.load(f)
        
        with open('cases_2024.json', 'r', encoding='utf-8') as f:
            cases_data = json.load(f)
        
        print(f"📋 Raw Data:")
        print(f"   Acts: {len(acts_data)} documents")
        print(f"   Cases: {len(cases_data)} documents")
        print(f"   Total: {len(acts_data) + len(cases_data)} documents")
        
        # Calculate total word count
        total_words_acts = sum(doc.get('wordCount', 0) for doc in acts_data)
        total_words_cases = sum(doc.get('wordCount', 0) for doc in cases_data)
        
        print(f"\n📝 Word Counts:")
        print(f"   Acts: {total_words_acts:,} words")
        print(f"   Cases: {total_words_cases:,} words")
        print(f"   Total: {total_words_acts + total_words_cases:,} words")
        
        # Estimate chunks (assuming 512 tokens per chunk, ~400 words per chunk)
        estimated_chunks = (total_words_acts + total_words_cases) // 400
        print(f"\n🔢 Estimated chunks: ~{estimated_chunks} chunks")
        
        return True
        
    except Exception as e:
        print(f"❌ Error showing statistics: {e}")
        return False

def main():
    """Main function to upload real legal documents"""
    print("=" * 70)
    print("📚 REAL LEGAL DOCUMENTS UPLOADER")
    print("=" * 70)
    
    # Show document statistics
    show_document_statistics()
    
    # Confirm with user
    print("\n⚠️  This will:")
    print("   1. Clear existing test data")
    print("   2. Process ALL legal documents (Acts and Cases)")
    print("   3. Generate real embeddings for each chunk")
    print("   4. Upload everything to Pinecone")
    print("   5. This may take 10-30 minutes depending on your system")
    
    confirm = input("\n🤔 Do you want to proceed? (y/n): ").lower().strip()
    
    if confirm != 'y':
        print("❌ Operation cancelled")
        return
    
    # Step 1: Clear existing data
    if not clear_existing_data():
        print("❌ Failed to clear existing data")
        return
    
    # Step 2: Process and upload real data
    print("\n🚀 Starting real data processing...")
    start_time = time.time()
    
    if process_and_upload_real_data():
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n🎉 SUCCESS!")
        print(f"✅ All real legal documents processed and uploaded")
        print(f"⏱️  Total time: {duration/60:.1f} minutes")
        print(f"🎯 Your Pinecone dashboard now shows REAL legal document data")
        
        print(f"\n💡 What was uploaded:")
        print(f"   📄 All Acts and Cases from your JSON files")
        print(f"   🔢 Each document split into chunks with sequence numbers")
        print(f"   🏷️ Legal classification tags applied")
        print(f"   🧠 Real embeddings generated using {config.EMBEDDING_MODEL_NAME}")
        print(f"   📊 Rich metadata for each chunk")
        
        print(f"\n🔍 You can now:")
        print(f"   1. See real data in Pinecone dashboard")
        print(f"   2. Search using natural language queries")
        print(f"   3. Filter by document type, category, language")
        print(f"   4. Proceed with RAG module development")
        
    else:
        print("\n❌ Failed to process real data")
        print("🔧 Please check the error messages above")

if __name__ == "__main__":
    main()
