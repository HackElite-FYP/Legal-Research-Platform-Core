"""
Pinecone Diagnostic Script
Check the actual status of Pinecone index and data storage
"""

from pinecone import Pinecone, ServerlessSpec
import json
import time
from typing import Dict, Any, List
import config

def check_pinecone_connection():
    """Check Pinecone connection and list indexes"""
    print("🔍 Checking Pinecone connection...")

    try:
        # Initialize Pinecone client
        pc = Pinecone(api_key=config.PINECONE_API_KEY)
        print(f"✅ Connected to Pinecone")

        # List all indexes
        indexes = pc.list_indexes().names()
        print(f"📋 Available indexes: {indexes}")

        return True, indexes, pc

    except Exception as e:
        print(f"❌ Pinecone connection failed: {e}")
        return False, [], None

def check_index_status(index_name: str, pc: Pinecone):
    """Check specific index status and statistics"""
    print(f"\n🔍 Checking index: {index_name}")

    try:
        # Check if index exists
        indexes = pc.list_indexes().names()
        if index_name not in indexes:
            print(f"❌ Index '{index_name}' does not exist!")
            print(f"Available indexes: {indexes}")
            return False

        # Connect to index
        index = pc.Index(index_name)
        print(f"✅ Connected to index: {index_name}")

        # Get index statistics
        stats = index.describe_index_stats()
        print(f"📊 Index Statistics:")
        print(f"   Total vectors: {stats.get('total_vector_count', 0)}")
        print(f"   Dimension: {stats.get('dimension', 0)}")
        print(f"   Index fullness: {stats.get('index_fullness', 0)}")

        # Check namespaces
        namespaces = stats.get('namespaces', {})
        if namespaces:
            print(f"   Namespaces: {list(namespaces.keys())}")
            for ns_name, ns_stats in namespaces.items():
                print(f"     {ns_name}: {ns_stats.get('vector_count', 0)} vectors")
        else:
            print("   No namespaces found")

        return stats.get('total_vector_count', 0) > 0

    except Exception as e:
        print(f"❌ Error checking index: {e}")
        return False

def test_index_operations(index_name: str, pc: Pinecone):
    """Test basic index operations"""
    print(f"\n🧪 Testing index operations for: {index_name}")

    try:
        index = pc.Index(index_name)

        # Test upsert with a sample vector
        test_vector = {
            'id': 'test-vector-001',
            'values': [0.1] * config.VECTOR_DIMENSION,
            'metadata': {
                'test': True,
                'document_type': 'test',
                'text_preview': 'This is a test vector'
            }
        }

        print("📤 Upserting test vector...")
        index.upsert(vectors=[test_vector])

        # Wait a moment for indexing
        time.sleep(2)

        # Check if vector was added
        stats = index.describe_index_stats()
        print(f"📊 Vectors after test upsert: {stats.get('total_vector_count', 0)}")

        # Test query
        print("🔍 Testing query...")
        query_results = index.query(
            vector=[0.1] * config.VECTOR_DIMENSION,
            top_k=5,
            include_metadata=True
        )

        print(f"📋 Query returned {len(query_results.get('matches', []))} results")

        # Clean up test vector
        print("🧹 Cleaning up test vector...")
        index.delete(ids=['test-vector-001'])

        return True

    except Exception as e:
        print(f"❌ Index operations test failed: {e}")
        return False

def create_index_if_missing(index_name: str, pc: Pinecone):
    """Create index if it doesn't exist"""
    print(f"\n🏗️ Creating index: {index_name}")

    try:
        indexes = pc.list_indexes().names()

        if index_name in indexes:
            print(f"✅ Index '{index_name}' already exists")
            return True

        print(f"📝 Creating new index with dimension {config.VECTOR_DIMENSION}...")
        pc.create_index(
            name=index_name,
            dimension=config.VECTOR_DIMENSION,
            metric=config.METRIC,
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

        print("⏳ Waiting for index to be ready...")
        time.sleep(30)  # Wait for index creation

        # Verify creation
        indexes = pc.list_indexes().names()
        if index_name in indexes:
            print(f"✅ Index '{index_name}' created successfully!")
            return True
        else:
            print(f"❌ Index creation failed")
            return False

    except Exception as e:
        print(f"❌ Error creating index: {e}")
        return False

def upload_sample_data(index_name: str):
    """Upload sample data to test the index"""
    print(f"\n📤 Uploading sample data to: {index_name}")
    
    try:
        from embedding_generator import EmbeddingGenerator
        from document_processor import DocumentProcessor
        
        # Process a few sample documents
        processor = DocumentProcessor()
        
        # Load sample data
        with open('acts_2024.json', 'r', encoding='utf-8') as f:
            acts_data = json.load(f)
        
        # Process first 3 documents
        sample_docs = acts_data[:3]
        processed_chunks = processor.process_documents(sample_docs, 'act')
        
        print(f"📋 Processed {len(processed_chunks)} chunks from {len(sample_docs)} documents")
        
        # Generate embeddings
        generator = EmbeddingGenerator()
        embedded_docs = generator.embed_documents(processed_chunks)
        
        print(f"🧠 Generated embeddings for {len(embedded_docs)} chunks")
        
        # Upload to Pinecone
        from vector_database import VectorDatabase
        db = VectorDatabase(index_name=index_name)
        db.upsert_documents(embedded_docs)
        
        print(f"✅ Uploaded {len(embedded_docs)} vectors to Pinecone")
        
        # Check final stats
        time.sleep(5)  # Wait for indexing
        stats = db.get_index_stats()
        print(f"📊 Final index stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error uploading sample data: {e}")
        return False

def main():
    """Main diagnostic function"""
    print("=" * 60)
    print("🔧 PINECONE DIAGNOSTIC TOOL")
    print("=" * 60)

    # Step 1: Check connection
    connected, indexes, pc = check_pinecone_connection()
    if not connected:
        print("\n❌ Cannot proceed without Pinecone connection")
        return

    # Step 2: Check target index
    index_name = config.PINECONE_INDEX_NAME
    print(f"\n🎯 Target index: {index_name}")

    has_data = check_index_status(index_name, pc)

    # Step 3: Create index if missing
    if index_name not in indexes:
        print(f"\n⚠️ Index '{index_name}' not found!")
        create_choice = input("Create the index? (y/n): ").lower().strip()

        if create_choice == 'y':
            if create_index_if_missing(index_name, pc):
                has_data = check_index_status(index_name, pc)
            else:
                print("❌ Failed to create index")
                return
        else:
            print("❌ Cannot proceed without index")
            return

    # Step 4: Test operations
    operations_ok = test_index_operations(index_name, pc)

    # Step 5: Upload sample data if empty
    if not has_data:
        print(f"\n⚠️ Index '{index_name}' is empty!")
        upload_choice = input("Upload sample data? (y/n): ").lower().strip()

        if upload_choice == 'y':
            upload_sample_data(index_name)
        else:
            print("ℹ️ Index remains empty")

    # Final summary
    print("\n" + "=" * 60)
    print("📋 DIAGNOSTIC SUMMARY")
    print("=" * 60)

    final_stats = check_index_status(index_name, pc)

    if final_stats:
        print("✅ Pinecone index is working and contains data")
        print("🎯 You should now see data in your Pinecone dashboard")
        print("\n💡 Tips for Pinecone dashboard:")
        print("   1. Refresh the dashboard page")
        print("   2. Check the correct project/environment")
        print("   3. Look for the index name:", index_name)
        print("   4. Data may take a few minutes to appear")
    else:
        print("❌ Issues found with Pinecone setup")
        print("\n🔧 Troubleshooting steps:")
        print("   1. Verify API key in config.py")
        print("   2. Check environment setting")
        print("   3. Ensure index exists and has data")
        print("   4. Try running the main pipeline: python main.py")

if __name__ == "__main__":
    main()
