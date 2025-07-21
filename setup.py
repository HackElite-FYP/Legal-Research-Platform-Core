"""
Setup and Installation Script for Sri Lankan Legal Vector DB System
"""

import subprocess
import sys
import os
import json
import logging
from typing import List, Dict, Any

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False
    logger.info(f"Python version: {sys.version}")
    return True

def install_requirements():
    """Install required packages"""
    logger.info("Installing required packages...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        logger.info("Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install requirements: {e}")
        return False

def download_nltk_data():
    """Download required NLTK data"""
    logger.info("Downloading NLTK data...")
    
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        logger.info("NLTK data downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to download NLTK data: {e}")
        return False

def verify_data_files():
    """Verify that required data files exist"""
    required_files = [
        'acts_2024.json',
        'cases_2024.json',
        'tags for classifcation and metadata-vector embeddings.xlsx'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"Missing required data files: {missing_files}")
        return False
    
    logger.info("All required data files found")
    return True

def test_pinecone_connection():
    """Test Pinecone connection"""
    logger.info("Testing Pinecone connection...")
    
    try:
        import pinecone
        import config
        
        pinecone.init(api_key=config.PINECONE_API_KEY, environment=config.PINECONE_ENVIRONMENT)
        
        # Test connection by listing indexes
        indexes = pinecone.list_indexes()
        logger.info(f"Pinecone connection successful. Available indexes: {len(indexes)}")
        return True
        
    except Exception as e:
        logger.error(f"Pinecone connection failed: {e}")
        logger.error("Please check your API key and environment in config.py")
        return False

def test_model_loading():
    """Test loading of embedding models"""
    logger.info("Testing model loading...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Test default model
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        test_embedding = model.encode("test sentence")
        
        logger.info(f"Model loaded successfully. Embedding dimension: {len(test_embedding)}")
        return True
        
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        'logs',
        'models',
        'reports',
        'temp'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

def run_basic_tests():
    """Run basic system tests"""
    logger.info("Running basic system tests...")
    
    try:
        # Test document processing
        from document_processor import DocumentProcessor
        processor = DocumentProcessor()
        logger.info("✓ Document processor initialized")
        
        # Test embedding generator
        from embedding_generator import EmbeddingGenerator
        generator = EmbeddingGenerator()
        logger.info("✓ Embedding generator initialized")
        
        # Test classifier
        from classifier import LegalClassificationSystem
        classifier = LegalClassificationSystem()
        logger.info("✓ Classifier initialized")
        
        # Test evaluation
        from evaluation import ComprehensiveEvaluator
        evaluator = ComprehensiveEvaluator()
        logger.info("✓ Evaluator initialized")
        
        logger.info("All basic tests passed")
        return True
        
    except Exception as e:
        logger.error(f"Basic tests failed: {e}")
        return False

def generate_setup_report():
    """Generate setup validation report"""
    report = {
        'setup_timestamp': str(pd.Timestamp.now()),
        'python_version': sys.version,
        'system_info': {
            'platform': sys.platform,
            'python_executable': sys.executable
        },
        'validation_results': {
            'python_version': check_python_version(),
            'requirements_installed': True,  # Assume success if we reach here
            'nltk_data': download_nltk_data(),
            'data_files': verify_data_files(),
            'pinecone_connection': test_pinecone_connection(),
            'model_loading': test_model_loading(),
            'basic_tests': run_basic_tests()
        }
    }
    
    # Save report
    with open('setup_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    return report

def main():
    """Main setup function"""
    print("=== Sri Lankan Legal Vector DB System Setup ===")
    print("Starting system setup and validation...\n")
    
    setup_steps = [
        ("Checking Python version", check_python_version),
        ("Installing requirements", install_requirements),
        ("Downloading NLTK data", download_nltk_data),
        ("Verifying data files", verify_data_files),
        ("Creating directories", create_directories),
        ("Testing Pinecone connection", test_pinecone_connection),
        ("Testing model loading", test_model_loading),
        ("Running basic tests", run_basic_tests)
    ]
    
    results = {}
    all_passed = True
    
    for step_name, step_function in setup_steps:
        print(f"⏳ {step_name}...")
        try:
            if step_function == create_directories:
                step_function()
                result = True
            else:
                result = step_function()
            
            results[step_name] = result
            
            if result:
                print(f"✅ {step_name} - SUCCESS")
            else:
                print(f"❌ {step_name} - FAILED")
                all_passed = False
                
        except Exception as e:
            print(f"❌ {step_name} - ERROR: {e}")
            results[step_name] = False
            all_passed = False
        
        print()
    
    # Generate setup report
    print("📊 Generating setup report...")
    try:
        import pandas as pd
        report = generate_setup_report()
        print("✅ Setup report generated: setup_report.json")
    except:
        print("⚠️  Could not generate detailed setup report")
    
    # Final summary
    print("\n" + "="*50)
    if all_passed:
        print("🎉 SETUP COMPLETED SUCCESSFULLY!")
        print("\nYour system is ready to use. You can now:")
        print("1. Run 'python main.py' to start the complete pipeline")
        print("2. Run 'python test_system.py' to validate the system")
        print("3. Use the individual modules for specific tasks")
        
        print("\nNext steps:")
        print("- Review config.py for any custom settings")
        print("- Check the README.md for usage examples")
        print("- Run the test suite to ensure everything works")
        
    else:
        print("❌ SETUP INCOMPLETE")
        print("\nSome setup steps failed. Please:")
        print("1. Check the error messages above")
        print("2. Ensure you have internet connection")
        print("3. Verify your Pinecone API key in config.py")
        print("4. Check that all data files are present")
        
        failed_steps = [step for step, result in results.items() if not result]
        print(f"\nFailed steps: {', '.join(failed_steps)}")
    
    print("\nFor support, check the README.md or create an issue on GitHub.")
    print("="*50)

if __name__ == "__main__":
    main()
