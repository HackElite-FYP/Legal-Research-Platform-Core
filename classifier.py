"""
BERT-based Legal Document Classifier
Classifies legal documents into predefined categories using tags
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import json
import config
import utils

logger = logging.getLogger(__name__)

class LegalDocumentClassifier(nn.Module):
    """BERT-based multi-label classifier for legal documents"""
    
    def __init__(self, model_name: str = config.CLASSIFICATION_MODEL, num_labels: int = config.NUM_LABELS):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Load BERT model and tokenizer
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Classification head
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
        logger.info(f"Initialized classifier with {model_name} for {num_labels} labels")
    
    def forward(self, input_ids, attention_mask):
        """Forward pass"""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class LegalClassificationSystem:
    """Complete classification system for legal documents"""
    
    def __init__(self):
        self.classification_tags = utils.load_classification_tags()
        self.all_tags = self._flatten_tags()
        self.label_encoder = MultiLabelBinarizer()
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Classification system initialized with {len(self.all_tags)} tags")
        logger.info(f"Using device: {self.device}")
    
    def _flatten_tags(self) -> List[str]:
        """Flatten all classification tags into a single list"""
        all_tags = []
        for category, tags in self.classification_tags.items():
            all_tags.extend(tags)
        return list(set(all_tags))  # Remove duplicates
    
    def _create_training_data(self, processed_docs: List[Dict[str, Any]]) -> Tuple[List[str], List[List[str]]]:
        """Create training data by matching text with relevant tags"""
        texts = []
        labels = []
        
        for doc in processed_docs:
            text = doc['text'].lower()
            doc_labels = []
            
            # Simple keyword matching for initial labeling
            for tag in self.all_tags:
                if tag.lower() in text:
                    doc_labels.append(tag)
            
            # Ensure at least one label per document
            if not doc_labels:
                # Assign based on document type or default category
                doc_type = doc['metadata']['document_type']
                if doc_type == 'act':
                    doc_labels = ['Property']  # Default for acts
                else:
                    doc_labels = ['Commercial']  # Default for cases
            
            texts.append(doc['text'])
            labels.append(doc_labels)
        
        return texts, labels
    
    def prepare_dataset(self, texts: List[str], labels: List[List[str]]) -> Dict[str, torch.Tensor]:
        """Prepare dataset for training"""
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=config.MAX_SEQUENCE_LENGTH,
            return_tensors='pt'
        )
        
        dataset = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': torch.tensor(encoded_labels, dtype=torch.float)
        }
        
        return dataset
    
    def train_classifier(self, processed_docs: List[Dict[str, Any]], epochs: int = 3, batch_size: int = 16):
        """Train the classification model"""
        logger.info("Starting classifier training...")
        
        # Create training data
        texts, labels = self._create_training_data(processed_docs)
        dataset = self.prepare_dataset(texts, labels)
        
        # Initialize model
        num_labels = len(self.label_encoder.classes_)
        self.model = LegalDocumentClassifier(num_labels=num_labels)
        self.tokenizer = self.model.tokenizer
        self.model.to(self.device)
        
        # Training setup
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        criterion = nn.BCEWithLogitsLoss()
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            num_batches = len(dataset['input_ids']) // batch_size
            
            for i in tqdm(range(0, len(dataset['input_ids']), batch_size), 
                         desc=f"Epoch {epoch+1}/{epochs}"):
                
                batch_input_ids = dataset['input_ids'][i:i+batch_size].to(self.device)
                batch_attention_mask = dataset['attention_mask'][i:i+batch_size].to(self.device)
                batch_labels = dataset['labels'][i:i+batch_size].to(self.device)
                
                optimizer.zero_grad()
                
                logits = self.model(batch_input_ids, batch_attention_mask)
                loss = criterion(logits, batch_labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / num_batches
            logger.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        logger.info("Training completed!")
    
    def classify_documents(self, processed_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Classify processed documents"""
        if self.model is None:
            logger.error("Model not trained. Please train the classifier first.")
            return processed_docs
        
        logger.info(f"Classifying {len(processed_docs)} documents...")
        
        self.model.eval()
        classified_docs = []
        
        with torch.no_grad():
            for doc in tqdm(processed_docs, desc="Classifying documents"):
                text = doc['text']
                
                # Tokenize
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding=True,
                    max_length=config.MAX_SEQUENCE_LENGTH,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                # Predict
                logits = self.model(input_ids, attention_mask)
                probabilities = torch.sigmoid(logits).cpu().numpy()[0]
                
                # Get predicted labels (threshold = 0.5)
                predicted_indices = np.where(probabilities > 0.5)[0]
                predicted_labels = [self.label_encoder.classes_[i] for i in predicted_indices]
                
                # Update document metadata
                doc['metadata']['classification_tags'] = predicted_labels
                doc['metadata']['classification_scores'] = {
                    self.label_encoder.classes_[i]: float(probabilities[i]) 
                    for i in range(len(probabilities))
                }
                
                # Determine primary category
                if predicted_labels:
                    primary_category = self._get_primary_category(predicted_labels)
                    doc['metadata']['primary_category'] = primary_category
                else:
                    doc['metadata']['primary_category'] = 'General'
                
                classified_docs.append(doc)
        
        logger.info("Classification completed!")
        return classified_docs
    
    def _get_primary_category(self, predicted_labels: List[str]) -> str:
        """Determine primary category from predicted labels"""
        category_counts = {'family': 0, 'property': 0, 'commercial': 0, 'labour': 0}
        
        for label in predicted_labels:
            for category, tags in self.classification_tags.items():
                if label in tags:
                    category_counts[category] += 1
        
        # Return category with highest count
        primary_category = max(category_counts, key=category_counts.get)
        return primary_category.capitalize()
    
    def save_model(self, path: str):
        """Save trained model"""
        if self.model is None:
            logger.error("No model to save")
            return
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'label_encoder': self.label_encoder,
            'classification_tags': self.classification_tags
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # Initialize model
            num_labels = len(checkpoint['label_encoder'].classes_)
            self.model = LegalDocumentClassifier(num_labels=num_labels)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            
            # Load other components
            self.label_encoder = checkpoint['label_encoder']
            self.classification_tags = checkpoint['classification_tags']
            self.tokenizer = self.model.tokenizer
            
            logger.info(f"Model loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")

if __name__ == "__main__":
    # Test the classifier
    from document_processor import DocumentProcessor
    
    # Process documents
    processor = DocumentProcessor()
    processed_acts, processed_cases = processor.process_all_documents()
    all_docs = processed_acts + processed_cases
    
    # Train classifier
    classifier = LegalClassificationSystem()
    classifier.train_classifier(all_docs[:100])  # Use subset for testing
    
    # Classify documents
    classified_docs = classifier.classify_documents(all_docs[:10])
    
    # Save model
    classifier.save_model('legal_classifier.pth')
    
    # Print sample results
    for doc in classified_docs[:3]:
        print(f"Document ID: {doc['id']}")
        print(f"Primary Category: {doc['metadata']['primary_category']}")
        print(f"Classification Tags: {doc['metadata']['classification_tags']}")
        print("---")
