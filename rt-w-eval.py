import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
import logging
from datetime import datetime, timedelta
import json
import re
from transformers import AutoTokenizer, AutoModelForMaskedLM
from neo4j import GraphDatabase
from pinecone import Pinecone
import torch
import nest_asyncio
import firebase_admin
from firebase_admin import credentials, firestore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pinecone credentials
PINECONE_API_KEY = "pcsk_2it9oG_RzRNfQdLGg9jUen7wW6viE9JpLRgVTHtWbTZhomuZKmhuyYnrh8GgMyrHJMz37Q"
PINECONE_CASE_DENSE_INDEX = f"law-cases-dense"
PINECONE_CASE_SPARSE_INDEX = f"law-cases-sparse"
PINECONE_ACTS_DENSE_INDEX = f"law-acts-dense"
PINECONE_ACTS_SPARSE_INDEX = f"law-acts-sparse"
PINECONE_DENSE_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PINECONE_SPARSE_EMBED_MODEL = "naver/splade-cocondenser-ensembledistil"

# Neo4j credentials and URI
NEO4J_URI = "neo4j+s://66d16355.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "G4UXZ6KLGd1dzo57rp6ITypJHZ37aM1fn-exAWdw3p8"
NEO4J_DATABASE = "neo4j"

FIRESTORE_PROJECT_ID = "legal-research-platform"
FIRESTORE_CREDENTIALS_PATH = "legal-research-platform-firebase-adminsdk-fbsvc-ab73d198d7.json"


class QueryIntent(Enum):
    DOCTRINAL = "doctrinal"
    PRECEDENTIAL = "precedential"
    PROCEDURAL = "procedural"
    COMPARATIVE = "comparative"
    MIXED = "mixed"

class DocumentType(Enum):
    CASE = "case"
    ACT = "act"

@dataclass
class QueryContext:
    raw_query: str
    intent: QueryIntent
    complexity_score: float
    legal_domains: List[str]
    jurisdictions: List[str]
    temporal_constraints: Optional[Dict[str, Any]] = None
    extracted_entities: List[str] = field(default_factory=list)

@dataclass
class DocumentMetadata:
    doc_id: str
    doc_type: DocumentType
    title: str
    court_level: Optional[int] = None  # 1=Supreme, 2=High, 3=District
    jurisdiction: str = ""
    date: Optional[datetime] = None
    legal_domains: List[str] = field(default_factory=list)
    act_sections: List[str] = field(default_factory=list)

@dataclass
class RetrievedChunk:
    doc_id: str
    chunk_id: str
    content: str
    metadata: DocumentMetadata
    vector_score: float
    chunk_index: int
    
class KGFeatures:
    def __init__(self):
        self.citation_count: int = 0
        self.act_references: int = 0
        self.judge_count: int = 0
        self.jurisdictional_weight: float = 0.0
        self.authority_score: float = 0.0
        self.recency_boost: float = 0.0

@dataclass
class EnhancedChunk(RetrievedChunk):
    kg_features: KGFeatures = field(default_factory=KGFeatures)
    final_score: float = 0.0

@dataclass
class ReasoningStep:
    step_type: str  # "major_premise", "minor_premise", "application", "conclusion"
    content: str
    doc_id: str
    confidence: float

@dataclass
class ReasoningChain:
    chain_id: str
    steps: List[ReasoningStep]
    chain_strength: float

@dataclass
class ContextBundle:
    core_documents: List[Dict[str, Any]]
    supporting_context: Dict[str, List[Dict[str, Any]]]
    reasoning_chains: List[ReasoningChain]
    provenance_graph: Dict[str, List[str]]
    total_tokens: int

@dataclass
class EvaluationMetrics:
    relevance_scores: List[float]
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    authority_coverage: float
    reasoning_completeness: float
    response_time: float

def embed_text_dense(text: str, model: SentenceTransformer) -> List[float]:
    return model.encode(text).tolist()

def embed_text_sparse(text: str, tokenizer, model) -> Dict[str, List]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs).logits.squeeze(0)
    scores = torch.log(1 + torch.relu(outputs))
    max_scores, _ = torch.max(scores, dim=0)
    non_zero_indices = torch.nonzero(max_scores).squeeze(1).tolist()
    non_zero_values = max_scores[non_zero_indices].tolist()
    tokens = tokenizer.convert_ids_to_tokens(non_zero_indices)
    indices = tokenizer.convert_tokens_to_ids(tokens)
    return {
        "values": non_zero_values,
        "indices": indices,
    }

class QueryAnalyzer:
    def __init__(self):
        self.legal_domain_keywords = {
            "contract": ["contract", "agreement", "breach", "consideration", "offer", "acceptance"],
            "tort": ["negligence", "liability", "damages", "duty", "breach"],
            "criminal": ["criminal", "offense", "punishment", "conviction", "sentence"],
            "constitutional": ["constitutional", "fundamental rights", "directive principles"],
            "property": ["property", "ownership", "title", "possession", "transfer"]
        }
        
        self.intent_patterns = {
            QueryIntent.DOCTRINAL: ["interpret", "meaning", "definition", "scope", "provisions"],
            QueryIntent.PRECEDENTIAL: ["precedent", "case law", "judicial", "ruling", "decided"],
            QueryIntent.PROCEDURAL: ["procedure", "process", "steps", "filing", "court"],
            QueryIntent.COMPARATIVE: ["compare", "difference", "similar", "contrast", "versus"]
        }

    def analyze_query(self, query: str) -> QueryContext:
        """Comprehensive query analysis with intent classification"""
        intent = self._classify_intent(query)
        complexity = self._assess_complexity(query)
        domains = self._extract_legal_domains(query)
        jurisdictions = self._extract_jurisdictions(query)
        entities = self._extract_legal_entities(query)
        
        return QueryContext(
            raw_query=query,
            intent=intent,
            complexity_score=complexity,
            legal_domains=domains,
            jurisdictions=jurisdictions,
            extracted_entities=entities
        )
    
    def _classify_intent(self, query: str) -> QueryIntent:
        """Classify query intent using pattern matching"""
        query_lower = query.lower()
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            intent_scores[intent] = score
        
        if not intent_scores or max(intent_scores.values()) == 0:
            return QueryIntent.MIXED
        
        return max(intent_scores, key=intent_scores.get)
    
    def _assess_complexity(self, query: str) -> float:
        """Assess query complexity based on various factors"""
        complexity_indicators = [
            len(query.split()) > 15,  # Long queries
            "and" in query.lower() or "or" in query.lower(),  # Logical operators
            query.count("?") > 1,  # Multiple questions
            any(word in query.lower() for word in ["compare", "analyze", "explain", "distinguish"]),
            re.search(r'\d{4}', query) is not None,  # Year mentions
        ]
        return sum(complexity_indicators) / len(complexity_indicators)
    
    def _extract_legal_domains(self, query: str) -> List[str]:
        """Extract legal domains from query"""
        query_lower = query.lower()
        domains = []
        
        for domain, keywords in self.legal_domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                domains.append(domain)
        
        return domains
    
    def _extract_jurisdictions(self, query: str) -> List[str]:
        """Extract jurisdictions mentioned in query"""
        jurisdictions = []
        jurisdiction_patterns = [
            r'supreme court', r'high court', r'district court',
        ]
        
        for pattern in jurisdiction_patterns:
            if re.search(pattern, query.lower()):
                jurisdictions.append(pattern.replace(r'\b', '').replace(r'\s+', ' '))
        
        return jurisdictions
    
    def _extract_legal_entities(self, query: str) -> List[str]:
        """Extract legal entities like case names, act names"""
        entities = []
        
        # Pattern for case citations (simplified)
        case_pattern = r'([A-Z][a-z]+ v\.? [A-Z][a-z]+)'
        cases = re.findall(case_pattern, query)
        entities.extend(cases)
        
        # Pattern for act names
        act_pattern = r'([A-Z][A-Za-z\s]+ Act,?\s*\d{4})'
        acts = re.findall(act_pattern, query)
        entities.extend(acts)
        
        return entities

class VectorDatabase(ABC):
    @abstractmethod
    async def search(self, query: str, top_k: int = 50) -> List[RetrievedChunk]:
        pass

class DenseVectorDB(VectorDatabase):
    def __init__(self, db_type: DocumentType, pinecone_client: Pinecone, model: SentenceTransformer):
        self.db_type = db_type
        self.model = model
        # Choose the correct index based on document type
        if self.db_type == DocumentType.CASE:
            self.index = pinecone_client.Index(PINECONE_CASE_DENSE_INDEX)
        else:
            self.index = pinecone_client.Index(PINECONE_ACTS_DENSE_INDEX)
        logger.info(f"Initialized DenseVectorDB for {db_type.value}")

    async def search(self, query: str, top_k: int = 50) -> List[RetrievedChunk]:
        """Performs a real dense vector search using Pinecone."""
        logger.info(f"Searching dense {self.db_type.value} DB for: {query[:50]}...")
        try:
            query_vector = embed_text_dense(query, self.model)
            query_response = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True
            )
            
            results = []
            for match in query_response.get('matches', []):
                metadata = match.get('metadata', {})
                doc_metadata = DocumentMetadata(
                    doc_id=metadata.get('doc_id', '').split('_')[0],  # Extract base doc_id
                    doc_type=self.db_type,
                    title=metadata.get('title', 'Untitled'),
                    court_level=metadata.get('court_level'),
                    jurisdiction=metadata.get('jurisdiction', ''),
                    date=datetime.fromisoformat(metadata.get('date')) if metadata.get('date') else None,
                    legal_domains=metadata.get('legal_domains', []),
                    act_sections=metadata.get('act_sections', [])
                )
                chunk = RetrievedChunk(
                    doc_id=doc_metadata.doc_id,
                    chunk_id=match.get('id', 'unknown_chunk'),
                    content=metadata.get('content', ''),
                    metadata=doc_metadata,
                    vector_score=match.get('score', 0.0),
                    chunk_index=metadata.get('chunk_index', -1)
                )
                results.append(chunk)
            
            logger.info(f"Found {len(results)} dense results.")
            return results
        except Exception as e:
            logger.error(f"Error in DenseVectorDB search: {e}")
            return []

class SparseVectorDB(VectorDatabase):
    def __init__(self, db_type: DocumentType, pinecone_client: Pinecone, tokenizer, model):
        self.db_type = db_type
        self.tokenizer = tokenizer
        self.model = model
        # Choose the correct index
        if self.db_type == DocumentType.CASE:
            self.index = pinecone_client.Index(PINECONE_CASE_SPARSE_INDEX)
        else:
            self.index = pinecone_client.Index(PINECONE_ACTS_SPARSE_INDEX)
        logger.info(f"Initialized SparseVectorDB for {db_type.value}")

    async def search(self, query: str, top_k: int = 50) -> List[RetrievedChunk]:
        """Performs sparse vector search using Pinecone."""
        logger.info(f"Searching sparse {self.db_type.value} DB for: {query[:50]}...")
        try:
            sparse_vector = embed_text_sparse(query, self.tokenizer, self.model)
            query_response = self.index.query(
                sparse_vector=sparse_vector,
                top_k=top_k,
                include_metadata=True
            )
            
            results = []
            for match in query_response.get('matches', []):
                metadata = match.get('metadata', {})
                doc_metadata = DocumentMetadata(
                    doc_id=metadata.get('doc_id', '').split('_')[0],  # Extract base doc_id
                    doc_type=self.db_type,
                    title=metadata.get('title', 'Untitled'),
                    court_level=metadata.get('court_level'),
                    jurisdiction=metadata.get('jurisdiction', ''),
                    date=datetime.fromisoformat(metadata.get('date')) if metadata.get('date') else None,
                    legal_domains=metadata.get('legal_domains', []),
                    act_sections=metadata.get('act_sections', [])
                )
                chunk = RetrievedChunk(
                    doc_id=doc_metadata.doc_id,
                    chunk_id=match.get('id', 'unknown_chunk'),
                    content=metadata.get('content', ''),
                    metadata=doc_metadata,
                    vector_score=match.get('score', 0.0),
                    chunk_index=metadata.get('chunk_index', -1)
                )
                results.append(chunk)

            logger.info(f"Found {len(results)} sparse results.")
            return results
        except Exception as e:
            logger.error(f"Error in SparseVectorDB search: {e}")
            return []

class KnowledgeGraph:
    def __init__(self, driver: GraphDatabase.driver):
        self.driver = driver
        self.driver.verify_connectivity()
        logger.info("Neo4j connection verified.")

    def _execute_query(self, query, params=None):
        """Helper function to run a read query."""
        with self.driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(query, parameters=params or {})
            return [dict(record) for record in result]
        
    async def get_claim_premise_data(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get claim-premise data for a document from Neo4j knowledge graph."""
        if not self.driver: 
            # Return mock data for testing
            return {
                "claims": [f"Mock claim for {doc_id}"],
                "premises": [f"Mock premise for {doc_id}"]
            }
        
        try:
            # Use the Neo4j driver session to query claims and premises
            with self.driver.session() as session:
                # Query for claims and premises using the REFERS_TO relationship
                cypher_query = """
                MATCH (doc {doc_id: $doc_id})
                RETURN doc.claims AS claims,
                        doc.premises AS premises
                """
                
                result = session.run(cypher_query, {"doc_id": doc_id}).single()
                if result:
                    return {
                        "claims": result["claims"] or [],
                        "premises": result["premises"] or []
                    }
                else:
                    logger.warning(f"No claim-premise data found for doc_id: {doc_id}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching claim-premise data for {doc_id} from Neo4j: {e}")
            return None

    def compute_kg_features(self, doc_id: str) -> KGFeatures:
        """Compute simplified KG features based on citation counts and relationships."""
        features = KGFeatures()
        
        try:
            # Query for Cases
            case_query = """
            MATCH (d:Case {doc_id: $doc_id})
            OPTIONAL MATCH (d)-[:REFERS_TO]->(act:Act)
            OPTIONAL MATCH (a)<-[:REFERS_TO]-(citing_case:Case)
            OPTIONAL MATCH (d)<-[r]-(judge:Person)
            RETURN 
                count(DISTINCT act) as act_references,
                count(DISTINCT citing_case) as citation_count,
                count(DISTINCT judge) as judge_count,
            """
            
            # Query for Acts
            act_query = """
            MATCH (d:Act {doc_id: $doc_id})
            OPTIONAL MATCH (case:Case)-[:REFERS_TO]->(d)
            RETURN 
                count(DISTINCT case) as citation_count,
                0 as act_references,
                0 as judge_count,
            """
            params = {"doc_id": doc_id}
            
            # Try Case first, then Act
            result = self._execute_query(case_query, params)
            if not result or result[0]['citation_count'] is None:
                result = self._execute_query(act_query, params)
            
            if result:
                data = result[0]
                features.citation_count = data.get('citation_count', 0) or 0
                features.act_references = data.get('act_references', 0) or 0
                features.judge_count = data.get('judge_count', 0) or 0
                
                # Compute jurisdictional weight based on court level
                court_level = data.get('court_level')
                if court_level:
                    features.jurisdictional_weight = 1.0 / court_level  # Supreme=1.0, High=0.5, District=0.33
                else:
                    features.jurisdictional_weight = 0.5  # Default for Acts
                
                # Compute recency boost
                doc_year = data.get('doc_year', 2000)
                if doc_year and isinstance(doc_year, int):
                    current_year = datetime.now().year
                    years_old = current_year - doc_year
                    features.recency_boost = max(0, 1.0 - (years_old / 10))  # Decay over 10 years
                else:
                    features.recency_boost = 0.0
                
                # Compute overall authority score
                features.authority_score = self._compute_authority_score(features)

        except Exception as e:
            logger.error(f"Error computing KG features for {doc_id}: {e}")
        
        return features
    
    def _compute_authority_score(self, features: KGFeatures) -> float:
        """Compute overall authority score from features"""
        # Normalize citation count (assume max of 50 citations)
        normalized_citations = min(features.citation_count / 50.0, 1.0)
        
        # Normalize act references (assume max of 10)
        normalized_act_refs = min(features.act_references / 10.0, 1.0)
        
        # Compute weighted score
        authority_score = (
            0.4 * normalized_citations +
            0.2 * normalized_act_refs +
            0.2 * features.jurisdictional_weight +
            0.1 * features.recency_boost +
            0.1 * min(features.judge_count / 5.0, 1.0)  # More judges = more authority
        )
        
        return authority_score
    
    def expand_context(self, doc_ids: List[str], max_depth: int = 2) -> Dict[str, List[Dict[str, Any]]]:
        """Intelligent context expansion via Neo4j graph traversal."""
        expanded_context = {
            "cited_authorities": [],
            "interpretive_cases": [],
            "related_judges": []
        }

        try:
            # Query for documents cited BY the core set
            cited_query = """
            UNWIND $doc_ids AS core_id
            MATCH (core {doc_id: core_id})-[:REFERS_TO]->(cited)
            RETURN cited.doc_id AS doc_id, 
                   labels(cited)[0] AS doc_type,
                   cited.title AS title
            LIMIT 20
            """
            
            # Query for documents that CITE the core set
            interpretive_query = """
            UNWIND $doc_ids AS core_id
            MATCH (interpretive)-[:REFERS_TO]->(core {doc_id: core_id})
            RETURN interpretive.doc_id AS doc_id,
                   labels(interpretive)[0] AS doc_type,
                   interpretive.title AS title
            LIMIT 20
            """
            
            # Query for related judges
            judges_query = """
            UNWIND $doc_ids AS core_id
            MATCH (core:Case {doc_id: core_id})<-[*]-(judge:Person)
            RETURN judge.name AS name
            LIMIT 10
            """
            
            cited_results = self._execute_query(cited_query, {"doc_ids": doc_ids})
            interpretive_results = self._execute_query(interpretive_query, {"doc_ids": doc_ids})
            judge_results = self._execute_query(judges_query, {"doc_ids": doc_ids})

            expanded_context["cited_authorities"] = cited_results
            expanded_context["interpretive_cases"] = interpretive_results
            expanded_context["related_judges"] = judge_results

        except Exception as e:
            logger.error(f"Error expanding context in KG: {e}")
            
        return expanded_context
    
    def extract_reasoning_chains(self, doc_ids: List[str]) -> List[ReasoningChain]:
        """Extract logical reasoning chains from graph paths"""
        chains = []
        
        try:
            for i, doc_id in enumerate(doc_ids[:3]):  # Limit to top 3 for reasoning
                chain_steps = []
                
                # Find cited acts (major premises)
                act_query = """
                MATCH (case:Case {doc_id: $doc_id})-[:REFERS_TO]->(act:Act)
                RETURN act.doc_id as act_id, act.title as act_title
                LIMIT 3
                """
                
                cited_acts = self._execute_query(act_query, {"doc_id": doc_id})
                
                for act in cited_acts:
                    chain_steps.append(ReasoningStep(
                        step_type="major_premise",
                        content=f"Legal provision from {act['act_title']}",
                        doc_id=act['act_id'],
                        confidence=0.9
                    ))
                
                # Application (current document)
                chain_steps.append(ReasoningStep(
                    step_type="application",
                    content=f"Application in case {doc_id}",
                    doc_id=doc_id,
                    confidence=0.8
                ))
                
                if len(chain_steps) >= 2:
                    chains.append(ReasoningChain(
                        chain_id=f"chain_{i}",
                        steps=chain_steps,
                        chain_strength=sum(step.confidence for step in chain_steps) / len(chain_steps)
                    ))
        
        except Exception as e:
            logger.error(f"Error extracting reasoning chains: {e}")
        
        return chains

class FirestoreClient:
    def __init__(self, project_id: Optional[str] = None, credential_path: Optional[str] = None):
        try:
            if not firebase_admin._apps:
                if credential_path:
                    cred = credentials.Certificate(credential_path)
                    firebase_admin.initialize_app(cred, {'projectId': project_id})
                else:
                    firebase_admin.initialize_app()
            
            self.db = firestore.client()
            logger.info("Firestore client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Firestore client: {e}")
            # Create a mock client for testing
            self.db = None

    async def get_presummary(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get pre-computed summary for a document from Firestore."""
        if not self.db:
            # Return mock data for testing
            return {
                "summary": f"Mock summary for document {doc_id}",
                "key_points": [f"Key point 1 for {doc_id}", f"Key point 2 for {doc_id}"]
            }
        
        try:
            doc_ref = self.db.collection('summaries').document(doc_id)
            doc_snapshot = doc_ref.get()
            
            if doc_snapshot.exists:
                return doc_snapshot.to_dict()
            else:
                logger.warning(f"No presummary found for doc_id: {doc_id}")
                return None
        except Exception as e:
            logger.error(f"Error fetching presummary for {doc_id}: {e}")
            return None

    async def get_claim_premise_data(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get claim-premise data for a document from Firestore."""
        if not self.db:
            # Return mock data for testing
            return {
                "claims": [f"Mock claim for {doc_id}"],
                "premises": [f"Mock premise for {doc_id}"]
            }
        
        try:
            doc_ref = self.db.collection('claim_premise_data').document(doc_id)
            doc_snapshot = doc_ref.get()
            
            if doc_snapshot.exists:
                return doc_snapshot.to_dict()
            else:
                logger.warning(f"No claim-premise data found for doc_id: {doc_id}")
                return None
        except Exception as e:
            logger.error(f"Error fetching claim-premise data for {doc_id}: {e}")
            return None

class AdaptiveRetriever:
    def __init__(self, pc_client, neo4j_driver, firestore_client, dense_model, sparse_tokenizer, sparse_model):
        self.dense_case_db = DenseVectorDB(DocumentType.CASE, pc_client, dense_model)
        self.sparse_case_db = SparseVectorDB(DocumentType.CASE, pc_client, sparse_tokenizer, sparse_model)
        self.dense_act_db = DenseVectorDB(DocumentType.ACT, pc_client, dense_model)
        self.sparse_act_db = SparseVectorDB(DocumentType.ACT, pc_client, sparse_tokenizer, sparse_model)
        
        self.kg = KnowledgeGraph(driver=neo4j_driver)
        self.firestore = firestore_client
        
    def _determine_retrieval_weights(self, query_context: QueryContext) -> Dict[str, float]:
        """Determine retrieval weights based on query context"""
        weights = {
            "dense_case": 0.25,
            "sparse_case": 0.25,
            "dense_act": 0.25,
            "sparse_act": 0.25
        }
        
        # Adjust based on intent
        if query_context.intent == QueryIntent.DOCTRINAL:
            weights["dense_act"] = 0.4
            weights["sparse_act"] = 0.3
            weights["dense_case"] = 0.2
            weights["sparse_case"] = 0.1
        elif query_context.intent == QueryIntent.PRECEDENTIAL:
            weights["dense_case"] = 0.4
            weights["sparse_case"] = 0.3
            weights["dense_act"] = 0.2
            weights["sparse_act"] = 0.1
        
        # Adjust based on complexity
        if query_context.complexity_score > 0.7:
            weights["dense_case"] *= 1.2
            weights["dense_act"] *= 1.2
            weights["sparse_case"] *= 0.8
            weights["sparse_act"] *= 0.8
        
        # Normalize weights
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}
    
    async def retrieve_candidates(self, query_context: QueryContext, top_k: int = 100) -> List[EnhancedChunk]:
        """Adaptive multi-vector retrieval"""
        weights = self._determine_retrieval_weights(query_context)
        
        # Determine how many results to get from each DB
        k_per_db = {
            "dense_case": max(1, int(top_k * weights["dense_case"])),
            "sparse_case": max(1, int(top_k * weights["sparse_case"])),
            "dense_act": max(1, int(top_k * weights["dense_act"])),
            "sparse_act": max(1, int(top_k * weights["sparse_act"]))
        }
        
        # Retrieve from all databases
        tasks = [
            self.dense_case_db.search(query_context.raw_query, k_per_db["dense_case"]),
            self.sparse_case_db.search(query_context.raw_query, k_per_db["sparse_case"]),
            self.dense_act_db.search(query_context.raw_query, k_per_db["dense_act"]),
            self.sparse_act_db.search(query_context.raw_query, k_per_db["sparse_act"])
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Combine and enhance with KG features, removing duplicates
        all_chunks = []
        seen_doc_ids = set()
        
        for chunk_list in results:
            for chunk in chunk_list:
                if chunk.doc_id not in seen_doc_ids:
                    seen_doc_ids.add(chunk.doc_id)
                    enhanced_chunk = EnhancedChunk(
                        doc_id=chunk.doc_id,
                        chunk_id=chunk.chunk_id,
                        content=chunk.content,
                        metadata=chunk.metadata,
                        vector_score=chunk.vector_score,
                        chunk_index=chunk.chunk_index,
                        kg_features=self.kg.compute_kg_features(chunk.doc_id) if chunk.doc_id else KGFeatures()
                    )
                    all_chunks.append(enhanced_chunk)
        
        return all_chunks

class EnhancedReranker:
    def __init__(self):
        self.feature_weights = {
            "vector_score": 0.4,
            "authority_score": 0.3,
            "metadata_alignment": 0.2,
            "kg_features": 0.1
        }
    
    def rerank(self, chunks: List[EnhancedChunk], query_context: QueryContext, top_k: int = 20) -> List[EnhancedChunk]:
        """Enhanced reranking with multiple signals"""
        
        for chunk in chunks:
            # Compute metadata alignment score
            metadata_score = self._compute_metadata_alignment(chunk, query_context)
            
            # Compute final score
            chunk.final_score = (
                self.feature_weights["vector_score"] * chunk.vector_score +
                self.feature_weights["authority_score"] * chunk.kg_features.authority_score +
                self.feature_weights["metadata_alignment"] * metadata_score +
                self.feature_weights["kg_features"] * (chunk.kg_features.citation_count / 50.0)
            )
        
        # Sort by final score and return top-k
        chunks.sort(key=lambda x: x.final_score, reverse=True)
        return chunks[:top_k]
    
    def _compute_metadata_alignment(self, chunk: EnhancedChunk, query_context: QueryContext) -> float:
        """Compute alignment between chunk metadata and query context"""
        score = 0.0
        
        # Domain alignment
        if query_context.legal_domains:
            chunk_domains = set(chunk.metadata.legal_domains)
            query_domains = set(query_context.legal_domains)
            if chunk_domains & query_domains:
                score += 0.5
        
        # Jurisdiction alignment
        if query_context.jurisdictions and chunk.metadata.jurisdiction:
            if any(jurisdiction in chunk.metadata.jurisdiction.lower() 
                   for jurisdiction in query_context.jurisdictions):
                score += 0.3
        
        # Document type preference based on intent
        if query_context.intent == QueryIntent.DOCTRINAL and chunk.metadata.doc_type == DocumentType.ACT:
            score += 0.2
        elif query_context.intent == QueryIntent.PRECEDENTIAL and chunk.metadata.doc_type == DocumentType.CASE:
            score += 0.2
        
        return min(score, 1.0)

class ContextAssembler:
    def __init__(self, kg: KnowledgeGraph, firestore: FirestoreClient):
        self.kg = kg
        self.firestore = firestore
        
    async def assemble_context(self, chunks: List[EnhancedChunk], query_context: QueryContext) -> ContextBundle:
        """Assemble comprehensive context bundle"""
        
        # Extract document IDs for KG expansion
        doc_ids = [chunk.doc_id for chunk in chunks]
        
        # Determine expansion depth based on query complexity
        expansion_depth = 1 if query_context.complexity_score < 0.5 else 2
        
        # Expand context via KG
        expanded_context = self.kg.expand_context(doc_ids, expansion_depth)
        
        # Extract reasoning chains
        reasoning_chains = self.kg.extract_reasoning_chains(doc_ids)
        
        # Fetch presummaries and claim-premise data
        core_documents = []
        for chunk in chunks:
            presummary = await self.firestore.get_presummary(chunk.doc_id)
            claim_premise = await self.kg.get_claim_premise_data(chunk.doc_id)
            
            doc_data = {
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "relevance_score": chunk.final_score,
                "authority_score": chunk.kg_features.authority_score,
                "citation_count": chunk.kg_features.citation_count,
                "presummary": presummary.get("summary", "") if presummary else "",
                "key_claims": claim_premise.get("claims", []) if claim_premise else [],
                "legal_premises": claim_premise.get("premises", []) if claim_premise else [],
                "provenance": "direct_match",
                "metadata": {
                    "doc_type": chunk.metadata.doc_type.value,
                    "title": chunk.metadata.title,
                    "court_level": chunk.metadata.court_level,
                    "date": chunk.metadata.date.isoformat() if chunk.metadata.date else None
                }
            }
            core_documents.append(doc_data)
        
        # Build provenance graph
        provenance_graph = {}
        for chunk in chunks:
            provenance_graph[chunk.doc_id] = [
                neighbor["doc_id"] for neighbor in 
                expanded_context.get("cited_authorities", [])[:3]
            ]
        
        # Estimate token count
        total_tokens = self._estimate_tokens(core_documents, expanded_context, reasoning_chains)
        
        return ContextBundle(
            core_documents=core_documents,
            supporting_context=expanded_context,
            reasoning_chains=reasoning_chains,
            provenance_graph=provenance_graph,
            total_tokens=total_tokens
        )
    
    def _estimate_tokens(self, core_docs: List[Dict], supporting: Dict, chains: List[ReasoningChain]) -> int:
        """Rough token estimation"""
        core_tokens = sum(len(doc.get("presummary", "").split()) * 1.3 for doc in core_docs)
        supporting_tokens = sum(
            len(str(docs)) * 0.3  # Rough estimate for supporting context
            for docs in supporting.values()
        )
        chain_tokens = sum(
            sum(len(step.content.split()) * 1.3 for step in chain.steps)
            for chain in chains
        )
        
        return int(core_tokens + supporting_tokens + chain_tokens)

class ResultEvaluator:
    """Evaluate retrieval and RAG system performance"""
    
    def __init__(self):
        self.ground_truth = {}  # Store ground truth relevance scores
    
    def add_ground_truth(self, query: str, relevant_doc_ids: List[str], relevance_scores: Dict[str, float] = None):
        """Add ground truth data for evaluation"""
        self.ground_truth[query] = {
            "relevant_docs": set(relevant_doc_ids),
            "scores": relevance_scores or {doc_id: 1.0 for doc_id in relevant_doc_ids}
        }
    
    def evaluate_retrieval(self, query: str, retrieved_chunks: List[EnhancedChunk], 
                          start_time: float = None, end_time: float = None) -> EvaluationMetrics:
        """Evaluate retrieval performance"""
        
        if query not in self.ground_truth:
            logger.warning(f"No ground truth available for query: {query}")
            return self._create_default_metrics(retrieved_chunks, start_time, end_time)
        
        gt = self.ground_truth[query]
        retrieved_doc_ids = [chunk.doc_id for chunk in retrieved_chunks]
        
        # Calculate relevance scores
        relevance_scores = []
        for doc_id in retrieved_doc_ids:
            if doc_id in gt["relevant_docs"]:
                relevance_scores.append(gt["scores"].get(doc_id, 1.0))
            else:
                relevance_scores.append(0.0)
        
        # Calculate precision and recall at different k values
        precision_at_k = {}
        recall_at_k = {}
        
        for k in [1, 3, 5, 10]:
            if k <= len(retrieved_doc_ids):
                relevant_at_k = sum(1 for i in range(k) if relevance_scores[i] > 0)
                precision_at_k[k] = relevant_at_k / k
                recall_at_k[k] = relevant_at_k / len(gt["relevant_docs"])
        
        # Authority coverage (how many high-authority docs were retrieved)
        high_authority_count = sum(1 for chunk in retrieved_chunks 
                                 if chunk.kg_features.authority_score > 0.7)
        authority_coverage = high_authority_count / len(retrieved_chunks) if retrieved_chunks else 0
        
        # Response time
        response_time = (end_time - start_time) if start_time and end_time else 0
        
        return EvaluationMetrics(
            relevance_scores=relevance_scores,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            authority_coverage=authority_coverage,
            reasoning_completeness=0.8,  # Placeholder
            response_time=response_time
        )
    
    def _create_default_metrics(self, retrieved_chunks: List[EnhancedChunk], 
                               start_time: float = None, end_time: float = None) -> EvaluationMetrics:
        """Create default metrics when no ground truth is available"""
        return EvaluationMetrics(
            relevance_scores=[0.5] * len(retrieved_chunks),  # Neutral scores
            precision_at_k={k: 0.5 for k in [1, 3, 5, 10]},
            recall_at_k={k: 0.5 for k in [1, 3, 5, 10]},
            authority_coverage=sum(1 for chunk in retrieved_chunks 
                                 if chunk.kg_features.authority_score > 0.5) / len(retrieved_chunks) if retrieved_chunks else 0,
            reasoning_completeness=0.5,
            response_time=(end_time - start_time) if start_time and end_time else 0
        )
    
    def print_evaluation_report(self, query: str, metrics: EvaluationMetrics):
        """Print a comprehensive evaluation report"""
        print(f"\n{'='*80}")
        print(f"EVALUATION REPORT FOR QUERY: {query}")
        print('='*80)
        
        print(f"Response Time: {metrics.response_time:.2f}s")
        print(f"Authority Coverage: {metrics.authority_coverage:.2%}")
        print(f"Reasoning Completeness: {metrics.reasoning_completeness:.2%}")
        
        print(f"\nPrecision@K:")
        for k, precision in metrics.precision_at_k.items():
            print(f"  P@{k}: {precision:.3f}")
        
        print(f"\nRecall@K:")
        for k, recall in metrics.recall_at_k.items():
            print(f"  R@{k}: {recall:.3f}")
        
        print(f"\nRelevance Scores Distribution:")
        relevant_count = sum(1 for score in metrics.relevance_scores if score > 0)
        print(f"  Relevant documents: {relevant_count}/{len(metrics.relevance_scores)}")
        print(f"  Average relevance: {np.mean(metrics.relevance_scores):.3f}")

class EnhancedLegalRAGSystem:
    def __init__(self, 
                 dense_model=None,
                 sparse_model=None,
                 sparse_tokenizer=None,
                 pinecone_client=None,
                 neo4j_driver=None,
                 firestore_project_id=None,
                 firestore_credentials=None):
        """Initialize the enhanced RAG system with models and database connections"""
        self.dense_model = dense_model
        self.sparse_model = sparse_model
        self.sparse_tokenizer = sparse_tokenizer
        self.pinecone_client = pinecone_client
        self.neo4j_driver = neo4j_driver
        
        # Initialize components
        self.query_analyzer = QueryAnalyzer()
        self.reranker = EnhancedReranker()
        self.evaluator = ResultEvaluator()
        
        # Initialize clients
        self.firestore_client = FirestoreClient(
            project_id=firestore_project_id,
            credential_path=firestore_credentials
        )
        
        self.kg = KnowledgeGraph(driver=self.neo4j_driver)
        
        self.context_assembler = ContextAssembler(
            kg=self.kg,
            firestore=self.firestore_client
        )
        
        self.retriever = AdaptiveRetriever(
            pc_client=self.pinecone_client,
            neo4j_driver=self.neo4j_driver,
            firestore_client=self.firestore_client,
            dense_model=self.dense_model,
            sparse_tokenizer=self.sparse_tokenizer,
            sparse_model=self.sparse_model
        )
    
    async def process_query(self, query: str, max_tokens: int = 99999, evaluate: bool = True) -> Tuple[ContextBundle, EvaluationMetrics]:
        """Main processing pipeline with evaluation"""
        start_time = asyncio.get_event_loop().time()
        
        logger.info(f"Processing query: {query}")
        
        # Step 1: Analyze query
        query_context = self.query_analyzer.analyze_query(query)
        logger.info(f"Query intent: {query_context.intent}, complexity: {query_context.complexity_score:.2f}")
                
        # Step 2: Adaptive retrieval
        candidates = await self.retriever.retrieve_candidates(query_context, top_k=100)
        logger.info(f"Retrieved {len(candidates)} candidates")
        
        # Step 3: Enhanced reranking
        top_chunks = self.reranker.rerank(candidates, query_context, top_k=20)
        logger.info(f"Reranked to top {len(top_chunks)} chunks")
        
        # Step 4: Context assembly
        context_bundle = await self.context_assembler.assemble_context(top_chunks, query_context)
        logger.info(f"Assembled context with {context_bundle.total_tokens} estimated tokens")
        
        # Step 5: Token budget management
        if context_bundle.total_tokens > max_tokens:
            context_bundle = self._trim_context(context_bundle, max_tokens)
        
        end_time = asyncio.get_event_loop().time()
        
        # Step 6: Evaluation
        evaluation_metrics = None
        if evaluate:
            evaluation_metrics = self.evaluator.evaluate_retrieval(
                query, top_chunks, start_time, end_time
            )
        
        return context_bundle, evaluation_metrics
    
    def _trim_context(self, context_bundle: ContextBundle, max_tokens: int) -> ContextBundle:
        """Intelligently trim context to fit token budget"""
        logger.info(f"Trimming context from {context_bundle.total_tokens} to {max_tokens} tokens")
        
        # Priority order: core documents > reasoning chains > supporting context
        target_core_ratio = 0.6
        target_chain_ratio = 0.25
        target_support_ratio = 0.15
        
        core_budget = int(max_tokens * target_core_ratio)
        
        # Trim core documents (keep highest scoring)
        trimmed_core = []
        for doc in sorted(context_bundle.core_documents, key=lambda x: x["relevance_score"], reverse=True):
            if len(trimmed_core) < 10:  # Keep top 10 documents
                trimmed_core.append(doc)
        
        # Trim reasoning chains (keep strongest)
        trimmed_chains = sorted(context_bundle.reasoning_chains, 
                               key=lambda x: x.chain_strength, reverse=True)[:3]
        
        # Trim supporting context
        trimmed_supporting = {}
        for context_type, docs in context_bundle.supporting_context.items():
            if isinstance(docs, list):
                trimmed_supporting[context_type] = docs[:3]  # Keep top 3 per category
            else:
                trimmed_supporting[context_type] = docs
        
        return ContextBundle(
            core_documents=trimmed_core,
            supporting_context=trimmed_supporting,
            reasoning_chains=trimmed_chains,
            provenance_graph=context_bundle.provenance_graph,
            total_tokens=self.context_assembler._estimate_tokens(
                trimmed_core, trimmed_supporting, trimmed_chains
            )
        )
    
    def add_evaluation_ground_truth(self, query: str, relevant_doc_ids: List[str], 
                                   relevance_scores: Dict[str, float] = None):
        """Add ground truth data for evaluation"""
        self.evaluator.add_ground_truth(query, relevant_doc_ids, relevance_scores)

# Usage Example and Testing
async def main():
    """Main function demonstrating the enhanced system with evaluation"""
    
    try:
        # Initialize all clients
        pc_client = Pinecone(api_key=PINECONE_API_KEY)
        neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))    
        dense_model_instance = SentenceTransformer(PINECONE_DENSE_EMBED_MODEL)
        sparse_tokenizer_instance = AutoTokenizer.from_pretrained(PINECONE_SPARSE_EMBED_MODEL)
        sparse_model_instance = AutoModelForMaskedLM.from_pretrained(PINECONE_SPARSE_EMBED_MODEL)

        # Initialize the system
        rag_system = EnhancedLegalRAGSystem(
            dense_model=dense_model_instance,
            sparse_model=sparse_model_instance,
            sparse_tokenizer=sparse_tokenizer_instance,
            pinecone_client=pc_client,
            neo4j_driver=neo4j_driver,
            firestore_project_id=FIRESTORE_PROJECT_ID,
            firestore_credentials=FIRESTORE_CREDENTIALS_PATH
        )
        
        # Add some ground truth data for evaluation
        rag_system.add_evaluation_ground_truth(
        "Who is the accused in the contempt of court case regarding bribery allegations against judges?",
        relevant_doc_ids=["doc_123", "doc_456", "doc_789"],
        relevance_scores={"doc_123": 1.0, "doc_456": 0.8, "doc_789": 0.6}
        )
        
        # Example legal queries
        test_queries = [
            "Who is the accused in the contempt of court case regarding bribery allegations against judges?",
            "What are the legal provisions regarding contempt of court in Sri Lankan law?",
            "What precedents exist for cases involving allegations against judicial officers?",
        ]
        
        # Process each query
        for i, query in enumerate(test_queries):
            print(f"\n{'='*100}")
            print(f"Processing Query {i+1}: {query}")
            print('='*100)
            
            try:
                # Process the query
                context_bundle, evaluation_metrics = await rag_system.process_query(
                    query, max_tokens=10000, evaluate=True
                )
                
                # Display results
                print(f"\nQuery Analysis:")
                query_context = rag_system.query_analyzer.analyze_query(query)
                print(f"  Intent: {query_context.intent.value}")
                print(f"  Complexity: {query_context.complexity_score:.2f}")
                print(f"  Domains: {query_context.legal_domains}")
                print(f"  Entities: {query_context.extracted_entities}")
                
                print(f"\nContext Bundle Summary:")
                print(f"  Core Documents: {len(context_bundle.core_documents)}")
                print(f"  Supporting Context Types: {len(context_bundle.supporting_context)}")
                print(f"  Reasoning Chains: {len(context_bundle.reasoning_chains)}")
                print(f"  Total Tokens: {context_bundle.total_tokens}")
                
                print(f"\nTop 3 Core Documents:")
                for j, doc in enumerate(context_bundle.core_documents[:3]):
                    print(f"  {j+1}. {doc['metadata']['title']}")
                    print(f"      Relevance: {doc['relevance_score']:.3f}")
                    print(f"      Authority: {doc['authority_score']:.3f}")
                    print(f"      Citations: {doc['citation_count']}")
                
                print(f"\nSupporting Context:")
                for context_type, items in context_bundle.supporting_context.items():
                    if isinstance(items, list) and items:
                        print(f"  {context_type}: {len(items)} items")
                        for item in items[:2]:  # Show first 2
                            if isinstance(item, dict) and 'doc_id' in item:
                                print(f"    - {item.get('title', item['doc_id'])}")
                
                print(f"\nReasoning Chains:")
                for chain in context_bundle.reasoning_chains:
                    print(f"  Chain {chain.chain_id} (Strength: {chain.chain_strength:.3f}):")
                    for step in chain.steps:
                        print(f"    - {step.step_type}: {step.content[:100]}...")
                
                # Print evaluation report
                if evaluation_metrics:
                    rag_system.evaluator.print_evaluation_report(query, evaluation_metrics)
                
            except Exception as e:
                print(f"Error processing query: {e}")
                logger.error(f"Error processing query '{query}': {e}", exc_info=True)

    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        
    finally:
        # Clean up connections
        if 'neo4j_driver' in locals() and neo4j_driver:
            neo4j_driver.close()
            logger.info("Neo4j driver closed.")

if __name__ == "__main__":
    nest_asyncio.apply()
    await main()