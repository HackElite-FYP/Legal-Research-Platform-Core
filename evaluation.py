"""
Evaluation Framework for Vector Database and Embedding Module
Implements standard IR metrics and legal expert validation
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import logging
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.stats import spearmanr
import json
import config

logger = logging.getLogger(__name__)

class RetrievalEvaluator:
    """Evaluate retrieval performance using standard IR metrics"""
    
    def __init__(self):
        self.metrics = {}
        logger.info("RetrievalEvaluator initialized")
    
    def precision_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """Calculate Precision@K"""
        if k <= 0 or not retrieved_docs:
            return 0.0
        
        retrieved_k = retrieved_docs[:k]
        relevant_retrieved = len(set(retrieved_k) & set(relevant_docs))
        return relevant_retrieved / min(k, len(retrieved_k))
    
    def recall_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """Calculate Recall@K"""
        if not relevant_docs or k <= 0:
            return 0.0
        
        retrieved_k = retrieved_docs[:k]
        relevant_retrieved = len(set(retrieved_k) & set(relevant_docs))
        return relevant_retrieved / len(relevant_docs)
    
    def f1_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """Calculate F1@K"""
        precision = self.precision_at_k(retrieved_docs, relevant_docs, k)
        recall = self.recall_at_k(retrieved_docs, relevant_docs, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def mean_reciprocal_rank(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """Calculate Mean Reciprocal Rank (MRR)"""
        if not relevant_docs or not retrieved_docs:
            return 0.0
        
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def ndcg_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], 
                  relevance_scores: Optional[Dict[str, float]] = None, k: int = 10) -> float:
        """Calculate Normalized Discounted Cumulative Gain (NDCG@K)"""
        if k <= 0 or not retrieved_docs:
            return 0.0
        
        # Default relevance scores (binary)
        if relevance_scores is None:
            relevance_scores = {doc: 1.0 for doc in relevant_docs}
        
        # Calculate DCG
        dcg = 0.0
        for i, doc in enumerate(retrieved_docs[:k]):
            if doc in relevance_scores:
                relevance = relevance_scores[doc]
                dcg += relevance / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG (Ideal DCG)
        ideal_relevances = sorted(relevance_scores.values(), reverse=True)
        idcg = 0.0
        for i, relevance in enumerate(ideal_relevances[:k]):
            idcg += relevance / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate_query(self, retrieved_docs: List[str], relevant_docs: List[str],
                      relevance_scores: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Evaluate a single query with all metrics"""
        results = {}
        
        # Calculate metrics for different k values
        for k in config.TOP_K_VALUES:
            results[f'precision@{k}'] = self.precision_at_k(retrieved_docs, relevant_docs, k)
            results[f'recall@{k}'] = self.recall_at_k(retrieved_docs, relevant_docs, k)
            results[f'f1@{k}'] = self.f1_at_k(retrieved_docs, relevant_docs, k)
            results[f'ndcg@{k}'] = self.ndcg_at_k(retrieved_docs, relevant_docs, relevance_scores, k)
        
        # Calculate MRR
        results['mrr'] = self.mean_reciprocal_rank(retrieved_docs, relevant_docs)
        
        return results
    
    def evaluate_multiple_queries(self, query_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate multiple queries and return average metrics"""
        if not query_results:
            return {}
        
        all_metrics = []
        
        for query_result in query_results:
            retrieved = query_result.get('retrieved_docs', [])
            relevant = query_result.get('relevant_docs', [])
            relevance_scores = query_result.get('relevance_scores')
            
            metrics = self.evaluate_query(retrieved, relevant, relevance_scores)
            all_metrics.append(metrics)
        
        # Calculate averages
        avg_metrics = {}
        if all_metrics:
            for metric_name in all_metrics[0].keys():
                avg_metrics[metric_name] = np.mean([m[metric_name] for m in all_metrics])
        
        return avg_metrics

class LegalExpertEvaluator:
    """Handle legal expert evaluation and validation"""
    
    def __init__(self):
        self.expert_ratings = {}
        logger.info("LegalExpertEvaluator initialized")
    
    def create_evaluation_dataset(self, search_results: List[Dict[str, Any]], 
                                 queries: List[str]) -> pd.DataFrame:
        """Create dataset for expert evaluation"""
        evaluation_data = []
        
        for i, (query, results) in enumerate(zip(queries, search_results)):
            for j, result in enumerate(results[:5]):  # Top 5 results for evaluation
                evaluation_data.append({
                    'query_id': i,
                    'query_text': query,
                    'document_id': result['id'],
                    'document_preview': result['metadata'].get('text_preview', ''),
                    'document_type': result['metadata'].get('document_type', ''),
                    'category': result['metadata'].get('primary_category', ''),
                    'similarity_score': result['score'],
                    'rank': j + 1,
                    'expert_rating': None,  # To be filled by experts
                    'expert_comments': None
                })
        
        return pd.DataFrame(evaluation_data)
    
    def calculate_expert_agreement(self, expert_ratings: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate inter-expert agreement using various metrics"""
        if len(expert_ratings) < 2:
            return {'agreement': 0.0}
        
        expert_names = list(expert_ratings.keys())
        agreements = {}
        
        # Pairwise correlations
        correlations = []
        for i in range(len(expert_names)):
            for j in range(i + 1, len(expert_names)):
                expert1_ratings = expert_ratings[expert_names[i]]
                expert2_ratings = expert_ratings[expert_names[j]]
                
                if len(expert1_ratings) == len(expert2_ratings):
                    corr, _ = spearmanr(expert1_ratings, expert2_ratings)
                    if not np.isnan(corr):
                        correlations.append(corr)
        
        agreements['spearman_correlation'] = np.mean(correlations) if correlations else 0.0
        
        # Calculate Fleiss' Kappa (simplified version)
        # This is a basic implementation - for production, use specialized libraries
        all_ratings = list(expert_ratings.values())
        if all_ratings and len(set(len(ratings) for ratings in all_ratings)) == 1:
            # All experts rated the same number of items
            agreements['fleiss_kappa'] = self._calculate_fleiss_kappa(all_ratings)
        
        return agreements
    
    def _calculate_fleiss_kappa(self, ratings_matrix: List[List[float]]) -> float:
        """Simplified Fleiss' Kappa calculation"""
        # This is a basic implementation
        # For production use, consider using specialized libraries like sklearn or statsmodels
        
        n_raters = len(ratings_matrix)
        n_items = len(ratings_matrix[0]) if ratings_matrix else 0
        
        if n_raters < 2 or n_items == 0:
            return 0.0
        
        # Convert to agreement matrix (simplified for continuous ratings)
        # In practice, you'd need to discretize ratings or use appropriate measures
        agreements = []
        for item_idx in range(n_items):
            item_ratings = [ratings[item_idx] for ratings in ratings_matrix]
            # Simple agreement: variance of ratings (lower is better)
            agreement = 1.0 / (1.0 + np.var(item_ratings))
            agreements.append(agreement)
        
        return np.mean(agreements)
    
    def analyze_expert_feedback(self, evaluation_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze expert evaluation results"""
        if evaluation_df.empty or 'expert_rating' not in evaluation_df.columns:
            return {}
        
        # Filter out non-rated items
        rated_df = evaluation_df.dropna(subset=['expert_rating'])
        
        if rated_df.empty:
            return {'error': 'No expert ratings found'}
        
        analysis = {
            'total_evaluations': len(rated_df),
            'average_rating': rated_df['expert_rating'].mean(),
            'rating_distribution': rated_df['expert_rating'].value_counts().to_dict(),
            'ratings_by_category': rated_df.groupby('category')['expert_rating'].mean().to_dict(),
            'ratings_by_document_type': rated_df.groupby('document_type')['expert_rating'].mean().to_dict(),
            'correlation_with_similarity': rated_df[['expert_rating', 'similarity_score']].corr().iloc[0, 1]
        }
        
        return analysis

class ComprehensiveEvaluator:
    """Comprehensive evaluation combining automatic metrics and expert validation"""
    
    def __init__(self):
        self.retrieval_evaluator = RetrievalEvaluator()
        self.expert_evaluator = LegalExpertEvaluator()
        logger.info("ComprehensiveEvaluator initialized")
    
    def evaluate_system(self, vector_db_manager, embedding_generator, 
                       test_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Comprehensive system evaluation
        
        Args:
            vector_db_manager: VectorDatabaseManager instance
            embedding_generator: EmbeddingGenerator instance
            test_queries: List of test queries with expected results
        """
        logger.info(f"Evaluating system with {len(test_queries)} test queries...")
        
        evaluation_results = {
            'automatic_metrics': {},
            'expert_evaluation': {},
            'system_performance': {}
        }
        
        # Automatic evaluation
        query_results = []
        all_search_results = []
        
        for query_data in test_queries:
            query_text = query_data['query']
            expected_docs = query_data.get('relevant_docs', [])
            
            # Perform search
            search_results = vector_db_manager.search_legal_documents(
                query_text, 
                embedding_generator,
                top_k=10
            )
            
            retrieved_docs = [result['id'] for result in search_results]
            
            query_results.append({
                'query': query_text,
                'retrieved_docs': retrieved_docs,
                'relevant_docs': expected_docs,
                'search_results': search_results
            })
            
            all_search_results.append(search_results)
        
        # Calculate automatic metrics
        evaluation_results['automatic_metrics'] = self.retrieval_evaluator.evaluate_multiple_queries(query_results)
        
        # Prepare expert evaluation dataset
        queries = [q['query'] for q in test_queries]
        expert_dataset = self.expert_evaluator.create_evaluation_dataset(all_search_results, queries)
        
        evaluation_results['expert_evaluation'] = {
            'dataset_size': len(expert_dataset),
            'evaluation_ready': True,
            'dataset_preview': expert_dataset.head().to_dict('records') if not expert_dataset.empty else []
        }
        
        # System performance summary
        auto_metrics = evaluation_results['automatic_metrics']
        evaluation_results['system_performance'] = {
            'overall_precision@5': auto_metrics.get('precision@5', 0),
            'overall_recall@5': auto_metrics.get('recall@5', 0),
            'overall_f1@5': auto_metrics.get('f1@5', 0),
            'mean_reciprocal_rank': auto_metrics.get('mrr', 0),
            'ndcg@10': auto_metrics.get('ndcg@10', 0)
        }
        
        return evaluation_results
    
    def compare_models(self, models_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare performance of different embedding models"""
        comparison = {
            'model_rankings': {},
            'best_model': None,
            'performance_summary': {}
        }
        
        # Calculate overall scores for each model
        model_scores = {}
        for model_name, results in models_results.items():
            auto_metrics = results.get('automatic_metrics', {})
            
            # Weighted score combining different metrics
            score = (
                auto_metrics.get('precision@5', 0) * 0.3 +
                auto_metrics.get('recall@5', 0) * 0.3 +
                auto_metrics.get('mrr', 0) * 0.2 +
                auto_metrics.get('ndcg@10', 0) * 0.2
            )
            
            model_scores[model_name] = score
        
        # Rank models
        ranked_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        comparison['model_rankings'] = {model: rank + 1 for rank, (model, _) in enumerate(ranked_models)}
        comparison['best_model'] = ranked_models[0][0] if ranked_models else None
        
        # Performance summary
        for model_name, results in models_results.items():
            comparison['performance_summary'][model_name] = results.get('system_performance', {})
        
        return comparison
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any], 
                                  output_file: str = 'evaluation_report.json'):
        """Generate comprehensive evaluation report"""
        report = {
            'evaluation_timestamp': pd.Timestamp.now().isoformat(),
            'system_configuration': {
                'embedding_model': config.EMBEDDING_MODEL_NAME,
                'vector_dimension': config.VECTOR_DIMENSION,
                'chunk_size': config.CHUNK_SIZE,
                'top_k_values': config.TOP_K_VALUES
            },
            'results': evaluation_results,
            'recommendations': self._generate_recommendations(evaluation_results)
        }
        
        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation report saved to {output_file}")
        return report
    
    def _generate_recommendations(self, evaluation_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on evaluation results"""
        recommendations = []
        
        auto_metrics = evaluation_results.get('automatic_metrics', {})
        
        # Check precision
        precision_5 = auto_metrics.get('precision@5', 0)
        if precision_5 < 0.5:
            recommendations.append("Consider improving embedding model or adding more training data")
        
        # Check recall
        recall_5 = auto_metrics.get('recall@5', 0)
        if recall_5 < 0.3:
            recommendations.append("Consider expanding the document corpus or improving chunking strategy")
        
        # Check MRR
        mrr = auto_metrics.get('mrr', 0)
        if mrr < 0.4:
            recommendations.append("Consider implementing hybrid search or query expansion")
        
        if not recommendations:
            recommendations.append("System performance is satisfactory")
        
        return recommendations

if __name__ == "__main__":
    # Test evaluation framework
    evaluator = ComprehensiveEvaluator()
    
    # Sample test data
    test_queries = [
        {
            'query': 'property ownership rights',
            'relevant_docs': ['act_123_0001', 'case_456_0002']
        },
        {
            'query': 'marriage and divorce laws',
            'relevant_docs': ['act_789_0001', 'act_789_0002']
        }
    ]
    
    # Mock evaluation (in practice, use real vector_db_manager and embedding_generator)
    print("Evaluation framework test completed")
    print("Ready for integration with vector database and embedding modules")
