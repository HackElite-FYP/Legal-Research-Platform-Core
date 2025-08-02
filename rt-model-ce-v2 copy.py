class DatabaseConnector:
    
    def get_kg_features(self, doc_id: str) -> Dict[str, Any]:
        """Get knowledge graph features for a document"""
        if not self.neo4j_driver:
            return self._mock_kg_features(doc_id)
        
        try:
            with self.neo4j_driver.session(database=NEO4J_DATABASE) as session:
                # Query for Cases
                case_query = """
                MATCH (d:Case {doc_id: $doc_id})
                OPTIONAL MATCH (d)-[:REFERS_TO]->(act:Act)
                OPTIONAL MATCH (d)<-[r]-(judge:Person)
                OPTIONAL MATCH (citing:Case)-[:REFERS_TO]->(d)
                RETURN 
                    count(DISTINCT act) as act_references,
                    count(DISTINCT judge) as judge_count,
                    count(DISTINCT citing) as citation_count,
                    d.doc_year as doc_date
                """
                
                # Query for Acts
                act_query = """
                MATCH (d:Act {doc_id: $doc_id})
                OPTIONAL MATCH (case:Case)-[:REFERS_TO]->(d)
                RETURN 
                    count(DISTINCT case) as citation_count,
                    0 as act_references,
                    0 as judge_count,
                    d.doc_year as doc_date
                """
                
                result = session.run(case_query, {"doc_id": doc_id}).single()
                if not result or result['act_references'] is None:
                    result = session.run(act_query, {"doc_id": doc_id}).single()
                
                if result:
                    # Compute authority score
                    citation_count = result.get('citation_count', 0) or 0
                    act_references = result.get('act_references', 0) or 0
                    judge_count = result.get('judge_count', 0) or 0
                    court_level = result.get('court_level', 3) or 3
                    
                    # Jurisdictional weight (higher court = higher weight)
                    jurisdictional_weight = 1.0 / court_level if court_level else 0.5
                    
                    # Authority score calculation
                    authority_score = (
                        0.4 * min(citation_count / 50.0, 1.0) +
                        0.2 * min(act_references / 10.0, 1.0) +
                        0.2 * jurisdictional_weight +
                        0.1 * min(judge_count / 5.0, 1.0) +
                        0.1 * 0.5  # baseline
                    )
                    
                    return {
                        'citation_count': citation_count,
                        'act_references': act_references,
                        'judge_count': judge_count,
                        'authority_score': authority_score,
                        'jurisdictional_weight': jurisdictional_weight,
                        'court_level': court_level
                    }
        
        except Exception as e:
            logger.error(f"Error fetching KG features for {doc_id}: {e}")
        
        return self._mock_kg_features(doc_id)

    
    def get_document_summary(self, doc_id: str) -> Optional[str]:
        """Get document summary from Firestore"""
        if not self.firestore_client:
            return f"Mock summary for {doc_id}"
        
        try:
            doc_ref = self.firestore_client.collection('summaries').document(doc_id)
            doc_snapshot = doc_ref.get()
            
            if doc_snapshot.exists:
                return doc_snapshot.to_dict().get('summary', '')
        except Exception as e:
            logger.error(f"Error fetching summary for {doc_id}: {e}")
        
        return f"Mock summary for {doc_id}"
    
    def close_connections(self):
        """Close all database connections"""
        if self.neo4j_driver:
            self.neo4j_driver.close()
        logger.info("Database connections closed")

class EnhancedLegalRerankingDataset(Dataset):
    """Enhanced dataset with database features"""
    
    def __init__(self, 
                 batches: List[RerankingBatch], 
                 tokenizer, 
                 db_connector: DatabaseConnector,
                 max_length: int = 512,
                 include_metadata: bool = True,
                 include_kg_features: bool = True):
        self.batches = batches
        self.tokenizer = tokenizer
        self.db_connector = db_connector
        self.max_length = max_length
        self.include_metadata = include_metadata
        self.include_kg_features = include_kg_features
        
        # Flatten batches into individual examples
        self.examples = []
        for batch in batches:
            for doc in batch.documents:
                self.examples.append((batch.query, doc))
        logger.info(f"Dataset created with {len(self.examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        query, document = self.examples[idx]
        
        # Tokenize the pair
        try:
            encoding = self.tokenizer(
                query,
                document.document_text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
        except Exception as e:
            logger.warning(f"Tokenization error for doc {document.doc_id}: {e}")
            encoding = self.tokenizer(
                "legal query",
                "legal document",
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
        
        # Enhanced metadata features (expanded to 16 features)
        metadata_features = torch.zeros(16)
        if self.include_metadata and document.metadata_features:
            # Original features (0-11)
            court_mapping = {
                'Supreme Court': 1.0, 'Court of Appeal': 0.8, 'High Court': 0.6,
                'District Court': 0.4, 'Magistrate\'s Court': 0.2, 'Primary Court': 0.1
            }
            jurisdiction = document.metadata_features.get('jurisdiction', 'District Court')
            metadata_features[0] = court_mapping.get(jurisdiction, 0.4)
            
            citation_count = document.metadata_features.get('citation_count', 0)
            metadata_features[1] = min(np.log1p(citation_count) / np.log1p(100), 1.0)
            
            metadata_features[2] = document.metadata_features.get('authority_score', 0.5)
            metadata_features[3] = min(document.metadata_features.get('recency_boost', 0.5), 1.0)
            
            doc_type = document.metadata_features.get('doc_type', 'case')
            type_scores = {'act': 1.0, 'regulation': 0.8, 'case': 0.6}
            metadata_features[4] = type_scores.get(doc_type, 0.3)
            
            domains = document.metadata_features.get('legal_domains', [])
            metadata_features[5] = min(len(domains) / 5.0, 1.0)
            metadata_features[6] = 1.0 if document.metadata_features.get('jurisdiction_match', False) else 0.0
            metadata_features[7] = min(document.metadata_features.get('act_references', 0) / 10.0, 1.0)
            metadata_features[8] = min(document.metadata_features.get('judge_count', 1) / 5.0, 1.0)
            metadata_features[9] = document.metadata_features.get('vector_score', 0.5)
            
            domains_str = ' '.join(domains).lower() if domains else ''
            metadata_features[10] = 1.0 if 'professional conduct' in domains_str else 0.0
            
            try:
                date_str = document.metadata_features.get('date', '2020-01-01')
                year = int(date_str.split('-')[0])
                metadata_features[11] = max(0.0, min(1.0, (year - 2020) / 4.0))
            except:
                metadata_features[11] = 0.5
            
            # Enhanced KG features (12-15)
            if self.include_kg_features:
                kg_features = self.db_connector.get_kg_features(document.doc_id)
                metadata_features[12] = min(kg_features.get('citation_count', 0) / 100.0, 1.0)
                metadata_features[13] = kg_features.get('authority_score', 0.5)
                metadata_features[14] = kg_features.get('jurisdictional_weight', 0.5)
                metadata_features[15] = min(kg_features.get('act_references', 0) / 20.0, 1.0)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'token_type_ids': encoding.get('token_type_ids', torch.zeros_like(encoding['input_ids'])).squeeze(),
            'metadata_features': metadata_features,
            'relevance_score': torch.tensor(float(document.relevance_score), dtype=torch.float),
            'doc_id': document.doc_id,
            'query': query,
            'document_text': document.document_text,
            'authority_score': torch.tensor(document.authority_score, dtype=torch.float),
            'citation_count': torch.tensor(document.citation_count, dtype=torch.float)
        }

class AdvancedNeuralLegalReranker(nn.Module):
    """Advanced cross-encoder with enhanced architecture"""
    
    def __init__(self, 
                 model_name: str = "nlpaueb/legal-bert-base-uncased",
                 metadata_dim: int = 16,
                 hidden_dim: int = 256,
                 dropout_rate: float = 0.15,
                 num_attention_heads: int = 8,
                 combine_strategy: str = "multi_head_attention"):
        super().__init__()
        
        self.model_name = model_name
        self.combine_strategy = combine_strategy
        
        # Initialize transformer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.transformer = AutoModel.from_pretrained(model_name)
            logger.info(f"Loaded model: {model_name}")
        except Exception as e:
            logger.warning(f"Could not load {model_name}, using BERT base: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.transformer = AutoModel.from_pretrained("bert-base-uncased")
            self.model_name = "bert-base-uncased"
        
        # Enhanced semantic processing with residual connections
        self.semantic_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.transformer.config.hidden_size if i == 0 else hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            ) for i in range(3)
        ])
        
        # Enhanced metadata processing
        self.metadata_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(metadata_dim if i == 0 else hidden_dim // 2, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            ) for i in range(2)
        ])
        
        # Fusion mechanisms
        if combine_strategy == "multi_head_attention":
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_attention_heads,
                dropout=dropout_rate,
                batch_first=True
            )
            self.fusion_norm = nn.LayerNorm(hidden_dim)
            
        elif combine_strategy == "gated_fusion":
            self.gate_network = nn.Sequential(
                nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim // 4),
                nn.GELU(),
                nn.Linear(hidden_dim // 4, 2),
                nn.Softmax(dim=-1)
            )
            
        # Final prediction layers
        final_input_dim = hidden_dim if combine_strategy == "multi_head_attention" else hidden_dim + hidden_dim // 2
        
        self.prediction_head = nn.Sequential(
            nn.Linear(final_input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate // 2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling"""
        for module in [self.semantic_layers, self.metadata_layers, self.prediction_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(self, input_ids, attention_mask, token_type_ids, metadata_features):
        batch_size = input_ids.size(0)
        
        # Get transformer outputs with gradient checkpointing for memory efficiency
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        # Enhanced semantic representation with residual connections
        semantic_repr = transformer_outputs.last_hidden_state[:, 0, :]  # CLS token
        
        for i, layer in enumerate(self.semantic_layers):
            residual = semantic_repr if i > 0 else None
            semantic_repr = layer(semantic_repr)
            if residual is not None and residual.shape == semantic_repr.shape:
                semantic_repr = semantic_repr + residual
        
        # Enhanced metadata representation with residual connections
        metadata_repr = metadata_features
        for i, layer in enumerate(self.metadata_layers):
            residual = metadata_repr if i > 0 else None
            metadata_repr = layer(metadata_repr)
            if residual is not None and residual.shape == metadata_repr.shape:
                metadata_repr = metadata_repr + residual
        
        # Fusion strategies
        if self.combine_strategy == "multi_head_attention":
            # Cross-attention between semantic and metadata
            semantic_expanded = semantic_repr.unsqueeze(1)  # [batch, 1, hidden]
            metadata_expanded = metadata_repr.unsqueeze(1)  # [batch, 1, hidden//2]
            
            # Pad metadata to match semantic dimension
            pad_size = semantic_expanded.size(-1) - metadata_expanded.size(-1)
            if pad_size > 0:
                metadata_padded = torch.cat([
                    metadata_expanded, 
                    torch.zeros(batch_size, 1, pad_size, device=metadata_expanded.device)
                ], dim=-1)
            else:
                metadata_padded = metadata_expanded
            
            attended_repr, _ = self.cross_attention(
                semantic_expanded, metadata_padded, metadata_padded
            )
            attended_repr = self.fusion_norm(attended_repr.squeeze(1))
            final_repr = attended_repr
            
        elif self.combine_strategy == "gated_fusion":
            # Gated fusion
            combined = torch.cat([semantic_repr, metadata_repr], dim=-1)
            gates = self.gate_network(combined)
            
            # Apply gates
            gated_semantic = gates[:, 0:1] * semantic_repr
            gated_metadata = gates[:, 1:2] * metadata_repr
            final_repr = torch.cat([gated_semantic, gated_metadata], dim=-1)
            
        else:
            # Simple concatenation
            final_repr = torch.cat([semantic_repr, metadata_repr], dim=-1)
        
        # Final prediction
        output = self.prediction_head(final_repr)
        return torch.sigmoid(output.squeeze(-1))
    
    def predict_batch(self, queries: List[str], documents: List[str], 
                     metadata_list: List[Dict] = None) -> np.ndarray:
        """Efficient batch prediction"""
        self.eval()
        
        if metadata_list is None:
            metadata_list = [{}] * len(documents)
        
        # Tokenize batch
        inputs = self.tokenizer(
            queries,
            documents,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Process metadata
        metadata_tensor = torch.zeros(len(queries), 16)
        # Add metadata processing logic here if needed
        
        with torch.no_grad():
            device = next(self.parameters()).device
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            token_type_ids = inputs.get('token_type_ids', torch.zeros_like(input_ids)).to(device)
            metadata_tensor = metadata_tensor.to(device)
            
            scores = self.forward(input_ids, attention_mask, token_type_ids, metadata_tensor)
            return scores.cpu().numpy()

class ComprehensiveEvaluator:
    """Comprehensive evaluation with multiple metrics"""
    
    def __init__(self, db_connector: DatabaseConnector):
        self.db_connector = db_connector
        self.ground_truth = {}
    
    def add_ground_truth(self, query: str, relevant_doc_ids: List[str], 
                        relevance_scores: Dict[str, float] = None):
        """Add ground truth data for evaluation"""
        self.ground_truth[query] = {
            "relevant_docs": set(relevant_doc_ids),
            "scores": relevance_scores or {doc_id: 1.0 for doc_id in relevant_doc_ids}
        }
    
    def evaluate_comprehensive(self, model: AdvancedNeuralLegalReranker,
                             val_loader: DataLoader,
                             detailed: bool = False) -> CrossEncoderEvaluationMetrics:
        """Comprehensive evaluation with database-enhanced metrics"""
        model.eval()
        
        start_time = datetime.now()
        
        all_predictions = []
        all_relevance = []
        all_queries = []
        all_doc_ids = []
        all_authority_scores = []
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    device = next(model.parameters()).device
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    token_type_ids = batch['token_type_ids'].to(device)
                    metadata_features = batch['metadata_features'].to(device)
                    
                    predictions = model(input_ids, attention_mask, token_type_ids, metadata_features)
                    
                    all_predictions.extend(predictions.cpu().numpy().flatten())
                    all_relevance.extend(batch['relevance_score'].numpy().flatten())
                    all_queries.extend(batch['query'])
                    all_doc_ids.extend(batch['doc_id'])
                    all_authority_scores.extend(batch['authority_score'].numpy().flatten())
                    
                except Exception as e:
                    logger.error(f"Error in evaluation batch: {e}")
                    continue
        
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()
        
        # Calculate comprehensive metrics
        metrics = self._calculate_all_metrics(
            all_predictions, all_relevance, all_queries, all_doc_ids, all_authority_scores
        )
        metrics.response_time = response_time
        
        if detailed:
            self._create_detailed_report(
                all_predictions, all_relevance, all_queries, all_doc_ids, all_authority_scores
            )
        
        return metrics
    
    def _calculate_all_metrics(self, predictions, relevance_scores, queries, doc_ids, authority_scores):
        """Calculate all evaluation metrics"""
        # Group by query
        query_groups = defaultdict(list)
        for pred, rel, query, doc_id, auth in zip(predictions, relevance_scores, queries, doc_ids, authority_scores):
            query_groups[query].append({
                'prediction': pred, 'relevance': rel, 'doc_id': doc_id, 'authority': auth
            })
        
        # Initialize metrics
        ndcg_3_scores = []
        ndcg_5_scores = []
        map_scores = []
        mrr_scores = []
        spearman_scores = []
        kendall_scores = []
        precision_at_k = {k: [] for k in [1, 3, 5, 10]}
        recall_at_k = {k: [] for k in [1, 3, 5, 10]}
        authority_coverages = []
        
        for query, docs in query_groups.items():
            if len(docs) < 2:
                continue
            
            # Sort by predicted relevance
            docs_sorted = sorted(docs, key=lambda x: x['prediction'], reverse=True)
            
            true_relevance = [doc['relevance'] for doc in docs_sorted]
            predicted_scores = [doc['prediction'] for doc in docs_sorted]
            authority_scores_sorted = [doc['authority'] for doc in docs_sorted]
            
            # NDCG scores
            if len(true_relevance) >= 3:
                ndcg_3 = ndcg_score([true_relevance], [predicted_scores], k=3)
                ndcg_3_scores.append(ndcg_3)
            
            if len(true_relevance) >= 5:
                ndcg_5 = ndcg_score([true_relevance], [predicted_scores], k=5)
                ndcg_5_scores.append(ndcg_5)
            
            # MAP
            ap_score = self._average_precision(true_relevance)
            map_scores.append(ap_score)
            
            # MRR
            mrr = self._reciprocal_rank(true_relevance)
            mrr_scores.append(mrr)
            
            # Precision and Recall at K
            for k in [1, 3, 5, 10]:
                if len(true_relevance) >= k:
                    prec_k = self._precision_at_k(true_relevance, k)
                    rec_k = self._recall_at_k(true_relevance, k)
                    precision_at_k[k].append(prec_k)
                    recall_at_k[k].append(rec_k)
            
            # Correlations
            original_relevance = [doc['relevance'] for doc in docs]
            original_predictions = [doc['prediction'] for doc in docs]
            
            if len(set(original_relevance)) > 1:
                spearman_corr, _ = stats.spearmanr(original_predictions, original_relevance)
                kendall_corr, _ = stats.kendalltau(original_predictions, original_relevance)
                
                if not np.isnan(spearman_corr):
                    spearman_scores.append(spearman_corr)
                if not np.isnan(kendall_corr):
                    kendall_scores.append(kendall_corr)
            
            # Authority coverage (top-k high authority documents retrieved)
            top_k_authority = np.mean(authority_scores_sorted[:5]) if len(authority_scores_sorted) >= 5 else 0
            authority_coverages.append(top_k_authority)
        
        # Aggregate metrics
        return CrossEncoderEvaluationMetrics(
            ndcg_at_3=np.mean(ndcg_3_scores) if ndcg_3_scores else 0.0,
            ndcg_at_5=np.mean(ndcg_5_scores) if ndcg_5_scores else 0.0,
            map_score=np.mean(map_scores) if map_scores else 0.0,
            mrr_score=np.mean(mrr_scores) if mrr_scores else 0.0,
            precision_at_k={k: np.mean(scores) if scores else 0.0 for k, scores in precision_at_k.items()},
            recall_at_k={k: np.mean(scores) if scores else 0.0 for k, scores in recall_at_k.items()},
            spearman_correlation=np.mean(spearman_scores) if spearman_scores else 0.0,
            kendall_tau=np.mean(kendall_scores) if kendall_scores else 0.0,
            authority_coverage=np.mean(authority_coverages) if authority_coverages else 0.0
        )
    
    def _average_precision(self, relevance_scores, threshold=0.5):
        """Calculate Average Precision"""
        relevant_count = 0
        precision_sum = 0.0
        
        for i, score in enumerate(relevance_scores):
            if score > threshold:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        total_relevant = sum(1 for score in relevance_scores if score > threshold)
        return precision_sum / total_relevant if total_relevant > 0 else 0.0
    
    def _reciprocal_rank(self, relevance_scores, threshold=0.5):
        """Calculate Reciprocal Rank"""
        for i, score in enumerate(relevance_scores):
            if score > threshold:
                return 1.0 / (i + 1)
        return 0.0
    
    def _precision_at_k(self, relevance_scores, k=5, threshold=0.5):
        """Calculate Precision@K"""
        top_k = relevance_scores[:min(k, len(relevance_scores))]
        relevant_count = sum(1 for score in top_k if score > threshold)
        return relevant_count / len(top_k) if top_k else 0.0
    
    def _recall_at_k(self, relevance_scores, k=10, threshold=0.5):
        """Calculate Recall@K"""
        top_k = relevance_scores[:min(k, len(relevance_scores))]
        relevant_in_top_k = sum(1 for score in top_k if score > threshold)
        total_relevant = sum(1 for score in relevance_scores if score > threshold)
        return relevant_in_top_k / total_relevant if total_relevant > 0 else 0.0
    
    def _create_detailed_report(self, predictions, relevance_scores, queries, doc_ids, authority_scores):
        """Create detailed evaluation report with visualizations"""
        print("\n" + "="*100)
        print("COMPREHENSIVE CROSS-ENCODER EVALUATION REPORT")
        print("="*100)
        
        df = pd.DataFrame({
            'prediction': predictions,
            'relevance': relevance_scores,
            'query': queries,
            'doc_id': doc_ids,
            'authority': authority_scores
        })
        
        # Overall statistics

        
        # Query-level analysis
        
        
        # Performance by authority level

        
        # Create visualizations
        self._create_evaluation_plots(df, query_stats_df)
    
    def _create_evaluation_plots(self, df, query_stats_df):
        """Create comprehensive evaluation visualizations"""
        try:
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(3, 3, figsize=(20, 15))
            
            # 1. Prediction vs Relevance scatter
            axes[0, 0].scatter(df['relevance'], df['prediction'], alpha=0.6, s=30)
            axes[0, 0].plot([0, 1], [0, 1], 'r--', label='Perfect correlation')
            axes[0, 0].set_xlabel('True Relevance')
            axes[0, 0].set_ylabel('Predicted Score')
            axes[0, 0].set_title('Predictions vs True Relevance')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Distribution comparison
            axes[0, 1].hist(df['prediction'], bins=30, alpha=0.7, label='Predictions', density=True)
            axes[0, 1].hist(df['relevance'], bins=30, alpha=0.7, label='True Relevance', density=True)
            axes[0, 1].set_xlabel('Score')
            axes[0, 1].set_ylabel('Density')
            axes[0, 1].set_title('Score Distributions')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Query-level correlation distribution
            axes[0, 2].hist(query_stats_df['correlation'], bins=20, alpha=0.7, edgecolor='black')
            axes[0, 2].set_xlabel('Correlation')
            axes[0, 2].set_ylabel('Number of Queries')
            axes[0, 2].set_title('Query-level Correlation Distribution')
            axes[0, 2].axvline(query_stats_df['correlation'].mean(), color='red', linestyle='--', 
                              label=f'Mean: {query_stats_df["correlation"].mean():.3f}')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
            
            # 4. Error analysis
            errors = df['prediction'] - df['relevance']
            axes[1, 0].hist(errors, bins=30, alpha=0.7, edgecolor='black')
            axes[1, 0].set_xlabel('Error (Pred - True)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Prediction Error Distribution')
            axes[1, 0].axvline(0, color='red', linestyle='--', label='Perfect prediction')
            axes[1, 0].axvline(errors.mean(), color='orange', linestyle='--', 
                              label=f'Mean error: {errors.mean():.3f}')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 5. Authority vs Performance
            axes[1, 1].scatter(df['authority'], df['prediction'], alpha=0.6, s=30, label='Prediction')
            axes[1, 1].scatter(df['authority'], df['relevance'], alpha=0.6, s=30, label='True Relevance')
            axes[1, 1].set_xlabel('Authority Score')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_title('Authority vs Relevance/Prediction')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            # 6. Query complexity analysis
            axes[1, 2].scatter(query_stats_df['num_docs'], query_stats_df['correlation'])
            axes[1, 2].set_xlabel('Number of Documents')
            axes[1, 2].set_ylabel('Correlation')
            axes[1, 2].set_title('Performance vs Query Complexity')
            axes[1, 2].grid(True, alpha=0.3)
            
            # 7. Precision-Recall analysis by relevance bins
            df['relevance_bin'] = pd.cut(df['relevance'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            bin_stats = df.groupby('relevance_bin')['prediction'].mean()
            axes[2, 0].bar(range(len(bin_stats)), bin_stats.values)
            axes[2, 0].set_xticks(range(len(bin_stats)))
            axes[2, 0].set_xticklabels(bin_stats.index, rotation=45)
            axes[2, 0].set_ylabel('Average Prediction')
            axes[2, 0].set_title('Prediction by Relevance Bins')
            axes[2, 0].grid(True, alpha=0.3)
            
            # 8. Calibration plot
            bin_boundaries = np.linspace(0, 1, 11)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            accuracies = []
            confidences = []
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (df['prediction'] > bin_lower) & (df['prediction'] <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = (df[in_bin]['relevance'] > 0.5).mean()
                    avg_confidence_in_bin = df[in_bin]['prediction'].mean()
                    accuracies.append(accuracy_in_bin)
                    confidences.append(avg_confidence_in_bin)
            
            axes[2, 1].plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
            if confidences and accuracies:
                axes[2, 1].plot(confidences, accuracies, 'o-', label='Model')
            axes[2, 1].set_xlabel('Mean Predicted Probability')
            axes[2, 1].set_ylabel('Fraction of Positives')
            axes[2, 1].set_title('Calibration Plot')
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)
            
            # 9. Performance heatmap by query characteristics
            query_heatmap_data = query_stats_df.pivot_table(
                values='correlation', 
                index=pd.cut(query_stats_df['num_docs'], bins=5),
                columns=pd.cut(query_stats_df['avg_authority'], bins=5),
                aggfunc='mean'
            )
            
            if not query_heatmap_data.empty:
                im = axes[2, 2].imshow(query_heatmap_data.values, cmap='RdYlBu', aspect='auto')
                axes[2, 2].set_title('Performance Heatmap\n(Docs vs Authority)')
                axes[2, 2].set_xlabel('Average Authority')
                axes[2, 2].set_ylabel('Number of Documents')
                plt.colorbar(im, ax=axes[2, 2])
            
            plt.tight_layout()
            plt.savefig('cross_encoder_evaluation_report.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            logger.error(f"Could not create evaluation plots: {e}")

class AdvancedTrainer:
    """Advanced trainer with database integration and comprehensive evaluation"""
    
    def __init__(self, 
                 model: AdvancedNeuralLegalReranker,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 db_connector: DatabaseConnector,
                 learning_rate: float = 2e-5,
                 weight_decay: float = 1e-5,
                 device: str = None,
                 use_advanced_loss: bool = True):
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.db_connector = db_connector
        self.use_advanced_loss = use_advanced_loss
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Advanced optimizer with different learning rates
        transformer_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'transformer' in name:
                transformer_params.append(param)
            else:
                other_params.append(param)
        
        self.optimizer = optim.AdamW([
            {'params': transformer_params, 'lr': learning_rate * 0.1},
            {'params': other_params, 'lr': learning_rate},
        ], weight_decay=weight_decay)
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.huber_loss = nn.HuberLoss(delta=0.1)
        
        # Advanced scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-7
        )
        
        # Evaluation
        self.evaluator = ComprehensiveEvaluator(db_connector)
        
        # Training history
        self.training_history = {
            'train_loss': [], 'val_loss': [], 'ndcg_at_3': [], 'ndcg_at_5': [],
            'map_score': [], 'mrr_score': [], 'spearman_correlation': [], 
            'authority_coverage': [], 'precision_at_5': [], 'recall_at_10': []
        }
    
    def listwise_loss(self, predictions, relevance_scores, temperature=2.0):
        """Enhanced ListNet loss for listwise learning"""
        if len(predictions) < 2:
            return torch.tensor(0.0, device=predictions.device)
        
        # Apply temperature scaling
        scaled_preds = predictions / temperature
        scaled_rels = relevance_scores / temperature
        
        # Compute softmax probabilities
        pred_probs = torch.softmax(scaled_preds, dim=0)
        true_probs = torch.softmax(scaled_rels, dim=0)
        
        # KL divergence loss with numerical stability
        kl_loss = torch.sum(true_probs * torch.log(true_probs / (pred_probs + 1e-8) + 1e-8))
        
        return kl_loss
    
    def pairwise_ranking_loss(self, predictions, relevance_scores, margin=0.1):
        """Enhanced pairwise ranking loss with authority weighting"""
        batch_size = predictions.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=predictions.device)
        
        loss = 0.0
        num_pairs = 0
        
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                rel_diff = relevance_scores[i] - relevance_scores[j]
                
                # Only consider pairs with significant relevance difference
                if abs(rel_diff) > 0.1:  # Increased threshold
                    pred_diff = predictions[i] - predictions[j]
                    
                    # Authority-weighted margin
                    dynamic_margin = margin * (1 + abs(rel_diff))
                    
                    if rel_diff > 0:  # i should be ranked higher than j
                        loss += torch.clamp(dynamic_margin - pred_diff, min=0.0)
                    else:  # j should be ranked higher than i
                        loss += torch.clamp(dynamic_margin + pred_diff, min=0.0)
                    
                    num_pairs += 1
        
        return loss / max(num_pairs, 1)
    
    def authority_aware_loss(self, predictions, relevance_scores, authority_scores):
        """Authority-aware loss function"""
        # Weight losses by authority scores
        weights = 1.0 + authority_scores  # Higher authority = higher weight
        
        # Weighted MSE loss
        mse_losses = (predictions - relevance_scores) ** 2
        weighted_mse = torch.mean(weights * mse_losses)
        
        return weighted_mse
    
    def train_epoch(self):
        """Enhanced training epoch with multiple loss components"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            try:
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                metadata_features = batch['metadata_features'].to(self.device)
                relevance_scores = batch['relevance_score'].to(self.device)
                authority_scores = batch['authority_score'].to(self.device)
                
                # Forward pass
                predictions = self.model(input_ids, attention_mask, token_type_ids, metadata_features)
                
                # Handle dimension issues
                if predictions.dim() == 0:
                    predictions = predictions.unsqueeze(0)
                if relevance_scores.dim() == 0:
                    relevance_scores = relevance_scores.unsqueeze(0)
                if authority_scores.dim() == 0:
                    authority_scores = authority_scores.unsqueeze(0)
                
                # Combined loss
                if self.use_advanced_loss:
                    # Primary loss (Huber for robustness)
                    main_loss = self.huber_loss(predictions, relevance_scores)
                    
                    # Authority-aware loss
                    auth_loss = self.authority_aware_loss(predictions, relevance_scores, authority_scores)
                    
                    # Listwise loss
                    listwise_loss = self.listwise_loss(predictions, relevance_scores)
                    
                    # Pairwise ranking loss
                    ranking_loss = self.pairwise_ranking_loss(predictions, relevance_scores)
                    
                    # Combined loss with weights
                    loss = (0.4 * main_loss + 
                           0.3 * auth_loss + 
                           0.2 * listwise_loss + 
                           0.1 * ranking_loss)
                else:
                    loss = self.mse_loss(predictions, relevance_scores)
                
                # Backward pass with gradient clipping
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    logger.info(f"Batch {batch_idx}: Loss = {loss.item():.4f}")
                    
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {e}")
                continue
        
        self.scheduler.step()
        return total_loss / max(num_batches, 1)
    
    def evaluate(self, detailed: bool = False) -> Tuple[float, CrossEncoderEvaluationMetrics]:
        """Comprehensive evaluation"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # Validation loss
        with torch.no_grad():
            for batch in self.val_loader:
                try:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    token_type_ids = batch['token_type_ids'].to(self.device)
                    metadata_features = batch['metadata_features'].to(self.device)
                    relevance_scores = batch['relevance_score'].to(self.device)
                    
                    predictions = self.model(input_ids, attention_mask, token_type_ids, metadata_features)
                    
                    if predictions.dim() == 0:
                        predictions = predictions.unsqueeze(0)
                    if relevance_scores.dim() == 0:
                        relevance_scores = relevance_scores.unsqueeze(0)
                    
                    loss = self.huber_loss(predictions, relevance_scores)
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    logger.error(f"Error in validation: {e}")
                    continue
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Comprehensive metrics
        comprehensive_metrics = self.evaluator.evaluate_comprehensive(
            self.model, self.val_loader, detailed=detailed
        )
        
        return avg_loss, comprehensive_metrics
    
    def train(self, num_epochs: int, save_path: str = None, early_stopping_patience: int = 10, save_best_model: bool = False):
        """Train with comprehensive monitoring"""
        best_metric = 0.0
        patience_counter = 0
        
        logger.info(f"Starting advanced training for {num_epochs} epochs on {self.device}")
        logger.info(f"Model: {self.model.model_name}")
        logger.info(f"Combine strategy: {self.model.combine_strategy}")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(num_epochs):
            try:
                # Training
                train_loss = self.train_epoch()
                
                # Evaluation
                val_loss, val_metrics = self.evaluate(detailed=(epoch == num_epochs - 1))
                
                # Early stopping based on NDCG@5
                current_metric = val_metrics.ndcg_at_5
                if current_metric > best_metric:
                    best_metric = current_metric
                    patience_counter = 0
                    
                    if save_best_model and save_path:
                        self.save_model(save_path)
                        logger.info(f"Saved best model (NDCG@5: {best_metric:.4f})")
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1} (best NDCG@5: {best_metric:.4f})")
                    break
                
            except Exception as e:
                logger.error(f"Error in epoch {epoch+1}: {e}")
                break
        
        return self.training_history
    
    def save_model(self, path: str):
        """Save model with comprehensive metadata"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'model_name': self.model.model_name,
                'combine_strategy': self.model.combine_strategy,
                'metadata_dim': 16,
                'hidden_dim': 256
            },
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'scheduler_state_dict': self.scheduler.state_dict()
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model with all components"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', self.training_history)
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f"Model loaded from {path}")

class DatabaseEnhancedDataGenerator:
    """Enhanced data generator with database integration"""
    
    def __init__(self, db_connector: DatabaseConnector):
        self.db_connector = db_connector
        
        self.legal_domains = [
            'constitutional', 'criminal', 'civil', 'commercial', 'administrative',
            'professional conduct', 'family law', 'property law', 'tort law'
        ]
        
        self.jurisdictions = [
            'Supreme Court', 'Court of Appeal', 'High Court', 
            'District Court', 'Magistrate\'s Court', 'Primary Court'
        ]
    
    def load_or_generate_data(self, json_file_path: str = "training_data.json") -> List[RerankingBatch]:
        """Load existing data or generate enhanced synthetic data"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            logger.info(f"Loaded {len(json_data)} queries from {json_file_path}")
            
            # Convert to enhanced batches with database features
            batches = []
            for item in json_data:
                if 'documents' not in item or len(item['documents']) < 2:
                    continue
                
                query = item['query']
                query_id = item.get('query_id', f"query_{len(batches)}")
                
                documents = []
                for doc_data in item['documents']:
                    # Get enhanced features from database
                    doc_id = doc_data.get('doc_id', f"doc_{len(documents)}")
                    kg_features = self.db_connector.get_kg_features(doc_id)
                    
                    # Enhance metadata with KG features
                    metadata_features = doc_data.get('metadata_features', {})
                    metadata_features.update({
                        'kg_citation_count': kg_features.get('citation_count', 0),
                        'kg_authority_score': kg_features.get('authority_score', 0.5),
                        'kg_court_level': kg_features.get('court_level', 3)
                    })
                    
                    example = RerankingExample(
                        query=query,
                        document_text=doc_data.get('document_text', 'Legal document text'),
                        doc_id=doc_id,
                        relevance_score=float(doc_data.get('relevance_score', 0.5)),
                        metadata_features=metadata_features,
                        kg_features=kg_features,
                        authority_score=kg_features.get('authority_score', 0.5),
                        citation_count=kg_features.get('citation_count', 0)
                    )
                    documents.append(example)
                
                if documents:
                    batch = RerankingBatch(query=query, documents=documents, query_id=query_id)
                    batches.append(batch)
            
            if len(batches) >= 10:
                logger.info(f"Successfully loaded {len(batches)} query batches with enhanced features")
                return batches
            else:
                logger.warning(f"Insufficient queries ({len(batches)}). Generating enhanced synthetic data...")
                return self.generate_enhanced_synthetic_data()
            
        except FileNotFoundError:
            logger.info(f"Training data file not found: {json_file_path}")
            return self.generate_enhanced_synthetic_data()
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return self.generate_enhanced_synthetic_data()

def create_enhanced_data_loaders(batches: List[RerankingBatch], 
                               tokenizer, 
                               db_connector: DatabaseConnector,
                               test_size: float = 0.2, 
                               batch_size: int = 8) -> Tuple[DataLoader, DataLoader]:
    """Create enhanced data loaders with database features"""
    
    # Filter valid batches
    valid_batches = [batch for batch in batches if len(batch.documents) >= 3]
    
    if len(valid_batches) < batch_size:
        raise ValueError(f"Insufficient valid batches: {len(valid_batches)}. Need at least {batch_size}.")
    
    # Split at query level
    train_batches, val_batches = train_test_split(
        valid_batches, test_size=test_size, random_state=42, shuffle=True
    )
    
    # Ensure minimum validation size
    if len(val_batches) < 2:
        val_batches = train_batches[:2]
        train_batches = train_batches[2:]
    
    logger.info(f"Enhanced split: {len(train_batches)} train queries, {len(val_batches)} val queries")
    
    # Create enhanced datasets
    train_dataset = EnhancedLegalRerankingDataset(
        train_batches, tokenizer, db_connector, max_length=512, 
        include_metadata=True, include_kg_features=True
    )
    val_dataset = EnhancedLegalRerankingDataset(
        val_batches, tokenizer, db_connector, max_length=512,
        include_metadata=True, include_kg_features=True
    )
    
    logger.info(f"Enhanced train dataset: {len(train_dataset)} examples")
    logger.info(f"Enhanced val dataset: {len(val_dataset)} examples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        drop_last=False,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        drop_last=False,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader