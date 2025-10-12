"""
Main metrics collector class - now simplified by delegating to specialized classes.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime

from data_models.models import LLMMetrics, LLMEvaluation
from evaluation.metrics_storage import MetricsStorage
from evaluation.metrics_analyzer import MetricsAnalyzer
from evaluation.report_generator import ReportGenerator
from config.settings import METRICS_DIR


class MetricsCollector:
    """Collects and manages LLM performance metrics with multi-judge evaluation"""
    
    def __init__(self, metrics_dir: str = None):
        self.metrics_dir = metrics_dir or METRICS_DIR
        self.metrics_storage = MetricsStorage(self.metrics_dir)
        self.current_session_metrics: List[LLMMetrics] = []
    
    def record_metrics(self, query: str, response: str, context: str, 
                     response_time: float, token_count: int, 
                     model: str, session_id: str,
                     evaluations: List[LLMEvaluation]) -> LLMMetrics:
        """Record metrics with multi-judge evaluation"""
        tokens_per_second = token_count / response_time if response_time > 0 else 0
        
        metrics = LLMMetrics(
            timestamp = datetime.now().isoformat(),
            query = query,
            response = response,
            context = context[:1000],  # Limit context length for storage
            response_time = response_time,
            token_count = token_count,
            tokens_per_second = tokens_per_second,
            model = model,
            session_id = session_id,
            evaluations = evaluations
        )
        
        self.current_session_metrics.append(metrics)
        
        # Auto-save each evaluation to JSON file
        self.metrics_storage.save_single_evaluation(metrics)
        
        return metrics
    
    def save_all_metrics_to_file(self, filename: str = None) -> str:
        """Save all metrics to a single JSON file"""
        return self.metrics_storage.save_all_metrics(self.current_session_metrics, filename)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary statistics"""
        return MetricsAnalyzer.calculate_session_summary(self.current_session_metrics)
    
    def generate_report(self) -> str:
        """Generate a comprehensive evaluation report"""
        summary = self.get_session_summary()
        model_name = self.current_session_metrics[0].model if self.current_session_metrics else "N/A"
        return ReportGenerator.generate_report(summary, model_name)
    
    def list_evaluation_files(self) -> List[str]:
        """List all evaluation files"""
        return self.metrics_storage.list_evaluation_files()
    
    def load_evaluation_file(self, filename: str) -> Dict[str, Any]:
        """Load evaluation data from file"""
        return self.metrics_storage.load_evaluation_file(filename)