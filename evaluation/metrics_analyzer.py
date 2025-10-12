"""
Metrics analysis and statistical calculations.
"""

from typing import Dict, Any, List
from data_models.models import LLMMetrics, LLMEvaluation


class MetricsAnalyzer:
    """Analyzes metrics data and calculates statistics"""
    
    @staticmethod
    def calculate_session_summary(metrics: List[LLMMetrics]) -> Dict[str, Any]:
        """Calculate comprehensive summary statistics from metrics"""
        if not metrics:
            return {}
        
        response_times = [m.response_time for m in metrics]
        token_counts = [m.token_count for m in metrics]
        tokens_per_second = [m.tokens_per_second for m in metrics]
        
        all_evaluations = []
        for metric in metrics:
            all_evaluations.extend(metric.evaluations)
        
        summary = {
            "total_interactions": len(metrics),
            "avg_response_time": round(sum(response_times) / len(response_times), 2),
            "min_response_time": round(min(response_times), 2),
            "max_response_time": round(max(response_times), 2),
            "avg_tokens_per_second": round(sum(tokens_per_second) / len(tokens_per_second), 2),
            "total_tokens_generated": sum(token_counts),
            "avg_tokens_per_response": round(sum(token_counts) / len(token_counts), 1),
            "total_evaluations": len(all_evaluations),
            "unique_judges": len(set(eval_obj.judge_model for eval_obj in all_evaluations))
        }
        
        if all_evaluations:
            summary.update(MetricsAnalyzer._calculate_evaluation_stats(all_evaluations))
        
        return summary
    
    @staticmethod
    def _calculate_evaluation_stats(evaluations: List[LLMEvaluation]) -> Dict[str, Any]:
        """Calculate evaluation-specific statistics"""
        stats = {
            "avg_faithfulness": round(sum(e.faithfulness for e in evaluations) / len(evaluations), 1),
            "avg_groundedness": round(sum(e.groundedness for e in evaluations) / len(evaluations), 1),
            "avg_factual_consistency": round(sum(e.factual_consistency for e in evaluations) / len(evaluations), 1),
            "avg_relevance": round(sum(e.relevance for e in evaluations) / len(evaluations), 1),
            "avg_completeness": round(sum(e.completeness for e in evaluations) / len(evaluations), 1),
            "avg_fluency": round(sum(e.fluency for e in evaluations) / len(evaluations), 1),
            "avg_overall_score": round(sum(e.overall_score for e in evaluations) / len(evaluations), 1),
        }
        
        # Calculate rating distribution
        rating_counts = MetricsAnalyzer._count_rating_categories(evaluations)
        stats["rating_distribution"] = rating_counts
        
        # Calculate scores by judge model
        judge_scores = MetricsAnalyzer._calculate_judge_scores(evaluations)
        stats["judge_models"] = judge_scores
        
        return stats
    
    @staticmethod
    def _count_rating_categories(evaluations: List[LLMEvaluation]) -> Dict[str, int]:
        """Count evaluations by rating category"""
        rating_counts = {"Excellent": 0, "Good": 0, "Fair": 0, "Average": 0, "Poor / Weak": 0}
        
        for eval_obj in evaluations:
            score = eval_obj.overall_score
            if score >= 9.0:
                rating_counts["Excellent"] += 1
            elif score >= 8.0:
                rating_counts["Good"] += 1
            elif score >= 6.5:
                rating_counts["Fair"] += 1
            elif score >= 5.0:
                rating_counts["Average"] += 1
            else:
                rating_counts["Poor / Weak"] += 1
        
        return rating_counts
    
    @staticmethod
    def _calculate_judge_scores(evaluations: List[LLMEvaluation]) -> Dict[str, Dict[str, Any]]:
        """Calculate average scores by judge model"""
        judge_scores = {}
        
        for eval_obj in evaluations:
            if eval_obj.judge_model not in judge_scores:
                judge_scores[eval_obj.judge_model] = []
            judge_scores[eval_obj.judge_model].append(eval_obj.overall_score)
        
        return {
            judge: {
                "avg_score": round(sum(scores) / len(scores), 1),
                "count": len(scores)
            }
            for judge, scores in judge_scores.items()
        }