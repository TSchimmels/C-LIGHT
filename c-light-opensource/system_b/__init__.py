"""
System B: SEAL-Style Self-Learning RAG
Learns from feedback, discovers patterns, improves over time
"""

from .seal_rag import SEALBasedRAG
from .pattern_learner import PatternLearner, CausalPattern
from .weight_optimizer import EdgeWeightOptimizer
from .active_learning import ActivePaperSelector
from .feedback_system import FeedbackCollector, QueryFeedback

__all__ = [
    'SEALBasedRAG',
    'PatternLearner',
    'CausalPattern',
    'EdgeWeightOptimizer',
    'ActivePaperSelector',
    'FeedbackCollector',
    'QueryFeedback'
]
