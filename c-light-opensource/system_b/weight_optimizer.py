"""
Edge Weight Optimizer for SEAL System
Learns which knowledge graph edges are reliable based on query success/failure
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class EdgeWeightOptimizer:
    """
    Optimizes edge weights in knowledge graph based on feedback

    Key idea: When a query succeeds using a certain path, reinforce those edges.
    When a query fails, penalize those edges.

    This is similar to reinforcement learning:
    - Successful path = positive reward → increase edge weights
    - Failed path = negative reward → decrease edge weights

    Over time, reliable edges get higher weights, unreliable edges get lower weights.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        decay_rate: float = 0.999,  # Slowly forget old adjustments
        storage_path: str = "/mnt/nvme/c-light/seal/weights"
    ):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Edge adjustments: (source, target) -> adjustment
        self.weight_adjustments: Dict[Tuple[str, str], float] = defaultdict(float)

        # Edge usage statistics
        self.edge_success_count: Dict[Tuple[str, str], int] = defaultdict(int)
        self.edge_failure_count: Dict[Tuple[str, str], int] = defaultdict(int)

        # Learning history
        self.update_history: List[Dict] = []

        # Load saved weights
        self._load_weights()

        logger.info("Edge weight optimizer initialized")

    def reinforce_path(
        self,
        path: List[str],
        reward: float = 1.0,
        rating: int = 5
    ):
        """
        Reinforce edges in a successful path

        Args:
            path: List of node IDs in the path
            reward: Reward magnitude (higher rating = higher reward)
            rating: User rating (1-5)
        """
        if len(path) < 2:
            return

        # Scale reward by rating
        scaled_reward = reward * (rating / 5.0)

        for i in range(len(path) - 1):
            edge = (path[i], path[i+1])

            # Increase weight
            self.weight_adjustments[edge] += self.learning_rate * scaled_reward

            # Update statistics
            self.edge_success_count[edge] += 1

            logger.debug(f"Reinforced edge {edge}: +{self.learning_rate * scaled_reward:.4f}")

        self.update_history.append({
            'type': 'reinforce',
            'path_length': len(path),
            'reward': scaled_reward,
            'rating': rating
        })

        # Save periodically
        if len(self.update_history) % 10 == 0:
            self._save_weights()

    def penalize_path(
        self,
        path: List[str],
        penalty: float = 1.0,
        rating: int = 1
    ):
        """
        Penalize edges in a failed path

        Args:
            path: List of node IDs in the path
            penalty: Penalty magnitude
            rating: User rating (1-5)
        """
        if len(path) < 2:
            return

        # Scale penalty by how bad the rating was
        scaled_penalty = penalty * ((5 - rating) / 4.0)  # rating 1 = max penalty

        for i in range(len(path) - 1):
            edge = (path[i], path[i+1])

            # Decrease weight
            self.weight_adjustments[edge] -= self.learning_rate * scaled_penalty

            # Update statistics
            self.edge_failure_count[edge] += 1

            logger.debug(f"Penalized edge {edge}: -{self.learning_rate * scaled_penalty:.4f}")

        self.update_history.append({
            'type': 'penalize',
            'path_length': len(path),
            'penalty': scaled_penalty,
            'rating': rating
        })

        if len(self.update_history) % 10 == 0:
            self._save_weights()

    def get_edge_weight(
        self,
        source: str,
        target: str,
        base_weight: float = 1.0
    ) -> float:
        """
        Get adjusted weight for an edge

        Args:
            source: Source node
            target: Target node
            base_weight: Original edge weight from evidence

        Returns:
            Adjusted weight
        """
        edge = (source, target)
        adjustment = self.weight_adjustments.get(edge, 0.0)

        # Combine base weight with learned adjustment
        adjusted_weight = base_weight + adjustment

        # Ensure non-negative
        return max(adjusted_weight, 0.01)

    def get_edge_confidence(
        self,
        source: str,
        target: str
    ) -> float:
        """
        Get confidence in an edge based on success/failure history

        Returns:
            Confidence score (0-1)
        """
        edge = (source, target)
        successes = self.edge_success_count.get(edge, 0)
        failures = self.edge_failure_count.get(edge, 0)

        total = successes + failures

        if total == 0:
            return 0.5  # No data, neutral confidence

        # Bayesian confidence with smoothing
        return (successes + 1) / (total + 2)

    def get_edge_reliability(
        self,
        source: str,
        target: str
    ) -> Dict[str, any]:
        """
        Get detailed reliability info for an edge

        Returns:
            Dict with successes, failures, confidence, adjustment
        """
        edge = (source, target)

        return {
            'successes': self.edge_success_count.get(edge, 0),
            'failures': self.edge_failure_count.get(edge, 0),
            'confidence': self.get_edge_confidence(source, target),
            'weight_adjustment': self.weight_adjustments.get(edge, 0.0),
            'usage_count': self.edge_success_count.get(edge, 0) + self.edge_failure_count.get(edge, 0)
        }

    def decay_weights(self):
        """
        Apply decay to weight adjustments

        This slowly forgets old adjustments, allowing the system to adapt
        to new information without being stuck in old patterns.
        """
        for edge in list(self.weight_adjustments.keys()):
            self.weight_adjustments[edge] *= self.decay_rate

            # Remove very small adjustments
            if abs(self.weight_adjustments[edge]) < 0.001:
                del self.weight_adjustments[edge]

        logger.debug(f"Decayed weights (remaining: {len(self.weight_adjustments)})")

    def get_top_edges(self, n: int = 20) -> List[Tuple[Tuple[str, str], float]]:
        """
        Get edges with highest confidence

        Returns:
            List of ((source, target), confidence) tuples
        """
        edges_with_confidence = [
            (edge, self.get_edge_confidence(edge[0], edge[1]))
            for edge in set(list(self.edge_success_count.keys()) + list(self.edge_failure_count.keys()))
        ]

        return sorted(edges_with_confidence, key=lambda x: x[1], reverse=True)[:n]

    def get_worst_edges(self, n: int = 20) -> List[Tuple[Tuple[str, str], float]]:
        """
        Get edges with lowest confidence (frequently failed)

        Returns:
            List of ((source, target), confidence) tuples
        """
        edges_with_confidence = [
            (edge, self.get_edge_confidence(edge[0], edge[1]))
            for edge in set(list(self.edge_success_count.keys()) + list(self.edge_failure_count.keys()))
        ]

        return sorted(edges_with_confidence, key=lambda x: x[1])[:n]

    def get_stats(self) -> Dict[str, any]:
        """Get optimization statistics"""
        adjustments = list(self.weight_adjustments.values())

        total_edges = len(set(
            list(self.edge_success_count.keys()) +
            list(self.edge_failure_count.keys())
        ))

        return {
            'total_edges_seen': total_edges,
            'edges_with_adjustments': len(self.weight_adjustments),
            'total_updates': len(self.update_history),
            'avg_adjustment': np.mean(adjustments) if adjustments else 0.0,
            'max_adjustment': max(adjustments) if adjustments else 0.0,
            'min_adjustment': min(adjustments) if adjustments else 0.0,
            'reinforcements': len([u for u in self.update_history if u['type'] == 'reinforce']),
            'penalizations': len([u for u in self.update_history if u['type'] == 'penalize'])
        }

    def analyze_improvement(self, window_size: int = 100) -> Dict[str, any]:
        """
        Analyze if the system is improving over time

        Compares recent performance to older performance

        Returns:
            Dict with improvement metrics
        """
        if len(self.update_history) < window_size * 2:
            return {'status': 'insufficient_data'}

        # Split into old and recent windows
        old_window = self.update_history[-(window_size*2):-window_size]
        recent_window = self.update_history[-window_size:]

        old_success_rate = len([u for u in old_window if u['type'] == 'reinforce']) / len(old_window)
        recent_success_rate = len([u for u in recent_window if u['type'] == 'reinforce']) / len(recent_window)

        improvement = recent_success_rate - old_success_rate

        return {
            'old_success_rate': old_success_rate,
            'recent_success_rate': recent_success_rate,
            'improvement': improvement,
            'improving': improvement > 0
        }

    def _save_weights(self):
        """Save weight adjustments to disk"""
        data = {
            'weight_adjustments': {
                f"{source}||{target}": weight
                for (source, target), weight in self.weight_adjustments.items()
            },
            'edge_success_count': {
                f"{source}||{target}": count
                for (source, target), count in self.edge_success_count.items()
            },
            'edge_failure_count': {
                f"{source}||{target}": count
                for (source, target), count in self.edge_failure_count.items()
            },
            'learning_rate': self.learning_rate,
            'total_updates': len(self.update_history)
        }

        with open(self.storage_path / "edge_weights.json", 'w') as f:
            json.dump(data, f, indent=2)

    def _load_weights(self):
        """Load saved weights from disk"""
        filepath = self.storage_path / "edge_weights.json"

        if not filepath.exists():
            return

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Load weight adjustments
            for edge_str, weight in data.get('weight_adjustments', {}).items():
                source, target = edge_str.split('||')
                self.weight_adjustments[(source, target)] = weight

            # Load success counts
            for edge_str, count in data.get('edge_success_count', {}).items():
                source, target = edge_str.split('||')
                self.edge_success_count[(source, target)] = count

            # Load failure counts
            for edge_str, count in data.get('edge_failure_count', {}).items():
                source, target = edge_str.split('||')
                self.edge_failure_count[(source, target)] = count

            logger.info(f"Loaded {len(self.weight_adjustments)} edge weight adjustments")

        except Exception as e:
            logger.error(f"Error loading weights: {e}")
