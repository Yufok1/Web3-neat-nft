"""
Language Evolution Pipeline for Constitutional AI Agents

This module implements genuine evolutionary language learning:
1. Character-level prediction (start simple)
2. Word-level generation (intermediate)
3. Sentence coherence (advanced)
4. Creative text generation (expert)

Constitutional traits influence:
- Learning rate and mutation parameters
- Creative vs conservative language generation
- Social/communicative focus vs technical precision
- Innovation drive affecting vocabulary diversity
"""

import random
import neat
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from ..identity import IdentityBundle
from ..neat_integration import ConstitutionalNEATRunner
from .base_capability import (
    TrainingData,
    CapabilityFitnessEvaluator,
    CapabilityTrainingPipeline,
    EvaluationResult,
)


@dataclass
class WordLevelLanguageTrainingData(TrainingData):
    """Word-level training data for conversational AI."""

    text: str
    word_to_int: Dict[str, int]
    int_to_word: Dict[int, str]
    vocab_size: int
    sequence_length: int

    def __post_init__(self):
        """Initialize parent TrainingData fields."""
        super().__init__(capability_type="language", data_size=len(self.text.split()))

    @classmethod
    def from_text(cls, text: str, sequence_length: int = 10):
        """Create word-level training data from conversational text."""
        # Clean and tokenize text
        import re

        text = re.sub(r"[^\w\s]", "", text.lower())
        # Remove punctuation, lowercase
        words = text.split()

        # Create word mappings
        unique_words = sorted(set(words))
        word_to_int = {word: i for i, word in enumerate(unique_words)}
        int_to_word = {i: word for i, word in enumerate(unique_words)}

        return cls(
            text=text,
            word_to_int=word_to_int,
            int_to_word=int_to_word,
            vocab_size=len(unique_words),
            sequence_length=sequence_length,
        )

    def get_training_sequences(
        self, num_sequences: int = 1000
    ) -> List[Tuple[List[int], int]]:
        """Generate word-level input-output sequence pairs for training."""
        sequences = []
        words = self.text.split()

        for _ in range(num_sequences):
            # Pick random starting position
            start = random.randint(0, len(words) - self.sequence_length - 1)

            # Input sequence (words as integers)
            input_seq = [
                self.word_to_int[words[i]]
                for i in range(start, start + self.sequence_length)
            ]

            # Target (next word)
            target = self.word_to_int[words[start + self.sequence_length]]

            sequences.append((input_seq, target))

        return sequences

    def get_training_samples(self, num_samples: int) -> List[Tuple[List[int], int]]:
        """Get training input-output pairs (implements TrainingData interface)."""
        return self.get_training_sequences(num_samples)

    def prepare_input(self, raw_input: List[int]) -> List[float]:
        """Convert word sequence to neural network input."""
        return [x / self.vocab_size for x in raw_input]

    def parse_output(self, network_output: List[float]) -> int:
        """Convert network output to word index."""
        word_idx = int(network_output[0] * self.vocab_size)
        return max(0, min(word_idx, self.vocab_size - 1))


class LanguageEvolutionFitness(CapabilityFitnessEvaluator):
    """
    Fitness evaluator for language evolution.

    Measures actual language generation capability:
    - Character prediction accuracy
    - Sequence coherence
    - Vocabulary diversity
    - Constitutional trait alignment
    """

    def __init__(
        self,
        training_data: WordLevelLanguageTrainingData,
        agent_identity: IdentityBundle,
    ):
        self.training_data = training_data
        self.agent_identity = agent_identity
        self.traits = agent_identity.constitution_result.constitution

        # Original trait-based evaluation weights
        self.creativity_weight = self._normalize_trait("InnovationDrive", 0.1, 0.5)
        self.accuracy_weight = (
            1.0 - self.creativity_weight + 0.5
        )  # Always value accuracy
        self.diversity_weight = self._normalize_trait("Curiosity", 0.1, 0.4)

        # Phase 1 Foundation Trait Modifiers
        self.critical_thinking_weight = self._normalize_trait(
            "CriticalThinking", 0.7, 1.5
        )
        self.pattern_recognition_weight = self._normalize_trait(
            "PatternRecognition", 0.7, 1.5
        )
        self.common_sense_weight = self._normalize_trait("CommonSense", 0.9, 1.2)
        self.resilience_weight = self._normalize_trait("Resilience", 0.8, 1.1)
        self.adaptability_weight = self._normalize_trait("Adaptability", 0.9, 1.4)

        # Phase 2 Reasoning Enhancement Trait Modifiers
        self.causal_reasoning_weight = self._normalize_trait(
            "CausalReasoning", 0.6, 1.6
        )
        self.abstract_thinking_weight = self._normalize_trait(
            "AbstractThinking", 0.8, 1.3
        )
        self.temporal_reasoning_weight = self._normalize_trait(
            "TemporalReasoning", 0.9, 1.4
        )
        self.spatial_reasoning_weight = self._normalize_trait(
            "SpatialReasoning", 0.7, 1.2
        )
        self.intuition_weight = self._normalize_trait("Intuition", 0.5, 1.8)

        # Phase 3 Social & Emotional Intelligence Trait Modifiers
        self.emotional_intelligence_weight = self._normalize_trait(
            "EmotionalIntelligence", 0.8, 1.4
        )
        self.empathy_weight = self._normalize_trait("Empathy", 0.9, 1.3)
        self.self_awareness_weight = self._normalize_trait("SelfAwareness", 0.7, 1.5)
        self.trustworthiness_weight = self._normalize_trait("Trustworthiness", 1.0, 1.2)
        self.cooperation_weight = self._normalize_trait("Cooperation", 0.8, 1.6)

        # Phase 4 Advanced Capabilities & Governance Trait Modifiers
        self.conflict_resolution_weight = self._normalize_trait(
            "ConflictResolution", 0.9, 1.3
        )
        self.cultural_intelligence_weight = self._normalize_trait(
            "CulturalIntelligence", 0.8, 1.4
        )
        self.leadership_weight = self._normalize_trait("Leadership", 0.7, 1.5)
        self.negotiation_weight = self._normalize_trait("Negotiation", 0.8, 1.4)
        self.goal_orientation_weight = self._normalize_trait(
            "GoalOrientation", 0.9, 1.6
        )
        self.autonomy_weight = self._normalize_trait("Autonomy", 0.6, 1.8)
        self.humor_weight = self._normalize_trait("Humor", 0.9, 1.2)
        self.ethical_reasoning_weight = self._normalize_trait(
            "EthicalReasoning", 1.0, 1.3
        )
        self.creativity_weight_advanced = self._normalize_trait("Creativity", 0.5, 1.9)

    def _normalize_trait(
        self, trait_name: str, min_weight: float, max_weight: float
    ) -> float:
        """Convert trait value to weight for fitness calculation."""
        trait_value = self.traits.get(trait_name, 5.0)

        if isinstance(trait_value, str):
            # Handle categorical traits
            trait_map = {
                "Minimal": 1.0,
                "Low": 3.0,
                "Moderate": 5.0,
                "High": 7.0,
                "Maximum": 10.0,
            }
            trait_value = trait_map.get(trait_value, 5.0)

        # Get trait domain from definitions for proper normalization
        from ..traits import COMPLETE_TRAIT_DEFINITIONS

        if trait_name in COMPLETE_TRAIT_DEFINITIONS:
            domain = COMPLETE_TRAIT_DEFINITIONS[trait_name].domain
            domain_min, domain_max = domain
            # Normalize to 0-1 range based on actual domain
            normalized = max(
                0.0, min(1.0, (trait_value - domain_min) / (domain_max - domain_min))
            )
        else:
            # Fallback for traits not in definitions (assume 0-10 scale)
            normalized = max(0.0, min(1.0, trait_value / 10.0))

        # Scale to weight range
        return min_weight + normalized * (max_weight - min_weight)

    def evaluate_network(
        self,
        network: neat.nn.FeedForwardNetwork,
        num_tests: int = 100,
        max_inputs: int = 4,  # Changed from 2 to 4 to match NEAT config
    ) -> float:
        """
        Evaluate a neural network's language generation capability.

        Returns fitness score based on:
        - Prediction accuracy
        - Output diversity
        - Trait-guided preferences
        """
        total_fitness = 0.0
        predictions = []
        test_sequences = self.training_data.get_training_sequences(num_tests)

        for input_seq, target in test_sequences:
            # Use only first max_inputs characters (adapt to network size)
            input_subset = input_seq[:max_inputs]

            # Normalize input to 0-1 range for neural network
            normalized_input = [x / self.training_data.vocab_size for x in input_subset]

            # Pad if necessary
            while len(normalized_input) < max_inputs:
                normalized_input.append(0.0)

            # Get network prediction
            output = network.activate(normalized_input)

            # Use first output for character prediction
            predicted_char_idx = int(output[0] * self.training_data.vocab_size)
            predicted_char_idx = max(
                0, min(predicted_char_idx, self.training_data.vocab_size - 1)
            )

            predictions.append(predicted_char_idx)

            # Accuracy component
            accuracy_score = 1.0 if predicted_char_idx == target else 0.0

            # Add to fitness
            total_fitness += accuracy_score * self.accuracy_weight

        # Calculate diversity bonus
        if predictions:
            unique_predictions = len(set(predictions))
            max_diversity = min(len(predictions), self.training_data.vocab_size)
            diversity_score = (
                unique_predictions / max_diversity if max_diversity > 0 else 0.0
            )
            total_fitness += diversity_score * self.diversity_weight

        # Apply Phase 1 Foundation Trait modifiers
        base_fitness = total_fitness / num_tests if num_tests > 0 else 0.0

        # Critical Thinking - improves accuracy evaluation and reduces noise
        base_fitness *= self.critical_thinking_weight

        # Additional critical thinking bonus for high accuracy
        if diversity_score > 0 and len(predictions) > 5:
            accuracy_rate = sum(
                1
                for i, (input_seq, target) in enumerate(test_sequences[:num_tests])
                if predictions[i] == target
            ) / len(predictions)
            if accuracy_rate > 0.3:  # Reward critical thinking for good perf
                critical_bonus = (
                    (self.critical_thinking_weight - 1.0) * accuracy_rate * 0.1
                )
                base_fitness += critical_bonus

        # Pattern Recognition - enhances learning from sequences
        pattern_bonus = self.pattern_recognition_weight - 1.0
        base_fitness += pattern_bonus * 0.1  # Small additive bonus

        # Common Sense - prevents nonsensical predictions (consistency)
        if len(predictions) > 1:
            consistency = 1.0 - (len(set(predictions)) / len(predictions))
            common_sense_bonus = consistency * (self.common_sense_weight - 1.0) * 0.05
            base_fitness += common_sense_bonus

        # Resilience - maintains performance under variation (no penalty)
        base_fitness *= self.resilience_weight

        # Adaptability - helps with diverse training examples
        if diversity_score > 0.5:  # Reward adaptable agents for diversity
            adaptability_bonus = (
                (self.adaptability_weight - 1.0) * diversity_score * 0.1
            )
            base_fitness += adaptability_bonus

        # Apply Phase 2 Reasoning Enhancement modifiers

        # Causal Reasoning - improves understanding of input-output relationships
        base_fitness *= self.causal_reasoning_weight

        # Abstract Thinking - enhances generalization beyond specific examples
        if diversity_score > 0.3:  # Reward abstract thinking for generalization
            abstract_bonus = (
                (self.abstract_thinking_weight - 1.0) * diversity_score * 0.08
            )
            base_fitness += abstract_bonus

        # Temporal Reasoning - improves sequence understanding (core to language)
        sequence_quality = 1.0 - abs(
            0.5 - diversity_score
        )  # Optimal diversity around 0.5
        temporal_bonus = (
            (self.temporal_reasoning_weight - 1.0) * sequence_quality * 0.12
        )
        base_fitness += temporal_bonus

        # Spatial Reasoning - less directly relevant to language, but helps with structure
        base_fitness *= 0.9 + 0.1 * self.spatial_reasoning_weight  # Smaller impact

        # Intuition - balances analytical with intuitive responses
        if len(predictions) > 3:
            # Measure response consistency as proxy for intuitive coherence
            prediction_variance = len(set(predictions)) / len(predictions)
            intuition_modifier = (
                1.0 + (self.intuition_weight - 1.0) * (1.0 - prediction_variance) * 0.05
            )
            base_fitness *= intuition_modifier

        # Apply Phase 3 Social & Emotional Intelligence modifiers

        # Emotional Intelligence - improves overall communication effectiveness
        base_fitness *= self.emotional_intelligence_weight

        # Empathy - enhances understanding and responsiveness (rewards consistent patterns)
        if len(predictions) > 5:
            # Higher empathy should produce more contextually appropriate responses
            contextual_appropriateness = 1.0 - abs(
                0.4 - diversity_score
            )  # Optimal around 0.4
            empathy_bonus = (
                (self.empathy_weight - 1.0) * contextual_appropriateness * 0.06
            )
            base_fitness += empathy_bonus

        # Self-Awareness - prevents overconfidence and improves calibration
        if diversity_score > 0.2:  # Only when agent shows some variation
            # Self-aware agents should show appropriate uncertainty
            uncertainty_appropriateness = (
                min(diversity_score, 1.0 - diversity_score) * 2
            )  # Peak at 0.5
            self_awareness_bonus = (
                (self.self_awareness_weight - 1.0) * uncertainty_appropriateness * 0.04
            )
            base_fitness += self_awareness_bonus

        # Trustworthiness - rewards consistency and reliability
        if len(predictions) > 1:
            # More trustworthy agents should show consistent quality
            consistency_score = 1.0 - (len(set(predictions)) / len(predictions))
            trustworthiness_bonus = (
                (self.trustworthiness_weight - 1.0) * consistency_score * 0.08
            )
            base_fitness += trustworthiness_bonus

        # Cooperation - enhances collaborative potential (measured via adaptability to corpus)
        if diversity_score > 0.1 and diversity_score < 0.8:  # Balanced cooperation
            cooperation_modifier = 1.0 + (self.cooperation_weight - 1.0) * 0.05
            base_fitness *= cooperation_modifier

        # Apply Phase 4 Advanced Capabilities & Governance modifiers

        # Conflict Resolution - improves handling of contradictory inputs
        if len(predictions) > 3:
            # Measure ability to balance different response patterns
            balance_score = 1.0 - abs(0.5 - diversity_score)  # Optimal balance at 0.5
            conflict_bonus = (
                (self.conflict_resolution_weight - 1.0) * balance_score * 0.05
            )
            base_fitness += conflict_bonus

        # Cultural Intelligence - enhances contextual adaptation
        base_fitness *= self.cultural_intelligence_weight

        # Leadership - amplifies overall performance (multiplicative effect)
        leadership_multiplier = 1.0 + (self.leadership_weight - 1.0) * 0.08
        base_fitness *= leadership_multiplier

        # Negotiation - optimizes for win-win outcomes (balanced diversity)
        if (
            diversity_score > 0.3 and diversity_score < 0.7
        ):  # Sweet spot for negotiation
            negotiation_bonus = (self.negotiation_weight - 1.0) * 0.06
            base_fitness += negotiation_bonus

        # Goal Orientation - enhances task completion focus
        base_fitness *= self.goal_orientation_weight

        # Autonomy - enables independent operation (reduces supervision needs)
        if diversity_score > 0.2:  # Only when showing independent variation
            autonomy_bonus = (self.autonomy_weight - 1.0) * 0.07
            base_fitness += autonomy_bonus

        # Humor - improves engagement and communication effectiveness
        if len(predictions) > 5:
            # Appropriate humor requires contextual awareness
            humor_appropriateness = min(
                diversity_score * 2, 1.0
            )  # Scale up to 0.5 -> 1.0
            humor_bonus = (self.humor_weight - 1.0) * humor_appropriateness * 0.04
            base_fitness += humor_bonus

        # Ethical Reasoning - ensures responsible AI behavior
        base_fitness *= self.ethical_reasoning_weight

        # Advanced Creativity - promotes novel solution generation
        if diversity_score > 0.4:  # Higher threshold for advanced creativity
            creativity_multiplier = (
                1.0 + (self.creativity_weight_advanced - 1.0) * diversity_score * 0.10
            )
            base_fitness *= creativity_multiplier

        return max(0.0, base_fitness)  # Ensure non-negative fitness

    def evaluate_generation_quality(
        self,
        network: neat.nn.FeedForwardNetwork,
        seed_text: str = "The",
        max_length: int = 50,
        max_inputs: int = 2,
    ) -> EvaluationResult:
        """
        Evaluate the quality of generated text.

        Returns metrics for analysis (not used in fitness directly).
        """
        if len(seed_text) < max_inputs:
            # Pad seed text
            seed_text = seed_text.ljust(max_inputs)

        # Generate text (word-level)
        words = seed_text.split()
        if len(words) < max_inputs:
            # Pad with common words
            words = ["the"] * (max_inputs - len(words)) + words

        current_sequence = [
            self.training_data.word_to_int.get(word.lower(), 0)
            for word in words[-max_inputs:]
        ]
        generated_text = seed_text

        for _ in range(max_length):
            # Normalize input
            normalized_input = [
                x / self.training_data.vocab_size for x in current_sequence
            ]

            # Predict next word (use first output)
            output = network.activate(normalized_input)
            next_word_idx = int(output[0] * self.training_data.vocab_size)
            next_word_idx = max(
                0, min(next_word_idx, self.training_data.vocab_size - 1)
            )

            next_word = self.training_data.int_to_word[next_word_idx]
            generated_text += " " + next_word

            # Update sequence
            current_sequence = current_sequence[1:] + [next_word_idx]

        # Calculate fitness score for this generation
        fitness_score = self.evaluate_network(
            network, num_tests=10, max_inputs=max_inputs
        )

        return EvaluationResult(
            fitness_score=fitness_score,
            sample_outputs={
                "generated_text": generated_text,
                "seed_text": seed_text,
            },
            performance_metrics={
                "length": len(generated_text),
                "unique_words": len(set(generated_text.split())),
                "vocab_coverage": len(set(generated_text.split()))
                / self.training_data.vocab_size,
            },
            evaluation_metadata={"max_length": max_length, "max_inputs": max_inputs},
        )


class LanguageTrainingPipeline(CapabilityTrainingPipeline):
    """
    Complete pipeline for evolving language capabilities in constitutional agents.

    Progressive training stages:
    1. Character prediction (foundation)
    2. Short sequence generation
    3. Longer coherent text
    4. Creative/diverse generation
    """

    def __init__(self, training_text: str, agent_identity: IdentityBundle):
        """
        Initialize language training pipeline.

        Args:
            training_text: Text corpus for training (e.g., Shakespeare, news, etc.)
            agent_identity: Constitutional agent to train
        """
        self.agent_identity = agent_identity
        self.training_data = WordLevelLanguageTrainingData.from_text(training_text)
        self.fitness_evaluator = LanguageEvolutionFitness(
            self.training_data, agent_identity
        )
        # Create NEAT runner with trait-configured parameters
        self.neat_runner = ConstitutionalNEATRunner(agent_identity)
        self.capability_type = "language"

    def create_fitness_evaluator(self):
        """Create the fitness evaluator for language capability."""
        return LanguageEvolutionFitness(self.training_data, self.agent_identity)

    def get_neat_config_params(self):
        """Return NEAT config parameters for language evolution."""
        # Example: population size and network structure can be trait-driven
        # For now, return some reasonable defaults
        return {
            "pop_size": 150,
            "num_inputs": 4,
            "num_outputs": 1,
            "activation": "sigmoid",
            "fitness_threshold": 0.95,
        }

    def get_relevant_traits(self):
        """Return list of constitutional traits most relevant to language capability."""
        return [
            "Innovation_Drive",
            "Curiosity",
            "Learning_Rate",
            "Communication_Style",
            "Working_Memory",
            "AttentionSpan",
        ]

    def create_fitness_function(self):
        """Create fitness function for NEAT evolution."""

        def fitness_function(genomes, config):
            for genome_id, genome in genomes:
                try:
                    # Create network
                    network = neat.nn.FeedForwardNetwork.create(genome, config)

                    # Evaluate language capability (use 4 input limit to match NEAT config)
                    fitness = self.fitness_evaluator.evaluate_network(
                        network, max_inputs=4
                    )

                    # Assign fitness
                    genome.fitness = fitness

                except Exception:
                    # Handle network creation/evaluation errors
                    genome.fitness = 0.0

        return fitness_function

    def train_language_capability(self, generations: int = 50) -> Dict[str, Any]:
        """
        Train the agent to develop language generation capability.

        Args:
            generations: Number of evolutionary generations

        Returns:
            Training results with best network and performance metrics
        """
        print(
            f"Training language capability for agent {self.agent_identity.id_hash[:12]}..."
        )
        print(f"Training corpus: {len(self.training_data.text)} characters")
        print(f"Vocabulary size: {self.training_data.vocab_size}")
        print("Constitutional traits influencing training:")
        print(
            f"  - Innovation Drive: {self.agent_identity.constitution_result.constitution.get('Innovation_Drive', 'N/A')}"
        )
        print(
            f"  - Curiosity: {self.agent_identity.constitution_result.constitution.get('Curiosity', 'N/A')}"
        )
        print(
            f"  - Learning Rate: {self.agent_identity.constitution_result.constitution.get('Learning_Rate', 'N/A')}"
        )

        # Configure NEAT for language task
        config_file = self.neat_runner.create_neat_config_file("language_config.txt")

        # Create fitness function
        fitness_func = self.create_fitness_function()

        # Run evolution
        print(f"\nStarting evolution for {generations} generations...")
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_file,
        )

        population = neat.Population(config)
        population.add_reporter(neat.StdOutReporter(True))

        # Evolve
        best_genome = population.run(fitness_func, generations)
        best_network = neat.nn.FeedForwardNetwork.create(best_genome, config)

        # Evaluate final capability
        final_fitness = self.fitness_evaluator.evaluate_network(
            best_network, num_tests=200
        )

        # Generate sample text
        sample_generation = self.fitness_evaluator.evaluate_generation_quality(
            best_network, seed_text="Test", max_length=100, max_inputs=4
        )

        # Cleanup
        import os

        if os.path.exists("language_config.txt"):
            os.remove("language_config.txt")

        return {
            "agent_id": self.agent_identity.id_hash,
            "identity_bundle": self.agent_identity,  # Include identity for persistence
            "final_fitness": final_fitness,
            "best_genome": best_genome,
            "best_network": best_network,
            "sample_generation": sample_generation,
            "training_generations": generations,
            "vocabulary_size": self.training_data.vocab_size,
        }


def create_language_training_corpus() -> str:
    """
    Create a substantial training corpus using HuggingFace datasets.

    This now loads real text data instead of tiny hardcoded samples.
    """
    try:
        from ..corpus_loader import get_language_corpus

        print("Loading substantial language corpus for training...")
        # Get 25MB of real text data for SERIOUS AI training
        corpus = get_language_corpus(25_000_000)  # 25MB corpus for production-level AI

        print(f"Loaded corpus: {len(corpus):,} characters")
        return corpus

    except ImportError:
        print("Corpus loader not available, using fallback")
        return _get_fallback_corpus()


def _get_fallback_corpus() -> str:
    """Fallback corpus if corpus_loader isn't available."""
    return (
        """
    The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.
    Language is a powerful tool for communication. Through words, we share ideas, emotions, and knowledge.
    Artificial intelligence systems can learn to understand and generate human language.
    Evolution shapes both biological and artificial intelligence through selection pressure.
    Constitutional traits guide how agents learn and develop their capabilities over time.
    Creative agents explore diverse solutions while analytical agents focus on precision.
    The relationship between genetics and behavior manifests in both DNA and digital genomes.
    Learning is not just memorization but the development of genuine understanding and capability.
    Each agent develops its own unique style based on its constitutional personality.
    Through practice and evolution, simple networks can develop sophisticated language abilities.
    """
        * 50  # Increased repetition for more training data
    )


# Factory function for easy pipeline creation
def train_agent_language_capability(
    agent_identity: IdentityBundle,
    training_text: Optional[str] = None,
    generations: int = 30,
) -> Dict[str, Any]:
    """
    Convenience function to train an agent's language capability.

    Args:
        agent_identity: Constitutional agent to train
        training_text: Text corpus (uses default if None)
        generations: Training generations

    Returns:
        Training results
    """
    if training_text is None:
        training_text = create_language_training_corpus()

    pipeline = LanguageTrainingPipeline(training_text, agent_identity)

    # Use CPU training (optimal for NEAT evolution)
    print("Using CPU training for language capability...")
    return pipeline.train_language_capability(generations)
