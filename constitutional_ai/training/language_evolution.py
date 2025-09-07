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
import numpy as np
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
class LanguageTrainingData(TrainingData):
    """Training data for language evolution."""

    text: str
    char_to_int: Dict[str, int]
    int_to_char: Dict[int, str]
    vocab_size: int
    sequence_length: int

    def __post_init__(self):
        """Initialize parent TrainingData fields."""
        super().__init__(capability_type="language", data_size=len(self.text))

    @classmethod
    def from_text(cls, text: str, sequence_length: int = 40):
        """Create training data from text corpus."""
        # Create character mappings
        chars = sorted(list(set(text)))
        char_to_int = {char: i for i, char in enumerate(chars)}
        int_to_char = {i: char for i, char in enumerate(chars)}

        return cls(
            text=text,
            char_to_int=char_to_int,
            int_to_char=int_to_char,
            vocab_size=len(chars),
            sequence_length=sequence_length,
        )

    def get_training_sequences(
        self, num_sequences: int = 1000
    ) -> List[Tuple[List[int], int]]:
        """Generate input-output sequence pairs for training."""
        sequences = []
        text_ints = [self.char_to_int[char] for char in self.text]

        for _ in range(num_sequences):
            # Pick random starting position
            start = random.randint(0, len(text_ints) - self.sequence_length - 1)

            # Input sequence
            input_seq = text_ints[start : start + self.sequence_length]

            # Target (next character)
            target = text_ints[start + self.sequence_length]

            sequences.append((input_seq, target))

        return sequences

    def get_training_samples(self, num_samples: int) -> List[Tuple[List[int], int]]:
        """Get training input-output pairs (implements TrainingData interface)."""
        return self.get_training_sequences(num_samples)

    def prepare_input(self, raw_input: List[int]) -> List[float]:
        """Convert character sequence to neural network input."""
        return [x / self.vocab_size for x in raw_input]

    def parse_output(self, network_output: List[float]) -> int:
        """Convert network output to character index."""
        char_idx = int(network_output[0] * self.vocab_size)
        return max(0, min(char_idx, self.vocab_size - 1))


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
        self, training_data: LanguageTrainingData, agent_identity: IdentityBundle
    ):
        self.training_data = training_data
        self.agent_identity = agent_identity
        self.traits = agent_identity.constitution_result.constitution

        # Trait-based evaluation weights
        self.creativity_weight = self._normalize_trait("Innovation_Drive", 0.1, 0.5)
        self.accuracy_weight = (
            1.0 - self.creativity_weight + 0.5
        )  # Always value accuracy
        self.diversity_weight = self._normalize_trait("Curiosity", 0.1, 0.4)

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

        # Normalize to 0-1 range (assuming traits are 1-10 scale)
        normalized = max(0.0, min(1.0, (trait_value - 1.0) / 9.0))

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

        return total_fitness / num_tests if num_tests > 0 else 0.0

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

        # Generate text
        current_sequence = [
            self.training_data.char_to_int.get(char, 0)
            for char in seed_text[-max_inputs:]
        ]
        generated_text = seed_text

        for _ in range(max_length):
            # Normalize input
            normalized_input = [
                x / self.training_data.vocab_size for x in current_sequence
            ]

            # Predict next character (use first output)
            output = network.activate(normalized_input)
            next_char_idx = int(output[0] * self.training_data.vocab_size)
            next_char_idx = max(
                0, min(next_char_idx, self.training_data.vocab_size - 1)
            )

            next_char = self.training_data.int_to_char[next_char_idx]
            generated_text += next_char

            # Update sequence
            current_sequence = current_sequence[1:] + [next_char_idx]

        # Calculate fitness score for this generation
        fitness_score = self.evaluate_network(
            network, num_tests=10, max_inputs=max_inputs
        )

        return {
            "final_fitness": fitness_score,
            "sample_generation": {
                "generated_text": generated_text,
                "seed_text": seed_text,
            },
            "performance_metrics": {
                "length": len(generated_text),
                "unique_chars": len(set(generated_text)),
                "vocab_coverage": len(set(generated_text))
                / self.training_data.vocab_size,
            },
            "evaluation_metadata": {"max_length": max_length, "max_inputs": max_inputs},
        }


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
        self.training_data = LanguageTrainingData.from_text(training_text)
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

                except Exception as e:
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
        print(f"Constitutional traits influencing training:")
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
            best_network, num_tests=200, max_inputs=4
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
        # Get 1MB of real text data for training
        corpus = get_language_corpus(1_000_000)  # 1MB corpus

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
    use_gpu: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function to train an agent's language capability.

    Args:
        agent_identity: Constitutional agent to train
        training_text: Text corpus (uses default if None)
        generations: Training generations
        use_gpu: Use GPU acceleration if available

    Returns:
        Training results
    """
    if training_text is None:
        training_text = create_language_training_corpus()

    pipeline = LanguageTrainingPipeline(training_text, agent_identity)

    # Use GPU acceleration if requested and available
    if use_gpu:
        try:
            from ..gpu_training import accelerate_agent_training

            print("Using GPU acceleration for language training...")
            result = accelerate_agent_training(
                agent_identity,
                pipeline.training_data,
                pipeline.fitness_evaluator,
                generations,
                batch_size=32,
            )

            # Add language-specific metadata
            result.update(
                {
                    "capability_type": "language",
                    "corpus_size": len(training_text),
                    "vocabulary_size": pipeline.training_data.vocab_size,
                }
            )

            return result

        except ImportError:
            print("GPU training not available, using CPU...")
        except Exception as e:
            print(f"GPU training failed ({e}), falling back to CPU...")

    # Fallback to CPU training
    print("Using CPU training for language capability...")
    return pipeline.train_language_capability(generations)
