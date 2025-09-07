"""
Base Capability Training Framework for Constitutional AI Agents

This module provides the abstract base class and interfaces for creating
new capability training modules. All capability modules (language, coding,
image generation, etc.) inherit from this framework to ensure consistency
and interoperability.

Key Principles:
1. Constitutional trait-guided learning
2. NEAT neural evolution integration  
3. Standardized fitness evaluation
4. Persistent training results
5. Modular and extensible design
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import neat
import numpy as np

from ..identity import IdentityBundle
from ..neat_integration import ConstitutionalNEATRunner


class TrainingData:
    """Base class for capability-specific training data."""
    
    def __init__(self, capability_type: str, data_size: int):
        self.capability_type = capability_type
        self.data_size = data_size
    
    @abstractmethod
    def get_training_samples(self, num_samples: int) -> List[Tuple[Any, Any]]:
        """Get training input-output pairs."""
        pass
    
    @abstractmethod
    def prepare_input(self, raw_input: Any) -> List[float]:
        """Convert raw input to neural network format."""
        pass
    
    @abstractmethod
    def parse_output(self, network_output: List[float]) -> Any:
        """Convert network output to capability-specific format."""
        pass


@dataclass  
class EvaluationResult:
    """Results from capability evaluation."""
    fitness_score: float
    sample_outputs: Dict[str, Any]
    performance_metrics: Dict[str, float]
    evaluation_metadata: Dict[str, Any]


class CapabilityFitnessEvaluator(ABC):
    """
    Abstract base class for capability-specific fitness evaluation.
    
    Each capability module implements this to define how agents are evaluated
    in their specific domain (language, coding, image generation, etc.).
    """
    
    def __init__(self, training_data: TrainingData, agent_identity: IdentityBundle):
        """
        Initialize fitness evaluator.
        
        Args:
            training_data: Domain-specific training data
            agent_identity: Constitutional agent identity
        """
        self.training_data = training_data
        self.agent_identity = agent_identity
        self.traits = agent_identity.constitution_result.constitution
        
        # Extract trait-based weights (subclasses can override)
        self.setup_trait_weights()
    
    def setup_trait_weights(self):
        """Setup evaluation weights based on constitutional traits."""
        # Default trait mappings - subclasses should customize
        self.creativity_weight = self._normalize_trait('InnovationDrive', 0.1, 0.4)
        self.accuracy_weight = 1.0 - self.creativity_weight + 0.5
        self.diversity_weight = self._normalize_trait('Curiosity', 0.1, 0.3)
        self.efficiency_weight = self._normalize_trait('EnergyEfficiency', 0.1, 0.2)
    
    def _normalize_trait(self, trait_name: str, min_weight: float, max_weight: float) -> float:
        """Convert trait value to weight for fitness calculation."""
        trait_value = self.traits.get(trait_name, 5.0)
        
        # Handle categorical traits
        if isinstance(trait_value, str):
            trait_map = {'Minimal': 1.0, 'Low': 3.0, 'Moderate': 5.0, 'High': 7.0, 'Maximum': 10.0}
            trait_value = trait_map.get(trait_value, 5.0)
        
        # Normalize to 0-1 range (assuming traits are 1-10 scale)
        normalized = max(0.0, min(1.0, (float(trait_value) - 1.0) / 9.0))
        
        # Scale to weight range
        return min_weight + normalized * (max_weight - min_weight)
    
    @abstractmethod
    def evaluate_network(self, network: neat.nn.FeedForwardNetwork, num_tests: int = 100) -> float:
        """
        Evaluate a neural network's capability in this domain.
        
        Returns:
            Fitness score (higher is better)
        """
        pass
    
    @abstractmethod
    def evaluate_generation_quality(self, network: neat.nn.FeedForwardNetwork, 
                                  **kwargs) -> EvaluationResult:
        """
        Evaluate the quality of network outputs for analysis.
        
        Returns:
            Detailed evaluation results with samples and metrics
        """
        pass


class CapabilityTrainingPipeline(ABC):
    """
    Abstract base class for capability training pipelines.
    
    Each capability module implements this to define the complete training
    process from raw data to evolved neural networks.
    """
    
    def __init__(self, training_data: TrainingData, agent_identity: IdentityBundle):
        """
        Initialize training pipeline.
        
        Args:
            training_data: Domain-specific training data
            agent_identity: Constitutional agent to train
        """
        self.agent_identity = agent_identity  
        self.training_data = training_data
        self.capability_type = training_data.capability_type
        
        # Create fitness evaluator (implemented by subclass)
        self.fitness_evaluator = self.create_fitness_evaluator()
        
        # Create NEAT runner with trait-configured parameters
        self.neat_runner = ConstitutionalNEATRunner(agent_identity)
    
    @abstractmethod
    def create_fitness_evaluator(self) -> CapabilityFitnessEvaluator:
        """Create the fitness evaluator for this capability."""
        pass
    
    @abstractmethod
    def get_neat_config_params(self) -> Dict[str, Any]:
        """Get capability-specific NEAT configuration parameters."""
        pass
    
    def create_fitness_function(self):
        """Create fitness function for NEAT evolution."""
        def fitness_function(genomes, config):
            for genome_id, genome in genomes:
                try:
                    # Create network
                    network = neat.nn.FeedForwardNetwork.create(genome, config)
                    
                    # Evaluate capability
                    fitness = self.fitness_evaluator.evaluate_network(network)
                    
                    # Assign fitness
                    genome.fitness = fitness
                    
                except Exception as e:
                    # Handle network creation/evaluation errors
                    genome.fitness = 0.0
        
        return fitness_function
    
    def train_capability(self, generations: int = 50, **kwargs) -> Dict[str, Any]:
        """
        Train the agent to develop this capability.
        
        Args:
            generations: Number of evolutionary generations
            **kwargs: Capability-specific training parameters
            
        Returns:
            Training results with best network and performance metrics
        """
        print(f"Training {self.capability_type} capability for agent {self.agent_identity.id_hash[:12]}...")
        print(f"Data size: {self.training_data.data_size}")
        print(f"Constitutional traits influencing training:")
        
        # Show relevant traits for this capability
        relevant_traits = self.get_relevant_traits()
        for trait_name in relevant_traits:
            trait_value = self.agent_identity.constitution_result.constitution.get(trait_name, 'N/A')
            print(f"  - {trait_name}: {trait_value}")
        
        # Get capability-specific NEAT config
        config_params = self.get_neat_config_params()
        
        # Create config file (the runner already handles trait-based configuration)
        config_file = self.neat_runner.create_neat_config_file(f"{self.capability_type}_config.txt")
        
        # Create fitness function
        fitness_func = self.create_fitness_function()
        
        # Run evolution
        print(f"\\nStarting evolution for {generations} generations...")
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           config_file)
        
        population = neat.Population(config)
        population.add_reporter(neat.StdOutReporter(True))
        
        # Evolve
        best_genome = population.run(fitness_func, generations)
        best_network = neat.nn.FeedForwardNetwork.create(best_genome, config)
        
        # Evaluate final capability
        final_fitness = self.fitness_evaluator.evaluate_network(best_network, num_tests=200)
        
        # Generate sample outputs
        sample_generation = self.fitness_evaluator.evaluate_generation_quality(
            best_network, **kwargs
        )
        
        # Cleanup config file
        import os
        if os.path.exists(f"{self.capability_type}_config.txt"):
            os.remove(f"{self.capability_type}_config.txt")
        
        return {
            "agent_id": self.agent_identity.id_hash,
            "identity_bundle": self.agent_identity,
            "capability_type": self.capability_type,
            "final_fitness": final_fitness,
            "best_genome": best_genome,
            "best_network": best_network,
            "sample_generation": sample_generation.sample_outputs,
            "performance_metrics": sample_generation.performance_metrics,
            "training_generations": generations,
            "data_size": self.training_data.data_size
        }
    
    @abstractmethod
    def get_relevant_traits(self) -> List[str]:
        """Get list of constitutional traits most relevant to this capability."""
        pass


# Factory function for easy capability training
def train_agent_capability(capability_type: str, 
                         agent_identity: IdentityBundle,
                         training_data: Optional[Any] = None,
                         generations: int = 30,
                         **kwargs) -> Dict[str, Any]:
    """
    Convenience function to train an agent's capability.
    
    Args:
        capability_type: Type of capability to train
        agent_identity: Constitutional agent to train
        training_data: Domain-specific training data (uses default if None)
        generations: Training generations
        **kwargs: Capability-specific parameters
        
    Returns:
        Training results
    """
    # Import capability modules dynamically
    capability_modules = {
        'language': ('constitutional_ai.training.language_evolution', 'LanguageTrainingPipeline'),
        # Future capabilities can be added here:
        # 'coding': ('constitutional_ai.training.coding_evolution', 'CodingTrainingPipeline'),  
        # 'image': ('constitutional_ai.training.image_evolution', 'ImageTrainingPipeline'),
        # 'reasoning': ('constitutional_ai.training.reasoning_evolution', 'ReasoningTrainingPipeline'),
    }
    
    if capability_type not in capability_modules:
        raise ValueError(f"Unsupported capability type: {capability_type}")
    
    module_name, class_name = capability_modules[capability_type]
    
    # Dynamic import
    import importlib
    module = importlib.import_module(module_name)
    pipeline_class = getattr(module, class_name)
    
    # Create appropriate training data if not provided
    if training_data is None:
        if capability_type == 'language':
            from .language_evolution import create_language_training_corpus
            training_data = create_language_training_corpus()
        else:
            raise ValueError(f"No default training data available for {capability_type}")
    
    # Create and run pipeline
    pipeline = pipeline_class(training_data, agent_identity)
    return pipeline.train_capability(generations, **kwargs)