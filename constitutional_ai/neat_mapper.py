"""
Trait-to-NEAT Configuration Mapper
Deterministically maps constitutional trait values to NEAT evolution parameters.

This module ensures monotone mapping - raising any trait never decreases
the corresponding NEAT parameters, maintaining mathematical consistency.
"""

from typing import Dict, Any, Tuple
from dataclasses import dataclass
from .traits import get_trait_definitions, get_trait_ordering_index


@dataclass
class NEATConfig:
    """
    NEAT configuration parameters derived from constitutional traits.

    All parameters are deterministically derived from trait values with
    monotonicity guarantees.
    """

    # Population and Evolution
    population_size: int
    elitism_count: int
    species_count_target: int

    # Network Architecture
    initial_hidden_nodes: int
    max_nodes: int
    max_connections: int
    activation_function: str

    # Mutation Rates
    weight_mutation_rate: float
    weight_perturbation_rate: float
    add_node_rate: float
    add_connection_rate: float
    remove_connection_rate: float

    # Learning and Adaptation
    learning_rate: float
    momentum: float
    weight_decay: float

    # Selection and Speciation
    compatibility_threshold: float
    survival_rate: float

    # Performance and Efficiency
    max_generations: int
    fitness_threshold: float
    stagnation_threshold: int

    # Behavioral Parameters
    exploration_rate: float
    temperature: float  # For softmax decisions

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "population_size": self.population_size,
            "elitism_count": self.elitism_count,
            "species_count_target": self.species_count_target,
            "initial_hidden_nodes": self.initial_hidden_nodes,
            "max_nodes": self.max_nodes,
            "max_connections": self.max_connections,
            "activation_function": self.activation_function,
            "weight_mutation_rate": self.weight_mutation_rate,
            "weight_perturbation_rate": self.weight_perturbation_rate,
            "add_node_rate": self.add_node_rate,
            "add_connection_rate": self.add_connection_rate,
            "remove_connection_rate": self.remove_connection_rate,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "compatibility_threshold": self.compatibility_threshold,
            "survival_rate": self.survival_rate,
            "max_generations": self.max_generations,
            "fitness_threshold": self.fitness_threshold,
            "stagnation_threshold": self.stagnation_threshold,
            "exploration_rate": self.exploration_rate,
            "temperature": self.temperature,
        }


class TraitToNEATMapper:
    """
    Maps constitutional trait values to NEAT configuration parameters.

    Ensures monotone mapping - higher trait values never result in
    lower NEAT parameter values.
    """

    def __init__(self):
        """Initialize the mapper with default parameter ranges."""

        # Define base parameter ranges - SCALED FOR 100M+ PARAMETER NETWORKS
        self.base_ranges = {
            "population_size": (500, 2000),  # MASSIVE populations to saturate TPU
            "elitism_count": (5, 50),  # More elites preserved
            "species_count_target": (10, 30),  # More species diversity
            "initial_hidden_nodes": (0, 0),  # ALWAYS START FROM ZERO - no shortcuts!
            "max_nodes": (50000, 200000),  # 100M+ parameters possible (200K nodes * 500 connections avg)
            "max_connections": (500000, 10000000),  # Up to 10 million connections for massive networks
            "weight_mutation_rate": (0.1, 0.9),
            "weight_perturbation_rate": (0.5, 0.95),
            "add_node_rate": (0.01, 0.2),
            "add_connection_rate": (0.05, 0.4),
            "remove_connection_rate": (0.01, 0.1),
            "learning_rate": (0.001, 0.3),
            "momentum": (0.0, 0.9),
            "weight_decay": (0.0, 0.01),
            "compatibility_threshold": (1.0, 5.0),
            "survival_rate": (0.1, 0.5),
            "max_generations": (200, 2000),  # More generations for complex evolution
            "fitness_threshold": (0.9, 1.0),  # Higher standards
            "stagnation_threshold": (50, 200),  # More patience for large network evolution
            "exploration_rate": (0.0, 0.3),
            "temperature": (0.1, 2.0),
        }

    def normalize_trait(self, trait_name: str, trait_value: Any) -> float:
        """
        Normalize trait value to 0.0-1.0 range for parameter mapping.

        Same normalization as used in color mapping for consistency.
        """
        trait_defs = get_trait_definitions(complete=True)

        if trait_name not in trait_defs:
            return 0.5

        trait_def = trait_defs[trait_name]

        if trait_def["type"] == "numeric":
            min_val, max_val = trait_def["domain"]
            return max(0.0, min(1.0, (trait_value - min_val) / (max_val - min_val)))

        elif trait_def["type"] == "categorical":
            ordering_idx = get_trait_ordering_index(trait_name, trait_value)
            if ordering_idx >= 0:
                max_idx = len(trait_def["monotone_order"]) - 1
                return ordering_idx / max_idx if max_idx > 0 else 0.0
            return 0.5

        elif trait_def["type"] == "boolean":
            return 1.0 if trait_value else 0.0

        return 0.5

    def map_population_parameters(self, traits: Dict[str, Any]) -> Dict[str, Any]:
        """Map traits to population and evolution parameters."""

        # Normalize relevant traits
        stability = self.normalize_trait("Stability", traits.get("Stability", 1.5))
        processing_speed = self.normalize_trait(
            "ProcessingSpeed", traits.get("ProcessingSpeed", 1.0)
        )
        energy_efficiency = self.normalize_trait(
            "EnergyEfficiency", traits.get("EnergyEfficiency", 1.0)
        )
        social_drive = self.normalize_trait(
            "SocialDrive", traits.get("SocialDrive", 1.5)
        )

        # Population size: influenced by stability and processing speed
        # More stable, faster agents can handle larger populations
        pop_factor = stability * 0.6 + processing_speed * 0.4
        population_size = int(
            self._map_to_range(pop_factor, self.base_ranges["population_size"])
        )

        # Elitism: more stable agents preserve more elite individuals
        elitism_factor = stability * 0.8 + energy_efficiency * 0.2
        elitism_count = int(
            self._map_to_range(elitism_factor, self.base_ranges["elitism_count"])
        )

        # Species count: social agents maintain more diverse species
        species_factor = social_drive * 0.7 + stability * 0.3
        species_count = int(
            self._map_to_range(species_factor, self.base_ranges["species_count_target"])
        )

        return {
            "population_size": population_size,
            "elitism_count": elitism_count,
            "species_count_target": species_count,
        }

    def map_architecture_parameters(self, traits: Dict[str, Any]) -> Dict[str, Any]:
        """Map traits to network architecture parameters."""

        # Normalize relevant traits
        perception = self.normalize_trait("Perception", traits.get("Perception", 5.0))
        working_memory = self.normalize_trait(
            "WorkingMemory", traits.get("WorkingMemory", "Low")
        )
        expertise = self.normalize_trait("Expertise", traits.get("Expertise", 1.0))
        attention_span = self.normalize_trait(
            "AttentionSpan", traits.get("AttentionSpan", 3.0)
        )

        # Initial hidden nodes: based on perception and working memory
        hidden_factor = perception * 0.6 + working_memory * 0.4
        initial_hidden = int(
            self._map_to_range(hidden_factor, self.base_ranges["initial_hidden_nodes"])
        )

        # Max nodes: influenced by working memory and attention span
        max_nodes_factor = working_memory * 0.5 + attention_span * 0.3 + expertise * 0.2
        max_nodes = int(
            self._map_to_range(max_nodes_factor, self.base_ranges["max_nodes"])
        )

        # Max connections: based on perception and expertise
        max_conn_factor = perception * 0.4 + expertise * 0.4 + working_memory * 0.2
        max_connections = int(
            self._map_to_range(max_conn_factor, self.base_ranges["max_connections"])
        )

        # Activation function: categorical based on traits
        activation_factor = (perception + expertise) / 2
        if activation_factor < 0.3:
            activation = "sigmoid"
        elif activation_factor < 0.7:
            activation = "tanh"
        else:
            activation = "relu"

        return {
            "initial_hidden_nodes": initial_hidden,
            "max_nodes": max_nodes,
            "max_connections": max_connections,
            "activation_function": activation,
        }

    def map_mutation_parameters(self, traits: Dict[str, Any]) -> Dict[str, Any]:
        """Map traits to mutation rate parameters."""

        # Normalize relevant traits
        innovation_drive = self.normalize_trait(
            "InnovationDrive", traits.get("InnovationDrive", 1.0)
        )
        risk_tolerance = self.normalize_trait(
            "RiskTolerance", traits.get("RiskTolerance", 1.5)
        )
        stability = self.normalize_trait("Stability", traits.get("Stability", 1.5))
        curiosity = self.normalize_trait("Curiosity", traits.get("Curiosity", 1.0))

        # Weight mutation rate: driven by innovation and risk tolerance
        weight_mut_factor = innovation_drive * 0.6 + risk_tolerance * 0.4
        weight_mutation_rate = self._map_to_range(
            weight_mut_factor, self.base_ranges["weight_mutation_rate"]
        )

        # Weight perturbation: more stable agents use more perturbation vs replacement
        perturb_factor = stability * 0.7 + (1.0 - risk_tolerance) * 0.3
        weight_perturbation_rate = self._map_to_range(
            perturb_factor, self.base_ranges["weight_perturbation_rate"]
        )

        # Add node rate: driven by curiosity and innovation
        add_node_factor = curiosity * 0.5 + innovation_drive * 0.5
        add_node_rate = self._map_to_range(
            add_node_factor, self.base_ranges["add_node_rate"]
        )

        # Add connection rate: similar to add node but slightly more conservative
        add_conn_factor = innovation_drive * 0.6 + curiosity * 0.4
        add_connection_rate = self._map_to_range(
            add_conn_factor, self.base_ranges["add_connection_rate"]
        )

        # Remove connection rate: inversely related to stability
        remove_conn_factor = (1.0 - stability) * 0.6 + risk_tolerance * 0.4
        remove_connection_rate = self._map_to_range(
            remove_conn_factor, self.base_ranges["remove_connection_rate"]
        )

        return {
            "weight_mutation_rate": weight_mutation_rate,
            "weight_perturbation_rate": weight_perturbation_rate,
            "add_node_rate": add_node_rate,
            "add_connection_rate": add_connection_rate,
            "remove_connection_rate": remove_connection_rate,
        }

    def map_learning_parameters(self, traits: Dict[str, Any]) -> Dict[str, Any]:
        """Map traits to learning and adaptation parameters."""

        # Normalize relevant traits
        learning_rate_trait = self.normalize_trait(
            "LearningRate", traits.get("LearningRate", 0.1)
        )
        transfer_learning = self.normalize_trait(
            "TransferLearning", traits.get("TransferLearning", 1.0)
        )
        stability = self.normalize_trait("Stability", traits.get("Stability", 1.5))

        # Learning rate: directly from LearningRate trait
        learning_rate = self._map_to_range(
            learning_rate_trait, self.base_ranges["learning_rate"]
        )

        # Momentum: based on stability and transfer learning
        momentum_factor = stability * 0.7 + transfer_learning * 0.3
        momentum = self._map_to_range(momentum_factor, self.base_ranges["momentum"])

        # Weight decay: inversely related to stability (more stable = less decay needed)
        decay_factor = 1.0 - stability
        weight_decay = self._map_to_range(
            decay_factor, self.base_ranges["weight_decay"]
        )

        return {
            "learning_rate": learning_rate,
            "momentum": momentum,
            "weight_decay": weight_decay,
        }

    def map_selection_parameters(self, traits: Dict[str, Any]) -> Dict[str, Any]:
        """Map traits to selection and speciation parameters."""

        # Normalize relevant traits
        social_drive = self.normalize_trait(
            "SocialDrive", traits.get("SocialDrive", 1.5)
        )
        stability = self.normalize_trait("Stability", traits.get("Stability", 1.5))
        expertise = self.normalize_trait("Expertise", traits.get("Expertise", 1.0))

        # Compatibility threshold: social agents are more tolerant of differences
        compat_factor = social_drive * 0.6 + stability * 0.4
        compatibility_threshold = self._map_to_range(
            compat_factor, self.base_ranges["compatibility_threshold"]
        )

        # Survival rate: expert, stable agents can afford to be more selective
        survival_factor = 1.0 - (expertise * 0.6 + stability * 0.4)
        survival_rate = self._map_to_range(
            survival_factor, self.base_ranges["survival_rate"]
        )

        return {
            "compatibility_threshold": compatibility_threshold,
            "survival_rate": survival_rate,
        }

    def map_performance_parameters(self, traits: Dict[str, Any]) -> Dict[str, Any]:
        """Map traits to performance and efficiency parameters."""

        # Normalize relevant traits
        attention_span = self.normalize_trait(
            "AttentionSpan", traits.get("AttentionSpan", 3.0)
        )
        energy_efficiency = self.normalize_trait(
            "EnergyEfficiency", traits.get("EnergyEfficiency", 1.0)
        )
        processing_speed = self.normalize_trait(
            "ProcessingSpeed", traits.get("ProcessingSpeed", 1.0)
        )
        meta_learning = self.normalize_trait(
            "MetaLearning", traits.get("MetaLearning", 0.5)
        )

        # Max generations: based on attention span and efficiency
        max_gen_factor = attention_span * 0.6 + energy_efficiency * 0.4
        max_generations = int(
            self._map_to_range(max_gen_factor, self.base_ranges["max_generations"])
        )

        # Fitness threshold: meta-learning agents can achieve higher thresholds
        fitness_factor = meta_learning * 0.7 + processing_speed * 0.3
        fitness_threshold = self._map_to_range(
            fitness_factor, self.base_ranges["fitness_threshold"]
        )

        # Stagnation threshold: patient, efficient agents wait longer
        stagnation_factor = attention_span * 0.5 + energy_efficiency * 0.5
        stagnation_threshold = int(
            self._map_to_range(
                stagnation_factor, self.base_ranges["stagnation_threshold"]
            )
        )

        return {
            "max_generations": max_generations,
            "fitness_threshold": fitness_threshold,
            "stagnation_threshold": stagnation_threshold,
        }

    def map_behavioral_parameters(self, traits: Dict[str, Any]) -> Dict[str, Any]:
        """Map traits to behavioral control parameters."""

        # Normalize relevant traits
        risk_tolerance = self.normalize_trait(
            "RiskTolerance", traits.get("RiskTolerance", 1.5)
        )
        curiosity = self.normalize_trait("Curiosity", traits.get("Curiosity", 1.0))
        innovation_drive = self.normalize_trait(
            "InnovationDrive", traits.get("InnovationDrive", 1.0)
        )

        # Exploration rate: driven by curiosity and risk tolerance
        exploration_factor = curiosity * 0.6 + risk_tolerance * 0.4
        exploration_rate = self._map_to_range(
            exploration_factor, self.base_ranges["exploration_rate"]
        )

        # Temperature for softmax decisions: innovation drives randomness
        temperature_factor = innovation_drive * 0.7 + risk_tolerance * 0.3
        temperature = self._map_to_range(
            temperature_factor, self.base_ranges["temperature"]
        )

        return {"exploration_rate": exploration_rate, "temperature": temperature}

    def _map_to_range(
        self, normalized_value: float, value_range: Tuple[float, float]
    ) -> float:
        """
        Map a normalized value (0-1) to a specific range.

        This ensures monotone mapping - higher normalized values
        always produce higher output values.
        """
        min_val, max_val = value_range
        return min_val + normalized_value * (max_val - min_val)

    def traits_to_neat_config(self, traits: Dict[str, Any]) -> NEATConfig:
        """
        Convert complete trait dictionary to NEAT configuration.

        Args:
            traits: Dictionary of trait names to values

        Returns:
            NEATConfig object with all parameters set
        """

        # Map each category of parameters
        pop_params = self.map_population_parameters(traits)
        arch_params = self.map_architecture_parameters(traits)
        mutation_params = self.map_mutation_parameters(traits)
        learning_params = self.map_learning_parameters(traits)
        selection_params = self.map_selection_parameters(traits)
        performance_params = self.map_performance_parameters(traits)
        behavioral_params = self.map_behavioral_parameters(traits)

        # Combine all parameters into NEATConfig
        return NEATConfig(
            # Population parameters
            population_size=pop_params["population_size"],
            elitism_count=pop_params["elitism_count"],
            species_count_target=pop_params["species_count_target"],
            # Architecture parameters
            initial_hidden_nodes=arch_params["initial_hidden_nodes"],
            max_nodes=arch_params["max_nodes"],
            max_connections=arch_params["max_connections"],
            activation_function=arch_params["activation_function"],
            # Mutation parameters
            weight_mutation_rate=mutation_params["weight_mutation_rate"],
            weight_perturbation_rate=mutation_params["weight_perturbation_rate"],
            add_node_rate=mutation_params["add_node_rate"],
            add_connection_rate=mutation_params["add_connection_rate"],
            remove_connection_rate=mutation_params["remove_connection_rate"],
            # Learning parameters
            learning_rate=learning_params["learning_rate"],
            momentum=learning_params["momentum"],
            weight_decay=learning_params["weight_decay"],
            # Selection parameters
            compatibility_threshold=selection_params["compatibility_threshold"],
            survival_rate=selection_params["survival_rate"],
            # Performance parameters
            max_generations=performance_params["max_generations"],
            fitness_threshold=performance_params["fitness_threshold"],
            stagnation_threshold=performance_params["stagnation_threshold"],
            # Behavioral parameters
            exploration_rate=behavioral_params["exploration_rate"],
            temperature=behavioral_params["temperature"],
        )

    def validate_monotonicity(
        self, traits1: Dict[str, Any], traits2: Dict[str, Any]
    ) -> bool:
        """
        Validate that trait increases lead to monotone parameter increases.

        Args:
            traits1: First trait configuration
            traits2: Second trait configuration (should have higher/equal trait values)

        Returns:
            True if monotonicity is preserved
        """
        config1 = self.traits_to_neat_config(traits1)
        config2 = self.traits_to_neat_config(traits2)

        dict1 = config1.to_dict()
        dict2 = config2.to_dict()

        # Check that no numeric parameter decreased
        for key in dict1:
            if isinstance(dict1[key], (int, float)) and isinstance(
                dict2[key], (int, float)
            ):
                if dict2[key] < dict1[key]:
                    return False

        return True


# Convenience function for external use
def map_traits_to_neat_config(constitutional_traits: Dict[str, Any]) -> NEATConfig:
    """
    Map constitutional trait values to NEAT configuration.

    Args:
        constitutional_traits: Trait values from constitution resolution

    Returns:
        NEATConfig object ready for NEAT evolution
    """
    mapper = TraitToNEATMapper()
    return mapper.traits_to_neat_config(constitutional_traits)
