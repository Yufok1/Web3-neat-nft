"""
Constitutional NEAT Integration
Bridges our constitutional system with neat-python for complete AI agent evolution.
"""

import neat
import pickle
import os
from typing import Dict, Any, Optional, Callable

from .genome import ConstitutionalGenome
from .identity import create_agent_identity, IdentityBundle


class ConstitutionalNEATRunner:
    """
    NEAT runner that uses constitutional trait-derived configurations.

    This class bridges our constitutional system with the neat-python library,
    allowing agents evolved through constitutional breeding to be further
    evolved as neural networks using NEAT.
    """

    def __init__(self, identity_bundle: IdentityBundle):
        """
        Initialize NEAT runner with constitutional agent identity.

        Args:
            identity_bundle: Complete agent identity with constitutional traits
        """
        self.identity = identity_bundle
        self.neat_config = identity_bundle.neat_config
        self.config_file = None
        self.population = None
        self.best_genome = None

    def create_neat_config_file(self, filename: str = "neat_config.txt") -> str:
        """
        Create a NEAT configuration file from constitutional traits.

        Args:
            filename: Name for the config file

        Returns:
            Path to the created config file
        """
        config_content = f"""[NEAT]
fitness_criterion     = max
fitness_threshold     = 100.0
pop_size              = {self.neat_config.population_size}
reset_on_extinction   = False

[DefaultGenome]
# node activation options
activation_default      = {self.neat_config.activation_function}
activation_mutate_rate  = 0.1
activation_options      = tanh sigmoid relu

# node aggregation options
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

# node bias options
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

# genome compatibility options
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

# connection add/remove rates
conn_add_prob           = {self.neat_config.add_connection_rate}
conn_delete_prob        = {self.neat_config.remove_connection_rate}

# connection enable options
enabled_default         = True
enabled_mutate_rate     = 0.01

feed_forward            = True
initial_connection      = full_nodirect

# node add/remove rates
node_add_prob           = {self.neat_config.add_node_rate}
node_delete_prob        = 0.0

# network parameters
num_hidden              = {self.neat_config.initial_hidden_nodes}
num_inputs              = 4
num_outputs             = 2

# node response options
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

# connection weight options
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate      = {self.neat_config.weight_mutation_rate}
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = {self.neat_config.compatibility_threshold}

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = {self.neat_config.elitism_count}

[DefaultReproduction]
elitism            = {self.neat_config.elitism_count}
survival_threshold = {self.neat_config.survival_rate}
"""

        config_path = os.path.abspath(filename)
        with open(config_path, "w") as f:
            f.write(config_content)

        self.config_file = config_path
        return config_path

    def evolve(
        self,
        fitness_function: Callable,
        generations: int = 100,
        num_inputs: int = 4,
        num_outputs: int = 2,
    ) -> Any:
        """
        Evolve neural networks using NEAT with constitutional configuration.

        Args:
            fitness_function: Function that evaluates genome fitness
            generations: Number of generations to evolve
            num_inputs: Number of input nodes
            num_outputs: Number of output nodes

        Returns:
            Best evolved genome
        """
        if not self.config_file:
            self.create_neat_config_file()

        # Update config file with correct input/output counts
        self._update_config_io(num_inputs, num_outputs)

        # Create NEAT configuration
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            self.config_file,
        )

        # Create population
        self.population = neat.Population(config)

        # Add reporters for monitoring
        self.population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        self.population.add_reporter(stats)

        # Run evolution
        winner = self.population.run(fitness_function, generations)
        self.best_genome = winner

        return winner

    def _update_config_io(self, num_inputs: int, num_outputs: int):
        """Update config file with correct input/output node counts."""
        with open(self.config_file, "r") as f:
            content = f.read()

        # Replace input/output counts
        content = content.replace(
            "num_inputs              = 4", f"num_inputs              = {num_inputs}"
        )
        content = content.replace(
            "num_outputs             = 2", f"num_outputs             = {num_outputs}"
        )

        with open(self.config_file, "w") as f:
            f.write(content)

    def create_network(self) -> Any:
        """Create a neural network from the best evolved genome."""
        if self.best_genome is None:
            raise ValueError("No evolved genome available. Run evolve() first.")

        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            self.config_file,
        )

        return neat.nn.FeedForwardNetwork.create(self.best_genome, config)

    def save_network(self, filename: str):
        """Save the evolved network to file."""
        if self.best_genome is None:
            raise ValueError("No evolved network to save. Run evolve() first.")

        with open(filename, "wb") as f:
            pickle.dump((self.best_genome, self.config_file), f)

    def run_evolution(
        self, fitness_function: Callable, generations: int = 100
    ) -> Dict[str, Any]:
        """
        Simplified interface for running evolution with default I/O sizes.

        Args:
            fitness_function: Function that evaluates genome fitness
            generations: Number of generations to evolve

        Returns:
            Evolution results with best genome and statistics
        """
        # Use default I/O sizes (can be overridden by fitness function)
        winner = self.evolve(fitness_function, generations, num_inputs=4, num_outputs=2)

        return {
            "best_genome": winner,
            "fitness": winner.fitness if winner else 0,
            "generations": generations,
            "config_file": self.config_file,
        }

    def cleanup(self):
        """Clean up temporary config file."""
        if self.config_file and os.path.exists(self.config_file):
            os.remove(self.config_file)


def evolve_constitutional_agent(
    genome: ConstitutionalGenome,
    fitness_function: Callable,
    generations: int = 50,
    num_inputs: int = 4,
    num_outputs: int = 2,
    seed_closure: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Complete pipeline: constitutional genome -> NEAT evolution -> trained agent.

    This is the main entry point for evolving complete AI agents that combine
    constitutional trait-based configuration with NEAT neural evolution.

    Args:
        genome: Constitutional genome with trait-based configuration
        fitness_function: Function to evaluate network fitness
        generations: Number of evolution generations
        num_inputs: Number of network inputs
        num_outputs: Number of network outputs
        seed_closure: Seed for constitutional resolution

    Returns:
        Dictionary with evolved agent and metadata
    """
    # Step 1: Create agent identity with constitutional traits
    identity = create_agent_identity(genome, seed_closure=seed_closure)

    # Step 2: Create NEAT runner with constitutional configuration
    neat_runner = ConstitutionalNEATRunner(identity)

    # Step 3: Evolve neural network using constitutional parameters
    best_genome = neat_runner.evolve(
        fitness_function, generations, num_inputs, num_outputs
    )

    # Step 4: Create trained network
    network = neat_runner.create_network()

    # Step 5: Return complete evolved agent
    result = {
        "identity": identity,
        "constitutional_genome": genome,
        "neat_genome": best_genome,
        "network": network,
        "neat_runner": neat_runner,
        "generations_evolved": generations,
        "final_fitness": getattr(best_genome, "fitness", None),
    }

    return result


def breed_and_evolve_agents(
    parent1: ConstitutionalGenome,
    parent2: ConstitutionalGenome,
    fitness_function: Callable,
    breeding_seed: Optional[int] = None,
    evolution_generations: int = 50,
) -> Dict[str, Any]:
    """
    Complete pipeline: breed constitutional genomes -> evolve neural networks.

    Args:
        parent1: First parent constitutional genome
        parent2: Second parent constitutional genome
        fitness_function: Function to evaluate neural network fitness
        breeding_seed: Seed for deterministic breeding
        evolution_generations: Number of NEAT evolution generations

    Returns:
        Dictionary with bred and evolved agent
    """
    from .breeder import ConstitutionalBreeder

    # Step 1: Breed the constitutional genomes
    breeder = ConstitutionalBreeder()
    breeding_result = breeder.breed_agents(parent1, parent2, seed=breeding_seed)
    offspring_genome = breeding_result.offspring

    # Step 2: Evolve the offspring with NEAT
    evolution_result = evolve_constitutional_agent(
        genome=offspring_genome,
        fitness_function=fitness_function,
        generations=evolution_generations,
        seed_closure=breeding_seed,
    )

    # Step 3: Combine results
    result = {
        **evolution_result,
        "breeding_result": breeding_result,
        "parent1_hash": parent1.compute_genome_hash(),
        "parent2_hash": parent2.compute_genome_hash(),
    }

    return result
