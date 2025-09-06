"""
Population class for NEAT algorithm - manages a collection of genomes.
"""

from typing import List, Dict, Callable
import random
from .genome import Genome
from .species import Species


class Population:
    """
    NEAT Population class that manages evolution of a collection of genomes.
    
    Each genome represents a neural network that can be evolved and potentially
    minted as an NFT based on its performance and uniqueness.
    """
    
    def __init__(self, size: int, input_size: int, output_size: int):
        self.size = size
        self.input_size = input_size
        self.output_size = output_size
        self.genomes: List[Genome] = []
        self.species: Dict[int, Species] = {}
        self.generation = 0
        self.global_innovation_number = output_size  # Start after initial nodes
        self.species_counter = 0
        
        # NEAT parameters
        self.compatibility_threshold = 3.0
        self.survival_rate = 0.25
        self.mutation_rates = {
            'add_node': 0.03,
            'add_connection': 0.05,
            'mutate_weights': 0.8
        }
        
        # Initialize population
        self._initialize_population()
    
    def _initialize_population(self):
        """Initialize the population with random genomes."""
        for _ in range(self.size):
            genome = Genome(self.input_size, self.output_size)
            
            # Add some initial random connections
            for _ in range(random.randint(1, 5)):
                self.global_innovation_number = genome.add_connection_mutation(
                    self.global_innovation_number
                )
            
            self.genomes.append(genome)
    
    def evaluate_fitness(self, fitness_function: Callable[[Genome], float]):
        """Evaluate fitness for all genomes in the population."""
        for genome in self.genomes:
            genome.fitness = fitness_function(genome)
    
    def speciate(self):
        """Divide population into species based on compatibility."""
        # Clear existing species assignments
        for species in self.species.values():
            species.genomes.clear()
        
        # Assign genomes to species
        for genome in self.genomes:
            assigned = False
            
            # Try to assign to existing species
            for species_id, species in self.species.items():
                if species.representative and self._compatibility_distance(
                    genome, species.representative
                ) < self.compatibility_threshold:
                    species.add_genome(genome)
                    genome.species_id = species_id
                    assigned = True
                    break
            
            # Create new species if not assigned
            if not assigned:
                new_species = Species(self.species_counter)
                new_species.add_genome(genome)
                genome.species_id = self.species_counter
                self.species[self.species_counter] = new_species
                self.species_counter += 1
        
        # Remove empty species
        empty_species = [sid for sid, species in self.species.items() 
                        if len(species.genomes) == 0]
        for sid in empty_species:
            del self.species[sid]
    
    def _compatibility_distance(self, genome1: Genome, genome2: Genome) -> float:
        """Calculate compatibility distance between two genomes."""
        # Simplified compatibility function
        # In full implementation, would consider excess, disjoint, and weight differences
        
        # Get all connection innovation numbers
        conn1 = set(genome1.connections.keys())
        conn2 = set(genome2.connections.keys())
        
        if not conn1 and not conn2:
            return 0.0
        
        # Calculate disjoint and excess connections
        all_innovations = conn1.union(conn2)
        matching = conn1.intersection(conn2)
        disjoint_excess = len(all_innovations) - len(matching)
        
        # Calculate average weight difference for matching connections
        weight_diff = 0.0
        if matching:
            weight_diff = sum(abs(genome1.connections[inn].weight - 
                                genome2.connections[inn].weight) 
                            for inn in matching) / len(matching)
        
        # Compatibility formula (simplified)
        c1, c2, c3 = 1.0, 1.0, 0.4
        N = max(len(conn1), len(conn2), 1)
        
        return (c1 * disjoint_excess / N) + (c3 * weight_diff)
    
    def evolve_generation(self):
        """Evolve the population for one generation."""
        self.generation += 1
        
        # Speciate the population
        self.speciate()
        
        # Calculate adjusted fitness and select parents
        self._adjust_fitness()
        
        # Generate offspring
        new_genomes = []
        
        for species in self.species.values():
            if len(species.genomes) > 0:
                # Calculate number of offspring for this species
                species_size = max(1, int(len(species.genomes) * self.survival_rate))
                
                # Sort by fitness and keep the best
                species.genomes.sort(key=lambda g: g.fitness, reverse=True)
                survivors = species.genomes[:species_size]
                
                # Generate offspring
                for _ in range(len(species.genomes)):
                    if random.random() < 0.75 and len(survivors) >= 2:
                        # Crossover
                        parent1 = random.choice(survivors)
                        parent2 = random.choice(survivors)
                        offspring = self._crossover(parent1, parent2)
                    else:
                        # Asexual reproduction (copy best)
                        offspring = self._copy_genome(survivors[0])
                    
                    # Mutate offspring
                    self._mutate(offspring)
                    new_genomes.append(offspring)
        
        # Replace population
        self.genomes = new_genomes[:self.size]
    
    def _adjust_fitness(self):
        """Adjust fitness based on species size (fitness sharing)."""
        for species in self.species.values():
            if len(species.genomes) > 0:
                for genome in species.genomes:
                    genome.fitness = genome.fitness / len(species.genomes)
    
    def _crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        """Create offspring through crossover of two parents."""
        # Choose the more fit parent as primary
        if parent1.fitness >= parent2.fitness:
            primary, secondary = parent1, parent2
        else:
            primary, secondary = parent2, parent1
        
        # Create offspring genome
        offspring = Genome(self.input_size, self.output_size)
        
        # Inherit nodes from primary parent
        offspring.nodes = {k: v for k, v in primary.nodes.items()}
        
        # Inherit connections
        offspring.connections = {}
        for inn, conn in primary.connections.items():
            if inn in secondary.connections:
                # Matching gene - randomly choose from either parent
                if random.random() < 0.5:
                    offspring.connections[inn] = conn
                else:
                    offspring.connections[inn] = secondary.connections[inn]
            else:
                # Excess/disjoint gene - inherit from more fit parent
                offspring.connections[inn] = conn
        
        return offspring
    
    def _copy_genome(self, genome: Genome) -> Genome:
        """Create a copy of a genome."""
        copy = Genome(genome.input_size, genome.output_size)
        copy.nodes = {k: v for k, v in genome.nodes.items()}
        copy.connections = {k: v for k, v in genome.connections.items()}
        copy.fitness = genome.fitness
        return copy
    
    def _mutate(self, genome: Genome):
        """Apply mutations to a genome."""
        # Add node mutation
        if random.random() < self.mutation_rates['add_node']:
            self.global_innovation_number = genome.add_node_mutation(
                self.global_innovation_number
            )
        
        # Add connection mutation
        if random.random() < self.mutation_rates['add_connection']:
            self.global_innovation_number = genome.add_connection_mutation(
                self.global_innovation_number
            )
        
        # Weight mutations
        if random.random() < self.mutation_rates['mutate_weights']:
            genome.mutate_weights()
    
    def get_best_genome(self) -> Genome:
        """Get the genome with highest fitness."""
        return max(self.genomes, key=lambda g: g.fitness)
    
    def get_species_champions(self) -> List[Genome]:
        """Get the best genome from each species."""
        champions = []
        for species in self.species.values():
            if species.genomes:
                champion = max(species.genomes, key=lambda g: g.fitness)
                champions.append(champion)
        return champions
