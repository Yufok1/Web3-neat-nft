"""
Species class for NEAT algorithm - groups similar genomes together.
"""

from typing import List, Optional
from .genome import Genome


class Species:
    """
    NEAT Species class that groups genetically similar genomes together.
    
    This helps maintain diversity in the population and prevents premature
    convergence to local optima.
    """
    
    def __init__(self, species_id: int):
        self.species_id = species_id
        self.genomes: List[Genome] = []
        self.representative: Optional[Genome] = None
        self.average_fitness = 0.0
        self.max_fitness = 0.0
        self.generations_without_improvement = 0
        self.age = 0
    
    def add_genome(self, genome: Genome):
        """Add a genome to this species."""
        self.genomes.append(genome)
        
        # Set representative if this is the first genome
        if not self.representative:
            self.representative = genome
    
    def update_fitness_stats(self):
        """Update fitness statistics for this species."""
        if not self.genomes:
            self.average_fitness = 0.0
            self.max_fitness = 0.0
            return
        
        fitnesses = [genome.fitness for genome in self.genomes]
        old_max_fitness = self.max_fitness
        
        self.average_fitness = sum(fitnesses) / len(fitnesses)
        self.max_fitness = max(fitnesses)
        
        # Update stagnation counter
        if self.max_fitness <= old_max_fitness:
            self.generations_without_improvement += 1
        else:
            self.generations_without_improvement = 0
        
        self.age += 1
    
    def select_representative(self):
        """Select a new representative genome for this species."""
        if self.genomes:
            # Choose the best genome as representative
            self.representative = max(self.genomes, key=lambda g: g.fitness)
    
    def is_stagnant(self, stagnation_threshold: int = 20) -> bool:
        """Check if this species has been stagnant for too long."""
        return self.generations_without_improvement >= stagnation_threshold
    
    def clear(self):
        """Clear all genomes from this species."""
        self.genomes.clear()
        self.representative = None
    
    def __len__(self) -> int:
        """Return the number of genomes in this species."""
        return len(self.genomes)
    
    def __repr__(self) -> str:
        """String representation of the species."""
        return f"Species(id={self.species_id}, size={len(self.genomes)}, avg_fitness={self.average_fitness:.2f})"
