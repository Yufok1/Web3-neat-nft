"""
Constitutional Breeder Service
Implements Mendelian breeding with crossover, mutation, linkage, and complete provenance tracking.

This module provides the breeding operations for creating new AI agents through
genetic recombination while maintaining mathematical consistency and deterministic results.
"""

import random
import copy
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from .genome import ConstitutionalGenome, Locus, Allele, AlleleType, ProvenanceRecord
from .traits import get_trait_definitions, validate_trait_value
from .identity import create_agent_identity, IdentityBundle


@dataclass
class BreedingConfig:
    """
    Configuration parameters for breeding operations.
    
    Controls mutation rates, linkage behavior, and other genetic parameters.
    """
    crossover_rate: float = 0.8      # Probability of crossover vs cloning
    mutation_rate: float = 0.1       # Per-allele mutation probability
    linkage_respect_rate: float = 0.9  # Probability of respecting linkage groups
    trait_drift_rate: float = 0.05   # Small random changes to trait values
    max_mutation_delta: float = 0.1  # Maximum change for numeric trait mutations
    
    def to_dict(self) -> dict:
        """Convert breeding config to dictionary."""
        return {
            'crossover_rate': self.crossover_rate,
            'mutation_rate': self.mutation_rate,
            'linkage_respect_rate': self.linkage_respect_rate,
            'trait_drift_rate': self.trait_drift_rate,
            'max_mutation_delta': self.max_mutation_delta
        }


@dataclass
class BreedingResult:
    """
    Result of a breeding operation.
    
    Contains the offspring genome and metadata about the breeding process.
    """
    offspring: ConstitutionalGenome
    crossover_points: List[str]      # Which loci underwent crossover
    mutations_applied: List[str]     # Which loci were mutated
    linkage_breaks: List[str]        # Where linkage groups were broken
    breeding_method: str             # 'crossover', 'clone', or 'random'
    
    def to_dict(self) -> dict:
        """Convert breeding result to dictionary."""
        return {
            'offspring': self.offspring.to_dict(),
            'crossover_points': self.crossover_points,
            'mutations_applied': self.mutations_applied,
            'linkage_breaks': self.linkage_breaks,
            'breeding_method': self.breeding_method
        }


class ConstitutionalBreeder:
    """
    Service for breeding constitutional AI agents through Mendelian genetics.
    
    Implements proper diploid crossover, mutation, linkage respect, and
    complete provenance tracking for deterministic breeding.
    """
    
    def __init__(self, config: Optional[BreedingConfig] = None):
        """
        Initialize the breeder with configuration.
        
        Args:
            config: Breeding configuration parameters
        """
        self.config = config or BreedingConfig()
        self.trait_definitions = get_trait_definitions(complete=True)
    
    def breed_agents(self, 
                    parent1: ConstitutionalGenome, 
                    parent2: ConstitutionalGenome,
                    seed: Optional[int] = None) -> BreedingResult:
        """
        Breed two parent genomes to create offspring.
        
        Args:
            parent1: First parent genome
            parent2: Second parent genome  
            seed: Random seed for deterministic breeding
            
        Returns:
            BreedingResult with offspring and metadata
        """
        if seed is not None:
            random.seed(seed)
        
        # Decide breeding method
        if random.random() < self.config.crossover_rate:
            return self._crossover_breeding(parent1, parent2, seed)
        else:
            # Clone one parent with possible mutations
            parent = random.choice([parent1, parent2])
            return self._clone_with_mutation(parent, seed)
    
    def _crossover_breeding(self, 
                          parent1: ConstitutionalGenome, 
                          parent2: ConstitutionalGenome,
                          seed: Optional[int]) -> BreedingResult:
        """
        Breed through genetic crossover (recombination).
        
        Implements proper Mendelian inheritance with linkage groups.
        """
        offspring = ConstitutionalGenome()
        crossover_points = []
        mutations_applied = []
        linkage_breaks = []
        
        # Get all trait names that exist in both parents
        common_traits = set(parent1.loci.keys()) & set(parent2.loci.keys())
        all_traits = set(parent1.loci.keys()) | set(parent2.loci.keys())
        
        # Group traits by linkage groups
        linkage_groups = self._organize_by_linkage(parent1, parent2)
        
        # Perform crossover for each linkage group
        for group_id, trait_names in linkage_groups.items():
            if self._should_respect_linkage(group_id):
                # Inherit entire linkage group from one parent
                donor_parent = random.choice([parent1, parent2])
                for trait_name in trait_names:
                    if trait_name in donor_parent.loci:
                        offspring.loci[trait_name] = copy.deepcopy(donor_parent.loci[trait_name])
            else:
                # Break linkage - crossover within the group
                linkage_breaks.extend(trait_names)
                for trait_name in trait_names:
                    if trait_name in common_traits:
                        new_locus = self._crossover_locus(
                            parent1.loci[trait_name], 
                            parent2.loci[trait_name]
                        )
                        offspring.loci[trait_name] = new_locus
                        crossover_points.append(trait_name)
        
        # Handle traits not in linkage groups (independent assortment)
        unlinked_traits = [t for t in all_traits 
                          if not any(t in group for group in linkage_groups.values())]
        
        for trait_name in unlinked_traits:
            if trait_name in common_traits:
                new_locus = self._crossover_locus(
                    parent1.loci[trait_name], 
                    parent2.loci[trait_name]
                )
                offspring.loci[trait_name] = new_locus
                crossover_points.append(trait_name)
            else:
                # Trait exists in only one parent
                parent = parent1 if trait_name in parent1.loci else parent2
                offspring.loci[trait_name] = copy.deepcopy(parent.loci[trait_name])
        
        # Apply mutations
        mutations_applied = self._apply_mutations(offspring)
        
        # Set provenance
        offspring.provenance = ProvenanceRecord(
            parents=[parent1.compute_genome_hash(), parent2.compute_genome_hash()],
            operation="crossover_breeding",
            generation=max(parent1.provenance.generation, parent2.provenance.generation) + 1,
            timestamp=datetime.utcnow().isoformat(),
            seed_used=seed
        )
        
        return BreedingResult(
            offspring=offspring,
            crossover_points=crossover_points,
            mutations_applied=mutations_applied,
            linkage_breaks=linkage_breaks,
            breeding_method="crossover"
        )
    
    def _clone_with_mutation(self, parent: ConstitutionalGenome, seed: Optional[int]) -> BreedingResult:
        """
        Create offspring by cloning parent with possible mutations.
        """
        offspring = parent.clone()
        mutations_applied = self._apply_mutations(offspring)
        
        # Update provenance
        offspring.provenance = ProvenanceRecord(
            parents=[parent.compute_genome_hash()],
            operation="clone_with_mutation",
            generation=parent.provenance.generation + 1,
            timestamp=datetime.utcnow().isoformat(),
            seed_used=seed
        )
        
        return BreedingResult(
            offspring=offspring,
            crossover_points=[],
            mutations_applied=mutations_applied,
            linkage_breaks=[],
            breeding_method="clone"
        )
    
    def _organize_by_linkage(self, 
                           parent1: ConstitutionalGenome, 
                           parent2: ConstitutionalGenome) -> Dict[int, List[str]]:
        """
        Organize traits by their linkage groups.
        
        Returns dictionary mapping linkage_group_id to list of trait names.
        """
        linkage_groups = {}
        
        # Collect linkage groups from both parents
        for parent in [parent1, parent2]:
            for trait_name, locus in parent.loci.items():
                if locus.linkage_group is not None:
                    if locus.linkage_group not in linkage_groups:
                        linkage_groups[locus.linkage_group] = []
                    if trait_name not in linkage_groups[locus.linkage_group]:
                        linkage_groups[locus.linkage_group].append(trait_name)
        
        return linkage_groups
    
    def _should_respect_linkage(self, linkage_group_id: Optional[int]) -> bool:
        """
        Determine if linkage should be respected for a given group.
        
        Returns True if the entire linkage group should be inherited together.
        """
        if linkage_group_id is None:
            return False
        
        return random.random() < self.config.linkage_respect_rate
    
    def _crossover_locus(self, locus1: Locus, locus2: Locus) -> Locus:
        """
        Perform crossover between two loci to create a new locus.
        
        Randomly selects maternal and paternal alleles from the two parents.
        """
        # Randomly select maternal allele from either parent
        if random.random() < 0.5:
            maternal_source = locus1
            paternal_source = locus2
        else:
            maternal_source = locus2
            paternal_source = locus1
        
        # Randomly select which allele from each source
        maternal_allele = random.choice([maternal_source.maternal_allele, maternal_source.paternal_allele])
        paternal_allele = random.choice([paternal_source.maternal_allele, paternal_source.paternal_allele])
        
        # Create new locus
        new_locus = Locus(
            name=locus1.name,
            maternal_allele=copy.deepcopy(maternal_allele),
            paternal_allele=copy.deepcopy(paternal_allele),
            linkage_group=locus1.linkage_group or locus2.linkage_group
        )
        
        return new_locus
    
    def _apply_mutations(self, genome: ConstitutionalGenome) -> List[str]:
        """
        Apply random mutations to a genome.
        
        Returns list of trait names that were mutated.
        """
        mutated_traits = []
        
        for trait_name, locus in genome.loci.items():
            # Check if this locus should be mutated
            if random.random() < self.config.mutation_rate:
                # Randomly choose maternal or paternal allele to mutate
                if random.random() < 0.5:
                    self._mutate_allele(locus.maternal_allele, trait_name)
                else:
                    self._mutate_allele(locus.paternal_allele, trait_name)
                
                mutated_traits.append(trait_name)
        
        return mutated_traits
    
    def _mutate_allele(self, allele: Allele, trait_name: str):
        """
        Mutate a single allele in place.
        
        Uses trait-appropriate mutation strategies.
        """
        if trait_name not in self.trait_definitions:
            return  # Can't mutate unknown traits
        
        trait_def = self.trait_definitions[trait_name]
        
        if trait_def["type"] == "numeric":
            self._mutate_numeric_allele(allele, trait_def)
        elif trait_def["type"] == "categorical":
            self._mutate_categorical_allele(allele, trait_def)
        elif trait_def["type"] == "boolean":
            self._mutate_boolean_allele(allele)
    
    def _mutate_numeric_allele(self, allele: Allele, trait_def: dict):
        """Mutate a numeric allele."""
        min_val, max_val = trait_def["domain"]
        current_val = allele.value
        
        # Apply small random change
        delta = random.uniform(-self.config.max_mutation_delta, self.config.max_mutation_delta)
        new_val = current_val + delta
        
        # Clamp to valid range
        new_val = max(min_val, min(max_val, new_val))
        
        # Update allele value (alleles are frozen, so we need to work around this)
        object.__setattr__(allele, 'value', new_val)
    
    def _mutate_categorical_allele(self, allele: Allele, trait_def: dict):
        """Mutate a categorical allele."""
        current_val = allele.value
        possible_values = list(trait_def["domain"])
        
        # Remove current value and pick randomly from remaining
        if current_val in possible_values:
            possible_values.remove(current_val)
        
        if possible_values:
            new_val = random.choice(possible_values)
            object.__setattr__(allele, 'value', new_val)
    
    def _mutate_boolean_allele(self, allele: Allele):
        """Mutate a boolean allele."""
        new_val = not allele.value
        object.__setattr__(allele, 'value', new_val)
    
    def create_random_population(self, 
                                size: int,
                                trait_set: str = "complete",
                                seed: Optional[int] = None) -> List[ConstitutionalGenome]:
        """
        Create a random population of genomes for initialization.
        
        Args:
            size: Number of genomes to create
            trait_set: "complete" for all 14 traits, "starter" for original 5
            seed: Random seed for deterministic generation
            
        Returns:
            List of randomly initialized genomes
        """
        if seed is not None:
            random.seed(seed)
        
        population = []
        trait_defs = get_trait_definitions(complete=(trait_set == "complete"))
        
        for i in range(size):
            genome = self._create_random_genome(trait_defs, seed + i if seed else None)
            genome.provenance = ProvenanceRecord(
                parents=[],
                operation="random_initialization",
                generation=0,
                timestamp=datetime.utcnow().isoformat(),
                seed_used=seed + i if seed else None
            )
            population.append(genome)
        
        return population
    
    def _create_random_genome(self, trait_defs: Dict[str, dict], seed: Optional[int]) -> ConstitutionalGenome:
        """Create a single random genome from trait definitions."""
        from .genome import create_random_genome
        return create_random_genome(trait_defs, seed)
    
    def breed_population(self,
                        parents: List[ConstitutionalGenome],
                        offspring_count: int,
                        seed: Optional[int] = None) -> List[BreedingResult]:
        """
        Breed a new generation from a parent population.
        
        Args:
            parents: List of parent genomes
            offspring_count: Number of offspring to create
            seed: Random seed for deterministic breeding
            
        Returns:
            List of breeding results
        """
        if seed is not None:
            random.seed(seed)
        
        if len(parents) < 2:
            raise ValueError("Need at least 2 parents for breeding")
        
        offspring = []
        
        for i in range(offspring_count):
            # Select two parents (can be the same for self-breeding)
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            
            # Breed them
            child_seed = seed + i if seed else None
            result = self.breed_agents(parent1, parent2, child_seed)
            offspring.append(result)
        
        return offspring
    
    def create_specialized_starters(self, seed: Optional[int] = None) -> Dict[str, IdentityBundle]:
        """
        Create the specialized starter AI agents.
        
        Args:
            seed: Random seed for deterministic creation
            
        Returns:
            Dictionary mapping starter names to their identity bundles
        """
        from .traits import create_starter_genome_definitions
        from .genome import ConstitutionalGenome, Locus, Allele, AlleleType
        
        if seed is not None:
            random.seed(seed)
        
        starter_definitions = create_starter_genome_definitions()
        starter_agents = {}
        
        for starter_name, genome_def in starter_definitions.items():
            # Create genome from definition
            genome = ConstitutionalGenome()
            
            for locus_name, locus_data in genome_def["loci"].items():
                maternal_allele = Allele(
                    value=locus_data["maternal_allele"]["value"],
                    allele_type=AlleleType(locus_data["maternal_allele"]["allele_type"]),
                    domain=locus_data["maternal_allele"]["domain"]
                )
                
                paternal_allele = Allele(
                    value=locus_data["paternal_allele"]["value"], 
                    allele_type=AlleleType(locus_data["paternal_allele"]["allele_type"]),
                    domain=locus_data["paternal_allele"]["domain"]
                )
                
                locus = Locus(
                    name=locus_name,
                    maternal_allele=maternal_allele,
                    paternal_allele=paternal_allele
                )
                
                genome.add_locus(locus)
            
            # Set provenance
            genome.provenance = ProvenanceRecord(
                parents=[],
                operation="starter_creation",
                generation=0,
                timestamp=datetime.utcnow().isoformat(),
                seed_used=seed
            )
            
            # Create identity bundle
            identity = create_agent_identity(genome, seed, seed)
            starter_agents[starter_name] = identity
        
        return starter_agents
    
    def get_breeding_statistics(self, results: List[BreedingResult]) -> Dict[str, Any]:
        """
        Get statistics about a set of breeding results.
        
        Args:
            results: List of breeding results to analyze
            
        Returns:
            Dictionary with breeding statistics
        """
        if not results:
            return {}
        
        total_results = len(results)
        crossover_count = sum(1 for r in results if r.breeding_method == "crossover")
        clone_count = sum(1 for r in results if r.breeding_method == "clone")
        
        total_mutations = sum(len(r.mutations_applied) for r in results)
        total_crossovers = sum(len(r.crossover_points) for r in results)
        total_linkage_breaks = sum(len(r.linkage_breaks) for r in results)
        
        return {
            'total_offspring': total_results,
            'crossover_rate': crossover_count / total_results if total_results > 0 else 0,
            'clone_rate': clone_count / total_results if total_results > 0 else 0,
            'avg_mutations_per_offspring': total_mutations / total_results if total_results > 0 else 0,
            'avg_crossovers_per_offspring': total_crossovers / total_results if total_results > 0 else 0,
            'avg_linkage_breaks_per_offspring': total_linkage_breaks / total_results if total_results > 0 else 0,
            'total_mutations': total_mutations,
            'total_crossovers': total_crossovers,
            'total_linkage_breaks': total_linkage_breaks
        }


# Convenience functions for external use
def breed_two_agents(parent1: ConstitutionalGenome, 
                    parent2: ConstitutionalGenome,
                    config: Optional[BreedingConfig] = None,
                    seed: Optional[int] = None) -> BreedingResult:
    """
    Convenience function to breed two agents.
    
    Args:
        parent1: First parent genome
        parent2: Second parent genome  
        config: Breeding configuration
        seed: Random seed
        
    Returns:
        BreedingResult with offspring
    """
    breeder = ConstitutionalBreeder(config)
    return breeder.breed_agents(parent1, parent2, seed)


def create_initial_population(size: int = 100,
                             trait_set: str = "complete", 
                             seed: Optional[int] = None) -> List[ConstitutionalGenome]:
    """
    Create initial population for evolution.
    
    Args:
        size: Population size
        trait_set: "complete" or "starter" trait set
        seed: Random seed
        
    Returns:
        List of random genomes
    """
    breeder = ConstitutionalBreeder()
    return breeder.create_random_population(size, trait_set, seed)
