"""
Emergent Constitution Engine
Uses only allelic stabilization behaviors to create trait resolution.

No additional rules or layers - everything emerges from the fundamental
allelic building blocks and their recursive stabilization behaviors.
"""

import copy
import random
import math
from typing import Dict, Any, Optional, List
from .genome import ConstitutionalGenome, StabilizationType


class EmergentConstitutionEngine:
    """
    Constitution engine that works purely through allelic emergent behavior.
    
    No additional rules - traits evolve based purely on the stabilization
    behaviors embedded in their constituent alleles.
    """
    
    def __init__(self):
        """Initialize emergent engine with minimal configuration."""
        self.max_iterations = 50  # Simpler convergence
        self.convergence_tolerance = 1e-6
    
    def resolve_to_emergent_constitution(self, genome: ConstitutionalGenome, 
                                       seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Resolve genome to constitutional traits using only allelic behaviors.
        
        Args:
            genome: Constitutional genome with alleles containing stabilization types
            seed: Random seed for reproducible behavior
            
        Returns:
            Dictionary with resolved constitutional traits and metadata
        """
        if seed is not None:
            random.seed(seed)
        
        # Start with expressed traits from allelic dominance
        current_traits = genome.get_expressed_traits()
        trait_history = [copy.deepcopy(current_traits)]
        
        # Get stabilization behaviors for each trait from their dominant alleles
        trait_behaviors = self._extract_trait_behaviors(genome)
        
        # Iterate based on allelic stabilization behaviors
        for iteration in range(self.max_iterations):
            new_traits = copy.deepcopy(current_traits)
            
            # Apply each trait's allelic stabilization behavior
            for trait_name, behavior in trait_behaviors.items():
                if trait_name in new_traits:
                    new_traits[trait_name] = self._apply_allelic_behavior(
                        trait_name, 
                        new_traits[trait_name], 
                        behavior,
                        iteration
                    )
            
            # Check for convergence
            if self._traits_converged(current_traits, new_traits):
                return {
                    "constitution": new_traits,
                    "converged": True,
                    "iterations": iteration + 1,
                    "trait_behaviors": {name: behavior.value for name, behavior in trait_behaviors.items()},
                    "history": trait_history
                }
            
            current_traits = new_traits
            trait_history.append(copy.deepcopy(current_traits))
        
        # Max iterations reached
        return {
            "constitution": current_traits,
            "converged": False,
            "iterations": self.max_iterations,
            "trait_behaviors": {name: behavior.value for name, behavior in trait_behaviors.items()},
            "history": trait_history
        }
    
    def _extract_trait_behaviors(self, genome: ConstitutionalGenome) -> Dict[str, StabilizationType]:
        """Extract the stabilization behavior for each trait from its dominant allele."""
        trait_behaviors = {}
        
        for trait_name, locus in genome.loci.items():
            # Get the dominant allele which carries the stabilization behavior
            dominant_allele = locus.get_dominant_allele()
            trait_behaviors[trait_name] = dominant_allele.stabilization_type
        
        return trait_behaviors
    
    def _apply_allelic_behavior(self, trait_name: str, current_value: Any, 
                              behavior: StabilizationType, iteration: int) -> Any:
        """
        Apply the allelic stabilization behavior to a trait value.
        
        This is where the recursion happens - each allele's behavior
        creates the trait-level dynamics.
        """
        from .traits import COMPLETE_TRAIT_DEFINITIONS
        
        # Skip unknown traits
        if trait_name not in COMPLETE_TRAIT_DEFINITIONS:
            return current_value
        
        trait_def = COMPLETE_TRAIT_DEFINITIONS[trait_name]
        
        # Only numeric traits can have complex behaviors for now
        if trait_def.allele_type != "numeric":
            return current_value
        
        min_val, max_val = trait_def.domain
        
        # Apply behavior based on stabilization type
        if behavior == StabilizationType.STATIC:
            # No change - allele seeks stability
            return current_value
            
        elif behavior == StabilizationType.PROGRESSIVE:
            # Small improvement toward maximum
            improvement = min(0.02, (max_val - current_value) * 0.05)
            return min(max_val, current_value + improvement)
            
        elif behavior == StabilizationType.OSCILLATORY:
            # Oscillate around current value
            amplitude = min(0.1, (max_val - min_val) * 0.02)
            phase = iteration * 0.3  # Phase based on iteration
            oscillation = amplitude * math.sin(phase)
            new_value = current_value + oscillation
            return max(min_val, min(max_val, new_value))
            
        elif behavior == StabilizationType.DEGENERATIVE:
            # Slow decay toward minimum
            decay = min(0.01, current_value * 0.02)
            return max(min_val, current_value - decay)
            
        elif behavior == StabilizationType.CHAOTIC:
            # Bounded chaotic behavior
            chaos_factor = (random.random() - 0.5) * 0.1
            new_value = current_value + chaos_factor
            return max(min_val, min(max_val, new_value))
            
        elif behavior == StabilizationType.MULTI_ATTRACTOR:
            # Switch between different target values based on context
            # For simplicity, oscillate between different stable points
            if iteration % 10 < 5:
                target = min_val + (max_val - min_val) * 0.3
            else:
                target = min_val + (max_val - min_val) * 0.7
            
            # Move slowly toward current target
            direction = 1 if target > current_value else -1
            step = min(0.05, abs(target - current_value) * 0.1)
            return current_value + (direction * step)
        
        return current_value
    
    def _traits_converged(self, traits1: Dict[str, Any], traits2: Dict[str, Any]) -> bool:
        """Check if traits have converged within tolerance."""
        if set(traits1.keys()) != set(traits2.keys()):
            return False
        
        for key in traits1:
            val1, val2 = traits1[key], traits2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                if abs(val1 - val2) > self.convergence_tolerance:
                    return False
            elif val1 != val2:
                return False
        
        return True


def create_emergent_agent_identity(genome: ConstitutionalGenome,
                                 seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Create agent identity using purely emergent constitution from allelic behaviors.
    
    Args:
        genome: Constitutional genome with stabilization behaviors in alleles
        seed: Random seed for reproducible constitution
        
    Returns:
        Dictionary with emergent constitutional traits and metadata
    """
    engine = EmergentConstitutionEngine()
    return engine.resolve_to_emergent_constitution(genome, seed)
