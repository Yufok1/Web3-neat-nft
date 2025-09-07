"""
Constitutional Trait System for NEAT Breeding
Defines the finite ordered domains for the 5 starter traits.

This module provides the trait definitions and domain specifications that serve
as the foundation for Kleene fixed-point trait closure and NEAT configuration mapping.
"""

from typing import Dict, Any, List, Tuple
from enum import Enum
from dataclasses import dataclass


class TraitLevel(Enum):
    """Standardized trait level representations."""
    UNKNOWN = "Unknown"
    MINIMAL = "Minimal" 
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    MAXIMUM = "Maximum"


@dataclass
class TraitDomain:
    """
    Specification for a trait's domain and characteristics.
    
    Defines the valid range/options for a trait and metadata for
    trait resolution and NEAT mapping.
    """
    name: str
    description: str
    allele_type: str  # "numeric", "categorical", "boolean"
    domain: Any  # Range tuple for numeric, list for categorical
    default_value: Any
    fixed_points: List[Any]  # Known stable values (Kleene fixed points)
    monotone_order: List[Any]  # Ordered from lowest to highest capability
    

# Define comprehensive AI system traits
COMPLETE_TRAIT_DEFINITIONS: Dict[str, TraitDomain] = {
    
    # Core Cognitive Abilities
    "Perception": TraitDomain(
        name="Perception",
        description="Ability to process and interpret sensory/input information",
        allele_type="numeric",
        domain=(2.0, 10.0),  # Scale from basic to highly sophisticated perception
        default_value=5.0,
        fixed_points=[2.0, 3.5, 5.0, 7.0, 8.5, 10.0],
        monotone_order=[2.0, 3.5, 5.0, 7.0, 8.5, 10.0]
    ),
    
    "WorkingMemory": TraitDomain(
        name="WorkingMemory", 
        description="Capacity to retain and manipulate information during processing",
        allele_type="categorical",
        domain=(TraitLevel.UNKNOWN.value, TraitLevel.MINIMAL.value, 
                TraitLevel.LOW.value, TraitLevel.MODERATE.value, 
                TraitLevel.HIGH.value),
        default_value=TraitLevel.LOW.value,
        fixed_points=[TraitLevel.UNKNOWN.value, TraitLevel.MINIMAL.value,
                     TraitLevel.LOW.value, TraitLevel.MODERATE.value, 
                     TraitLevel.HIGH.value],
        monotone_order=[TraitLevel.UNKNOWN.value, TraitLevel.MINIMAL.value,
                       TraitLevel.LOW.value, TraitLevel.MODERATE.value, 
                       TraitLevel.HIGH.value]
    ),
    
    "Expertise": TraitDomain(
        name="Expertise",
        description="Domain-specific knowledge and specialized competence",
        allele_type="numeric", 
        domain=(0.0, 3.0),  # 0=novice, 1=competent, 2=proficient, 3=expert
        default_value=1.0,
        fixed_points=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        monotone_order=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    ),
    
    # Learning & Adaptation
    "LearningRate": TraitDomain(
        name="LearningRate",
        description="Speed of adaptation to new information and experiences",
        allele_type="numeric",
        domain=(0.001, 1.0),  # Very slow to very fast learning
        default_value=0.1,
        fixed_points=[0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0],
        monotone_order=[0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
    ),
    
    "TransferLearning": TraitDomain(
        name="TransferLearning",
        description="Ability to apply knowledge across different domains",
        allele_type="numeric",
        domain=(0.0, 3.0),  # 0=domain-specific, 3=highly generalizable
        default_value=1.0,
        fixed_points=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        monotone_order=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    ),
    
    # Behavioral Control
    "AttentionSpan": TraitDomain(
        name="AttentionSpan",
        description="Duration and focus of sustained attention on tasks",
        allele_type="numeric",
        domain=(0.1, 10.0),  # Very scattered to extremely focused
        default_value=3.0,
        fixed_points=[0.1, 0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
        monotone_order=[0.1, 0.5, 1.0, 3.0, 5.0, 7.0, 10.0]
    ),
    
    "RiskTolerance": TraitDomain(
        name="RiskTolerance",
        description="Willingness to take risks vs preference for safe choices",
        allele_type="numeric",
        domain=(0.0, 3.0),  # 0=very conservative, 3=high risk-taking
        default_value=1.5,
        fixed_points=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        monotone_order=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    ),
    
    "InnovationDrive": TraitDomain(
        name="InnovationDrive", 
        description="Tendency to explore novel solutions and take creative risks",
        allele_type="numeric",
        domain=(0.0, 3.0),  # 0=conventional, 3=highly creative
        default_value=1.0,
        fixed_points=[0.0, 0.75, 1.5, 2.25, 3.0],
        monotone_order=[0.0, 0.75, 1.5, 2.25, 3.0]
    ),
    
    # System Performance
    "ProcessingSpeed": TraitDomain(
        name="ProcessingSpeed",
        description="Raw computational throughput and response time",
        allele_type="numeric",
        domain=(0.1, 5.0),  # Very slow to very fast processing
        default_value=1.0,
        fixed_points=[0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0],
        monotone_order=[0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0]
    ),
    
    "Stability": TraitDomain(
        name="Stability",
        description="Consistency and robustness of performance across contexts", 
        allele_type="numeric",
        domain=(0.0, 3.0),  # 0=chaotic, 3=highly consistent
        default_value=1.5,
        fixed_points=[0.0, 0.6, 1.2, 1.8, 2.4, 3.0],
        monotone_order=[0.0, 0.6, 1.2, 1.8, 2.4, 3.0]
    ),
    
    "EnergyEfficiency": TraitDomain(
        name="EnergyEfficiency",
        description="Resource optimization and computational efficiency",
        allele_type="numeric",
        domain=(0.1, 3.0),  # Very wasteful to highly efficient
        default_value=1.0,
        fixed_points=[0.1, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0],
        monotone_order=[0.1, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0]
    ),
    
    # Social & Communication
    "SocialDrive": TraitDomain(
        name="SocialDrive",
        description="Tendency toward collaborative vs independent behavior",
        allele_type="numeric",
        domain=(0.0, 3.0),  # 0=highly independent, 3=highly collaborative
        default_value=1.5,
        fixed_points=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        monotone_order=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    ),
    
    "CommunicationStyle": TraitDomain(
        name="CommunicationStyle",
        description="Approach to information sharing and interaction",
        allele_type="categorical",
        domain=("Minimal", "Technical", "Balanced", "Expressive", "Verbose"),
        default_value="Balanced",
        fixed_points=["Minimal", "Technical", "Balanced", "Expressive", "Verbose"],
        monotone_order=["Minimal", "Technical", "Balanced", "Expressive", "Verbose"]
    ),
    
    # Advanced Capabilities
    "MetaLearning": TraitDomain(
        name="MetaLearning",
        description="Ability to learn how to learn more effectively",
        allele_type="numeric",
        domain=(0.0, 3.0),  # 0=no meta-learning, 3=highly adaptive learning
        default_value=0.5,
        fixed_points=[0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0],
        monotone_order=[0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0]
    ),
    
    "Curiosity": TraitDomain(
        name="Curiosity",
        description="Drive to explore and understand new information",
        allele_type="numeric",
        domain=(0.0, 3.0),  # 0=no curiosity, 3=extremely curious
        default_value=1.0,
        fixed_points=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        monotone_order=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    )
}

# Keep the original 5 traits as a subset for compatibility
STARTER_TRAIT_DEFINITIONS: Dict[str, TraitDomain] = {
    k: v for k, v in COMPLETE_TRAIT_DEFINITIONS.items() 
    if k in ["Perception", "WorkingMemory", "Expertise", "InnovationDrive", "Stability"]
}


def get_trait_definitions(complete: bool = False) -> Dict[str, dict]:
    """
    Get trait definitions in format suitable for genome creation.
    
    Args:
        complete: If True, return all traits. If False, return only starter traits.
    
    Returns dictionary mapping trait names to their specifications.
    """
    definitions = {}
    
    trait_set = COMPLETE_TRAIT_DEFINITIONS if complete else STARTER_TRAIT_DEFINITIONS
    
    for name, trait_domain in trait_set.items():
        definitions[name] = {
            "type": trait_domain.allele_type,
            "domain": trait_domain.domain,
            "description": trait_domain.description,
            "default_value": trait_domain.default_value,
            "fixed_points": trait_domain.fixed_points,
            "monotone_order": trait_domain.monotone_order
        }
    
    return definitions


def validate_trait_value(trait_name: str, value: Any) -> bool:
    """
    Validate that a trait value is within its defined domain.
    
    Args:
        trait_name: Name of the trait
        value: Value to validate
        
    Returns:
        True if value is valid for the trait domain
    """
    if trait_name not in STARTER_TRAIT_DEFINITIONS:
        return False
        
    trait_domain = STARTER_TRAIT_DEFINITIONS[trait_name]
    
    if trait_domain.allele_type == "numeric":
        min_val, max_val = trait_domain.domain
        return isinstance(value, (int, float)) and min_val <= value <= max_val
        
    elif trait_domain.allele_type == "categorical":
        return value in trait_domain.domain
        
    elif trait_domain.allele_type == "boolean":
        return isinstance(value, bool)
    
    return False


def get_trait_ordering_index(trait_name: str, value: Any) -> int:
    """
    Get the monotone ordering index for a trait value.
    
    This is used to ensure monotone mapping to NEAT parameters.
    
    Args:
        trait_name: Name of the trait
        value: Trait value
        
    Returns:
        Index in the monotone ordering, or -1 if value not found
    """
    if trait_name not in STARTER_TRAIT_DEFINITIONS:
        return -1
    
    trait_domain = STARTER_TRAIT_DEFINITIONS[trait_name]
    
    try:
        # For numeric traits, find closest value in monotone order
        if trait_domain.allele_type == "numeric":
            closest_idx = 0
            min_diff = abs(value - trait_domain.monotone_order[0])
            
            for i, ordered_val in enumerate(trait_domain.monotone_order):
                diff = abs(value - ordered_val)
                if diff < min_diff:
                    min_diff = diff
                    closest_idx = i
            
            return closest_idx
        
        # For categorical traits, exact match
        else:
            return trait_domain.monotone_order.index(value)
            
    except (ValueError, IndexError):
        return -1


def get_specialized_starter_genomes() -> Dict[str, Dict[str, dict]]:
    """
    Get genome specifications for specialized starter AIs.
    
    Each starter is optimized for 1-2 traits while being weak in others,
    encouraging breeding to create well-rounded agents.
    
    Returns:
        Dictionary mapping starter names to their genome specifications
    """
    starters = {
        "Perceptor": {
            "description": "Highly perceptive but lacks memory and stability",
            "specializations": ["Perception"],
            "trait_values": {
                "Perception": 9.0,           # High perception
                "WorkingMemory": TraitLevel.MINIMAL.value,  # Poor memory
                "Expertise": 0.5,            # Low expertise
                "InnovationDrive": 0.75,     # Conservative
                "Stability": 0.6             # Unstable
            }
        },
        
        "Scholar": {
            "description": "Expert with excellent memory but poor innovation",
            "specializations": ["Expertise", "WorkingMemory"], 
            "trait_values": {
                "Perception": 4.0,           # Average perception
                "WorkingMemory": TraitLevel.HIGH.value,     # Excellent memory
                "Expertise": 2.5,            # High expertise
                "InnovationDrive": 0.0,      # Very conservative
                "Stability": 2.4             # Very stable
            }
        },
        
        "Innovator": {
            "description": "Highly creative but unstable and inexperienced", 
            "specializations": ["InnovationDrive"],
            "trait_values": {
                "Perception": 6.0,           # Good perception
                "WorkingMemory": TraitLevel.MODERATE.value, # Decent memory
                "Expertise": 0.0,            # Novice level
                "InnovationDrive": 3.0,      # Maximum creativity
                "Stability": 0.0             # Very chaotic
            }
        },
        
        "Stabilizer": {
            "description": "Highly consistent but lacks innovation and expertise",
            "specializations": ["Stability"],
            "trait_values": {
                "Perception": 5.0,           # Average perception
                "WorkingMemory": TraitLevel.MODERATE.value, # Decent memory
                "Expertise": 1.0,            # Basic competence
                "InnovationDrive": 0.0,      # Very conservative
                "Stability": 3.0             # Maximum stability
            }
        },
        
        "Generalist": {
            "description": "Balanced but not exceptional in any area",
            "specializations": [],  # No specialization
            "trait_values": {
                "Perception": 5.0,           # Average
                "WorkingMemory": TraitLevel.LOW.value,      # Below average memory
                "Expertise": 1.0,            # Basic
                "InnovationDrive": 1.5,      # Balanced
                "Stability": 1.2             # Slightly below average
            }
        },
        
        "Savant": {
            "description": "Incredible perception and expertise but poor social traits",
            "specializations": ["Perception", "Expertise"],
            "trait_values": {
                "Perception": 10.0,          # Maximum perception
                "WorkingMemory": TraitLevel.UNKNOWN.value,  # Unpredictable memory
                "Expertise": 3.0,            # Maximum expertise  
                "InnovationDrive": 2.25,     # High innovation
                "Stability": 0.6             # Unstable
            }
        }
    }
    
    return starters


def create_starter_genome_definitions() -> Dict[str, dict]:
    """
    Create genome definitions for all starter AIs.
    
    Returns:
        Dictionary suitable for creating ConstitutionalGenomes
    """
    trait_defs = get_trait_definitions()
    starters = get_specialized_starter_genomes()
    
    genome_definitions = {}
    
    for starter_name, starter_spec in starters.items():
        genome_def = {
            "description": starter_spec["description"],
            "specializations": starter_spec["specializations"],
            "loci": {}
        }
        
        # Create loci for each trait
        for trait_name, trait_def in trait_defs.items():
            # Get the specialized value or default
            trait_value = starter_spec["trait_values"].get(trait_name, trait_def["default_value"])
            
            # For simplicity, make both alleles the same value (homozygous)
            # This ensures predictable expression of the designed traits
            genome_def["loci"][trait_name] = {
                "name": trait_name,
                "maternal_allele": {
                    "value": trait_value,
                    "allele_type": trait_def["type"],
                    "domain": trait_def["domain"]
                },
                "paternal_allele": {
                    "value": trait_value, 
                    "allele_type": trait_def["type"],
                    "domain": trait_def["domain"]
                }
            }
        
        genome_definitions[starter_name] = genome_def
    
    return genome_definitions
