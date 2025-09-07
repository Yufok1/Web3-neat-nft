"""
Constitutional AI Breeding System
Revolutionary Web3 platform for evolving complete AI agents through genetic algorithms.

Key Components:
- Diploid genome system with 6 stabilization types at allelic level
- Emergent constitution engine using Kleene fixed points
- True Mendelian breeding with crossover and mutation
- Immutable identity with cryptographic verification
- Integration with NEAT neural evolution
- Visual DNA system for unique NFT artwork
"""

from .genome import ConstitutionalGenome, create_random_genome
from .traits import COMPLETE_TRAIT_DEFINITIONS
from .breeder import ConstitutionalBreeder
from .identity import create_agent_identity
from .emergent_constitution import create_emergent_agent_identity

__version__ = "1.0.0"
__all__ = [
    "ConstitutionalGenome",
    "create_random_genome", 
    "COMPLETE_TRAIT_DEFINITIONS",
    "ConstitutionalBreeder",
    "create_agent_identity",
    "create_emergent_agent_identity"
]