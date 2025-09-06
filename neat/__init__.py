"""
Web3 NEAT - NeuroEvolution of Augmenting Topologies for NFT AI Systems

A Python implementation of NEAT algorithm for evolving neural networks 
that can be minted as NFTs on the blockchain.
"""

__version__ = "1.0.0"
__author__ = "Web3 NEAT Team"

from .core.genome import Genome
from .core.species import Species
from .core.population import Population
from .evolution.neat_algorithm import NEATAlgorithm

__all__ = [
    "Genome",
    "Species", 
    "Population",
    "NEATAlgorithm"
]
