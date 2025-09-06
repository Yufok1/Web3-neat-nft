"""
Genome class for NEAT algorithm - represents a neural network genotype.
Each genome can be evolved and eventually minted as an NFT.
"""

import random
from typing import Dict, List, Set, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class NodeGene:
    """Represents a node in the neural network genome."""
    innovation_number: int
    node_type: str  # 'input', 'hidden', 'output'
    activation: str = 'relu'


@dataclass  
class ConnectionGene:
    """Represents a connection between nodes in the neural network genome."""
    innovation_number: int
    input_node: int
    output_node: int
    weight: float
    enabled: bool = True


class Genome:
    """
    NEAT Genome class representing the genotype of a neural network.
    
    This genome can be evolved through mutations and crossover, and eventually
    serialized for storage on blockchain as NFT metadata.
    """
    
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        self.nodes: Dict[int, NodeGene] = {}
        self.connections: Dict[int, ConnectionGene] = {}
        self.fitness: float = 0.0
        self.species_id: int = -1
        
        # Initialize input and output nodes
        self._initialize_nodes()
        
    def _initialize_nodes(self):
        """Initialize input and output nodes."""
        node_id = 0
        
        # Input nodes
        for i in range(self.input_size):
            self.nodes[node_id] = NodeGene(node_id, 'input')
            node_id += 1
            
        # Output nodes
        for i in range(self.output_size):
            self.nodes[node_id] = NodeGene(node_id, 'output')
            node_id += 1
    
    def add_node_mutation(self, global_innovation_number: int) -> int:
        """Add a new hidden node by splitting an existing connection."""
        if not self.connections:
            return global_innovation_number
            
        # Choose random connection to split
        conn = random.choice(list(self.connections.values()))
        if not conn.enabled:
            return global_innovation_number
            
        # Disable the original connection
        conn.enabled = False
        
        # Add new hidden node
        new_node_id = max(self.nodes.keys()) + 1
        self.nodes[new_node_id] = NodeGene(new_node_id, 'hidden')
        
        # Add two new connections
        # Connection from input to new node (weight = 1.0)
        self.connections[global_innovation_number] = ConnectionGene(
            global_innovation_number, conn.input_node, new_node_id, 1.0, True
        )
        global_innovation_number += 1
        
        # Connection from new node to output (original weight)
        self.connections[global_innovation_number] = ConnectionGene(
            global_innovation_number, new_node_id, conn.output_node, conn.weight, True
        )
        global_innovation_number += 1
        
        return global_innovation_number
    
    def add_connection_mutation(self, global_innovation_number: int) -> int:
        """Add a new random connection between nodes."""
        nodes = list(self.nodes.keys())
        
        # Try multiple times to find a valid connection
        for _ in range(100):
            input_node = random.choice(nodes)
            output_node = random.choice(nodes)
            
            # Check if connection is valid and doesn't already exist
            if self._is_valid_connection(input_node, output_node):
                weight = random.uniform(-2.0, 2.0)
                self.connections[global_innovation_number] = ConnectionGene(
                    global_innovation_number, input_node, output_node, weight, True
                )
                return global_innovation_number + 1
                
        return global_innovation_number
    
    def _is_valid_connection(self, input_node: int, output_node: int) -> bool:
        """Check if a connection between two nodes is valid."""
        # No self-connections
        if input_node == output_node:
            return False
            
        # No connections from output nodes
        if self.nodes[input_node].node_type == 'output':
            return False
            
        # No connections to input nodes  
        if self.nodes[output_node].node_type == 'input':
            return False
            
        # Check if connection already exists
        for conn in self.connections.values():
            if conn.input_node == input_node and conn.output_node == output_node:
                return False
                
        return True
    
    def mutate_weights(self, mutation_rate: float = 0.1, perturbation_rate: float = 0.9):
        """Mutate connection weights."""
        for conn in self.connections.values():
            if random.random() < mutation_rate:
                if random.random() < perturbation_rate:
                    # Perturb existing weight
                    conn.weight += random.uniform(-0.5, 0.5)
                else:
                    # Assign new random weight
                    conn.weight = random.uniform(-2.0, 2.0)
    
    def to_dict(self) -> dict:
        """Convert genome to dictionary for serialization."""
        return {
            'input_size': self.input_size,
            'output_size': self.output_size,
            'nodes': {k: {'innovation_number': v.innovation_number, 
                         'node_type': v.node_type, 
                         'activation': v.activation} 
                     for k, v in self.nodes.items()},
            'connections': {k: {'innovation_number': v.innovation_number,
                               'input_node': v.input_node,
                               'output_node': v.output_node, 
                               'weight': v.weight,
                               'enabled': v.enabled}
                           for k, v in self.connections.items()},
            'fitness': self.fitness,
            'species_id': self.species_id
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Genome':
        """Create genome from dictionary."""
        genome = cls(data['input_size'], data['output_size'])
        genome.fitness = data['fitness']
        genome.species_id = data['species_id']
        
        # Rebuild nodes
        genome.nodes = {}
        for k, v in data['nodes'].items():
            genome.nodes[int(k)] = NodeGene(
                v['innovation_number'], v['node_type'], v['activation']
            )
        
        # Rebuild connections
        genome.connections = {}
        for k, v in data['connections'].items():
            genome.connections[int(k)] = ConnectionGene(
                v['innovation_number'], v['input_node'], v['output_node'],
                v['weight'], v['enabled']
            )
            
        return genome
