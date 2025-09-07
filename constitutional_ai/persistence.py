"""
Agent Persistence System for Constitutional AI

Handles saving and loading of evolved agents with their:
- Constitutional genomes and identities
- Trained neural networks and performance
- Training history and lineage
- Capability-specific metadata

Enables cross-run breeding, performance analysis, and agent reuse.
"""

import json
import pickle
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import neat

from .identity import IdentityBundle


@dataclass
class AgentCapabilityRecord:
    """Record of an agent's capability in a specific domain."""

    capability_type: str  # e.g., "language", "coding", "image_generation"
    training_generations: int
    final_fitness: float
    training_date: str
    sample_outputs: Dict[str, Any]  # Example generations/responses
    performance_metrics: Dict[str, float]
    training_config: Dict[str, Any]


@dataclass
class AgentRecord:
    """Complete record of a constitutional agent."""

    agent_id: str
    identity_bundle: IdentityBundle
    capabilities: Dict[str, AgentCapabilityRecord]
    creation_date: str
    lineage: Dict[str, str]  # Parent IDs, breeding info
    notes: str = ""

    def get_capability_names(self) -> List[str]:
        """Get list of capabilities this agent has been trained in."""
        return list(self.capabilities.keys())

    def has_capability(self, capability_type: str) -> bool:
        """Check if agent has been trained in a specific capability."""
        return capability_type in self.capabilities

    def get_best_capability(self) -> Optional[Tuple[str, AgentCapabilityRecord]]:
        """Get the capability with highest fitness score."""
        if not self.capabilities:
            return None

        best_name = max(
            self.capabilities.keys(), key=lambda k: self.capabilities[k].final_fitness
        )
        return best_name, self.capabilities[best_name]


class AgentPersistence:
    """
    System for saving and loading constitutional agents with their trained capabilities.

    File structure:
    agents/
    ├── {agent_id}/
    │   ├── record.json          # Agent metadata and capability records
    │   ├── genome.pkl           # Constitutional genome
    │   ├── identity.pkl         # Identity bundle
    │   └── capabilities/
    │       ├── language_network.pkl     # Trained networks
    │       ├── coding_network.pkl
    │       └── ...
    """

    def __init__(self, agents_dir: str = "agents"):
        """
        Initialize persistence system.

        Args:
            agents_dir: Directory for storing agent data
        """
        self.agents_dir = agents_dir
        self.ensure_directory_structure()

    def ensure_directory_structure(self):
        """Create necessary directories."""
        os.makedirs(self.agents_dir, exist_ok=True)

    def save_agent(self, agent_record: AgentRecord) -> str:
        """
        Save a complete agent record to persistent storage.

        Args:
            agent_record: Complete agent information

        Returns:
            Path to saved agent directory
        """
        agent_dir = os.path.join(self.agents_dir, agent_record.agent_id)
        os.makedirs(agent_dir, exist_ok=True)

        # Create capabilities directory
        capabilities_dir = os.path.join(agent_dir, "capabilities")
        os.makedirs(capabilities_dir, exist_ok=True)

        # Save agent metadata (JSON for human readability)
        record_data = {
            "agent_id": agent_record.agent_id,
            "genome_hash": agent_record.identity_bundle.genome.compute_genome_hash(),
            "creation_date": agent_record.creation_date,
            "lineage": agent_record.lineage,
            "notes": agent_record.notes,
            "visual_dna": agent_record.identity_bundle.visual_identity.primary_color_hex,
            "personality_traits": {
                k: f"{v:.2f}" if isinstance(v, (int, float)) else str(v)
                for k, v in agent_record.identity_bundle.constitution_result.constitution.items()
            },
            "capabilities": {
                name: {
                    "capability_type": cap.capability_type,
                    "training_generations": cap.training_generations,
                    "final_fitness": cap.final_fitness,
                    "training_date": cap.training_date,
                    "sample_outputs": cap.sample_outputs,
                    "performance_metrics": cap.performance_metrics,
                    "training_config": cap.training_config,
                }
                for name, cap in agent_record.capabilities.items()
            },
        }

        record_path = os.path.join(agent_dir, "record.json")
        with open(record_path, "w") as f:
            json.dump(record_data, f, indent=2)

        # Save constitutional genome (pickle for exact reproduction)
        genome_path = os.path.join(agent_dir, "genome.pkl")
        with open(genome_path, "wb") as f:
            pickle.dump(agent_record.identity_bundle.genome, f)

        # Save identity bundle
        identity_path = os.path.join(agent_dir, "identity.pkl")
        with open(identity_path, "wb") as f:
            pickle.dump(agent_record.identity_bundle, f)

        return agent_dir

    def save_trained_network(
        self,
        agent_id: str,
        capability_type: str,
        network: neat.nn.FeedForwardNetwork,
        genome: neat.DefaultGenome,
    ) -> str:
        """
        Save a trained neural network for an agent capability.

        Args:
            agent_id: Agent identifier
            capability_type: Type of capability (language, coding, etc.)
            network: Trained NEAT network
            genome: NEAT genome that created the network

        Returns:
            Path to saved network file
        """
        agent_dir = os.path.join(self.agents_dir, agent_id)
        capabilities_dir = os.path.join(agent_dir, "capabilities")
        os.makedirs(capabilities_dir, exist_ok=True)

        # Save both network and genome for complete reproduction
        network_data = {
            "network": network,
            "genome": genome,
            "capability_type": capability_type,
            "save_date": datetime.now().isoformat(),
        }

        network_path = os.path.join(capabilities_dir, f"{capability_type}_network.pkl")
        with open(network_path, "wb") as f:
            pickle.dump(network_data, f)

        return network_path

    def load_agent_record(self, agent_id: str) -> Optional[AgentRecord]:
        """
        Load a complete agent record from storage.

        Args:
            agent_id: Agent identifier

        Returns:
            AgentRecord if found, None otherwise
        """
        agent_dir = os.path.join(self.agents_dir, agent_id)
        record_path = os.path.join(agent_dir, "record.json")

        if not os.path.exists(record_path):
            return None

        try:
            # Load metadata
            with open(record_path, "r") as f:
                record_data = json.load(f)

            # Load identity
            identity_path = os.path.join(agent_dir, "identity.pkl")
            with open(identity_path, "rb") as f:
                identity_bundle = pickle.load(f)

            # Reconstruct capability records
            capabilities = {}
            for name, cap_data in record_data.get("capabilities", {}).items():
                capabilities[name] = AgentCapabilityRecord(
                    capability_type=cap_data["capability_type"],
                    training_generations=cap_data["training_generations"],
                    final_fitness=cap_data["final_fitness"],
                    training_date=cap_data["training_date"],
                    sample_outputs=cap_data["sample_outputs"],
                    performance_metrics=cap_data["performance_metrics"],
                    training_config=cap_data["training_config"],
                )

            return AgentRecord(
                agent_id=record_data["agent_id"],
                identity_bundle=identity_bundle,
                capabilities=capabilities,
                creation_date=record_data["creation_date"],
                lineage=record_data.get("lineage", {}),
                notes=record_data.get("notes", ""),
            )

        except Exception as e:
            print(f"Error loading agent {agent_id}: {e}")
            return None

    def load_trained_network(
        self, agent_id: str, capability_type: str
    ) -> Optional[Tuple[neat.nn.FeedForwardNetwork, neat.DefaultGenome]]:
        """
        Load a trained neural network for an agent capability.

        Args:
            agent_id: Agent identifier
            capability_type: Type of capability

        Returns:
            (network, genome) tuple if found, None otherwise
        """
        capabilities_dir = os.path.join(self.agents_dir, agent_id, "capabilities")
        network_path = os.path.join(capabilities_dir, f"{capability_type}_network.pkl")

        if not os.path.exists(network_path):
            return None

        try:
            with open(network_path, "rb") as f:
                network_data = pickle.load(f)

            return network_data["network"], network_data["genome"]

        except Exception as e:
            print(f"Error loading network for {agent_id}/{capability_type}: {e}")
            return None

    def list_agents(self) -> List[str]:
        """Get list of all saved agent IDs."""
        if not os.path.exists(self.agents_dir):
            return []

        agents = []
        for item in os.listdir(self.agents_dir):
            agent_path = os.path.join(self.agents_dir, item)
            record_path = os.path.join(agent_path, "record.json")
            if os.path.isdir(agent_path) and os.path.exists(record_path):
                agents.append(item)

        return sorted(agents)

    def get_agents_by_capability(self, capability_type: str) -> List[Tuple[str, float]]:
        """
        Get agents trained in a specific capability, sorted by fitness.

        Args:
            capability_type: Type of capability to filter by

        Returns:
            List of (agent_id, fitness) tuples, sorted by fitness descending
        """
        agents_with_capability = []

        for agent_id in self.list_agents():
            record = self.load_agent_record(agent_id)
            if record and record.has_capability(capability_type):
                fitness = record.capabilities[capability_type].final_fitness
                agents_with_capability.append((agent_id, fitness))

        return sorted(agents_with_capability, key=lambda x: x[1], reverse=True)

    def create_agent_record_from_training(
        self,
        identity_bundle: IdentityBundle,
        training_result: Dict[str, Any],
        capability_type: str,
        lineage: Optional[Dict[str, str]] = None,
    ) -> AgentRecord:
        """
        Create an AgentRecord from training results.

        Args:
            identity_bundle: Agent's constitutional identity
            training_result: Results from training pipeline
            capability_type: Type of capability trained
            lineage: Parent/breeding information

        Returns:
            Complete AgentRecord ready for saving
        """
        # Create capability record
        capability_record = AgentCapabilityRecord(
            capability_type=capability_type,
            training_generations=training_result.get("training_generations", 0),
            final_fitness=training_result.get("final_fitness", 0.0),
            training_date=datetime.now().isoformat(),
            sample_outputs=training_result.get("sample_generation", {}),
            performance_metrics={
                "final_fitness": training_result.get("final_fitness", 0.0),
                "vocabulary_size": training_result.get("vocabulary_size", 0),
            },
            training_config={},
        )

        return AgentRecord(
            agent_id=identity_bundle.id_hash,
            identity_bundle=identity_bundle,
            capabilities={capability_type: capability_record},
            creation_date=datetime.now().isoformat(),
            lineage=lineage or {},
            notes=f"Trained in {capability_type} with fitness {
                training_result.get(
                    'final_fitness', 0.0):.3f}",
        )


# Convenience functions for common operations
def save_training_result(
    training_result: Dict[str, Any], capability_type: str, agents_dir: str = "agents"
) -> str:
    """
    Save a training result as a persistent agent.

    Args:
        training_result: Result from training pipeline
        capability_type: Type of capability trained
        agents_dir: Directory for agent storage

    Returns:
        Agent ID of saved agent
    """
    persistence = AgentPersistence(agents_dir)

    # Extract identity from training result
    identity_bundle = training_result.get("identity_bundle")
    if not identity_bundle:
        # Create from available data
        raise ValueError("Training result must contain identity_bundle")

    # Create agent record
    agent_record = persistence.create_agent_record_from_training(
        identity_bundle, training_result, capability_type
    )

    # Save agent
    _ = persistence.save_agent(agent_record)

    # Save trained network if available
    if "best_network" in training_result and "best_genome" in training_result:
        persistence.save_trained_network(
            agent_record.agent_id,
            capability_type,
            training_result["best_network"],
            training_result["best_genome"],
        )

    return agent_record.agent_id


def load_agent(agent_id: str, agents_dir: str = "agents") -> Optional[AgentRecord]:
    """Load an agent by ID."""
    persistence = AgentPersistence(agents_dir)
    return persistence.load_agent_record(agent_id)


def list_all_agents(agents_dir: str = "agents") -> List[str]:
    """Get list of all saved agent IDs."""
    persistence = AgentPersistence(agents_dir)
    return persistence.list_agents()
