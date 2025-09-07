"""
Identity System for Constitutional NEAT Agents
Provides canonical serialization, immutable ID hashing, and complete provenance tracking.

This module ensures that each AI agent has a unique, verifiable identity based on
their complete constitutional state, enabling blockchain immutability and no-extraction policies.
"""

import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .genome import ConstitutionalGenome
from .emergent_constitution import create_emergent_agent_identity
from .neat_mapper import map_traits_to_neat_config, NEATConfig
from .color_mapping_simple import (
    traits_to_simple_color,
    get_simple_color_description,
    traits_to_hsv_simple,
)


@dataclass
class SeedChain:
    """
    Tracks all random seeds used in agent creation and evolution.

    Ensures complete reproducibility by recording every random decision point.
    """

    seed_breed: Optional[int] = None  # Seed used for breeding/crossover
    seed_closure: Optional[int] = None  # Seed used for trait closure
    seed_build: Optional[int] = None  # Seed used for NEAT config building
    seed_eval: List[int] = field(default_factory=list)  # Seeds used for evaluation runs

    def to_dict(self) -> dict:
        """Convert seed chain to dictionary for serialization."""
        return {
            "seed_breed": self.seed_breed,
            "seed_closure": self.seed_closure,
            "seed_build": self.seed_build,
            "seed_eval": self.seed_eval,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SeedChain":
        """Create seed chain from dictionary."""
        return cls(
            seed_breed=data.get("seed_breed"),
            seed_closure=data.get("seed_closure"),
            seed_build=data.get("seed_build"),
            seed_eval=data.get("seed_eval", []),
        )


@dataclass
class VersionInfo:
    """
    Tracks all software versions used in agent creation.

    Critical for reproducibility - same versions must produce identical results.
    """

    traits_version: str = "1.0.0"
    rules_version: str = "1.0.0"
    builder_version: str = "1.0.0"
    neat_version: str = "1.0.0"
    env_version: str = "1.0.0"

    def to_dict(self) -> dict:
        """Convert version info to dictionary."""
        return {
            "traits_version": self.traits_version,
            "rules_version": self.rules_version,
            "builder_version": self.builder_version,
            "neat_version": self.neat_version,
            "env_version": self.env_version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VersionInfo":
        """Create version info from dictionary."""
        return cls(
            traits_version=data.get("traits_version", "1.0.0"),
            rules_version=data.get("rules_version", "1.0.0"),
            builder_version=data.get("builder_version", "1.0.0"),
            neat_version=data.get("neat_version", "1.0.0"),
            env_version=data.get("env_version", "1.0.0"),
        )


@dataclass
class ConstitutionResult:
    """
    Results from trait closure resolution.

    Contains the final constitutional trait values and metadata about the resolution process.
    """

    constitution: Dict[str, Any]
    converged: bool
    iterations: int
    applied_rules: List[str]

    def to_dict(self) -> dict:
        """Convert constitution result to dictionary."""
        return {
            "constitution": self.constitution,
            "converged": self.converged,
            "iterations": self.iterations,
            "applied_rules": self.applied_rules,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConstitutionResult":
        """Create constitution result from dictionary."""
        return cls(
            constitution=data["constitution"],
            converged=data["converged"],
            iterations=data["iterations"],
            applied_rules=data["applied_rules"],
        )


@dataclass
class VisualIdentity:
    """
    Visual representation derived from constitutional traits.

    Includes color information and descriptive elements for NFT generation.
    """

    primary_color_hex: str
    hsv_values: Tuple[float, float, float]
    color_description: str

    def to_dict(self) -> dict:
        """Convert visual identity to dictionary."""
        return {
            "primary_color_hex": self.primary_color_hex,
            "hsv_values": self.hsv_values,
            "color_description": self.color_description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VisualIdentity":
        """Create visual identity from dictionary."""
        return cls(
            primary_color_hex=data["primary_color_hex"],
            hsv_values=tuple(data["hsv_values"]),
            color_description=data["color_description"],
        )


class IdentityBundle:
    """
    Complete identity bundle for a constitutional AI agent.

    Contains all information needed to uniquely identify, reproduce, and verify an agent.
    """

    def __init__(
        self,
        genome: ConstitutionalGenome,
        constitution_result: ConstitutionResult,
        neat_config: NEATConfig,
        seed_chain: SeedChain,
        version_info: Optional[VersionInfo] = None,
    ):
        """
        Initialize identity bundle.

        Args:
            genome: Constitutional genome with diploid loci
            constitution_result: Results from trait closure
            neat_config: NEAT configuration derived from traits
            seed_chain: Complete seed chain for reproducibility
            version_info: Software version information
        """
        self.genome = genome
        self.constitution_result = constitution_result
        self.neat_config = neat_config
        self.seed_chain = seed_chain
        self.version_info = version_info or VersionInfo()

        # Generate visual identity from constitutional traits
        self.visual_identity = self._generate_visual_identity()

        # Generate creation timestamp
        self.created_at = datetime.utcnow().isoformat()

        # Calculate identity hash (this is the agent's unique ID)
        self.id_hash = self._calculate_identity_hash()

    def _generate_visual_identity(self) -> VisualIdentity:
        """Generate visual identity from constitutional traits."""
        traits = self.constitution_result.constitution

        hsv = traits_to_hsv_simple(traits)
        hex_color = traits_to_simple_color(traits)
        description = get_simple_color_description(hex_color)

        return VisualIdentity(
            primary_color_hex=hex_color, hsv_values=hsv, color_description=description
        )

    def _calculate_identity_hash(self) -> str:
        """
        Calculate the canonical identity hash for this agent.

        This hash uniquely identifies the agent and is used for all references.
        """
        canonical_data = self.get_canonical_representation()
        identity_json = json.dumps(
            canonical_data, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(identity_json.encode("utf-8")).hexdigest()

    def get_canonical_representation(self) -> dict:
        """
        Get canonical representation for deterministic hashing.

        Order is critical - identical agents must produce identical representations.
        """
        return {
            "genome": self.genome.get_canonical_representation(),
            "constitution": {
                "traits": {
                    k: v
                    for k, v in sorted(self.constitution_result.constitution.items())
                },
                "converged": self.constitution_result.converged,
                "iterations": self.constitution_result.iterations,
                "applied_rules": sorted(self.constitution_result.applied_rules),
            },
            "neat_config": {
                k: v for k, v in sorted(self.neat_config.to_dict().items())
            },
            "seed_chain": {k: v for k, v in sorted(self.seed_chain.to_dict().items())},
            "versions": {k: v for k, v in sorted(self.version_info.to_dict().items())},
        }

    def to_dict(self) -> dict:
        """Convert complete identity bundle to dictionary for serialization."""
        return {
            "id_hash": self.id_hash,
            "created_at": self.created_at,
            "genome": self.genome.to_dict(),
            "constitution_result": self.constitution_result.to_dict(),
            "neat_config": self.neat_config.to_dict(),
            "seed_chain": self.seed_chain.to_dict(),
            "version_info": self.version_info.to_dict(),
            "visual_identity": self.visual_identity.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IdentityBundle":
        """Create identity bundle from dictionary."""
        genome = ConstitutionalGenome.from_dict(data["genome"])
        constitution_result = ConstitutionResult.from_dict(data["constitution_result"])
        neat_config = NEATConfig(**data["neat_config"])
        seed_chain = SeedChain.from_dict(data["seed_chain"])
        version_info = VersionInfo.from_dict(data["version_info"])

        # Create bundle (this will recalculate derived fields)
        bundle = cls(genome, constitution_result, neat_config, seed_chain, version_info)

        # Override with stored values for consistency
        bundle.created_at = data["created_at"]
        bundle.id_hash = data["id_hash"]
        bundle.visual_identity = VisualIdentity.from_dict(data["visual_identity"])

        return bundle

    def verify_integrity(self) -> bool:
        """
        Verify that the identity bundle is internally consistent.

        Returns:
            True if all components are consistent and hash matches
        """
        try:
            # Verify genome hash matches
            expected_genome_hash = self.genome.compute_genome_hash()
            if expected_genome_hash != self.genome.compute_genome_hash():
                return False

            # Verify identity hash matches canonical representation
            expected_id_hash = self._calculate_identity_hash()
            if expected_id_hash != self.id_hash:
                return False

            # Verify visual identity matches constitution
            expected_hex = traits_to_simple_color(self.constitution_result.constitution)
            if expected_hex != self.visual_identity.primary_color_hex:
                return False

            return True

        except Exception:
            return False

    def get_summary(self) -> dict:
        """Get a summary of the agent for display purposes."""

        return {
            "id_hash": self.id_hash[:12] + "...",  # Shortened for display
            "created_at": self.created_at,
            "generation": self.genome.provenance.generation,
            "converged": self.constitution_result.converged,
            "color": self.visual_identity.color_description,
            "dominant_traits": self._get_dominant_traits(),
            "neat_summary": {
                "population_size": self.neat_config.population_size,
                "max_nodes": self.neat_config.max_nodes,
                "learning_rate": self.neat_config.learning_rate,
                "activation": self.neat_config.activation_function,
            },
        }

    def _get_dominant_traits(self) -> List[str]:
        """Identify the agent's strongest traits for summary."""
        constitution = self.constitution_result.constitution

        # Normalize all trait values and find top 3
        trait_scores = []

        for trait_name, trait_value in constitution.items():
            # Simple scoring - could be more sophisticated
            if isinstance(trait_value, (int, float)):
                # For numeric traits, assume higher is more dominant
                if trait_value > 2.0:  # Threshold for "high"
                    trait_scores.append((trait_name, trait_value))
            elif isinstance(trait_value, str):
                # For categorical traits, consider certain values dominant
                if trait_value in ["High", "Maximum", "Expert", "Verbose"]:
                    trait_scores.append((trait_name, 1.0))

        # Sort by score and return top 3 names
        trait_scores.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in trait_scores[:3]]


class IdentityRegistry:
    """
    Registry for managing and verifying agent identities.

    Ensures no duplicate agents and maintains integrity of the identity system.
    """

    def __init__(self):
        """Initialize empty registry."""
        self.agents: Dict[str, IdentityBundle] = {}
        self.genome_hashes: Dict[str, str] = {}  # genome_hash -> id_hash mapping

    def register_agent(self, identity_bundle: IdentityBundle) -> bool:
        """
        Register a new agent identity.

        Args:
            identity_bundle: Complete identity bundle for the agent

        Returns:
            True if successfully registered, False if duplicate or invalid
        """
        if not identity_bundle.verify_integrity():
            return False

        id_hash = identity_bundle.id_hash
        genome_hash = identity_bundle.genome.compute_genome_hash()

        # Check for duplicates
        if id_hash in self.agents:
            return False  # Duplicate identity

        if genome_hash in self.genome_hashes:
            return False  # Duplicate genome

        # Register the agent
        self.agents[id_hash] = identity_bundle
        self.genome_hashes[genome_hash] = id_hash

        return True

    def get_agent(self, id_hash: str) -> Optional[IdentityBundle]:
        """Get agent by ID hash."""
        return self.agents.get(id_hash)

    def verify_agent(self, id_hash: str) -> bool:
        """Verify an agent's identity and integrity."""
        agent = self.agents.get(id_hash)
        if not agent:
            return False

        return agent.verify_integrity()

    def get_registry_stats(self) -> dict:
        """Get statistics about the registry."""
        return {
            "total_agents": len(self.agents),
            "unique_genomes": len(self.genome_hashes),
            "integrity_valid": sum(
                1 for agent in self.agents.values() if agent.verify_integrity()
            ),
        }


def create_agent_identity(
    genome: ConstitutionalGenome,
    seed_closure: Optional[int] = None,
    seed_build: Optional[int] = None,
    version_info: Optional[VersionInfo] = None,
) -> IdentityBundle:
    """
    Create a complete identity bundle for an agent.

    This is the main entry point for creating agent identities from genomes.

    Args:
        genome: Constitutional genome
        seed_closure: Seed for trait closure resolution
        seed_build: Seed for NEAT config building
        version_info: Software version information

    Returns:
        Complete IdentityBundle ready for registration
    """

    # Step 1: Resolve constitution from genome traits using emergent engine
    constitution_result = create_emergent_agent_identity(genome, seed_closure)

    # Step 2: Map constitutional traits to NEAT config
    neat_config = map_traits_to_neat_config(constitution_result["constitution"])

    # Step 3: Create seed chain
    seed_chain = SeedChain(
        seed_breed=genome.provenance.seed_used,
        seed_closure=seed_closure,
        seed_build=seed_build,
        seed_eval=[],
    )

    # Step 4: Create constitution result object
    constitution_obj = ConstitutionResult(
        constitution=constitution_result["constitution"],
        converged=constitution_result["converged"],
        iterations=constitution_result["iterations"],
        applied_rules=constitution_result.get("applied_rules", []),
    )

    # Step 5: Create complete identity bundle
    return IdentityBundle(
        genome=genome,
        constitution_result=constitution_obj,
        neat_config=neat_config,
        seed_chain=seed_chain,
        version_info=version_info,
    )


# Global registry instance
_global_registry = IdentityRegistry()


def get_global_registry() -> IdentityRegistry:
    """Get the global identity registry."""
    return _global_registry
