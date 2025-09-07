"""
Constitutional Genome System for NEAT Breeding
Implements diploid loci with alleles and Kleene fixed-point trait closure.

This module provides the foundational genome representation for the Constitutional
NEAT Breeding System, supporting Mendelian inheritance with mathematical guarantees.
"""

import json
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import random


class AlleleType(Enum):
    """Types of alleles that can exist at a locus."""

    NUMERIC = "numeric"  # Float values with min/max bounds
    CATEGORICAL = "categorical"  # Discrete string choices
    BOOLEAN = "boolean"  # True/False values


class StabilizationType(Enum):
    """The 6 fundamental stabilization behaviors for alleles."""

    STATIC = "static"  # f(x) = x - stable, unchanging
    PROGRESSIVE = "progressive"  # f(x) > x - continuous improvement
    OSCILLATORY = "oscillatory"  # f(f(x)) = x - stable cycles
    DEGENERATIVE = "degenerative"  # f(x) < x - controlled decay
    CHAOTIC = "chaotic"  # bounded non-repeating trajectories
    MULTI_ATTRACTOR = "multi_attractor"  # context-dependent behavior


@dataclass(frozen=True)
class Allele:
    """
    Immutable allele representation with recursive stabilization behavior.

    Each allele is a Kleene fixed-point with its own stabilization behavior.
    All higher-level behaviors emerge from these fundamental building blocks.
    """

    value: Any
    allele_type: AlleleType
    stabilization_type: StabilizationType  # The fundamental recursive behavior
    domain: Optional[Tuple] = (
        None  # (min, max) for numeric, tuple of choices for categorical
    )

    def __post_init__(self):
        """Validate allele value against its type and domain."""
        if self.allele_type == AlleleType.NUMERIC:
            if not isinstance(self.value, (int, float)):
                raise ValueError(
                    f"Numeric allele must have numeric value, got {type(self.value)}"
                )
            if self.domain and not (self.domain[0] <= self.value <= self.domain[1]):
                raise ValueError(
                    f"Numeric allele {self.value} outside domain {self.domain}"
                )

        elif self.allele_type == AlleleType.CATEGORICAL:
            if self.domain and self.value not in self.domain:
                raise ValueError(
                    f"Categorical allele {self.value} not in domain {self.domain}"
                )

        elif self.allele_type == AlleleType.BOOLEAN:
            if not isinstance(self.value, bool):
                raise ValueError(
                    f"Boolean allele must have boolean value, got {type(self.value)}"
                )

    def to_dict(self) -> dict:
        """Convert allele to dictionary for serialization."""
        return {
            "value": self.value,
            "allele_type": self.allele_type.value,
            "stabilization_type": self.stabilization_type.value,
            "domain": self.domain,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Allele":
        """Create allele from dictionary."""
        return cls(
            value=data["value"],
            allele_type=AlleleType(data["allele_type"]),
            stabilization_type=StabilizationType(data["stabilization_type"]),
            domain=data.get("domain"),
        )


@dataclass
class Locus:
    """
    A genetic locus containing two alleles (diploid).

    Represents a single heritable trait position with maternal and paternal alleles.
    """

    name: str
    maternal_allele: Allele
    paternal_allele: Allele
    linkage_group: Optional[int] = None  # For linked inheritance

    def get_dominant_allele(self) -> Allele:
        """
        Get the dominant allele for trait expression.

        Combines allelic values and their stabilization behaviors.
        The emergent behavior comes from the interaction of the two alleles.
        """
        # For numeric traits: average value, but stabilization behavior from stronger allele
        if (
            self.maternal_allele.allele_type == AlleleType.NUMERIC
            and self.paternal_allele.allele_type == AlleleType.NUMERIC
        ):
            avg_value = (self.maternal_allele.value + self.paternal_allele.value) / 2

            # Stabilization behavior determined by which allele is more "active"
            # Progressive, Chaotic, and Oscillatory are "dominant" over Static and Degenerative
            dominant_stabilization = self._resolve_stabilization_dominance(
                self.maternal_allele.stabilization_type,
                self.paternal_allele.stabilization_type,
            )

            return Allele(
                value=avg_value,
                allele_type=AlleleType.NUMERIC,
                stabilization_type=dominant_stabilization,
                domain=self.maternal_allele.domain,
            )

        # Categorical/Boolean: maternal dominance for now
        return self.maternal_allele

    def _resolve_stabilization_dominance(
        self, maternal_stab: StabilizationType, paternal_stab: StabilizationType
    ) -> StabilizationType:
        """
        Resolve which stabilization type dominates when alleles combine.

        This creates the emergent behavior from allelic interactions.
        """
        # If both are the same, use that
        if maternal_stab == paternal_stab:
            return maternal_stab

        # Dominance hierarchy: Active behaviors dominate passive ones
        dominance_order = {
            StabilizationType.CHAOTIC: 6,  # Most active/dominant
            StabilizationType.PROGRESSIVE: 5,
            StabilizationType.OSCILLATORY: 4,
            StabilizationType.MULTI_ATTRACTOR: 3,
            StabilizationType.STATIC: 2,
            StabilizationType.DEGENERATIVE: 1,  # Least active/dominant
        }

        maternal_rank = dominance_order.get(maternal_stab, 0)
        paternal_rank = dominance_order.get(paternal_stab, 0)

        # Return the more dominant stabilization type
        return maternal_stab if maternal_rank >= paternal_rank else paternal_stab

    def to_dict(self) -> dict:
        """Convert locus to dictionary for serialization."""
        return {
            "name": self.name,
            "maternal_allele": self.maternal_allele.to_dict(),
            "paternal_allele": self.paternal_allele.to_dict(),
            "linkage_group": self.linkage_group,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Locus":
        """Create locus from dictionary."""
        return cls(
            name=data["name"],
            maternal_allele=Allele.from_dict(data["maternal_allele"]),
            paternal_allele=Allele.from_dict(data["paternal_allele"]),
            linkage_group=data.get("linkage_group"),
        )


@dataclass
class ProvenanceRecord:
    """Track the breeding history and lineage of a genome."""

    parents: List[str] = field(default_factory=list)  # Parent genome ID hashes
    operation: str = "initial"  # breeding, mutation, etc.
    generation: int = 0
    timestamp: Optional[str] = None
    seed_used: Optional[int] = None


class ConstitutionalGenome:
    """
    Constitutional genome with diploid loci and deterministic breeding.

    This genome supports Mendelian inheritance patterns and provides the foundation
    for trait-based AI agent breeding with mathematical guarantees.
    """

    def __init__(self, loci: Optional[Dict[str, Locus]] = None):
        """
        Initialize a constitutional genome.

        Args:
            loci: Dictionary of locus name -> Locus object
        """
        self.loci: Dict[str, Locus] = loci or {}
        self.provenance: ProvenanceRecord = ProvenanceRecord()
        self.version: str = "1.0.0"  # Track schema version

    def add_locus(self, locus: Locus) -> None:
        """Add a locus to the genome."""
        self.loci[locus.name] = locus

    def get_expressed_traits(self) -> Dict[str, Any]:
        """
        Get the expressed trait values from all loci.

        Returns the dominant allele value for each locus.
        """
        traits = {}
        for name, locus in self.loci.items():
            traits[name] = locus.get_dominant_allele().value
        return traits

    def get_canonical_representation(self) -> dict:
        """
        Get canonical representation for deterministic hashing.

        Ensures identical genomes always produce identical hashes regardless
        of creation order or internal state.
        """
        # Sort loci by name for deterministic ordering
        sorted_loci = {}
        for name in sorted(self.loci.keys()):
            locus = self.loci[name]
            sorted_loci[name] = {
                "name": name,
                "maternal": locus.maternal_allele.to_dict(),
                "paternal": locus.paternal_allele.to_dict(),
                "linkage_group": locus.linkage_group,
            }

        return {
            "version": self.version,
            "loci": sorted_loci,
            "provenance": {
                "parents": sorted(self.provenance.parents),
                "operation": self.provenance.operation,
                "generation": self.provenance.generation,
                "seed_used": self.provenance.seed_used,
            },
        }

    def compute_genome_hash(self) -> str:
        """
        Compute deterministic hash of genome for immutable identity.

        Returns SHA-256 hash of canonical representation.
        """
        canonical = self.get_canonical_representation()
        genome_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(genome_json.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        """Convert genome to dictionary for serialization."""
        return {
            "version": self.version,
            "loci": {name: locus.to_dict() for name, locus in self.loci.items()},
            "provenance": {
                "parents": self.provenance.parents,
                "operation": self.provenance.operation,
                "generation": self.provenance.generation,
                "timestamp": self.provenance.timestamp,
                "seed_used": self.provenance.seed_used,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConstitutionalGenome":
        """Create genome from dictionary."""
        genome = cls()
        genome.version = data.get("version", "1.0.0")

        # Rebuild loci
        for name, locus_data in data["loci"].items():
            genome.loci[name] = Locus.from_dict(locus_data)

        # Rebuild provenance
        prov_data = data["provenance"]
        genome.provenance = ProvenanceRecord(
            parents=prov_data.get("parents", []),
            operation=prov_data.get("operation", "initial"),
            generation=prov_data.get("generation", 0),
            timestamp=prov_data.get("timestamp"),
            seed_used=prov_data.get("seed_used"),
        )

        return genome

    def clone(self) -> "ConstitutionalGenome":
        """Create a deep copy of the genome."""
        return ConstitutionalGenome.from_dict(self.to_dict())


def create_random_genome(
    trait_definitions: Dict[str, Any], seed: Optional[int] = None
) -> ConstitutionalGenome:
    """
    Create a random genome from trait definitions.

    Args:
        trait_definitions: Dict mapping trait names to TraitDomain objects
        seed: Random seed for reproducible generation

    Returns:
        ConstitutionalGenome with randomly initialized loci
    """
    if seed is not None:
        random.seed(seed)

    genome = ConstitutionalGenome()

    for trait_name, trait_spec in trait_definitions.items():
        allele_type = AlleleType(trait_spec.allele_type)
        domain = trait_spec.domain

        # Generate two random alleles with stabilization behaviors
        if allele_type == AlleleType.NUMERIC:
            min_val, max_val = domain
            maternal_val = random.uniform(min_val, max_val)
            paternal_val = random.uniform(min_val, max_val)

            # Randomly assign stabilization types
            maternal_stab = random.choice(list(StabilizationType))
            paternal_stab = random.choice(list(StabilizationType))

            maternal_allele = Allele(maternal_val, allele_type, maternal_stab, domain)
            paternal_allele = Allele(paternal_val, allele_type, paternal_stab, domain)

        elif allele_type == AlleleType.CATEGORICAL:
            maternal_val = random.choice(domain)
            paternal_val = random.choice(domain)

            maternal_stab = random.choice(list(StabilizationType))
            paternal_stab = random.choice(list(StabilizationType))

            maternal_allele = Allele(maternal_val, allele_type, maternal_stab, domain)
            paternal_allele = Allele(paternal_val, allele_type, paternal_stab, domain)

        elif allele_type == AlleleType.BOOLEAN:
            maternal_stab = random.choice(list(StabilizationType))
            paternal_stab = random.choice(list(StabilizationType))

            maternal_allele = Allele(
                random.choice([True, False]), allele_type, maternal_stab
            )
            paternal_allele = Allele(
                random.choice([True, False]), allele_type, paternal_stab
            )

        locus = Locus(
            name=trait_name,
            maternal_allele=maternal_allele,
            paternal_allele=paternal_allele,
            linkage_group=getattr(trait_spec, "linkage_group", None),
        )

        genome.add_locus(locus)

    genome.provenance.operation = "random_generation"
    genome.provenance.seed_used = seed

    return genome
