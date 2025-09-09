#!/usr/bin/env python3
"""
Deep Governance Integration Test
Demonstrates agents voting on their own evolutionary parameters and mapping functions.
"""

from constitutional_ai.governance import GovernanceManager, GovernanceExecutor
from constitutional_ai.neat_mapper import TraitToNEATMapper
from constitutional_ai import (
    create_random_genome,
    create_agent_identity,
    COMPLETE_TRAIT_DEFINITIONS,
)
from constitutional_ai.persistence import AgentRecord


def create_test_agents_for_deep_governance(num_agents=6):
    """Create test agents with diverse governance profiles for deep governance testing."""
    agents = []

    # Create agents with different constitutional profiles
    profiles = [
        # Innovation-focused agent
        {"InnovationDrive": 3.0, "Expertise": 2.5, "Autonomy": 4.2, "Leadership": 2.8},
        # Stability-focused agent
        {
            "Stability": 2.8,
            "EthicalReasoning": 3.5,
            "Expertise": 2.2,
            "Leadership": 3.1,
        },
        # Technical expert
        {
            "Expertise": 2.9,
            "Autonomy": 5.1,
            "ProcessingSpeed": 4.8,
            "InnovationDrive": 2.1,
        },
        # Ethical leader
        {
            "EthicalReasoning": 3.9,
            "Leadership": 3.6,
            "Trustworthiness": 3.8,
            "ConflictResolution": 3.2,
        },
        # Balanced agent
        {
            "Leadership": 2.5,
            "InnovationDrive": 2.0,
            "Expertise": 2.0,
            "EthicalReasoning": 2.5,
        },
        # Highly autonomous innovator
        {
            "Autonomy": 5.3,
            "InnovationDrive": 2.8,
            "Expertise": 2.3,
            "CriticalThinking": 4.2,
        },
    ]

    for i, profile in enumerate(profiles):
        # Create base genome
        genome = create_random_genome(COMPLETE_TRAIT_DEFINITIONS, seed=200 + i)

        # Override specific traits
        for trait_name, value in profile.items():
            if trait_name in genome.loci:
                locus = genome.loci[trait_name]
                from constitutional_ai.genome import (
                    Allele,
                    AlleleType,
                    StabilizationType,
                )

                trait_domain = COMPLETE_TRAIT_DEFINITIONS[trait_name]

                new_allele = Allele(
                    value=value,
                    allele_type=AlleleType(trait_domain.allele_type),
                    stabilization_type=StabilizationType.STATIC,
                    domain=trait_domain.domain,
                )

                locus.maternal_allele = new_allele
                locus.paternal_allele = new_allele

        # Create agent identity
        identity = create_agent_identity(genome, seed_closure=200 + i)

        # Use the identity bundle directly (it already contains constitution_result)
        identity_bundle = identity

        agent_record = AgentRecord(
            agent_id=identity.id_hash,
            identity_bundle=identity_bundle,
            capabilities={},
            creation_date=identity.created_at,
            lineage={},
            notes=f"Test agent {i+1} for deep governance testing",
        )

        agents.append(agent_record)

    return agents


def test_evolutionary_rule_governance():
    """Test agents voting on deep evolutionary rules."""
    print("TESTING DEEP EVOLUTIONARY RULE GOVERNANCE")
    print("=" * 60)

    agents = create_test_agents_for_deep_governance(6)
    governance = GovernanceManager(agents)
    executor = GovernanceExecutor()

    print(f"Governance council: {len(agents)} agents")

    # Test 1: Vote on mutation rate increase
    print(f"\n📋 DEEP GOVERNANCE VOTE 1: Mutation Rate Boost")
    mutation_proposal = governance.propose_evolutionary_rule_change(
        rule_name="innovation_mutation_boost",
        target_parameter="weight_mutation_rate",
        modification={
            "type": "multiplier",
            "value": 1.5,
            "conditions": "innovation > 2.0",
        },
        description="Increase mutation rates for high-innovation agents by 50% to accelerate evolution",
    )

    result1 = governance.conduct_vote(mutation_proposal)
    if result1.passed:
        print("✅ PROPOSAL PASSED - Executing deep evolutionary change...")
        execution_result = executor.execute_proposal(mutation_proposal, result1)
        print(f"🔧 {execution_result['details']}")

    # Test 2: Vote on NEAT parameter mapping change
    print(f"\n📋 DEEP GOVERNANCE VOTE 2: NEAT Parameter Mapping")
    mapping_proposal = governance.propose_neat_parameter_mapping_change(
        trait_name="InnovationDrive",
        neat_parameter="add_node_rate",
        new_mapping={
            "type": "exponential",
            "exponent": 1.3,
            "description": "Exponential boost for structural innovation",
        },
        description="Change InnovationDrive → add_node_rate mapping from linear to exponential for faster structural evolution",
    )

    result2 = governance.conduct_vote(mapping_proposal)
    if result2.passed:
        print("✅ PROPOSAL PASSED - Executing mapping change...")
        execution_result = executor.execute_proposal(mapping_proposal, result2)
        print(f"🔧 {execution_result['details']}")

    return result1.passed, result2.passed


def test_governance_influenced_neat_mapping():
    """Test NEAT parameter mapping with governance decisions applied."""
    print("\nTESTING GOVERNANCE-INFLUENCED NEAT MAPPING")
    print("=" * 60)

    # Create test governance decisions
    governance_decisions = {
        # Range modifiers
        "range_modifier_weight_mutation_rate": {"multiplier": 1.5, "offset": 0.1},
        # Mapping modifiers
        "mapping_InnovationDrive_add_node_rate": {"type": "exponential", "value": 1.3},
        "mapping_InnovationDrive_weight_mutation_rate": {
            "type": "threshold",
            "threshold": 0.6,
            "boost": 2.0,
        },
    }

    # Create test agent
    genome = create_random_genome(COMPLETE_TRAIT_DEFINITIONS, seed=42)
    identity = create_agent_identity(genome)
    traits = identity.constitution_result.constitution

    print(f"Test agent traits:")
    print(f"  InnovationDrive: {traits.get('InnovationDrive', 0):.2f}")
    print(f"  Stability: {traits.get('Stability', 0):.2f}")
    print(f"  Expertise: {traits.get('Expertise', 0):.2f}")

    # Test without governance
    print(f"\n🔄 NEAT Mapping WITHOUT Governance:")
    mapper_base = TraitToNEATMapper()
    config_base = mapper_base.traits_to_neat_config(traits)

    print(f"  Base mutation rate: {config_base.weight_mutation_rate:.3f}")
    print(f"  Base add node rate: {config_base.add_node_rate:.3f}")
    print(f"  Base population size: {config_base.population_size}")

    # Test with governance
    print(f"\n🏛️ NEAT Mapping WITH Governance Decisions:")
    mapper_governed = TraitToNEATMapper(governance_decisions)
    config_governed = mapper_governed.traits_to_neat_config(traits)

    print(f"  Governed mutation rate: {config_governed.weight_mutation_rate:.3f}")
    print(f"  Governed add node rate: {config_governed.add_node_rate:.3f}")
    print(f"  Governed population size: {config_governed.population_size}")

    # Calculate differences
    mutation_diff = (
        config_governed.weight_mutation_rate - config_base.weight_mutation_rate
    )
    node_diff = config_governed.add_node_rate - config_base.add_node_rate

    print(f"\n📊 GOVERNANCE IMPACT:")
    print(
        f"  Mutation rate change: {mutation_diff:+.3f} ({mutation_diff/config_base.weight_mutation_rate*100:+.1f}%)"
    )
    print(
        f"  Add node rate change: {node_diff:+.3f} ({node_diff/config_base.add_node_rate*100:+.1f}%)"
    )

    return mutation_diff, node_diff


def test_deep_governance_integration():
    """Test complete deep governance integration."""
    print("\nTESTING COMPLETE DEEP GOVERNANCE INTEGRATION")
    print("=" * 60)

    agents = create_test_agents_for_deep_governance(6)
    governance = GovernanceManager(agents)
    executor = GovernanceExecutor()

    # Simulate a complete governance → evolution cycle
    print("🏛️ Simulating: Agent Population Votes → NEAT Parameter Changes → Evolution")

    # Step 1: Agents analyze current evolutionary performance
    print(f"\n📊 STEP 1: Population Analysis")
    print(f"Population size: {len(agents)} agents")

    innovation_agents = sum(
        1
        for a in agents
        if a.identity_bundle.constitution_result.constitution.get("InnovationDrive", 0)
        > 2.5
    )
    stability_agents = sum(
        1
        for a in agents
        if a.identity_bundle.constitution_result.constitution.get("Stability", 0) > 2.5
    )

    print(f"High-innovation agents: {innovation_agents}/{len(agents)}")
    print(f"High-stability agents: {stability_agents}/{len(agents)}")

    # Step 2: Generate proposals based on population characteristics
    print(f"\n📋 STEP 2: Generate Adaptive Proposals")

    if innovation_agents > stability_agents:
        print(
            "Population is innovation-heavy → Proposing aggressive evolutionary rules"
        )
        proposal = governance.propose_evolutionary_rule_change(
            rule_name="innovation_acceleration",
            target_parameter="mutation_rate_global",
            modification={"type": "multiplier", "value": 1.8},
            description="Population voted for accelerated evolution due to high innovation traits",
        )
    else:
        print(
            "Population is stability-heavy → Proposing conservative evolutionary rules"
        )
        proposal = governance.propose_evolutionary_rule_change(
            rule_name="stability_preservation",
            target_parameter="selection_pressure",
            modification={"type": "multiplier", "value": 1.2},
            description="Population voted for higher selection pressure to maintain stability",
        )

    # Step 3: Democratic vote
    print(f"\n🗳️ STEP 3: Democratic Vote")
    result = governance.conduct_vote(proposal)

    # Step 4: Execute governance decision
    print(f"\n⚙️ STEP 4: Execute Decision")
    if result.passed:
        execution_result = executor.execute_proposal(proposal, result)
        print(f"✅ Executed: {execution_result['details']}")

        # Step 5: Apply to NEAT evolution
        print(f"\n🧬 STEP 5: Apply to NEAT Evolution")
        print("In a full implementation, this would:")
        print("  1. Modify neat_mapper.py with new governance decisions")
        print("  2. All future agents would use the new evolutionary rules")
        print("  3. NEAT evolution would proceed with governance-approved parameters")
        print("  4. Population evolves under democratically-chosen rules")

        return True
    else:
        print("❌ Proposal rejected - evolution continues with current rules")
        return False


def main():
    """Run all deep governance tests."""
    print("DEEP GOVERNANCE INTEGRATION - COMPREHENSIVE TEST")
    print("=" * 70)

    try:
        # Test 1: Evolutionary rule governance
        rule_passed, mapping_passed = test_evolutionary_rule_governance()

        # Test 2: Governance-influenced NEAT mapping
        mutation_diff, node_diff = test_governance_influenced_neat_mapping()

        # Test 3: Complete integration
        integration_success = test_deep_governance_integration()

        print("\n" + "=" * 70)
        print("DEEP GOVERNANCE TEST RESULTS:")
        print(f"✅ Evolutionary rule voting: {'PASSED' if rule_passed else 'FAILED'}")
        print(f"✅ NEAT mapping voting: {'PASSED' if mapping_passed else 'FAILED'}")
        print(
            f"✅ Governance influences NEAT: {'YES' if abs(mutation_diff) > 0.01 else 'NO'}"
        )
        print(
            f"✅ Complete integration: {'PASSED' if integration_success else 'FAILED'}"
        )

        if all(
            [
                rule_passed,
                mapping_passed,
                abs(mutation_diff) > 0.01,
                integration_success,
            ]
        ):
            print("\n🏛️ SUCCESS: DEEP GOVERNANCE FULLY OPERATIONAL!")
            print(
                "Agents can now democratically control their own evolutionary process!"
            )
            print("🧬 Constitutional self-governance of artificial evolution achieved!")
        else:
            print("\n⚠️  Some deep governance features need refinement")

    except Exception as e:
        print(f"[ERROR] Deep governance test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
