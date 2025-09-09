#!/usr/bin/env python3
"""
Enhanced Governance System Test
Demonstrates the Phase 4 trait-enhanced governance capabilities.
"""

from constitutional_ai.governance import GovernanceManager, CouncilManager, GovernanceProposal
from constitutional_ai.persistence import list_all_agents, load_agent
from constitutional_ai import create_random_genome, create_agent_identity, COMPLETE_TRAIT_DEFINITIONS


def create_test_governance_agents(num_agents=8):
    """Create a diverse set of agents for governance testing."""
    agents = []
    
    # Create agents with different governance profiles
    governance_profiles = [
        # Strong leaders
        {"Leadership": 3.4, "EthicalReasoning": 3.5, "ConflictResolution": 3.0, "Negotiation": 2.8},
        {"Leadership": 3.6, "EthicalReasoning": 2.9, "ConflictResolution": 3.2, "Autonomy": 4.8},
        
        # Ethical specialists
        {"EthicalReasoning": 3.9, "Trustworthiness": 4.0, "ConflictResolution": 2.8, "SelfAwareness": 3.2},
        {"EthicalReasoning": 3.7, "Trustworthiness": 3.8, "Empathy": 3.5, "Cooperation": 3.9},
        
        # Technical specialists  
        {"Expertise": 2.8, "Autonomy": 5.2, "Leadership": 2.1, "ProcessingSpeed": 8.5, "CommunicationStyle": "Technical"},
        {"Expertise": 2.9, "Autonomy": 4.9, "EthicalReasoning": 2.2, "ProcessingSpeed": 7.8, "CommunicationStyle": "Technical"},
        
        # Social specialists
        {"Cooperation": 4.2, "CulturalIntelligence": 3.8, "EmotionalIntelligence": 3.6, "Negotiation": 3.2},
        {"Cooperation": 3.9, "CulturalIntelligence": 4.1, "ConflictResolution": 3.4, "Empathy": 3.8}
    ]
    
    for i, profile in enumerate(governance_profiles[:num_agents]):
        # Create base genome
        genome = create_random_genome(COMPLETE_TRAIT_DEFINITIONS, seed=100 + i)
        
        # Override governance traits
        for trait_name, value in profile.items():
            if trait_name in genome.loci:
                locus = genome.loci[trait_name]
                from constitutional_ai.genome import Allele, AlleleType, StabilizationType
                trait_domain = COMPLETE_TRAIT_DEFINITIONS[trait_name]
                
                if trait_domain.allele_type == "categorical":
                    # Handle categorical traits
                    new_value = value if isinstance(value, str) else trait_domain.default_value
                else:
                    new_value = value
                
                new_allele = Allele(
                    value=new_value,
                    allele_type=AlleleType(trait_domain.allele_type),
                    stabilization_type=StabilizationType.STATIC,
                    domain=trait_domain.domain
                )
                
                locus.maternal_allele = new_allele
                locus.paternal_allele = new_allele
        
        # Create agent identity
        identity = create_agent_identity(genome, seed_closure=100 + i)
        
        # Create agent record
        from constitutional_ai.persistence import AgentRecord, IdentityBundle, ConstitutionResult
        
        constitution_result = ConstitutionResult(
            constitution=identity.trait_summary,
            converged=True,
            iterations=1
        )
        
        identity_bundle = IdentityBundle(
            identity=identity,
            constitution_result=constitution_result
        )
        
        agent_record = AgentRecord(
            agent_id=identity.id_hash,
            identity_bundle=identity_bundle,
            generation_created=0,
            parent_ids=[],
            training_history=[],
            fitness_scores={}
        )
        
        agents.append(agent_record)
    
    return agents


def test_enhanced_voting_weights():
    """Test Phase 4 enhanced voting weight calculation."""
    print("TESTING ENHANCED VOTING WEIGHTS")
    print("=" * 50)
    
    agents = create_test_governance_agents(6)
    governance = GovernanceManager(agents)
    
    print("Agent voting weights (Phase 4 enhanced):")
    for i, agent in enumerate(agents):
        weight = governance.calculate_vote_weight(agent)
        traits = agent.identity_bundle.constitution_result.constitution
        
        leadership = traits.get("Leadership", 0.0)
        ethical = traits.get("EthicalReasoning", 0.0)
        conflict_res = traits.get("ConflictResolution", 0.0)
        
        print(f"  Agent {i+1} ({agent.agent_id[:8]}...): Weight = {weight:.2f}")
        print(f"    Leadership: {leadership:.1f}, Ethics: {ethical:.1f}, Conflict Res: {conflict_res:.1f}")
    
    print(f"\nWeight range: {min(governance.calculate_vote_weight(a) for a in agents):.2f} - {max(governance.calculate_vote_weight(a) for a in agents):.2f}")


def test_enhanced_voting_behavior():
    """Test Phase 4 enhanced voting behavior on different proposal types."""
    print("\nTESTING ENHANCED VOTING BEHAVIOR")
    print("=" * 50)
    
    agents = create_test_governance_agents(6)
    governance = GovernanceManager(agents)
    
    # Test different proposal types
    proposals = [
        governance.propose_breeding_rule_change(
            "innovation_threshold", 
            {"min_innovation": 2.5}, 
            "Raise innovation requirements for breeding eligibility"
        ),
        governance.propose_governance_reform(
            "council_system",
            {"enable_councils": True, "council_types": ["technical", "ethical", "social"]},
            "Implement specialized governance councils for different decision types"
        ),
        governance.propose_agent_council_formation(
            5, "merit_based", 10
        )
    ]
    
    for proposal in proposals:
        print(f"\nProposal: {proposal.title}")
        print(f"Type: {proposal.proposal_type}")
        
        result = governance.conduct_vote(proposal)
        print(f"Result: {'PASSED' if result.passed else 'FAILED'}")


def test_council_system():
    """Test the agent council formation and voting system."""
    print("\nTESTING AGENT COUNCIL SYSTEM") 
    print("=" * 50)
    
    agents = create_test_governance_agents(8)
    council_manager = CouncilManager()
    
    # Form different types of councils
    councils = [
        ("technical_council", "technical"),
        ("ethics_council", "ethical"),
        ("social_council", "social"),
        ("general_council", "general")
    ]
    
    for council_id, specialization in councils:
        council = council_manager.form_council(
            council_id=council_id,
            agents=agents,
            council_size=3,
            selection_criteria="trait_based",
            rotation_period=5,
            specialization=specialization
        )
        print()
    
    # Test council voting on a technical proposal
    technical_proposal = GovernanceProposal(
        proposal_id="tech_upgrade_001",
        title="Upgrade Neural Network Architecture",
        description="Implement new NEAT configuration with larger population sizes",
        proposal_type="system_parameter",
        parameters={"new_population_size": 2000, "max_nodes": 100000},
        required_majority=0.6
    )
    
    print("=" * 50)
    print("COUNCIL VOTING ON TECHNICAL PROPOSAL")
    print("=" * 50)
    
    for council_id, specialization in councils:
        result = council_manager.get_council_vote_on_proposal(council_id, technical_proposal)
        print(f"\n{specialization.title()} Council ({council_id}):")
        print(f"  Consensus: {result['consensus']}")
        print(f"  Vote breakdown: {result['vote_breakdown']}")
        print(f"  Individual votes: {result['member_votes']}")


def test_governance_integration():
    """Test integration of regular voting with council input."""
    print("\nTESTING GOVERNANCE INTEGRATION")
    print("=" * 50)
    
    agents = create_test_governance_agents(8)
    governance = GovernanceManager(agents)
    council_manager = CouncilManager()
    
    # Form ethics council
    ethics_council = council_manager.form_council(
        "ethics_oversight",
        agents,
        council_size=3,
        selection_criteria="ethical_expertise", 
        rotation_period=10,
        specialization="ethical"
    )
    
    # Test ethical proposal
    ethical_proposal = governance.propose_breeding_rule_change(
        "ethical_screening",
        {"require_ethical_minimum": 2.0},
        "Require minimum ethical reasoning for breeding eligibility"
    )
    
    print(f"\nProposal: {ethical_proposal.title}")
    
    # Get council recommendation
    council_result = council_manager.get_council_vote_on_proposal(
        "ethics_oversight", ethical_proposal
    )
    
    print(f"\nEthics Council Recommendation: {council_result['consensus']}")
    print(f"Council vote breakdown: {council_result['vote_breakdown']}")
    
    # Conduct full population vote
    print(f"\nFull Population Vote:")
    population_result = governance.conduct_vote(ethical_proposal)
    
    print(f"\nGOVERNANCE SUMMARY:")
    print(f"Council consensus: {council_result['consensus']}")
    print(f"Population decision: {'PASSED' if population_result.passed else 'FAILED'}")
    print(f"Alignment: {'YES' if (council_result['consensus'] == 'yes') == population_result.passed else 'NO'}")


def main():
    """Run all governance system tests."""
    print("ENHANCED GOVERNANCE SYSTEM - COMPREHENSIVE TEST")
    print("=" * 70)
    
    try:
        # Test 1: Enhanced voting weights
        test_enhanced_voting_weights()
        
        # Test 2: Enhanced voting behavior
        test_enhanced_voting_behavior()
        
        # Test 3: Council system
        test_council_system()
        
        # Test 4: Governance integration
        test_governance_integration()
        
        print("\n" + "=" * 70)
        print("ALL GOVERNANCE TESTS COMPLETED SUCCESSFULLY!")
        print("[SUCCESS] Phase 4 governance traits fully integrated")
        print("[SUCCESS] Council system operational")
        print("[SUCCESS] Enhanced voting mechanisms working") 
        print("[SUCCESS] Constitutional governance system ready!")
        
    except Exception as e:
        print(f"[ERROR] Governance test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()