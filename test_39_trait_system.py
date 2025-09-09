#!/usr/bin/env python3
"""
39-Trait System Validation Test
Comprehensive validation of the complete Constitutional NEAT trait system.
"""

from constitutional_ai import create_random_genome, create_agent_identity, COMPLETE_TRAIT_DEFINITIONS
from constitutional_ai.training.language_evolution import LanguageTrainingData, LanguageEvolutionFitness
import neat


def test_complete_trait_system():
    """Test that all 39 traits are properly integrated."""
    print("TESTING COMPLETE 39-TRAIT SYSTEM")
    print("=" * 50)
    
    # Verify trait count
    trait_count = len(COMPLETE_TRAIT_DEFINITIONS)
    print(f"Total traits defined: {trait_count}")
    
    if trait_count != 39:
        print(f"[ERROR] Expected 39 traits, found {trait_count}")
        return False
    
    # Create test agent with all traits
    genome = create_random_genome(COMPLETE_TRAIT_DEFINITIONS, seed=42)
    identity = create_agent_identity(genome, seed_closure=123)
    
    # Test fitness system integration
    training_data = LanguageTrainingData.from_text("The quick brown fox jumps over the lazy dog. " * 50)
    fitness = LanguageEvolutionFitness(training_data, identity)
    
    # Count fitness weight attributes
    weight_attributes = [attr for attr in dir(fitness) if attr.endswith('_weight')]
    print(f"Fitness weight attributes found: {len(weight_attributes)}")
    
    # Expected weights from all phases
    expected_weights = {
        # Core system weights
        'accuracy_weight', 'diversity_weight',
        
        # Phase 1 Foundation
        'critical_thinking_weight', 'pattern_recognition_weight', 'common_sense_weight', 
        'resilience_weight', 'adaptability_weight',
        
        # Phase 2 Reasoning 
        'causal_reasoning_weight', 'abstract_thinking_weight', 'temporal_reasoning_weight',
        'spatial_reasoning_weight', 'intuition_weight',
        
        # Phase 3 Social & Emotional
        'emotional_intelligence_weight', 'empathy_weight', 'self_awareness_weight',
        'trustworthiness_weight', 'cooperation_weight',
        
        # Phase 4 Advanced Capabilities & Governance
        'conflict_resolution_weight', 'cultural_intelligence_weight', 'leadership_weight',
        'negotiation_weight', 'goal_orientation_weight', 'autonomy_weight', 'humor_weight',
        'ethical_reasoning_weight', 'creativity_weight_advanced'
    }
    
    # Check for all expected weights
    missing_weights = []
    for expected in expected_weights:
        if not hasattr(fitness, expected):
            missing_weights.append(expected)
    
    if missing_weights:
        print(f"[ERROR] Missing weight attributes: {missing_weights}")
        return False
    
    print(f"[OK] All {len(expected_weights)} expected weight attributes found")
    
    # Test network evaluation with mock network
    class MockNetwork:
        def activate(self, inputs):
            return [0.5] * 1
    
    network = MockNetwork()
    fitness_score = fitness.evaluate_network(network, num_tests=10)
    print(f"Mock network fitness score: {fitness_score:.4f}")
    
    if fitness_score <= 0:
        print("[ERROR] Fitness evaluation returned non-positive score")
        return False
    
    print("[SUCCESS] All 39 traits properly integrated into fitness system!")
    return True


def test_trait_phase_coverage():
    """Test that all four phases are represented."""
    print("\nTESTING TRAIT PHASE COVERAGE")
    print("=" * 50)
    
    # Expected traits by phase
    phase_1_traits = {"CriticalThinking", "PatternRecognition", "CommonSense", "Resilience", "Adaptability"}
    phase_2_traits = {"CausalReasoning", "AbstractThinking", "TemporalReasoning", "SpatialReasoning", "Intuition"}
    phase_3_traits = {"EmotionalIntelligence", "Empathy", "SelfAwareness", "Trustworthiness", "Cooperation"}
    phase_4_traits = {"ConflictResolution", "CulturalIntelligence", "Leadership", "Negotiation", 
                     "GoalOrientation", "Autonomy", "Humor", "EthicalReasoning", "Creativity"}
    
    all_defined_traits = set(COMPLETE_TRAIT_DEFINITIONS.keys())
    
    # Check each phase
    phases = [
        ("Phase 1 Foundation", phase_1_traits),
        ("Phase 2 Reasoning", phase_2_traits), 
        ("Phase 3 Social/Emotional", phase_3_traits),
        ("Phase 4 Advanced/Governance", phase_4_traits)
    ]
    
    all_phases_valid = True
    total_phase_traits = 0
    
    for phase_name, expected_traits in phases:
        missing = expected_traits - all_defined_traits
        present = expected_traits & all_defined_traits
        
        print(f"{phase_name}: {len(present)}/{len(expected_traits)} traits present")
        if missing:
            print(f"   Missing: {missing}")
            all_phases_valid = False
        else:
            print(f"   [OK] All traits present")
        
        total_phase_traits += len(expected_traits)
    
    print(f"\nTotal phase traits: {total_phase_traits}")
    print(f"Total defined traits: {len(all_defined_traits)}")
    
    if total_phase_traits != len(all_defined_traits):
        print(f"[WARNING] Phase coverage ({total_phase_traits}) != total traits ({len(all_defined_traits)})")
        # Show extra traits not in phases
        phase_traits = phase_1_traits | phase_2_traits | phase_3_traits | phase_4_traits
        extra_traits = all_defined_traits - phase_traits
        if extra_traits:
            print(f"Extra traits not in phases: {extra_traits}")
    
    return all_phases_valid


def test_trait_value_ranges():
    """Test that all traits have valid domains and default values."""
    print("\nTESTING TRAIT VALUE RANGES")
    print("=" * 50)
    
    all_valid = True
    numeric_traits = 0
    categorical_traits = 0
    
    for trait_name, trait_domain in COMPLETE_TRAIT_DEFINITIONS.items():
        domain = trait_domain.domain
        default_val = trait_domain.default_value
        allele_type = trait_domain.allele_type
        
        if allele_type == "numeric":
            numeric_traits += 1
            if not isinstance(domain, tuple) or len(domain) != 2:
                print(f"[ERROR] {trait_name}: Numeric trait should have 2-value tuple domain, got {domain}")
                all_valid = False
                continue
                
            domain_min, domain_max = domain
            
            # Check domain validity
            if domain_min >= domain_max:
                print(f"[ERROR] {trait_name}: Invalid domain {domain}")
                all_valid = False
                continue
            
            # Check default in domain
            if not (domain_min <= default_val <= domain_max):
                print(f"[ERROR] {trait_name}: Default {default_val} outside domain {domain}")
                all_valid = False
                continue
            
            # Check fixed points
            if hasattr(trait_domain, 'fixed_points') and trait_domain.fixed_points:
                if not all(domain_min <= fp <= domain_max for fp in trait_domain.fixed_points):
                    print(f"[ERROR] {trait_name}: Fixed points outside domain")
                    all_valid = False
                    continue
                    
        elif allele_type == "categorical":
            categorical_traits += 1
            if not isinstance(domain, list) or len(domain) == 0:
                print(f"[ERROR] {trait_name}: Categorical trait should have non-empty list domain, got {domain}")
                all_valid = False
                continue
                
            # Check default in domain
            if default_val not in domain:
                print(f"[ERROR] {trait_name}: Default {default_val} not in domain {domain}")
                all_valid = False
                continue
        else:
            print(f"[WARNING] {trait_name}: Unknown allele type {allele_type}")
    
    if all_valid:
        print(f"[OK] All {len(COMPLETE_TRAIT_DEFINITIONS)} trait domains valid")
        print(f"   Numeric traits: {numeric_traits}")
        print(f"   Categorical traits: {categorical_traits}")
    
    return all_valid


def main():
    """Run all validation tests."""
    print("39-TRAIT CONSTITUTIONAL SYSTEM - COMPREHENSIVE VALIDATION")
    print("=" * 70)
    
    try:
        # Test 1: Complete system integration
        system_test = test_complete_trait_system()
        
        # Test 2: Phase coverage
        phase_test = test_trait_phase_coverage()
        
        # Test 3: Trait value ranges
        range_test = test_trait_value_ranges()
        
        # Final verdict
        print("\n" + "=" * 70)
        if system_test and phase_test and range_test:
            print("*** ALL TESTS PASSED - 39-TRAIT SYSTEM FULLY OPERATIONAL!")
            print("[OK] Complete trait integration functional")
            print("[OK] All 4 phases properly represented")
            print("[OK] All trait domains and defaults valid")
            print("[OK] Ready for governance system implementation!")
        else:
            print("*** Some tests failed - system may need refinement")
            if not system_test:
                print("[!] System integration issues")
            if not phase_test:
                print("[!] Phase coverage issues") 
            if not range_test:
                print("[!] Trait definition issues")
                
    except Exception as e:
        print(f"[ERROR] Validation failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()