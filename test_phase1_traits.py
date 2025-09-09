#!/usr/bin/env python3
"""
Phase 1 Foundation Traits - Validation Test
Demonstrates that new traits create meaningful evolutionary differences.
"""

from constitutional_ai import create_random_genome, create_agent_identity, COMPLETE_TRAIT_DEFINITIONS
from constitutional_ai.training.language_evolution import LanguageTrainingData, LanguageEvolutionFitness
import neat
import random

def create_test_agent_with_traits(trait_values, seed=None):
    """Create an agent with specific Phase 1 trait values for testing."""
    if seed:
        random.seed(seed)
    
    # Create base genome
    genome = create_random_genome(COMPLETE_TRAIT_DEFINITIONS, seed=seed)
    
    # Override Phase 1 trait values by modifying the genome directly
    for trait_name, target_value in trait_values.items():
        if trait_name in genome.loci:
            locus = genome.loci[trait_name]
            
            # Create new alleles with the target value
            from constitutional_ai.genome import Allele, AlleleType, StabilizationType
            trait_domain = COMPLETE_TRAIT_DEFINITIONS[trait_name]
            
            new_allele = Allele(
                value=target_value,
                allele_type=AlleleType(trait_domain.allele_type),
                stabilization_type=StabilizationType.STATIC,  # Use static for predictable testing
                domain=trait_domain.domain
            )
            
            # Set both alleles to same value for predictable expression
            locus.maternal_allele = new_allele
            locus.paternal_allele = new_allele
    
    return create_agent_identity(genome, seed_closure=seed or 123)

def test_trait_impact_on_fitness():
    """Test that Phase 1 traits meaningfully impact fitness evaluation."""
    print("TESTING PHASE 1 TRAIT IMPACT ON FITNESS")
    print("=" * 50)
    
    # Create test training data
    test_text = "The quick brown fox jumps over the lazy dog. " * 100
    training_data = LanguageTrainingData.from_text(test_text)
    
    # Create dummy network for testing
    class MockNetwork:
        def activate(self, inputs):
            # Simple mock that returns consistent output
            return [0.5] * 1
    
    network = MockNetwork()
    
    # Test Case 1: High Critical Thinking vs Low Critical Thinking
    print("\n1. CRITICAL THINKING IMPACT")
    
    high_critical = create_test_agent_with_traits({"CriticalThinking": 5.0}, seed=1)
    low_critical = create_test_agent_with_traits({"CriticalThinking": 0.0}, seed=2) 
    
    fitness_high = LanguageEvolutionFitness(training_data, high_critical)
    fitness_low = LanguageEvolutionFitness(training_data, low_critical)
    
    score_high = fitness_high.evaluate_network(network, num_tests=10)
    score_low = fitness_low.evaluate_network(network, num_tests=10)
    
    print(f"   High Critical Thinking (5.0): Fitness = {score_high:.4f}")
    print(f"   Low Critical Thinking (0.0):  Fitness = {score_low:.4f}")
    print(f"   Difference: {abs(score_high - score_low):.4f}")
    
    # Test Case 2: High Pattern Recognition vs Low Pattern Recognition  
    print("\n2. PATTERN RECOGNITION IMPACT")
    
    high_pattern = create_test_agent_with_traits({"PatternRecognition": 8.0}, seed=3)
    low_pattern = create_test_agent_with_traits({"PatternRecognition": 1.0}, seed=4)
    
    fitness_high_p = LanguageEvolutionFitness(training_data, high_pattern) 
    fitness_low_p = LanguageEvolutionFitness(training_data, low_pattern)
    
    score_high_p = fitness_high_p.evaluate_network(network, num_tests=10)
    score_low_p = fitness_low_p.evaluate_network(network, num_tests=10)
    
    print(f"   High Pattern Recognition (8.0): Fitness = {score_high_p:.4f}")
    print(f"   Low Pattern Recognition (1.0):  Fitness = {score_low_p:.4f}")
    print(f"   Difference: {abs(score_high_p - score_low_p):.4f}")
    
    # Test Case 3: Combined trait effects
    print("\n3. COMBINED TRAIT EFFECTS")
    
    optimal_agent = create_test_agent_with_traits({
        "CriticalThinking": 5.0,
        "PatternRecognition": 8.0,
        "CommonSense": 4.0,
        "Resilience": 3.0,
        "Adaptability": 4.0
    }, seed=5)
    
    minimal_agent = create_test_agent_with_traits({
        "CriticalThinking": 0.0,
        "PatternRecognition": 1.0,
        "CommonSense": 0.0,
        "Resilience": 0.1,
        "Adaptability": 0.0
    }, seed=6)
    
    fitness_optimal = LanguageEvolutionFitness(training_data, optimal_agent)
    fitness_minimal = LanguageEvolutionFitness(training_data, minimal_agent)
    
    score_optimal = fitness_optimal.evaluate_network(network, num_tests=10)
    score_minimal = fitness_minimal.evaluate_network(network, num_tests=10)
    
    print(f"   Optimal Phase 1 Agent:  Fitness = {score_optimal:.4f}")
    print(f"   Minimal Phase 1 Agent:  Fitness = {score_minimal:.4f}")
    print(f"   Performance Gap: {abs(score_optimal - score_minimal):.4f}")
    
    # Validation
    print("\nVALIDATION RESULTS")
    critical_impact = abs(score_high - score_low) > 0.01
    pattern_impact = abs(score_high_p - score_low_p) > 0.01
    combined_impact = abs(score_optimal - score_minimal) > 0.05
    
    print(f"   Critical Thinking creates fitness difference: {critical_impact}")
    print(f"   Pattern Recognition creates fitness difference: {pattern_impact}")  
    print(f"   Combined traits create significant difference: {combined_impact}")
    
    if critical_impact and pattern_impact and combined_impact:
        print("\n*** SUCCESS: Phase 1 traits meaningfully impact evolution!")
        return True
    else:
        print("\n*** Phase 1 traits may need adjustment for stronger evolutionary pressure")
        return False

def test_trait_weights():
    """Test that trait weights are properly calculated."""
    print("\nTESTING TRAIT WEIGHT CALCULATION") 
    print("=" * 50)
    
    # Test extreme values
    max_traits = create_test_agent_with_traits({
        "CriticalThinking": 5.0,
        "PatternRecognition": 8.0,
        "CommonSense": 4.0,
        "Resilience": 3.0,
        "Adaptability": 4.0
    })
    
    min_traits = create_test_agent_with_traits({
        "CriticalThinking": 0.0,
        "PatternRecognition": 1.0,
        "CommonSense": 0.0,
        "Resilience": 0.1,
        "Adaptability": 0.0
    })
    
    training_data = LanguageTrainingData.from_text("Test text")
    
    fitness_max = LanguageEvolutionFitness(training_data, max_traits)
    fitness_min = LanguageEvolutionFitness(training_data, min_traits)
    
    print("Maximum trait values -> Fitness weights:")
    print(f"   Critical Thinking: {fitness_max.critical_thinking_weight:.3f}")
    print(f"   Pattern Recognition: {fitness_max.pattern_recognition_weight:.3f}")
    print(f"   Common Sense: {fitness_max.common_sense_weight:.3f}")
    print(f"   Resilience: {fitness_max.resilience_weight:.3f}")
    print(f"   Adaptability: {fitness_max.adaptability_weight:.3f}")
    
    print("\nMinimum trait values -> Fitness weights:")
    print(f"   Critical Thinking: {fitness_min.critical_thinking_weight:.3f}")
    print(f"   Pattern Recognition: {fitness_min.pattern_recognition_weight:.3f}")
    print(f"   Common Sense: {fitness_min.common_sense_weight:.3f}")
    print(f"   Resilience: {fitness_min.resilience_weight:.3f}")
    print(f"   Adaptability: {fitness_min.adaptability_weight:.3f}")
    
    # Verify weights are within expected ranges (updated for actual weight ranges)
    weights_valid = (
        0.7 <= fitness_min.critical_thinking_weight <= 1.5 and
        0.7 <= fitness_min.pattern_recognition_weight <= 1.5 and
        0.9 <= fitness_min.common_sense_weight <= 1.2 and
        0.8 <= fitness_min.resilience_weight <= 1.1 and
        0.9 <= fitness_min.adaptability_weight <= 1.4
    )
    
    print(f"\n[OK] Trait weights within expected ranges: {weights_valid}")
    return weights_valid

def main():
    """Run all Phase 1 validation tests."""
    print("PHASE 1 FOUNDATION TRAITS - VALIDATION SUITE")
    print("=" * 60)
    
    try:
        # Test 1: Trait impact on fitness
        fitness_test = test_trait_impact_on_fitness()
        
        # Test 2: Weight calculation
        weight_test = test_trait_weights()
        
        # Final verdict
        print("\n" + "=" * 60)
        if fitness_test and weight_test:
            print("*** ALL TESTS PASSED - PHASE 1 TRAITS FULLY FUNCTIONAL!")
            print("[OK] Traits create meaningful evolutionary pressure")
            print("[OK] Weight calculation working correctly")
            print("[OK] Ready for breeding experiments!")
        else:
            print("*** Some tests failed - Phase 1 may need refinement")
            
    except Exception as e:
        print(f"[ERROR] Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()