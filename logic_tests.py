#!/usr/bin/env python3
"""
Constitutional Agent Logic Test Battery
Progressive difficulty XOR and logic tests to benchmark agent learning capabilities.

Usage: python logic_tests.py [agent_id] [--all]
"""

import sys
import argparse
import time
from typing import Dict, List, Tuple, Any
import neat

sys.path.append('C:/Users/Jeff Towers/projects/web3-neat-nft')

from constitutional_ai.persistence import list_all_agents, load_agent
from constitutional_ai.neat_integration import evolve_constitutional_agent


class LogicTestSuite:
    """Progressive logic tests from simple to complex."""
    
    @staticmethod
    def basic_xor():
        """Level 1: Basic 2-input XOR (easiest)"""
        return {
            'name': 'Basic XOR',
            'difficulty': 1,
            'inputs': 2,
            'outputs': 1,
            'test_cases': [
                ((0.0, 0.0), 0.0),
                ((0.0, 1.0), 1.0),
                ((1.0, 0.0), 1.0),
                ((1.0, 1.0), 0.0)
            ],
            'description': 'Classic XOR - different inputs = 1'
        }
    
    @staticmethod
    def basic_and():
        """Level 2: AND gate (simple pattern)"""
        return {
            'name': 'AND Gate',
            'difficulty': 2,
            'inputs': 2,
            'outputs': 1,
            'test_cases': [
                ((0.0, 0.0), 0.0),
                ((0.0, 1.0), 0.0),
                ((1.0, 0.0), 0.0),
                ((1.0, 1.0), 1.0)
            ],
            'description': 'AND - both inputs must be 1'
        }
    
    @staticmethod
    def basic_nand():
        """Level 3: NAND gate (harder than XOR)"""
        return {
            'name': 'NAND Gate',
            'difficulty': 3,
            'inputs': 2,
            'outputs': 1,
            'test_cases': [
                ((0.0, 0.0), 1.0),
                ((0.0, 1.0), 1.0),
                ((1.0, 0.0), 1.0),
                ((1.0, 1.0), 0.0)
            ],
            'description': 'NAND - opposite of AND'
        }
    
    @staticmethod
    def three_input_xor():
        """Level 4: 3-input XOR (much harder)"""
        return {
            'name': '3-Input XOR',
            'difficulty': 4,
            'inputs': 3,
            'outputs': 1,
            'test_cases': [
                ((0.0, 0.0, 0.0), 0.0),  # Even number of 1s = 0
                ((0.0, 0.0, 1.0), 1.0),  # Odd number of 1s = 1
                ((0.0, 1.0, 0.0), 1.0),
                ((0.0, 1.0, 1.0), 0.0),
                ((1.0, 0.0, 0.0), 1.0),
                ((1.0, 0.0, 1.0), 0.0),
                ((1.0, 1.0, 0.0), 0.0),
                ((1.0, 1.0, 1.0), 1.0)
            ],
            'description': '3-input XOR - odd number of 1s = 1'
        }
    
    @staticmethod
    def majority_gate():
        """Level 5: 3-input majority (complex logic)"""
        return {
            'name': 'Majority Gate',
            'difficulty': 5,
            'inputs': 3,
            'outputs': 1,
            'test_cases': [
                ((0.0, 0.0, 0.0), 0.0),  # 0 ones - minority
                ((0.0, 0.0, 1.0), 0.0),  # 1 one - minority
                ((0.0, 1.0, 0.0), 0.0),  # 1 one - minority
                ((0.0, 1.0, 1.0), 1.0),  # 2 ones - majority
                ((1.0, 0.0, 0.0), 0.0),  # 1 one - minority
                ((1.0, 0.0, 1.0), 1.0),  # 2 ones - majority
                ((1.0, 1.0, 0.0), 1.0),  # 2 ones - majority
                ((1.0, 1.0, 1.0), 1.0)   # 3 ones - majority
            ],
            'description': 'Majority - output 1 if majority of inputs are 1'
        }
    
    @staticmethod
    def four_input_parity():
        """Level 6: 4-input parity (very complex)"""
        return {
            'name': '4-Input Parity',
            'difficulty': 6,
            'inputs': 4,
            'outputs': 1,
            'test_cases': [
                # Even parity - output 1 if even number of 1s
                ((0.0, 0.0, 0.0, 0.0), 1.0),  # 0 ones (even)
                ((0.0, 0.0, 0.0, 1.0), 0.0),  # 1 one (odd)
                ((0.0, 0.0, 1.0, 1.0), 1.0),  # 2 ones (even)
                ((0.0, 1.0, 1.0, 1.0), 0.0),  # 3 ones (odd)
                ((1.0, 1.0, 1.0, 1.0), 1.0),  # 4 ones (even)
                ((1.0, 0.0, 0.0, 0.0), 0.0),  # 1 one (odd)
                ((1.0, 1.0, 0.0, 0.0), 1.0),  # 2 ones (even)
                ((1.0, 1.0, 1.0, 0.0), 0.0),  # 3 ones (odd)
                # Add more test cases for complete coverage
                ((1.0, 0.0, 1.0, 0.0), 1.0),  # 2 ones (even)
                ((0.0, 1.0, 0.0, 1.0), 1.0),  # 2 ones (even)
                ((1.0, 0.0, 0.1, 1.0), 0.0),  # 3 ones (odd)
                ((0.0, 1.0, 1.0, 0.0), 1.0),  # 2 ones (even)
            ],
            'description': '4-input Parity - output 1 if even number of 1s'
        }
    
    @staticmethod
    def get_all_tests():
        """Get all logic tests in order of difficulty."""
        return [
            LogicTestSuite.basic_xor(),
            LogicTestSuite.basic_and(), 
            LogicTestSuite.basic_nand(),
            LogicTestSuite.three_input_xor(),
            LogicTestSuite.majority_gate(),
            LogicTestSuite.four_input_parity()
        ]


def create_fitness_function(test_spec):
    """Create NEAT fitness function for a specific logic test."""
    def fitness_function(genomes, config):
        for genome_id, genome in genomes:
            genome.fitness = len(test_spec['test_cases'])  # Start with perfect score
            
            try:
                # Create neural network
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                
                # Test on all cases
                for inputs, expected in test_spec['test_cases']:
                    output = net.activate(inputs)
                    error = (output[0] - expected) ** 2
                    genome.fitness -= error
                
                # Ensure non-negative fitness
                genome.fitness = max(0.0, genome.fitness)
                
            except Exception:
                genome.fitness = 0.0
    
    return fitness_function


def test_agent_on_logic(agent_id: str, test_spec: Dict, generations: int = 20) -> Dict[str, Any]:
    """Test a single agent on a specific logic problem."""
    print(f"Testing agent {agent_id[:12]}... on {test_spec['name']}")
    
    try:
        # Load agent
        agent_record = load_agent(agent_id)
        identity = agent_record.identity_bundle
        
        # Create fitness function
        fitness_func = create_fitness_function(test_spec)
        
        # Run evolution
        start_time = time.time()
        result = evolve_constitutional_agent(
            genome=identity.genome,
            fitness_function=fitness_func,
            generations=generations,
            num_inputs=test_spec['inputs'],
            num_outputs=test_spec['outputs'],
            seed_closure=hash(agent_id) % 10000
        )
        training_time = time.time() - start_time
        
        # Test final network
        network = result['network']
        correct = 0
        total = len(test_spec['test_cases'])
        
        print(f"  Final fitness: {result['final_fitness']:.3f}")
        print(f"  Testing accuracy:")
        
        for inputs, expected in test_spec['test_cases']:
            output = network.activate(inputs)[0]
            is_correct = abs(output - expected) < 0.5  # Threshold for binary classification
            correct += is_correct
            status = "PASS" if is_correct else "FAIL"
            print(f"    {status} {inputs} -> {output:.3f} (expected {expected})")
        
        accuracy = correct / total
        print(f"  Accuracy: {correct}/{total} ({accuracy*100:.1f}%)")
        print(f"  Training time: {training_time:.1f}s")
        
        # Cleanup
        result['neat_runner'].cleanup()
        
        return {
            'test_name': test_spec['name'],
            'difficulty': test_spec['difficulty'],
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'final_fitness': result['final_fitness'],
            'training_time': training_time,
            'generations': generations,
            'passed': accuracy >= 0.8  # 80% threshold for "passing"
        }
        
    except Exception as e:
        print(f"  Error testing agent: {e}")
        return {
            'test_name': test_spec['name'],
            'difficulty': test_spec['difficulty'],
            'accuracy': 0.0,
            'error': str(e),
            'passed': False
        }


def run_logic_test_battery(agent_id: str, max_difficulty: int = 6) -> List[Dict]:
    """Run progressive logic tests on an agent until failure or completion."""
    print(f"LOGIC TEST BATTERY FOR AGENT {agent_id[:12]}...")
    print("=" * 60)
    
    tests = LogicTestSuite.get_all_tests()
    results = []
    
    for test_spec in tests:
        if test_spec['difficulty'] > max_difficulty:
            break
        
        print(f"\nLevel {test_spec['difficulty']}: {test_spec['name']}")
        print(f"Description: {test_spec['description']}")
        print("-" * 40)
        
        result = test_agent_on_logic(agent_id, test_spec, generations=5)
        results.append(result)
        
        if not result['passed']:
            print(f"\nX Agent failed at Level {test_spec['difficulty']} - stopping tests")
            break
        else:
            print(f"\n+ Agent passed Level {test_spec['difficulty']}!")
    
    return results


def compare_agents_on_logic():
    """Compare all agents on logic test performance."""
    print("CONSTITUTIONAL AGENT LOGIC COMPARISON")
    print("=" * 50)
    
    agent_ids = list_all_agents()
    if not agent_ids:
        print("No agents found.")
        return
    
    all_results = {}
    
    # Test each agent on basic XOR first
    xor_test = LogicTestSuite.basic_xor()
    
    for agent_id in agent_ids[:5]:  # Test top 5 agents
        print(f"\nTesting agent {agent_id[:12]}...")
        try:
            agent_record = load_agent(agent_id)
            identity = agent_record.identity_bundle
            
            # Show agent's key traits
            constitution = identity.constitution_result.constitution
            print(f"  LearningRate: {constitution.get('LearningRate', 0):.3f}")
            print(f"  Innovation: {constitution.get('InnovationDrive', 0):.3f}")
            print(f"  Stability: {constitution.get('Stability', 0):.3f}")
            
            # Run basic XOR test
            result = test_agent_on_logic(agent_id, xor_test, generations=5)
            all_results[agent_id] = result
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Show comparison
    print(f"\n" + "="*60)
    print("LOGIC TEST COMPARISON (Basic XOR)")
    print("="*60)
    
    # Sort by accuracy
    sorted_results = sorted(all_results.items(), 
                          key=lambda x: x[1]['accuracy'], reverse=True)
    
    for i, (agent_id, result) in enumerate(sorted_results, 1):
        accuracy = result['accuracy'] * 100
        time_taken = result['training_time']
        fitness = result['final_fitness']
        
        print(f"{i}. {agent_id[:12]}... - {accuracy:5.1f}% accuracy, {time_taken:4.1f}s, fitness {fitness:.2f}")
    
    # Show breeding recommendations
    if len(sorted_results) >= 2:
        best_agent = sorted_results[0][0]
        second_agent = sorted_results[1][0]
        
        print(f"\nBREEDING RECOMMENDATION:")
        print(f"Best performers: {best_agent[:8]} x {second_agent[:8]}")
        print(f"Command: python agent_browser.py breed {best_agent[:8]} {second_agent[:8]}")


def main():
    """Main logic test runner."""
    parser = argparse.ArgumentParser(description='Constitutional Agent Logic Tests')
    parser.add_argument('agent_id', nargs='?', help='Agent ID to test (optional)')
    parser.add_argument('--all', action='store_true', help='Compare all agents')
    parser.add_argument('--max-level', type=int, default=6, help='Maximum difficulty level')
    
    args = parser.parse_args()
    
    if args.all:
        compare_agents_on_logic()
    elif args.agent_id:
        agent_ids = list_all_agents()
        matching = [aid for aid in agent_ids if aid.startswith(args.agent_id)]
        
        if not matching:
            print(f"No agent found starting with: {args.agent_id}")
            return
        
        if len(matching) > 1:
            print(f"Multiple matches: {[aid[:12] for aid in matching[:3]]}")
            return
        
        agent_id = matching[0]
        results = run_logic_test_battery(agent_id, args.max_level)
        
        # Summary
        passed_levels = sum(1 for r in results if r['passed'])
        print(f"\n" + "="*60)
        print(f"AGENT PERFORMANCE SUMMARY")
        print(f"Agent: {agent_id[:12]}...")
        print(f"Levels passed: {passed_levels}/{len(results)}")
        print(f"Highest difficulty: {passed_levels}")
        
        if passed_levels == len(LogicTestSuite.get_all_tests()):
            print("*** PERFECT SCORE - Agent mastered all logic tests!")
        elif passed_levels >= 4:
            print("*** EXCELLENT - Advanced logic capabilities")  
        elif passed_levels >= 2:
            print("*** GOOD - Solid basic logic learning")
        else:
            print("*** BASIC - Needs more training")
    
    else:
        # Show available tests
        print("AVAILABLE LOGIC TESTS:")
        print("=" * 30)
        
        for test in LogicTestSuite.get_all_tests():
            print(f"Level {test['difficulty']}: {test['name']}")
            print(f"  {test['description']}")
            print(f"  Inputs: {test['inputs']}, Test cases: {len(test['test_cases'])}")
            print()
        
        print("USAGE:")
        print("python logic_tests.py [agent_id]     # Test specific agent")
        print("python logic_tests.py --all          # Compare all agents")


if __name__ == "__main__":
    main()