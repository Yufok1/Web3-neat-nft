#!/usr/bin/env python3
"""
PARALLEL EVOLUTIONARY TRAINING - MAX TPU UTILIZATION
Trains multiple agents simultaneously to saturate available compute resources.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from constitutional_ai.training.language_evolution import train_agent_language_capability
from constitutional_ai import create_random_genome, create_agent_identity, COMPLETE_TRAIT_DEFINITIONS
from constitutional_ai.persistence import save_training_result


def train_single_agent(agent_id, generations=150):
    """Train a single agent with error handling."""
    try:
        print(f"🚀 STARTING Agent {agent_id}: {generations} generations")
        
        # Create constitutional genome
        seed = int(time.time() * 1000) % 100000 + agent_id * 1000
        genome = create_random_genome(COMPLETE_TRAIT_DEFINITIONS, seed=seed)
        identity = create_agent_identity(genome)
        
        print(f"🧬 Agent {agent_id}: {identity.id_hash[:12]}, Pop: {identity.neat_config.population_size}")
        
        # Train with full power
        result = train_agent_language_capability(identity, generations=generations)
        
        # Save results
        save_training_result(result, "language")
        
        print(f"✅ COMPLETED Agent {agent_id}: fitness {result['final_fitness']:.3f}")
        
        # Cleanup
        result['neat_runner'].cleanup()
        
        return {
            'agent_id': agent_id,
            'identity_hash': identity.id_hash[:12],
            'fitness': result['final_fitness'],
            'population_size': identity.neat_config.population_size,
            'success': True
        }
        
    except Exception as e:
        print(f"❌ FAILED Agent {agent_id}: {e}")
        return {
            'agent_id': agent_id,
            'error': str(e),
            'success': False
        }


def parallel_evolution_training(num_agents=6, generations=150, max_workers=4):
    """
    Train multiple agents in parallel to maximize TPU utilization.
    
    Args:
        num_agents: Number of agents to train simultaneously
        generations: Generations per agent
        max_workers: Max parallel threads (adjust based on TPU cores)
    """
    print("=" * 80)
    print("🔥 PARALLEL EVOLUTIONARY TRAINING - MAXIMUM TPU SATURATION")
    print(f"📊 Training {num_agents} agents × {generations} generations")
    print(f"🔧 Max parallel workers: {max_workers}")
    print("=" * 80)
    
    start_time = time.time()
    results = []
    
    # Submit all training jobs
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Start all agents
        futures = {
            executor.submit(train_single_agent, i, generations): i 
            for i in range(1, num_agents + 1)
        }
        
        # Collect results as they complete
        for future in as_completed(futures):
            agent_id = futures[future]
            try:
                result = future.result()
                results.append(result)
                
                if result['success']:
                    print(f"🎯 Agent {agent_id} COMPLETE: {result['identity_hash']} (fitness: {result['fitness']:.3f})")
                else:
                    print(f"💥 Agent {agent_id} FAILED: {result['error']}")
                    
            except Exception as e:
                print(f"🔥 Agent {agent_id} CRASHED: {e}")
    
    # Final summary
    total_time = time.time() - start_time
    successful_agents = [r for r in results if r.get('success', False)]
    
    print("\n" + "=" * 80)
    print("🏁 PARALLEL TRAINING COMPLETE")
    print(f"⏱️  Total time: {total_time/3600:.2f} hours")
    print(f"✅ Successful agents: {len(successful_agents)}/{num_agents}")
    
    if successful_agents:
        avg_fitness = sum(r['fitness'] for r in successful_agents) / len(successful_agents)
        best_agent = max(successful_agents, key=lambda x: x['fitness'])
        
        print(f"📈 Average fitness: {avg_fitness:.3f}")
        print(f"🏆 Best agent: {best_agent['identity_hash']} (fitness: {best_agent['fitness']:.3f})")
        
        # Population sizes
        pop_sizes = [r['population_size'] for r in successful_agents]
        avg_pop = sum(pop_sizes) / len(pop_sizes)
        print(f"🔢 Average population size: {avg_pop:.0f}")
        
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    # OPTIMIZED FOR 3 AGENTS
    results = parallel_evolution_training(
        num_agents=3,       # 3 agents simultaneously  
        generations=150,    # Standard training duration
        max_workers=3       # One worker per agent
    )
    
    print("🎉 EVOLUTION COMPLETE - Check your agents directory!")