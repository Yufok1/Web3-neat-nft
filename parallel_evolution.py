#!/usr/bin/env python3
"""
PARALLEL EVOLUTIONARY TRAINING - MAXIMUM CPU UTILIZATION
Trains multiple agents simultaneously to saturate available compute resources.
"""

import psutil
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from constitutional_ai.training.language_evolution import (
    train_agent_language_capability,
)
from constitutional_ai import (
    create_random_genome,
    create_agent_identity,
    COMPLETE_TRAIT_DEFINITIONS,
)
from constitutional_ai.persistence import (
    AgentPersistence,
    AgentRecord,
    AgentCapabilityRecord,
    list_all_agents,
    load_agent,
)
from constitutional_ai.governance import (
    GovernanceManager,
    GovernanceExecutor,
    create_governance_manager_from_agent_list,
)
from datetime import datetime


def log_system_resources():
    """Log current system resource usage."""
    try:
        memory = psutil.virtual_memory()
        ram_used = memory.used / 1024 / 1024 / 1024
        ram_total = memory.total / 1024 / 1024 / 1024
        print(f"🖥️  RAM: {ram_used:.1f}GB / {ram_total:.1f}GB ({memory.percent}%)")

        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"⚡ CPU: {cpu_percent}%")

    except Exception as e:
        print(f"⚠️  Resource monitoring error: {e}")


def train_single_agent(agent_id, generations=300):
    """Train a single agent with error handling."""
    try:
        print(f"🚀 STARTING Agent {agent_id}: {generations} generations")

        # Create constitutional genome
        seed = int(time.time() * 1000) % 100000 + agent_id * 1000
        genome = create_random_genome(COMPLETE_TRAIT_DEFINITIONS, seed=seed)
        identity = create_agent_identity(genome)

        print(f"🧬 Agent {agent_id}: {identity.id_hash[:12]}")
        print(f"   Population: {identity.neat_config.population_size}")

        # Add delay to stagger dataset loading and reduce I/O contention
        import random

        delay = random.uniform(0, 5)  # Random delay 0-5 seconds
        print(f"   Staggering start by {delay:.1f}s to reduce I/O contention...")
        time.sleep(delay)

        # Train with full power
        result = train_agent_language_capability(identity, generations=generations)

        # Create capability record
        capability_record = AgentCapabilityRecord(
            capability_type="language",
            training_generations=generations,
            final_fitness=result.get("final_fitness", 0.0),
            training_date=datetime.now().isoformat(),
            sample_outputs=result.get("sample_generation", {}),
            performance_metrics=result.get("performance_metrics", {}),
            training_config={},
        )

        # Create agent record
        agent_record = AgentRecord(
            agent_id=identity.id_hash,
            identity_bundle=identity,
            capabilities={"language": capability_record},
            creation_date=datetime.now().isoformat(),
            lineage={"source": "parallel_evolution", "agent_number": agent_id},
            notes=f"Parallel evolution agent {agent_id}",
        )

        # Save agent
        persistence = AgentPersistence()
        persistence.save_agent(agent_record)

        # Save trained network if available
        if "best_network" in result and "best_genome" in result:
            persistence.save_trained_network(
                identity.id_hash,
                "language",
                result["best_network"],
                result["best_genome"],
            )

        fitness = result.get("final_fitness", 0.0)
        print(f"✅ COMPLETED Agent {agent_id}: fitness {fitness:.3f}")

        return {
            "agent_id": agent_id,
            "identity_hash": identity.id_hash[:12],
            "fitness": fitness,
            "population_size": identity.neat_config.population_size,
            "success": True,
        }

    except Exception as e:
        print(f"❌ FAILED Agent {agent_id}: {e}")
        return {"agent_id": agent_id, "error": str(e), "success": False}


def parallel_evolution_training(num_agents=8, generations=300, max_workers=None):
    """
    Train multiple agents in parallel to maximize CPU utilization.

    Args:
        num_agents: Number of agents to train simultaneously
        generations: Generations per agent
        max_workers: Max parallel processes (auto-detects if None)
    """
    if max_workers is None:
        max_workers = multiprocessing.cpu_count() - 1
    print("=" * 80)
    print("🔥 PARALLEL EVOLUTIONARY TRAINING - MAXIMUM CPU UTILIZATION")
    print(f"📊 Training {num_agents} agents × {generations} generations")
    print(f"🔧 Max parallel workers: {max_workers}")
    print("=" * 80)

    # Log initial system resources
    print("📈 INITIAL SYSTEM RESOURCES:")
    log_system_resources()

    start_time = time.time()
    results = []

    # Submit all training jobs
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
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

                if result["success"]:
                    print(
                        f"🎯 Agent {agent_id} COMPLETE: {result['identity_hash']} (fitness: {result['fitness']:.3f})"
                    )
                else:
                    print(f"💥 Agent {agent_id} FAILED: {result['error']}")

            except Exception as e:
                print(f"🔥 Agent {agent_id} CRASHED: {e}")

    # Final summary
    total_time = time.time() - start_time
    successful_agents = [r for r in results if r.get("success", False)]

    print("\n" + "=" * 80)
    print("🏁 PARALLEL TRAINING COMPLETE")
    print(f"⏱️  Total time: {total_time/3600:.2f} hours")
    print(f"✅ Successful agents: {len(successful_agents)}/{num_agents}")

    if successful_agents:
        avg_fitness = sum(r["fitness"] for r in successful_agents) / len(
            successful_agents
        )
        best_agent = max(successful_agents, key=lambda x: x["fitness"])

        print(f"📈 Average fitness: {avg_fitness:.3f}")
        print(
            f"🏆 Best agent: {best_agent['identity_hash']} (fitness: {best_agent['fitness']:.3f})"
        )

        pop_sizes = [r["population_size"] for r in successful_agents]
        avg_pop = sum(pop_sizes) / len(pop_sizes)
        print(f"🔢 Average population size: {avg_pop:.0f}")

    print("=" * 80)
    return results


def governance_controlled_parallel_evolution(
    initial_num_agents=8,
    initial_generations=300,
    max_workers=None,
    enable_governance=True,
):
    """
    Run parallel evolution with governance control from existing agents.

    Existing agents vote on training parameters for new agents.
    """
    if max_workers is None:
        max_workers = multiprocessing.cpu_count() - 1
    print("=" * 80)
    print("🏛️  GOVERNANCE-CONTROLLED PARALLEL EVOLUTION")
    print("=" * 80)

    # Load existing agents for governance
    agent_ids = list_all_agents()
    if len(agent_ids) < 3 and enable_governance:
        print(
            "⚠️  Need at least 3 existing agents for governance. Running without governance..."
        )
        enable_governance = False

    # Set up governance parameters
    num_agents = initial_num_agents
    generations = initial_generations

    if enable_governance:
        print(f"🗳️  Loading {len(agent_ids)} existing agents for governance...")

        # Create governance manager
        governance_manager = create_governance_manager_from_agent_list(
            agent_ids[:10]
        )  # Use first 10 for performance
        governance_executor = GovernanceExecutor()

        print(f"📊 Governance council: {len(governance_manager.agents)} agents")

        # Proposal 1: Number of agents to train
        agent_proposal = governance_manager.propose_system_parameter_change(
            parameter_name="parallel_training_scale",
            new_parameters={"num_agents": num_agents},
            description=f"Train {num_agents} agents in parallel for population expansion",
        )

        print(f"\n📋 GOVERNANCE VOTE: Training Scale")
        agent_result = governance_manager.conduct_vote(agent_proposal)

        if not agent_result.passed:
            # Agents voted against current scale, reduce it
            num_agents = max(3, num_agents // 2)
            print(f"🔽 Agents voted to reduce training scale to {num_agents} agents")

        # Proposal 2: Training intensity
        intensity_proposal = governance_manager.propose_system_parameter_change(
            parameter_name="parallel_training_intensity",
            new_parameters={"generations": generations},
            description=f"Train each agent for {generations} generations",
        )

        print(f"\n📋 GOVERNANCE VOTE: Training Intensity")
        intensity_result = governance_manager.conduct_vote(intensity_proposal)

        if not intensity_result.passed:
            # Agents voted against current intensity
            generations = max(50, generations // 2)
            print(
                f"🔽 Agents voted to reduce training intensity to {generations} generations"
            )

        print(f"\n✅ GOVERNANCE DECISIONS APPLIED:")
        print(f"   Agents to train: {num_agents}")
        print(f"   Generations per agent: {generations}")
        print("=" * 50)

    # Execute the parallel training with governance-approved parameters
    print(f"🚀 Starting parallel training with governance-approved parameters...")
    results = parallel_evolution_training(
        num_agents=num_agents, generations=generations, max_workers=max_workers
    )

    # Post-training governance evaluation
    if enable_governance and results:
        successful_results = [r for r in results if r.get("success", False)]
        if successful_results:
            avg_fitness = sum(r["fitness"] for r in successful_results) / len(
                successful_results
            )

            print(f"\n🏛️  POST-TRAINING GOVERNANCE ASSESSMENT")
            print(f"📊 Average fitness achieved: {avg_fitness:.3f}")

            # Agents evaluate if the training was successful
            assessment_proposal = governance_manager.propose_system_parameter_change(
                parameter_name="training_assessment",
                new_parameters={
                    "avg_fitness": avg_fitness,
                    "success_rate": len(successful_results) / len(results),
                },
                description=f"Assess parallel training results: {avg_fitness:.3f} avg fitness, {len(successful_results)}/{len(results)} success rate",
            )

            print(f"\n📋 GOVERNANCE ASSESSMENT: Training Results")
            assessment_result = governance_manager.conduct_vote(assessment_proposal)

            if assessment_result.passed:
                print(
                    "✅ Agents approved the training results - parameters were effective"
                )
            else:
                print(
                    "❌ Agents disapproved the training results - consider adjusting parameters"
                )

    return results


if __name__ == "__main__":
    import sys

    # Dynamic resource allocation based on available hardware
    available_cores = multiprocessing.cpu_count()
    available_memory_gb = psutil.virtual_memory().total / (1024**3)

    # Calculate optimal parameters based on hardware
    optimal_workers = max(1, available_cores - 1)
    optimal_agents = min(32, optimal_workers * 2)  # 2 agents per core max
    optimal_generations = 150 if available_memory_gb < 16 else 200  # Adjust for memory

    print(f"🔧 HARDWARE DETECTED:")
    print(f"   CPU Cores: {available_cores}")
    print(f"   RAM: {available_memory_gb:.1f}GB")
    print(f"   Optimal Workers: {optimal_workers}")
    print(f"   Optimal Agents: {optimal_agents}")
    print(f"   Optimal Generations: {optimal_generations}")
    print("=" * 60)

    if len(sys.argv) > 1 and sys.argv[1] == "--governance":
        # GOVERNANCE-CONTROLLED PARALLEL EVOLUTION
        print("🏛️  Running governance-controlled parallel evolution...")
        results = governance_controlled_parallel_evolution(
            initial_num_agents=optimal_agents,
            initial_generations=optimal_generations,
            max_workers=optimal_workers,
            enable_governance=True,
        )
    else:
        # MAXIMUM CPU UTILIZATION - Parallel Processing (original)
        print("🚀 Running standard parallel evolution...")
        results = parallel_evolution_training(
            num_agents=optimal_agents,
            generations=optimal_generations,
            max_workers=optimal_workers,
        )

    print("🎉 EVOLUTION COMPLETE - Check your agents directory!")
    print(
        "💡 Tip: Use '--governance' flag to enable agent voting on training parameters"
    )
