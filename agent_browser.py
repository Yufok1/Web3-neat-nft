#!/usr/bin/env python3
"""
Constitutional Agent Browser - Interactive Multi-Capability AI Management

A comprehensive CLI for managing, training, testing, and breeding constitutional AI agents.
Supports multiple capabilities (language, coding) with cross-generational evolution.

Features:
- Agent persistence and loading
- Multi-capability training (language, coding, extensible)
- Cross-run breeding with genetic inheritance
- Interactive testing and evaluation
- Visual DNA and trait analysis

Usage Examples:
    python agent_browser.py list
    python agent_browser.py train [agent_id] language --generations 5
    python agent_browser.py breed [parent1] [parent2] --count 2
    python agent_browser.py test-coding [agent_id] "def hello"
"""

import argparse
from typing import Optional, List
from constitutional_ai.persistence import (
    load_agent,
    list_all_agents,
    AgentPersistence,
    save_training_result,
)
from constitutional_ai.training.language_evolution import (
    LanguageTrainingPipeline,
    train_agent_language_capability,
)
from constitutional_ai.training.coding_evolution import (
    CodingTrainingPipeline,
    train_agent_coding_capability,
)
from constitutional_ai.breeder import ConstitutionalBreeder
from constitutional_ai.identity import create_agent_identity


# ANSI color codes for terminal output
def hex_to_ansi(hex_color: str) -> str:
    """Convert hex color to closest ANSI terminal color."""
    # Remove # if present
    hex_color = hex_color.lstrip("#")

    # Convert to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    # Convert to ANSI 256-color
    # Use 16 + 36*r + 6*g + b formula for RGB cube
    ansi_r = round(r / 255 * 5)
    ansi_g = round(g / 255 * 5)
    ansi_b = round(b / 255 * 5)

    color_code = 16 + 36 * ansi_r + 6 * ansi_g + ansi_b
    return f"\033[38;5;{color_code}m"


def colorize_text(text: str, hex_color: str) -> str:
    """Colorize text using agent's visual DNA color."""
    color_code = hex_to_ansi(hex_color)
    reset_code = "\033[0m"
    return f"{color_code}{text}{reset_code}"


def print_agent_text(text: str, agent_record, prefix: str = ""):
    """Print text in the agent's visual DNA color."""
    if agent_record and hasattr(agent_record, "identity_bundle"):
        color = agent_record.identity_bundle.visual_identity.primary_color_hex
        colored_text = colorize_text(text, color)
        if prefix:
            print(f"{prefix}{colored_text}")
        else:
            print(colored_text)
    else:
        print(f"{prefix}{text}" if prefix else text)


def list_agents_command():
    """List all saved agents with their capabilities."""
    agents = list_all_agents()

    if not agents:
        print("No saved agents found.")
        return

    print(f"\nFound {len(agents)} saved agents:")
    print("=" * 80)

    persistence = AgentPersistence()

    for i, agent_id in enumerate(agents, 1):
        record = persistence.load_agent_record(agent_id)
        if record:
            print(f"\n{i}. Agent: {agent_id[:16]}...")
            print(
                f"   Visual DNA: {record.identity_bundle.visual_identity.primary_color_hex}"
            )
            print(f"   Creation: {record.creation_date[:19]}")
            print(f"   Capabilities: {', '.join(record.get_capability_names())}")

            # Show best capability
            best = record.get_best_capability()
            if best:
                cap_name, cap_record = best
                print(f"   Best: {cap_name} (fitness: {cap_record.final_fitness:.3f})")

            print(f"   Notes: {record.notes or 'No notes'}")


def show_agent_details(agent_id: str):
    """Show detailed information about a specific agent."""
    # Try partial ID matching
    agents = list_all_agents()
    matching_agents = [a for a in agents if a.startswith(agent_id)]

    if not matching_agents:
        print(f"No agent found with ID starting with: {agent_id}")
        return

    if len(matching_agents) > 1:
        print(f"Multiple agents match '{agent_id}':")
        for agent in matching_agents[:5]:
            print(f"  {agent[:16]}...")
        return

    full_agent_id = matching_agents[0]
    record = load_agent(full_agent_id)

    if not record:
        print(f"Failed to load agent: {full_agent_id}")
        return

    print(f"\nAgent Details: {full_agent_id[:16]}...")
    print("=" * 80)

    # Basic info
    print(f"Full ID: {record.agent_id}")
    print(f"Visual DNA: {record.identity_bundle.visual_identity.primary_color_hex}")
    print(f"Creation Date: {record.creation_date}")
    print(f"Notes: {record.notes or 'No notes'}")

    # Constitutional traits (top 5)
    traits = record.identity_bundle.constitution_result.constitution
    print(f"\nConstitutional Traits (top 5):")
    numeric_traits = {
        k: float(v)
        for k, v in traits.items()
        if isinstance(v, (int, float))
        or (isinstance(v, str) and v.replace(".", "").replace("-", "").isdigit())
    }

    if numeric_traits:
        sorted_traits = sorted(
            numeric_traits.items(), key=lambda x: x[1], reverse=True
        )[:5]
        for trait, value in sorted_traits:
            print(f"  {trait}: {value:.2f}")

    # Capabilities
    print(f"\nCapabilities ({len(record.capabilities)}):")
    for cap_name, cap_record in record.capabilities.items():
        print(f"  {cap_name}:")
        print(f"    Fitness: {cap_record.final_fitness:.3f}")
        print(f"    Generations: {cap_record.training_generations}")
        print(f"    Trained: {cap_record.training_date[:19]}")

        # Show sample output if available
        if cap_record.sample_outputs:
            sample = cap_record.sample_outputs.get("generated_text", "No sample")
            if len(sample) > 100:
                sample = sample[:100] + "..."
            print(f'    Sample: "{sample}"')


def test_agent_language(agent_id: str, prompt: str = "The quick brown"):
    """Test an agent's language generation capability."""
    # Find agent
    agents = list_all_agents()
    matching_agents = [a for a in agents if a.startswith(agent_id)]

    if not matching_agents:
        print(f"No agent found with ID starting with: {agent_id}")
        return

    full_agent_id = matching_agents[0]
    record = load_agent(full_agent_id)

    if not record:
        print(f"Failed to load agent: {full_agent_id}")
        return

    if "language" not in record.capabilities:
        print(f"Agent {full_agent_id[:16]}... has no language capability")
        return

    print(f"\nTesting language generation for agent {full_agent_id[:16]}...")
    print(f"Visual DNA: {record.identity_bundle.visual_identity.primary_color_hex}")

    # Load trained network
    persistence = AgentPersistence()
    network_data = persistence.load_trained_network(full_agent_id, "language")

    if not network_data:
        print("No trained language network found for this agent.")
        return

    network, genome = network_data

    # Create a pipeline to test with
    training_text = (
        """
    The quick brown fox jumps over the lazy dog. Language is a powerful tool for communication.
    Through words we share ideas emotions and knowledge. Artificial intelligence can learn language patterns.
    """
        * 5
    )

    try:
        pipeline = LanguageTrainingPipeline(training_text, record.identity_bundle)
        # Generate text
        generation_result = pipeline.fitness_evaluator.evaluate_generation_quality(
            network, seed_text=prompt, max_length=150, max_inputs=4
        )
        print(f'\nPrompt: "{prompt}"')
        print("Generated text:")
        sample = generation_result.get("sample_generation", {}).get(
            "generated_text", "No sample"
        )
        print_agent_text(f'"{sample}"', record, "  ")
        metrics = generation_result.get("performance_metrics", {})
        print(f"Length: {metrics.get('length', 0)} characters")
        print(f"Unique chars: {metrics.get('unique_chars', 0)}")
    except Exception as e:
        print(f"Error testing agent: {e}")


def interactive_chat_with_agent(agent_id: str):
    """Start an interactive chat session with an agent."""
    # Find agent
    agents = list_all_agents()
    matching_agents = [a for a in agents if a.startswith(agent_id)]

    if not matching_agents:
        print(f"No agent found with ID starting with: {agent_id}")
        return

    full_agent_id = matching_agents[0]
    record = load_agent(full_agent_id)

    if not record or "language" not in record.capabilities:
        print(f"Agent has no language capability")
        return

    # Load network
    persistence = AgentPersistence()
    network_data = persistence.load_trained_network(full_agent_id, "language")

    if not network_data:
        print("No trained language network found.")
        return

    network, genome = network_data

    # Setup pipeline
    training_text = "The quick brown fox jumps over the lazy dog. " * 20
    pipeline = LanguageTrainingPipeline(training_text, record.identity_bundle)

    print(f"\nChatting with Agent {full_agent_id[:16]}...")
    print(f"Visual DNA: {record.identity_bundle.visual_identity.primary_color_hex}")
    print(f"Language fitness: {record.capabilities['language'].final_fitness:.3f}")
    print("\nType 'quit' to exit, or enter prompts to see agent responses.")
    print("-" * 60)

    while True:
        try:
            prompt = input("\nYou: ").strip()

            if prompt.lower() in ["quit", "exit", "q"]:
                print("Chat ended.")
                break

            if not prompt:
                continue

            # Generate response
            result = pipeline.fitness_evaluator.evaluate_generation_quality(
                network, seed_text=prompt[:4], max_length=100, max_inputs=4
            )

            response = result["generated_text"][len(prompt[:4]) :]
            if len(response) > 200:
                response = response[:200] + "..."

            print(f"Agent: {response}")

        except KeyboardInterrupt:
            print("\nChat ended.")
            break
        except Exception as e:
            print(f"Error: {e}")


def breed_agents(
    parent1_id: str,
    parent2_id: str,
    count: int = 1,
    train_language: bool = True,
    generations: int = 3,
):
    """
    Breed two agents to create offspring with optional language training.
    
    Args:
        parent1_id: First parent agent ID (partial match supported)
        parent2_id: Second parent agent ID (partial match supported) 
        count: Number of offspring to create (default: 1)
        train_language: Whether to train offspring in language capability (default: True)
        generations: Training generations for offspring (default: 3)
    """
    print("CONSTITUTIONAL AGENT BREEDING")
    print("=" * 50)
    
    # Find parent agents
    agents = list_all_agents()
    parent1_matches = [a for a in agents if a.startswith(parent1_id)]
    parent2_matches = [a for a in agents if a.startswith(parent2_id)]
    
    if not parent1_matches:
        print(f"No agent found matching: {parent1_id}")
        return
    if not parent2_matches:
        print(f"No agent found matching: {parent2_id}")
        return
        
    if len(parent1_matches) > 1:
        print(f"Multiple matches for {parent1_id}: {[a[:12] for a in parent1_matches[:3]]}")
        return
    if len(parent2_matches) > 1:
        print(f"Multiple matches for {parent2_id}: {[a[:12] for a in parent2_matches[:3]]}")
        return
    
    # Load parent records
    parent1_record = load_agent(parent1_matches[0])
    parent2_record = load_agent(parent2_matches[0])
    
    if not parent1_record or not parent2_record:
        print("Could not load parent records")
        return
    
    # Get best fitness for each parent
    parent1_fitness = "unknown"
    parent2_fitness = "unknown"
    if parent1_record.capabilities:
        best_cap1 = parent1_record.get_best_capability()
        if best_cap1:
            parent1_fitness = f"{best_cap1[1].final_fitness:.3f}"
    if parent2_record.capabilities:
        best_cap2 = parent2_record.get_best_capability()
        if best_cap2:
            parent2_fitness = f"{best_cap2[1].final_fitness:.3f}"
            
    print(f"Parent 1: {parent1_record.agent_id[:12]}... (fitness: {parent1_fitness})")
    print(f"Parent 2: {parent2_record.agent_id[:12]}... (fitness: {parent2_fitness})")
    print(f"Creating {count} offspring with {generations} training generations each")
    print("-" * 50)
    
    # Initialize breeder
    breeder = ConstitutionalBreeder()
    successful_offspring = []
    
    for i in range(count):
        print(f"\nOffspring {i+1}/{count}:")
        print("-" * 20)
        
        try:
            # Breed constitutional genomes
            breeding_result = breeder.breed_agents(
                parent1_record.identity_bundle.genome,
                parent2_record.identity_bundle.genome,
                seed=hash(f"{parent1_record.agent_id}{parent2_record.agent_id}{i}") % 10000
            )
            
            offspring_genome = breeding_result.offspring
            print(f"  Bred genome: {offspring_genome.compute_genome_hash()[:12]}...")
            print(f"  Method: {breeding_result.breeding_method}")
            
            # Create identity for offspring
            offspring_identity = create_agent_identity(
                offspring_genome, 
                seed_closure=hash(f"offspring_{i}") % 10000
            )
            print(f"  Identity: {offspring_identity.id_hash[:12]}...")
            print(f"  Visual DNA: {offspring_identity.visual_identity.primary_color_hex}")
            
            if train_language:
                print(f"  Training language capability ({generations} generations)...")
                try:
                    # Train language capability
                    training_result = train_agent_language_capability(
                        offspring_identity, 
                        generations=generations
                    )
                    
                    # Add identity_bundle to training result for persistence
                    training_result['identity_bundle'] = offspring_identity
                    
                    # Save trained agent
                    agent_id = save_training_result(training_result, 'language')
                    print(f"  Trained and saved: {agent_id[:12]}...")
                    print(f"  Final fitness: {training_result.get('final_fitness', 'unknown')}")
                    
                    successful_offspring.append(agent_id)
                    
                except Exception as e:
                    print(f"  Training failed: {e}")
                    # Still save the untrained agent
                    from constitutional_ai.persistence import AgentRecord, AgentPersistence
                    from datetime import datetime
                    
                    record = AgentRecord(
                        agent_id=offspring_identity.id_hash,
                        identity_bundle=offspring_identity,
                        capabilities={},
                        creation_date=datetime.now().isoformat(),
                        lineage={
                            "parent1": parent1_record.agent_id,
                            "parent2": parent2_record.agent_id,
                            "breeding_method": breeding_result.breeding_method
                        },
                        notes=f"Untrained offspring of {parent1_record.agent_id[:12]}... and {parent2_record.agent_id[:12]}..."
                    )
                    persistence = AgentPersistence()
                    persistence.save_agent_record(record)
                    successful_offspring.append(offspring_identity.id_hash)
                    print(f"  Saved untrained: {offspring_identity.id_hash[:12]}...")
            else:
                # Save without training
                from constitutional_ai.persistence import AgentRecord, AgentPersistence
                from datetime import datetime
                
                record = AgentRecord(
                    agent_id=offspring_identity.id_hash,
                    identity_bundle=offspring_identity,
                    capabilities={},
                    creation_date=datetime.now().isoformat(),
                    lineage={
                        "parent1": parent1_record.agent_id,
                        "parent2": parent2_record.agent_id,
                        "breeding_method": breeding_result.breeding_method
                    },
                    notes=f"Offspring of {parent1_record.agent_id[:12]}... and {parent2_record.agent_id[:12]}... (no training)"
                )
                persistence = AgentPersistence()
                persistence.save_agent_record(record)
                successful_offspring.append(offspring_identity.id_hash)
                print(f"  Saved: {offspring_identity.id_hash[:12]}...")
                
        except Exception as e:
            print(f"  Failed to create offspring {i+1}: {e}")
            continue
    
    # Summary
    print("\n" + "=" * 50)
    print(f"BREEDING COMPLETE: {len(successful_offspring)}/{count} successful")
    if successful_offspring:
        print("New agents:")
        for agent_id in successful_offspring:
            print(f"  {agent_id[:12]}...")
        print(f"\nTo view: python agent_browser.py show {successful_offspring[0][:8]}")
        print(f"To breed further: python agent_browser.py breed {successful_offspring[0][:8]} {successful_offspring[-1][:8] if len(successful_offspring) > 1 else parent1_matches[0][:8]}")


def progressive_breed_agents(
    parent1_id: str,
    parent2_id: str,
    rounds: int = 3,
    offspring_per_round: int = 5,
    test_prompts: List[str] = None,
    min_fitness_threshold: float = 0.05,
):
    """
    Progressive breeding with comprehensive testing and multi-round selection.

    This creates a tournament-style breeding process where:
    1. Breed initial offspring from parents
    2. Test and evaluate all offspring
    3. Select top performers for next round breeding
    4. Repeat for specified number of rounds
    5. Return final generation statistics

    Args:
        parent1_id: First parent agent ID
        parent2_id: Second parent agent ID
        rounds: Number of breeding rounds (default: 3)
        offspring_per_round: Offspring per round (default: 5)
        test_prompts: Custom test prompts (default: standard set)
        min_fitness_threshold: Minimum fitness to continue breeding (default: 0.05)
    """

    if test_prompts is None:
        test_prompts = [
            "Hello world",
            "The quick brown",
            "Artificial intelligence",
            "Machine learning",
            "Neural networks",
        ]

    print("*** PROGRESSIVE BREEDING TOURNAMENT ***")
    print("=" * 60)
    print(f"Parents: {parent1_id[:12]}... + {parent2_id[:12]}...")
    print(f"Rounds: {rounds} | Offspring/Round: {offspring_per_round}")
    print(f"Min Fitness Threshold: {min_fitness_threshold}")
    print("=" * 60)

    # Track all generations
    all_generations = []
    elite_agents = []

    # Round 0: Initial parents
    print("\n*** ROUND 0: PARENT EVALUATION ***")
    print("-" * 40)

    # Find and load parents
    agents = list_all_agents()
    parent1_matches = [a for a in agents if a.startswith(parent1_id)]
    parent2_matches = [a for a in agents if a.startswith(parent2_id)]

    if not parent1_matches or not parent2_matches:
        print("❌ Could not find parent agents")
        return

    parent1_record = load_agent(parent1_matches[0])
    parent2_record = load_agent(parent2_matches[0])

    if not parent1_record or not parent2_record:
        print("❌ Could not load parent records")
        return

    # Test parent fitness
    parent1_fitness = test_agent_fitness(parent1_record, test_prompts[:3])
    parent2_fitness = test_agent_fitness(parent2_record, test_prompts[:3])

    print(f"  Parent 1 Fitness: {parent1_fitness:.3f}")
    print(f"  Parent 2 Fitness: {parent2_fitness:.3f}")
    elite_agents.extend([parent1_record, parent2_record])

    # Progressive breeding rounds
    for round_num in range(1, rounds + 1):
        print(f"\n*** ROUND {round_num}: BREEDING & TESTING ***")
        print("-" * 40)

        current_elite = elite_agents[-2:]  # Use last 2 elites as parents
        round_offspring = []

        print(
            f"Breeding from: {current_elite[0].agent_id[:12]}... + {current_elite[1].agent_id[:12]}..."
        )

        # Create offspring for this round
        for i in range(offspring_per_round):
            print(f"\n  >> Creating offspring {i+1}/{offspring_per_round}...")

            try:
                # Breed new offspring
                breeder = ConstitutionalBreeder()
                breeding_result = breeder.breed_agents(
                    current_elite[0].identity_bundle.genome,
                    current_elite[1].identity_bundle.genome,
                    seed=42 + round_num * 100 + i,
                )

                # Create identity
                offspring_identity = create_agent_identity(
                    breeding_result.offspring,
                    seed_closure=123 + round_num * 100 + i,
                    seed_build=456 + round_num * 100 + i,
                )

                # Train the offspring
                print(f"    🧠 Training language capability (3 generations)...")
                training_result = train_agent_language_capability(
                    offspring_identity, generations=3
                )

                # Save the trained agent
                offspring_id = save_training_result(training_result, "language")
                offspring_record = load_agent(offspring_id)

                # Test the offspring
                print(f"    🧪 Testing offspring performance...")
                offspring_fitness = test_agent_fitness(offspring_record, test_prompts)

                print(f"    📊 Fitness: {offspring_fitness:.3f}")
                print(
                    f"    🧬 DNA: {offspring_record.identity_bundle.visual_identity.primary_color_hex}"
                )
                # Show sample generation
                if "sample_generation" in training_result:
                    sample = training_result["sample_generation"]["generated_text"]
                    if len(sample) > 60:
                        sample = sample[:60] + "..."
                    print(f'    💬 Sample: "{sample}"')

                round_offspring.append((offspring_record, offspring_fitness))

            except Exception as e:
                print(f"    ❌ Failed to create offspring {i+1}: {e}")

        # Evaluate round results
        if round_offspring:
            # Sort by fitness
            round_offspring.sort(key=lambda x: x[1], reverse=True)

            print("\n📊 ROUND RESULTS:")
            for i, (agent, fitness) in enumerate(round_offspring[:3], 1):  # Top 3
                status = "👑 ELITE" if i == 1 else "🥈" if i == 2 else "🥉"
                print(f"  {status}: {agent.agent_id[:12]}... | Fitness: {fitness:.3f}")
            # Add top performers to elite pool
            best_offspring, best_fitness = round_offspring[0]
            if best_fitness > min_fitness_threshold:
                elite_agents.append(best_offspring)
                print(f"    ➕ Added to elite pool (fitness: {best_fitness:.3f})")
            else:
                print(
                    f"    ❌ Below threshold ({best_fitness:.3f} < {min_fitness_threshold}), stopping"
                )
                break  # Stop if we can't improve

            all_generations.append(
                {
                    "round": round_num,
                    "offspring_count": len(round_offspring),
                    "best_fitness": best_fitness,
                    "avg_fitness": sum(f for _, f in round_offspring)
                    / len(round_offspring),
                    "elite_count": len(elite_agents),
                }
            )
        else:
            print("❌ No successful offspring this round")
            break

    # Final tournament results
    print("\n🎉 **TOURNAMENT COMPLETE** 🎉")
    print("=" * 60)

    if all_generations:
        print("📈 GENERATION PROGRESS:")
        for gen in all_generations:
            print(
                f"  Round {gen['round']}: {gen['offspring_count']} offspring | "
                ".3f"
                f" | {gen['elite_count']} elites"
            )

    if elite_agents:
        print("\n*** FINAL ELITE AGENTS: ***")
        for i, agent in enumerate(elite_agents[-3:], 1):  # Show last 3 elites
            elite_marker = (
                ">>> CHAMPION" if i == len(elite_agents[-3:]) else f">>> ELITE #{i}"
            )
            print(
                f"  {elite_marker}: {agent.agent_id[:16]}... "
                f"(DNA: {agent.identity_bundle.visual_identity.primary_color_hex})"
            )

        print("\n🧪 TEST YOUR CHAMPIONS:")
        for agent in elite_agents[-2:]:  # Test the final two
            print(
                f"  python agent_browser.py test {agent.agent_id[:8]} 'Hello champion!'"
            )
            print(f"  python agent_browser.py show {agent.agent_id[:8]}")

    return {
        "generations": all_generations,
        "elite_agents": [agent.agent_id for agent in elite_agents],
        "total_offspring": sum(gen["offspring_count"] for gen in all_generations),
    }


def test_agent_fitness(agent_record, test_prompts: List[str]) -> float:
    """
    Test an agent's fitness across multiple prompts.

    Returns average fitness score across all test prompts.
    """
    if not agent_record or "language" not in agent_record.capabilities:
        return 0.0

    try:
        # Load trained network
        persistence = AgentPersistence()
        network_data = persistence.load_trained_network(
            agent_record.agent_id, "language"
        )

        if not network_data:
            return 0.0

        network, genome = network_data

        # Create test pipeline
        training_text = "The quick brown fox jumps over the lazy dog. " * 10
        pipeline = LanguageTrainingPipeline(training_text, agent_record.identity_bundle)

        total_fitness = 0.0
        successful_tests = 0

        for prompt in test_prompts:
            try:
                # Test generation quality
                result = pipeline.fitness_evaluator.evaluate_generation_quality(
                    network, seed_text=prompt[:4], max_length=50, max_inputs=4
                )

                # Simple fitness based on generation length and coherence
                generated_text = result.get("generated_text", "")
                if len(generated_text) > 10:
                    # Basic coherence check (has some letters)
                    coherence = sum(1 for c in generated_text if c.isalpha()) / len(
                        generated_text
                    )
                    total_fitness += coherence
                    successful_tests += 1

            except Exception:
                continue

        return total_fitness / max(successful_tests, 1)

    except Exception:
        return 0.0


def train_agent_capability(
    agent_id: str, capability: str, generations: int = 5, use_gpu: bool = None
):
    """Train an existing agent in a new capability."""

    # Find agent
    agents = list_all_agents()
    matching_agents = [a for a in agents if a.startswith(agent_id)]

    if not matching_agents:
        print(f"No agent found with ID starting with: {agent_id}")
        return

    if len(matching_agents) > 1:
        print(
            f"Multiple agents match '{agent_id}': {[a[:16] + '...' for a in matching_agents[:3]]}"
        )
        return

    full_agent_id = matching_agents[0]
    record = load_agent(full_agent_id)

    if not record:
        print(f"Failed to load agent: {full_agent_id}")
        return

    print(f"\\nTraining Agent {full_agent_id[:16]}... in {capability} capability")
    print(f"Visual DNA: {record.identity_bundle.visual_identity.primary_color_hex}")
    print(
        f"Current capabilities: {', '.join(record.capabilities.keys()) if record.capabilities else 'None'}"
    )

    if capability in record.capabilities:
        print(
            f"Agent already has {capability} capability (fitness: {record.capabilities[capability].final_fitness:.3f})"
        )
        print("Training will improve existing capability...")

    try:
        # Determine GPU usage
        actual_use_gpu = True  # Default to GPU
        if use_gpu is not None:
            actual_use_gpu = use_gpu

        if actual_use_gpu:
            print(f"🚀 Using GPU acceleration for {capability} training...")
        else:
            print(f"💻 Using CPU for {capability} training...")

        # Train in the requested capability
        if capability == "language":
            training_result = train_agent_language_capability(
                record.identity_bundle, generations=generations, use_gpu=actual_use_gpu
            )
        elif capability == "coding":
            training_result = train_agent_coding_capability(
                record.identity_bundle, generations=generations, use_gpu=actual_use_gpu
            )
        else:
            print(f"Unsupported capability: {capability}")
            print("Supported capabilities: language, coding")
            return

        print(f"\\nTraining complete!")
        print(
            f"Final fitness: {training_result.get('final_fitness', training_result.get('fitness', 0.0)):.3f}"
        )

        # Show sample output
        if "sample_generation" in training_result:
            if capability == "language":
                sample = training_result["sample_generation"].get(
                    "generated_text", "No sample"
                )
            elif capability == "coding":
                sample = training_result["sample_generation"].get(
                    "generated_code", "No sample"
                )
            else:
                sample = str(training_result["sample_generation"])

            if len(sample) > 100:
                sample = sample[:100] + "..."
            print(f'Sample output: "{sample}"')

        # Update the agent record with new capability
        from constitutional_ai.persistence import AgentCapabilityRecord
        from datetime import datetime

        new_capability = AgentCapabilityRecord(
            capability_type=capability,
            training_generations=training_result.get(
                "training_generations", generations
            ),
            final_fitness=training_result.get("final_fitness", 0.0),
            training_date=datetime.now().isoformat(),
            sample_outputs=training_result.get("sample_generation", {}),
            performance_metrics=training_result.get("performance_metrics", {}),
            training_config={},
        )

        # Add/update capability
        record.capabilities[capability] = new_capability
        record.notes = f"Trained in {', '.join(record.capabilities.keys())}"

        # Save updated agent
        persistence = AgentPersistence()
        persistence.save_agent(record)

        # Save trained network
        if "best_network" in training_result and "best_genome" in training_result:
            persistence.save_trained_network(
                record.agent_id,
                capability,
                training_result["best_network"],
                training_result["best_genome"],
            )

        print(f"\\nAgent updated successfully!")
        print(f"Test the new capability with:")
        print(f"  python agent_browser.py test-{capability} {agent_id[:8]}")

    except Exception as e:
        print(f"Training failed: {e}")


def test_agent_coding(agent_id: str, prompt: str = "def hello"):
    """Test an agent's coding generation capability."""
    # Find agent
    agents = list_all_agents()
    matching_agents = [a for a in agents if a.startswith(agent_id)]

    if not matching_agents:
        print(f"No agent found with ID starting with: {agent_id}")
        return

    full_agent_id = matching_agents[0]
    record = load_agent(full_agent_id)

    if not record:
        print(f"Failed to load agent: {full_agent_id}")
        return

    if "coding" not in record.capabilities:
        print(f"Agent {full_agent_id[:16]}... has no coding capability")
        print("Train it first with:")
        print(f"  python agent_browser.py train {agent_id[:8]} coding")
        return

    print(f"\\nTesting coding generation for agent {full_agent_id[:16]}...")
    print(f"Visual DNA: {record.identity_bundle.visual_identity.primary_color_hex}")

    # Load trained network
    persistence = AgentPersistence()
    network_data = persistence.load_trained_network(full_agent_id, "coding")

    if not network_data:
        print("No trained coding network found for this agent.")
        return

    network, genome = network_data

    # Create a pipeline to test with
    from constitutional_ai.training.coding_evolution import (
        create_coding_training_corpus,
        CodingTrainingPipeline,
    )

    training_data = create_coding_training_corpus()

    try:
        pipeline = CodingTrainingPipeline(training_data, record.identity_bundle)

        # Generate code
        generation_result = pipeline.fitness_evaluator.evaluate_generation_quality(
            network, prompt=prompt, max_length=80
        )

        print(f'\\nPrompt: "{prompt}"')
        print(f"Generated Code:")
        print("-" * 40)
        generated_code = generation_result.sample_outputs["generated_code"]
        print(generated_code)
        print("-" * 40)
        print(f"Length: {generation_result.performance_metrics['length']} characters")
        print(
            f"Syntax score: {generation_result.performance_metrics['syntax_score']:.3f}"
        )
        print(
            f"Pattern score: {generation_result.performance_metrics['pattern_score']:.3f}"
        )

    except Exception as e:
        print(f"Error testing agent: {e}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Constitutional Agent Browser and Tester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agent_browser.py list                           # List all saved agents
  python agent_browser.py show 9488b0a1                  # Show agent details
  python agent_browser.py test 9488b0a1                  # Test language generation
  python agent_browser.py test 9488b0a1 "Hello world"    # Test with custom prompt
  python agent_browser.py chat 9488b0a1                  # Interactive chat
  python agent_browser.py breed 9488b0a1 725af723        # Breed two agents
  python agent_browser.py progressive-breed 9488b0a1 725af723 --rounds 3 --offspring-per-round 5  # Progressive tournament breeding
  python agent_browser.py train 9488b0a1 coding          # Train agent in coding
  python agent_browser.py test-coding 9488b0a1           # Test coding capability
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # List command
    subparsers.add_parser("list", help="List all saved agents")

    # Show command
    show_parser = subparsers.add_parser("show", help="Show agent details")
    show_parser.add_argument("agent_id", help="Agent ID (partial match supported)")

    # Test command
    test_parser = subparsers.add_parser("test", help="Test agent language generation")
    test_parser.add_argument("agent_id", help="Agent ID (partial match supported)")
    test_parser.add_argument(
        "prompt",
        nargs="?",
        default="The quick brown",
        help='Prompt for generation (default: "The quick brown")',
    )

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive chat with agent")
    chat_parser.add_argument("agent_id", help="Agent ID (partial match supported)")

    # Breed command
    breed_parser = subparsers.add_parser(
        "breed", help="Breed two agents to create offspring"
    )
    breed_parser.add_argument(
        "parent1_id", help="First parent agent ID (partial match supported)"
    )
    breed_parser.add_argument(
        "parent2_id", help="Second parent agent ID (partial match supported)"
    )
    breed_parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of offspring to create (default: 1)",
    )
    breed_parser.add_argument(
        "--no-train", action="store_true", help="Skip language training for offspring"
    )
    breed_parser.add_argument(
        "--generations", type=int, default=3, help="Training generations (default: 3)"
    )

    # Train command
    train_parser = subparsers.add_parser(
        "train", help="Train agent in a new capability"
    )
    train_parser.add_argument("agent_id", help="Agent ID (partial match supported)")
    train_parser.add_argument(
        "capability", choices=["language", "coding"], help="Capability to train"
    )
    train_parser.add_argument(
        "--generations", type=int, default=5, help="Training generations (default: 5)"
    )
    train_parser.add_argument(
        "--gpu", action="store_true", help="Use GPU acceleration (default: auto-detect)"
    )
    train_parser.add_argument(
        "--no-gpu", action="store_true", help="Force CPU training (disable GPU)"
    )

    # Progressive breed command
    progressive_parser = subparsers.add_parser(
        "progressive-breed",
        help="Progressive tournament breeding with comprehensive testing",
    )
    progressive_parser.add_argument(
        "parent1_id", help="First parent agent ID (partial match supported)"
    )
    progressive_parser.add_argument(
        "parent2_id", help="Second parent agent ID (partial match supported)"
    )
    progressive_parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of breeding rounds (default: 3)",
    )
    progressive_parser.add_argument(
        "--offspring-per-round",
        type=int,
        default=5,
        help="Offspring per round (default: 5)",
    )
    progressive_parser.add_argument(
        "--min-fitness",
        type=float,
        default=0.05,
        help="Minimum fitness threshold to continue (default: 0.05)",
    )
    progressive_parser.add_argument(
        "--test-prompts",
        nargs="+",
        help="Custom test prompts (default: standard set)",
    )

    args = parser.parse_args()

    if args.command == "list":
        list_agents_command()
    elif args.command == "show":
        show_agent_details(args.agent_id)
    elif args.command == "test":
        test_agent_language(args.agent_id, args.prompt)
    elif args.command == "chat":
        interactive_chat_with_agent(args.agent_id)
    elif args.command == "breed":
        breed_agents(
            args.parent1_id,
            args.parent2_id,
            count=args.count,
            train_language=not args.no_train,
            generations=args.generations,
        )
    elif args.command == "train":
        # Determine GPU usage from command line flags
        use_gpu = None
        if args.no_gpu:
            use_gpu = False
        elif args.gpu:
            use_gpu = True
        # If neither flag is set, use_gpu remains None (auto-detect)

        train_agent_capability(
            args.agent_id, args.capability, args.generations, use_gpu
        )
    elif args.command == "progressive-breed":
        progressive_breed_agents(
            args.parent1_id,
            args.parent2_id,
            rounds=args.rounds,
            offspring_per_round=args.offspring_per_round,
            test_prompts=args.test_prompts,
            min_fitness_threshold=args.min_fitness,
        )


if __name__ == "__main__":
    main()
