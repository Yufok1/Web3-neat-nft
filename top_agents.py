#!/usr/bin/env python3
"""
Top Agents Display - Show agents with breeding commands
Usage: python top_agents.py
"""

from constitutional_ai.persistence import list_all_agents, load_agent
from constitutional_ai.color_mapping_simple import (
    traits_to_simple_color,
    get_simple_color_description,
)


def show_top_agents():
    """Show agents with detailed constitutional traits."""
    print("TOP 5 CONSTITUTIONAL AGENTS WITH TRAITS")
    print("=" * 60)

    try:
        # Get all agent IDs
        agent_ids = list_all_agents()

        if not agent_ids:
            print(
                "❌ No agents found. Run evolutionary_cycle.py or "
                "parallel_evolution.py first."
            )
            return

        print(f"📊 Found {len(agent_ids)} agents\n")

        # Load and analyze all agents
        agents_data = []
        for agent_id in agent_ids:
            try:
                agent = load_agent(agent_id)
                if agent:
                    fitness = 0.0
                    if agent.capabilities and "language" in agent.capabilities:
                        lang_cap = agent.capabilities["language"]
                        fitness = getattr(lang_cap, "final_fitness", 0.0)

                    # Get visual DNA
                    if hasattr(agent, "identity_bundle"):
                        identity_obj = agent.identity_bundle
                        traits = identity_obj.constitution_result.constitution
                        visual_dna = traits_to_simple_color(traits)

                        agents_data.append(
                            {
                                "id": agent_id,
                                "fitness": fitness,
                                "traits": traits,
                                "visual_dna": visual_dna,
                            }
                        )
            except Exception as e:
                print(f"⚠️  Error loading agent {agent_id[:12]}: {e}")
                continue

        if not agents_data:
            print("❌ No agents could be loaded successfully.")
            return

        # Sort by fitness (highest first)
        agents_data.sort(key=lambda x: x["fitness"], reverse=True)

        # Display top 5 agents with traits
        for i, agent_data in enumerate(agents_data[:5]):
            agent_id = agent_data["id"]
            fitness = agent_data["fitness"]
            traits = agent_data["traits"]
            visual_dna = agent_data["visual_dna"]
            color_desc = get_simple_color_description(visual_dna)

            print(f"{i+1}. Agent: {agent_id[:12]}...")
            print(
                f"   Visual DNA: {visual_dna} ({color_desc}) | "
                f"Fitness: {fitness:.3f}"
            )

            # Show ALL traits in alphabetical order
            sorted_traits = sorted(traits.items())
            print("   All Traits:")
            for name, value in sorted_traits:
                if isinstance(value, (int, float)):
                    print(f"     {name}: {value:.3f}")
                else:
                    print(f"     {name}: {value}")
            print()

        # Show breeding and training commands
        if len(agents_data) >= 2:
            top1 = agents_data[0]["id"]
            top2 = agents_data[1]["id"]

            print("🧬 BREEDING COMMANDS:")
            print(
                f"python agent_browser.py breed {top1[:12]} "
                f"{top2[:12]} --count 3 --generations 5"
            )
            print()
            print("🎯 TRAINING COMMANDS:")
            print(
                f"python agent_browser.py train {top1[:12]} "
                f"language --generations 10"
            )
            print(
                f"python agent_browser.py train {top2[:12]} " f"coding --generations 8"
            )
            print()
            print("📊 GOVERNANCE COMMANDS:")
            print("python test_deep_governance.py")
            print()
            print("🚀 EVOLUTION COMMANDS:")
            print("python evolutionary_cycle.py  # Continuous evolution")
            print("python parallel_evolution.py --governance  # Parallel")

        else:
            print("📝 Need at least 2 agents for breeding. Run:")
            print("python parallel_evolution.py")
            print("python evolutionary_cycle.py")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Ensure agents directory exists with agent files")
        print("2. Check that constitutional_ai modules are properly installed")
        print("3. Run: python quick_test.py to verify system")
        print("\n📋 Manual commands:")
        print("python agent_browser.py list")
        print("python agent_browser.py show [agent_id]")
        print("python agent_browser.py breed [id1] [id2]")


if __name__ == "__main__":
    show_top_agents()
