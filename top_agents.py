#!/usr/bin/env python3
"""
Top Agents Display - Show agents with breeding commands
Usage: python top_agents.py
"""

import subprocess
import sys


def show_top_agents():
    """Show agents with detailed constitutional traits."""
    print("TOP 5 CONSTITUTIONAL AGENTS WITH TRAITS")
    print("=" * 60)

    # Use the working agent browser to list agents
    try:
        list_result = subprocess.run(
            [sys.executable, "agent_browser.py", "list"],
            capture_output=True,
            text=True,
            cwd="C:/Users/Jeff Towers/projects/web3-neat-nft",
        )

        if list_result.returncode != 0:
            print("Error running agent browser list")
            return

        # Parse agent IDs from list output
        agent_ids = []
        agent_info = []

        for line in list_result.stdout.strip().split("\n"):
            if "Agent:" in line and "..." in line:
                # Extract: "1. Agent: 231466cbedc6817a..."
                parts = line.split("Agent: ")
                if len(parts) > 1:
                    agent_part = parts[1].split("...")[0]
                    agent_ids.append(agent_part)

                    # Extract basic info from list output
                    rank = line.split(".")[0].strip()
                    visual_dna = ""
                    fitness = ""

                    # Look for Visual DNA and fitness in next few lines
                    lines = list_result.stdout.strip().split("\n")
                    idx = lines.index(line)
                    if idx + 1 < len(lines) and "Visual DNA:" in lines[idx + 1]:
                        visual_dna = lines[idx + 1].split("Visual DNA: ")[1].split()[0]

                    # Look for fitness in format: "Best: base (fitness: 0.560)"
                    for i in range(1, 6):  # Check next 5 lines
                        if idx + i < len(lines):
                            line_check = lines[idx + i]
                            if "(fitness:" in line_check:
                                fitness_part = line_check.split("(fitness: ")[1]
                                fitness = fitness_part.split(")")[0]
                                break

                    agent_info.append(
                        {
                            "rank": rank,
                            "id": agent_part,
                            "visual_dna": visual_dna,
                            "fitness": fitness,
                        }
                    )

        print(f"Found {len(agent_ids)} agents\n")

        # Get detailed traits for each agent
        for i, info in enumerate(agent_info[:5]):
            print(f"{info['rank']}. Agent: {info['id'][:12]}...")
            print(f"   Visual DNA: {info['visual_dna']} | Fitness: {info['fitness']}")

            # Get detailed constitutional traits
            show_result = subprocess.run(
                [sys.executable, "agent_browser.py", "show", info["id"][:8]],
                capture_output=True,
                text=True,
                cwd="C:/Users/Jeff Towers/projects/web3-neat-nft",
            )

            if show_result.returncode == 0:
                # Parse constitutional traits from show output
                lines = show_result.stdout.strip().split("\n")
                traits_section = False
                traits = []

                for line in lines:
                    if "Constitutional Traits" in line:
                        traits_section = True
                        continue
                    elif traits_section and line.strip().startswith("Capabilities"):
                        break
                    elif traits_section and ":" in line:
                        # Parse trait line: "  Perception: 5.19"
                        trait_line = line.strip()
                        if trait_line and ":" in trait_line:
                            trait_name, trait_value = trait_line.split(":", 1)
                            traits.append(f"{trait_name.strip()}={trait_value.strip()}")

                # Display traits in rows of 3
                print("   Constitutional Traits:")
                for j in range(0, len(traits), 3):
                    trait_row = traits[j : j + 3]
                    trait_str = " | ".join(trait_row)
                    print(f"     {trait_str}")

            print()  # Blank line between agents

        # Show breeding and training commands
        if len(agent_ids) >= 2:
            print("BREEDING COMMANDS (Top Performers):")
            print("-" * 35)

            # Show top 3 breeding combinations
            for i in range(min(3, len(agent_ids))):
                for j in range(i + 1, min(3, len(agent_ids))):
                    id1 = agent_ids[i][:8]
                    id2 = agent_ids[j][:8]
                    print(f"python agent_browser.py breed {id1} {id2}")

            print(f"\nTRAINING COMMANDS (Language Focus):")
            print("-" * 32)

            # Show training commands for all agents
            for info in agent_info[:5]:
                short_id = info["id"][:8]
                print(
                    f"python agent_browser.py train {short_id} language --generations 5"
                )

        else:
            print("Error running agent browser:")
            print(list_result.stderr)

    except Exception as e:
        print(f"Error: {e}")
        print("\nFallback - use these commands manually:")
        print("python agent_browser.py list")
        print("python agent_browser.py show [agent_id]")
        print("python agent_browser.py breed [id1] [id2]")
        print("python agent_browser.py train [id] language --generations 5")


if __name__ == "__main__":
    show_top_agents()
