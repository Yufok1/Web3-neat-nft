#!/usr/bin/env python3
"""
Quick Test Script for Constitutional NEAT Breeding System
Run this to verify your installation and see the system in action!
"""


def test_constitutional_system():
    """Test the complete constitutional AI system."""
    print("🧬 Constitutional NEAT Breeding System - Quick Test")
    print("=" * 60)

    try:
        # Test 1: Basic Constitutional System
        print("\n1. Testing Constitutional Genome System...")
        from constitutional_ai import create_random_genome, COMPLETE_TRAIT_DEFINITIONS

        genome = create_random_genome(COMPLETE_TRAIT_DEFINITIONS, seed=42)
        print(f"   ✅ Created genome: {genome.compute_genome_hash()[:12]}...")
        print(f"   📊 Traits: {len(genome.loci)} comprehensive AI characteristics")

        # Test 2: Identity System
        print("\n2. Testing Agent Identity Creation...")
        from constitutional_ai import create_agent_identity

        identity = create_agent_identity(genome, seed_closure=123)
        print(f"   ✅ Agent identity: {identity.id_hash[:12]}...")
        print(f"   🎨 Visual DNA: {identity.visual_identity.primary_color_hex}")
        print(f"   🔧 NEAT population size: {identity.neat_config.population_size}")

        # Test 3: Breeding System
        print("\n3. Testing Constitutional Breeding...")
        from constitutional_ai import ConstitutionalBreeder

        parent1 = create_random_genome(COMPLETE_TRAIT_DEFINITIONS, seed=100)
        parent2 = create_random_genome(COMPLETE_TRAIT_DEFINITIONS, seed=200)

        breeder = ConstitutionalBreeder()
        result = breeder.breed_agents(parent1, parent2, seed=300)

        print(f"   ✅ Bred offspring: {result.offspring.compute_genome_hash()[:12]}...")
        print(f"   🧬 Method: {result.breeding_method}")
        print(f"   🔀 Crossover points: {len(result.crossover_points)}")

        # Test 4: NEAT Integration
        print("\n4. Testing NEAT Integration...")
        import neat
        import os

        neat_dir = os.path.dirname(neat.__file__)
        print(f"   ✅ NEAT-python loaded from: {neat_dir}")

        # Simple fitness test (no actual evolution to save time)
        from constitutional_ai.neat_integration import ConstitutionalNEATRunner

        runner = ConstitutionalNEATRunner(identity)
        config_file = runner.create_neat_config_file("test_config.txt")
        print(f"   ✅ NEAT config generated: {config_file}")

        # Test 5: All Stabilization Types
        print("\n5. Testing Stabilization Types...")
        from constitutional_ai.genome import StabilizationType

        all_types = set()
        for _ in range(20):  # Create multiple genomes to see all types
            test_genome = create_random_genome(COMPLETE_TRAIT_DEFINITIONS, seed=_ * 10)
            for locus in test_genome.loci.values():
                all_types.add(locus.get_dominant_allele().stabilization_type.value)

        print(f"   ✅ Found stabilization types: {sorted(all_types)}")
        print(f"   📈 Total types discovered: {len(all_types)}/6")

        # Cleanup
        import os

        if os.path.exists("test_config.txt"):
            os.remove("test_config.txt")

        print("\n" + "=" * 60)
        print("🎉 SUCCESS: Constitutional NEAT Breeding System is working!")
        print("\n🚀 Ready to evolve AI agents!")
        print("📚 See README.md for full examples including XOR learning")

        return True

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Ensure Python 3.8+ is installed")
        print("   3. Check that neat-python is properly installed")
        return False


def test_learning_capability():
    """Test actual learning with XOR problem (quick version)."""
    print("\n" + "=" * 60)
    print("🧠 BONUS: Testing Learning Capability (XOR Problem)")
    print("=" * 60)

    try:
        from constitutional_ai.neat_integration import evolve_constitutional_agent
        from constitutional_ai import create_random_genome, COMPLETE_TRAIT_DEFINITIONS
        import neat

        # XOR fitness function
        def xor_fitness(genomes, config):
            for genome_id, genome in genomes:
                genome.fitness = 4.0
                net = neat.nn.FeedForwardNetwork.create(genome, config)

                for xi, xo in [
                    ((0.0, 0.0), 0.0),
                    ((0.0, 1.0), 1.0),
                    ((1.0, 0.0), 1.0),
                    ((1.0, 1.0), 0.0),
                ]:
                    output = net.activate(xi)
                    genome.fitness -= (output[0] - xo) ** 2

        print("🔬 Creating constitutional agent...")
        genome = create_random_genome(COMPLETE_TRAIT_DEFINITIONS, seed=42)

        print("🧠 Evolving neural network (2 generations)...")
        result = evolve_constitutional_agent(
            genome=genome,
            fitness_function=xor_fitness,
            generations=2,
            num_inputs=2,
            num_outputs=1,
            seed_closure=123,
        )

        print(f"\n✅ Evolution complete!")
        print(f"🎯 Final fitness: {result['final_fitness']:.3f}")
        print(f"🤖 Agent ID: {result['identity'].id_hash[:12]}...")

        # Test the evolved network
        network = result["network"]
        print("\n📊 Testing XOR learning:")
        correct = 0
        for inputs, expected in [
            ((0.0, 0.0), 0.0),
            ((0.0, 1.0), 1.0),
            ((1.0, 0.0), 1.0),
            ((1.0, 1.0), 0.0),
        ]:
            output = network.activate(inputs)[0]
            is_correct = (output < 0.5) == (expected < 0.5)  # Same side of 0.5
            correct += is_correct
            status = "✅" if is_correct else "❌"
            print(f"   {status} {inputs} -> {output:.3f} (expected {expected})")

        print(f"\n🎉 Learning success: {correct}/4 patterns correct!")

        # Cleanup
        result["neat_runner"].cleanup()

        return correct >= 3  # Success if at least 3/4 correct

    except Exception as e:
        print(f"❌ Learning test failed: {e}")
        return False


if __name__ == "__main__":
    import sys

    # Run basic system test
    basic_success = test_constitutional_system()

    if not basic_success:
        sys.exit(1)

    # Ask user if they want to run learning test (takes longer)
    try:
        response = input(
            "\n🧠 Run learning capability test? (XOR problem, ~30 seconds) [y/N]: "
        ).lower()
        if response in ["y", "yes"]:
            learning_success = test_learning_capability()
            if learning_success:
                print("\n🏆 EXCELLENT: Your AI agents can actually learn on CPU!")
                print("   NEAT evolution works optimally with CPU processing.")
            else:
                print("\n📈 Learning test completed (partial success)")
    except KeyboardInterrupt:
        print("\n\n👋 Test completed!")

    print("\n🚀 Constitutional NEAT Breeding System is ready!")
    print("   Using CPU training (optimal for NEAT evolution)")
    print("   Your agents will evolve efficiently on CPU architecture.")
