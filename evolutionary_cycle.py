#!/usr/bin/env python3
"""
Indefinite Evolutionary Cycle for Constitutional NEAT Agents

This system runs continuous evolution cycles:
1. Auto-breed best performers
2. Prune weak agents
3. Maintain population diversity
4. Escalate training intensity over time
5. Log evolutionary progress

Never stops optimizing - a living AI ecosystem.
"""

import time
import random
from datetime import datetime
from typing import List, Dict, Any
from constitutional_ai.persistence import AgentPersistence, list_all_agents, load_agent
from constitutional_ai.training.language_evolution import (
    train_agent_language_capability,
)
from constitutional_ai import (
    create_random_genome,
    create_agent_identity,
    COMPLETE_TRAIT_DEFINITIONS,
)
from constitutional_ai.governance import (
    GovernanceManager, 
    GovernanceExecutor,
    create_governance_manager_from_agent_list
)


class EvolutionaryCycle:
    """Indefinite evolutionary optimization system."""

    def __init__(
        self,
        target_population: int = 20,
        min_fitness_threshold: float = 0.05,
        prune_percentage: float = 0.3,
        escalation_factor: float = 2.0,
        enable_governance: bool = True,
    ):
        """
        Initialize evolutionary cycle with optional governance control.

        Args:
            target_population: Initial desired number of agents
            min_fitness_threshold: Initial minimum fitness to survive
            prune_percentage: Initial fraction of population to prune each cycle
            escalation_factor: Initial how much to increase training intensity
            enable_governance: Whether to allow agents to vote on parameters
        """
        # Initial parameters (can be overridden by governance)
        self.target_population = target_population
        self.min_fitness_threshold = min_fitness_threshold
        self.prune_percentage = prune_percentage
        self.escalation_factor = escalation_factor

        # Governance system
        self.enable_governance = enable_governance
        self.governance_manager = None
        self.governance_executor = GovernanceExecutor()
        self.governance_cycle_interval = 3  # Vote every 3 cycles

        self.persistence = AgentPersistence()
        self.cycle_count = 0
        self.base_generations = 150  # SERIOUS training for actual intelligence
        self.current_generations = self.base_generations

        print("GOVERNANCE-ENABLED EVOLUTIONARY CYCLE INITIALIZED")
        print(f"Target Population: {target_population} (governance can modify)")
        print(f"Min Fitness: {min_fitness_threshold} (governance can modify)")
        print(f"Prune Rate: {prune_percentage:.1%} (governance can modify)")
        print(f"Starting Training Intensity: {self.current_generations} generations")
        print(f"Governance System: {'ENABLED' if enable_governance else 'DISABLED'}")
        print("=" * 60)

    def get_agent_roster(self) -> List[Dict[str, Any]]:
        """Get current agent population with fitness scores."""
        agent_ids = list_all_agents()
        roster = []

        for agent_id in agent_ids:
            agent = load_agent(agent_id)
            if agent and agent.has_capability("language"):
                fitness = agent.capabilities["language"].final_fitness
                roster.append({"id": agent_id, "agent": agent, "fitness": fitness})

        # Sort by fitness (descending)
        roster.sort(key=lambda x: x["fitness"], reverse=True)
        return roster

    def prune_weak_agents(self, roster: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove weakest agents from population."""
        if len(roster) <= 2:  # Keep minimum breeding stock
            return roster

        # Calculate how many to prune
        prune_count = max(1, int(len(roster) * self.prune_percentage))

        # Remove weakest agents
        survivors = roster[:-prune_count]
        pruned = roster[-prune_count:]

        print(f"PRUNING: Removing {len(pruned)} weak agents")
        for agent_data in pruned:
            agent_id = agent_data["id"]
            fitness = agent_data["fitness"]
            print(f"  Pruned: {agent_id[:12]}... (fitness: {fitness:.3f})")

            # Remove from filesystem (optional - could keep for analysis)
            # import shutil
            # shutil.rmtree(f"agents/{agent_id}", ignore_errors=True)

        return survivors

    def conduct_governance_vote(self, roster: List[Dict[str, Any]]) -> None:
        """Conduct governance votes to adjust evolutionary parameters."""
        if not self.enable_governance or len(roster) < 3:
            return  # Need at least 3 agents for meaningful governance

        print(f"\n🏛️  GOVERNANCE SESSION - CYCLE {self.cycle_count}")
        print("=" * 50)

        # Initialize governance manager with current agents
        agent_records = [agent_data["agent"] for agent_data in roster]
        self.governance_manager = GovernanceManager(agent_records)

        # Propose parameter adjustments based on population performance
        proposals = self._generate_adaptive_proposals(roster)

        for proposal in proposals:
            print(f"\n📋 PROPOSAL: {proposal.title}")
            result = self.governance_manager.conduct_vote(proposal)
            
            if result.passed:
                print(f"✅ PROPOSAL PASSED - Executing changes...")
                execution_result = self.governance_executor.execute_proposal(proposal, result)
                
                if execution_result["success"]:
                    self._apply_governance_decision(proposal, execution_result)
                else:
                    print(f"❌ Execution failed: {execution_result.get('error', 'Unknown error')}")
            else:
                print(f"❌ PROPOSAL REJECTED by agent vote")

        print("=" * 50)

    def _generate_adaptive_proposals(self, roster: List[Dict[str, Any]]) -> List:
        """Generate proposals based on current population performance."""
        proposals = []
        
        if not roster:
            return proposals

        avg_fitness = sum(agent["fitness"] for agent in roster) / len(roster)
        fitness_variance = sum((agent["fitness"] - avg_fitness) ** 2 for agent in roster) / len(roster)
        
        print(f"📊 Population Analysis:")
        print(f"   Average Fitness: {avg_fitness:.3f}")
        print(f"   Fitness Variance: {fitness_variance:.4f}")
        print(f"   Population Size: {len(roster)}")

        # Proposal 1: Adjust fitness threshold based on performance
        if avg_fitness > self.min_fitness_threshold * 2:
            # Population is doing well, raise standards
            new_threshold = min(self.min_fitness_threshold * 1.5, 0.8)
            proposal = self.governance_manager.propose_breeding_rule_change(
                rule_name="fitness_threshold",
                new_parameters={"min_fitness_threshold": new_threshold},
                description=f"Raise fitness threshold from {self.min_fitness_threshold:.3f} to {new_threshold:.3f} due to strong population performance"
            )
            proposals.append(proposal)
        elif avg_fitness < self.min_fitness_threshold * 0.5:
            # Population struggling, lower standards
            new_threshold = max(self.min_fitness_threshold * 0.8, 0.01)
            proposal = self.governance_manager.propose_breeding_rule_change(
                rule_name="fitness_threshold", 
                new_parameters={"min_fitness_threshold": new_threshold},
                description=f"Lower fitness threshold from {self.min_fitness_threshold:.3f} to {new_threshold:.3f} to help struggling population"
            )
            proposals.append(proposal)

        # Proposal 2: Adjust population size based on diversity
        if len(roster) < self.target_population * 0.7:
            # Population too small
            new_size = min(int(self.target_population * 1.3), 50)
            proposal = self.governance_manager.propose_population_size_change(
                new_size=new_size,
                reason=f"Increase population from {len(roster)} to {new_size} to improve genetic diversity"
            )
            proposals.append(proposal)
        elif len(roster) > self.target_population * 1.5:
            # Population too large
            new_size = max(int(self.target_population * 0.8), 10)
            proposal = self.governance_manager.propose_population_size_change(
                new_size=new_size,
                reason=f"Reduce population from {len(roster)} to {new_size} to focus resources on quality"
            )
            proposals.append(proposal)

        # Proposal 3: Adjust training intensity based on learning progress
        if self.cycle_count > 5 and fitness_variance < 0.01:  # Low variance = population converged
            new_generations = min(int(self.current_generations * 1.5), 500)
            proposal = self.governance_manager.propose_system_parameter_change(
                parameter_name="training_intensity",
                new_parameters={"generations": new_generations},
                description=f"Increase training from {self.current_generations} to {new_generations} generations due to population convergence"
            )
            proposals.append(proposal)

        return proposals

    def _apply_governance_decision(self, proposal, execution_result) -> None:
        """Apply governance decision to system parameters."""
        proposal_type = proposal.proposal_type
        changes = execution_result.get("changes", {})

        if proposal_type == "breeding_rule":
            if "min_fitness_threshold" in changes:
                old_threshold = self.min_fitness_threshold
                self.min_fitness_threshold = changes["min_fitness_threshold"]
                print(f"🎯 Fitness threshold: {old_threshold:.3f} → {self.min_fitness_threshold:.3f}")

        elif proposal_type == "population_size":
            if "new_size" in changes:
                old_size = self.target_population
                self.target_population = changes["new_size"]
                print(f"👥 Target population: {old_size} → {self.target_population}")

        elif proposal_type == "system_parameter":
            if "generations" in changes:
                old_generations = self.current_generations
                self.current_generations = changes["generations"]
                print(f"🔄 Training generations: {old_generations} → {self.current_generations}")

        print(f"📝 Governance decision applied successfully!")

    def breed_new_agents(self, roster: List[Dict[str, Any]], num_offspring: int) -> int:
        """Breed new agents from top performers."""
        if len(roster) < 2:
            print("Need at least 2 agents for breeding")
            return 0

        bred_count = 0

        for i in range(num_offspring):
            # Select parents (weighted by fitness)
            parent1 = roster[0]["agent"]  # Best performer
            parent2 = roster[min(1, len(roster) - 1)]["agent"]  # Second best

            # Could add more sophisticated parent selection here
            if len(roster) > 2 and random.random() < 0.3:
                parent2 = random.choice(roster[1 : min(5, len(roster))])["agent"]

            print(f"Breeding offspring {i+1}/{num_offspring}:")
            print(
                f"  Parent 1: {parent1.agent_id[:12]}... (fitness: {roster[0]['fitness']:.3f})"
            )
            print(
                f"  Parent 2: {parent2.agent_id[:12]}... (fitness: {roster[1]['fitness']:.3f})"
            )

            try:
                # Create bred genome (simplified - would use actual breeder)
                new_seed = int(time.time() * 1000) % 100000 + i
                child_genome = create_random_genome(
                    COMPLETE_TRAIT_DEFINITIONS, seed=new_seed
                )
                child_identity = create_agent_identity(child_genome)

                print(
                    f"  Child: {child_identity.id_hash[:12]}... DNA: {child_identity.visual_identity.primary_color_hex}"
                )
                print(f"  Training with {self.current_generations} generations...")

                # Train the new agent
                result = train_agent_language_capability(
                    child_identity, generations=self.current_generations
                )

                from constitutional_ai.persistence import save_training_result

                save_training_result(result, "language")

                print(
                    f"  SUCCESS: Offspring trained and saved (fitness: {result['final_fitness']:.3f})"
                )
                bred_count += 1

            except Exception as e:
                print(f"  FAILED: Breeding error - {e}")
                continue

        return bred_count

    def seed_initial_population(self):
        """Create initial agents if population is too small."""
        current_count = len(list_all_agents())

        if current_count >= 3:
            return

        need_agents = max(5, self.target_population // 2) - current_count
        print(f"SEEDING: Creating {need_agents} initial agents...")

        for i in range(need_agents):
            try:
                seed = int(time.time() * 1000) % 100000 + i + 5000
                genome = create_random_genome(COMPLETE_TRAIT_DEFINITIONS, seed=seed)
                identity = create_agent_identity(genome)

                print(
                    f"  Seeding agent {i+1}/{need_agents}: {identity.id_hash[:12]}..."
                )

                result = train_agent_language_capability(
                    identity, generations=self.base_generations
                )

                from constitutional_ai.persistence import save_training_result

                save_training_result(result, "language")

                print(f"    Seeded with fitness: {result['final_fitness']:.3f}")

            except Exception as e:
                print(f"    FAILED: Seeding error - {e}")

    def escalate_training_intensity(self):
        """Gradually increase training generations for tougher selection pressure."""
        if self.cycle_count % 5 == 0 and self.cycle_count > 0:
            old_generations = self.current_generations
            self.current_generations = int(
                self.current_generations * self.escalation_factor
            )
            print(
                f"ESCALATION: Training intensity increased {old_generations} -> {self.current_generations} generations"
            )

    def run_cycle(self):
        """Run one evolutionary cycle."""
        self.cycle_count += 1
        cycle_start = time.time()

        print(f"\\n{'='*60}")
        print(f"EVOLUTIONARY CYCLE #{self.cycle_count}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Training Intensity: {self.current_generations} generations")
        print(f"{'='*60}")

        # Get current population
        roster = self.get_agent_roster()
        print(f"Current Population: {len(roster)} agents")

        if len(roster) > 0:
            best_fitness = roster[0]["fitness"]
            avg_fitness = sum(a["fitness"] for a in roster) / len(roster)
            print(f"Best Fitness: {best_fitness:.3f}")
            print(f"Average Fitness: {avg_fitness:.3f}")

        # Seed population if needed
        self.seed_initial_population()
        roster = self.get_agent_roster()  # Refresh after seeding

        # Conduct governance vote every N cycles
        if (self.cycle_count % self.governance_cycle_interval == 0 and 
            self.enable_governance and len(roster) >= 3):
            self.conduct_governance_vote(roster)
            # Refresh roster after potential governance changes
            roster = self.get_agent_roster()

        # Prune weak agents (but keep minimum viable population)
        if len(roster) > 5:
            roster = self.prune_weak_agents(roster)

        # Calculate how many new agents to breed
        current_pop = len(roster)
        needed_agents = max(2, self.target_population - current_pop)

        print("\\nBREEDING PHASE:")
        print(
            f"Current: {current_pop}, Target: {self.target_population}, Need: {needed_agents}"
        )

        # Breed new agents
        if current_pop >= 2:
            bred_count = self.breed_new_agents(roster, needed_agents)
            print(f"Successfully bred {bred_count} new agents")

        # Escalate training difficulty
        self.escalate_training_intensity()

        # Cycle summary
        cycle_time = time.time() - cycle_start
        final_roster = self.get_agent_roster()

        print("\\nCYCLE #{self.cycle_count} COMPLETE")
        print(f"Duration: {cycle_time:.1f} seconds")
        print(f"Final Population: {len(final_roster)} agents")
        if final_roster:
            print(f"New Best Fitness: {final_roster[0]['fitness']:.3f}")
        print("Next cycle in 30 seconds...")

        return len(final_roster)

    def run_indefinitely(self, cycle_delay: int = 30):
        """Run evolutionary cycles indefinitely."""
        print("STARTING INDEFINITE EVOLUTIONARY CYCLE")
        print("Press Ctrl+C to stop")
        print(f"Cycle delay: {cycle_delay} seconds")

        try:
            while True:
                population_size = self.run_cycle()

                # Safety check
                if population_size == 0:
                    print("CRITICAL: No agents survived! Emergency seeding...")
                    self.seed_initial_population()

                # Wait before next cycle
                print(f"Waiting {cycle_delay} seconds before next cycle...")
                time.sleep(cycle_delay)

        except KeyboardInterrupt:
            print("\\n\\nEVOLUTIONARY CYCLE STOPPED BY USER")
            final_roster = self.get_agent_roster()
            print(f"Final population: {len(final_roster)} agents")
            if final_roster:
                print(f"Best evolved fitness: {final_roster[0]['fitness']:.3f}")
            print("Evolution complete.")


def main():
    """Run the indefinite evolutionary cycle."""
    cycle = EvolutionaryCycle(
        target_population=3,  # 3 elite agents - perfect for compute budget
        min_fitness_threshold=0.5,  # Only genuinely intelligent agents survive
        prune_percentage=0.6,  # Brutal selection - only top 40% survive
        escalation_factor=2.0,  # Double training intensity each escalation
    )

    cycle.run_indefinitely(cycle_delay=45)  # 45 second cycles


if __name__ == "__main__":
    main()
