"""
Constitutional Governance System for NEAT Agent Networks

This module implements trait-weighted voting for system-level decisions,
keeping governance separate from NEAT fitness calculations to avoid confusion.

Based on associative AI system analysis and recommendations.
"""

from typing import List, Dict, Any
from dataclasses import dataclass
from .persistence import AgentRecord


@dataclass
class GovernanceProposal:
    """A proposal for system-level decision making."""

    proposal_id: str
    title: str
    description: str
    proposal_type: str  # "breeding_rule", "population_size", "system_parameter"
    parameters: Dict[str, Any]
    required_majority: float = 0.5  # Simple majority by default


@dataclass
class VoteResult:
    """Result of a governance vote."""

    proposal_id: str
    passed: bool
    total_weight: float
    yes_weight: float
    no_weight: float
    abstain_weight: float
    voter_count: int


class GovernanceManager:
    """
    Manages constitutional governance for agent networks.

    Implements trait-weighted voting while keeping governance separate
    from NEAT evolution to prevent fitness calculation interference.
    """

    def __init__(self, agents: List[AgentRecord]):
        """
        Initialize governance manager.

        Args:
            agents: List of agent records to govern
        """
        self.agents = agents
        self.vote_history: List[VoteResult] = []

    def calculate_vote_weight(self, agent: AgentRecord) -> float:
        """
        Calculate voting weight based on constitutional traits.

        Uses social traits + stability for governance authority,
        avoiding interference with capability-specific traits.

        Args:
            agent: Agent record with constitutional identity

        Returns:
            Voting weight (0.0 to 10.0 range typically)
        """
        try:
            traits = agent.identity_bundle.constitution_result.constitution

            # Base voting weight from social governance traits
            social_drive = traits.get("SocialDrive", 0.0)
            stability = traits.get("Stability", 0.0)
            communication_style = traits.get("CommunicationStyle", "Unknown")

            # Technical communication style gets slight boost for system decisions
            comm_bonus = 1.2 if communication_style == "Technical" else 1.0

            # Weight = social responsibility * stability * communication effectiveness
            base_weight = social_drive * stability * comm_bonus

            # Normalize to reasonable range (0.1 to 5.0)
            normalized_weight = max(0.1, min(5.0, base_weight))

            return normalized_weight

        except (AttributeError, KeyError) as e:
            # Fallback for malformed agent data
            print(
                f"Warning: Could not calculate vote weight for agent {agent.agent_id[:12]}: {e}"
            )
            return 0.1  # Minimal voting weight

    def get_agent_vote_on_proposal(
        self, agent: AgentRecord, proposal: GovernanceProposal
    ) -> str:
        """
        Determine how an agent would vote on a proposal.

        Uses constitutional traits to predict voting behavior.
        This is a simplified model - could be extended with learning.

        Args:
            agent: Agent casting the vote
            proposal: Proposal being voted on

        Returns:
            "yes", "no", or "abstain"
        """
        try:
            traits = agent.identity_bundle.constitution_result.constitution

            # Different proposal types appeal to different traits
            if proposal.proposal_type == "breeding_rule":
                # Innovation-driven agents favor breeding changes
                innovation = traits.get("InnovationDrive", 0.0)
                risk_tolerance = traits.get("RiskTolerance", 0.0)

                if innovation > 2.5 and risk_tolerance > 2.0:
                    return "yes"
                elif innovation < 1.5 or risk_tolerance < 1.0:
                    return "no"
                else:
                    return "abstain"

            elif proposal.proposal_type == "population_size":
                # Stable agents prefer smaller, manageable populations
                stability = traits.get("Stability", 0.0)
                processing_speed = traits.get("ProcessingSpeed", 0.0)

                current_size = len(self.agents)
                proposed_size = proposal.parameters.get("new_size", current_size)

                if stability > 3.0 and proposed_size > current_size * 1.5:
                    return "no"  # Too much change
                elif processing_speed > 3.0:
                    return "yes"  # Can handle larger populations
                else:
                    return "abstain"

            elif proposal.proposal_type == "system_parameter":
                # Technical agents more likely to vote on system changes
                expertise = traits.get("Expertise", 0.0)
                communication_style = traits.get("CommunicationStyle", "Unknown")

                if expertise > 2.5 and communication_style == "Technical":
                    return "yes"
                elif expertise < 1.5:
                    return "abstain"  # Not qualified to judge
                else:
                    return "no"  # Conservative default

            else:
                # Unknown proposal type - abstain
                return "abstain"

        except (AttributeError, KeyError):
            # Fallback for malformed data
            return "abstain"

    def conduct_vote(self, proposal: GovernanceProposal) -> VoteResult:
        """
        Conduct a trait-weighted vote on a proposal.

        Args:
            proposal: Proposal to vote on

        Returns:
            Vote result with weighted tallies
        """
        total_weight = 0.0
        yes_weight = 0.0
        no_weight = 0.0
        abstain_weight = 0.0

        print(f"🗳️  Conducting governance vote: {proposal.title}")
        print(f"   Proposal: {proposal.description}")
        print(f"   Required majority: {proposal.required_majority:.1%}")
        print()

        for agent in self.agents:
            weight = self.calculate_vote_weight(agent)
            vote = self.get_agent_vote_on_proposal(agent, proposal)

            total_weight += weight

            if vote == "yes":
                yes_weight += weight
            elif vote == "no":
                no_weight += weight
            else:  # abstain
                abstain_weight += weight

            print(f"   Agent {agent.agent_id[:12]}: {vote:>7} (weight: {weight:.2f})")

        # Determine if proposal passes
        # Only count yes/no votes for majority calculation
        voting_weight = yes_weight + no_weight
        if voting_weight > 0:
            yes_percentage = yes_weight / voting_weight
            passed = yes_percentage >= proposal.required_majority
        else:
            # All abstained - proposal fails
            passed = False

        result = VoteResult(
            proposal_id=proposal.proposal_id,
            passed=passed,
            total_weight=total_weight,
            yes_weight=yes_weight,
            no_weight=no_weight,
            abstain_weight=abstain_weight,
            voter_count=len(self.agents),
        )

        self.vote_history.append(result)

        print()
        print(f"📊 Vote Result: {'✅ PASSED' if passed else '❌ FAILED'}")
        print(f"   Yes: {yes_weight:.2f} ({yes_weight/total_weight:.1%})")
        print(f"   No:  {no_weight:.2f} ({no_weight/total_weight:.1%})")
        print(f"   Abstain: {abstain_weight:.2f} ({abstain_weight/total_weight:.1%})")

        return result

    def propose_breeding_rule_change(
        self, rule_name: str, new_parameters: Dict[str, Any], description: str
    ) -> GovernanceProposal:
        """
        Create a proposal to change breeding rules.

        Args:
            rule_name: Name of breeding rule to change
            new_parameters: New parameter values
            description: Human-readable description

        Returns:
            Governance proposal ready for voting
        """
        return GovernanceProposal(
            proposal_id=f"breeding_rule_{rule_name}_{len(self.vote_history)}",
            title=f"Change Breeding Rule: {rule_name}",
            description=description,
            proposal_type="breeding_rule",
            parameters={"rule_name": rule_name, **new_parameters},
            required_majority=0.6,  # Slightly higher threshold for breeding changes
        )

    def propose_population_size_change(
        self, new_size: int, reason: str
    ) -> GovernanceProposal:
        """
        Create a proposal to change population size.

        Args:
            new_size: Proposed new population size
            reason: Justification for the change

        Returns:
            Governance proposal ready for voting
        """
        current_size = len(self.agents)

        return GovernanceProposal(
            proposal_id=f"population_size_{new_size}_{len(self.vote_history)}",
            title=f"Change Population Size: {current_size} → {new_size}",
            description=f"Reason: {reason}",
            proposal_type="population_size",
            parameters={
                "current_size": current_size,
                "new_size": new_size,
                "reason": reason,
            },
            required_majority=0.55,  # Moderate threshold for population changes
        )

    def get_governance_summary(self) -> Dict[str, Any]:
        """
        Get summary of governance system state.

        Returns:
            Summary statistics and recent activity
        """
        if not self.vote_history:
            return {
                "total_agents": len(self.agents),
                "total_votes": 0,
                "recent_activity": "No votes conducted yet",
            }

        recent_votes = self.vote_history[-5:]  # Last 5 votes
        passed_votes = sum(1 for v in recent_votes if v.passed)

        return {
            "total_agents": len(self.agents),
            "total_votes": len(self.vote_history),
            "recent_votes": len(recent_votes),
            "recent_pass_rate": passed_votes / len(recent_votes) if recent_votes else 0,
            "average_participation": (
                sum(v.voter_count for v in recent_votes) / len(recent_votes)
                if recent_votes
                else 0
            ),
            "last_vote": recent_votes[-1].proposal_id if recent_votes else None,
        }


def create_governance_manager_from_agent_list(
    agent_ids: List[str],
) -> GovernanceManager:
    """
    Create a governance manager from a list of agent IDs.

    Args:
        agent_ids: List of agent ID strings

    Returns:
        Initialized governance manager
    """
    from .persistence import load_agent

    agents = []
    for agent_id in agent_ids:
        try:
            agent = load_agent(agent_id)
            if agent:
                agents.append(agent)
        except Exception as e:
            print(f"Warning: Could not load agent {agent_id[:12]}: {e}")

    return GovernanceManager(agents)


# Example usage functions for testing
def example_breeding_rule_vote():
    """Example of a breeding rule change vote."""
    from .persistence import list_all_agents

    agent_ids = list_all_agents()
    if len(agent_ids) < 2:
        print("Need at least 2 agents for governance voting")
        return

    governance = create_governance_manager_from_agent_list(
        agent_ids[:5]
    )  # Use first 5 agents

    proposal = governance.propose_breeding_rule_change(
        rule_name="min_fitness_threshold",
        new_parameters={"min_fitness": 0.8},
        description="Raise minimum fitness threshold to improve breeding quality",
    )

    result = governance.conduct_vote(proposal)
    print(
        f"\nGovernance decision: {'Implement change' if result.passed else 'Keep current rules'}"
    )


if __name__ == "__main__":
    # Test the governance system
    example_breeding_rule_vote()
