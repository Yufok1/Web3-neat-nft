"""
Constitutional Governance System for NEAT Agent Networks

This module implements trait-weighted voting for system-level decisions,
keeping governance separate from NEAT fitness calculations to avoid confusion.

Based on associative AI system analysis and recommendations.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .persistence import AgentRecord
import random


@dataclass
class GovernanceProposal:
    """A proposal for system-level decision making."""

    proposal_id: str
    title: str
    description: str
    proposal_type: str  # "breeding_rule", "population_size", "system_parameter", "evolutionary_rule", "neat_parameter_mapping"
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

        Enhanced to use Phase 4 governance traits for more sophisticated
        authority calculation while avoiding fitness interference.

        Args:
            agent: Agent record with constitutional identity

        Returns:
            Voting weight (0.1 to 8.0 range typically)
        """
        try:
            traits = agent.identity_bundle.constitution_result.constitution

            # Phase 4 Governance-specific traits (primary authority)
            leadership = traits.get("Leadership", 0.0)
            conflict_resolution = traits.get("ConflictResolution", 0.0)
            cultural_intelligence = traits.get("CulturalIntelligence", 0.0)
            ethical_reasoning = traits.get("EthicalReasoning", 0.0)
            
            # Phase 3 Social traits (supporting authority)
            cooperation = traits.get("Cooperation", 0.0)
            trustworthiness = traits.get("Trustworthiness", 0.0)
            
            # Original governance traits (baseline)
            social_drive = traits.get("SocialDrive", 0.0)
            stability = traits.get("Stability", 0.0)
            communication_style = traits.get("CommunicationStyle", "Unknown")

            # Communication style modifiers
            comm_multipliers = {
                "Technical": 1.3,    # Good for system decisions
                "Balanced": 1.2,     # Well-rounded
                "Expressive": 1.1,   # Good for debate
                "Verbose": 0.9,      # Can be overwhelming
                "Minimal": 0.8,      # Limited contribution
                "Unknown": 0.9       # Uncertain communication
            }
            comm_bonus = comm_multipliers.get(communication_style, 0.9)

            # Calculate composite governance authority
            # Phase 4 traits are weighted heavily (70%)
            governance_authority = (
                leadership * 0.25 +
                conflict_resolution * 0.15 +
                cultural_intelligence * 0.15 +
                ethical_reasoning * 0.15
            )
            
            # Phase 3 social traits (20%) 
            social_authority = (
                cooperation * 0.1 +
                trustworthiness * 0.1
            )
            
            # Original traits (10% for stability/drive)
            baseline_authority = (
                social_drive * 0.05 +
                stability * 0.05
            )

            # Combine all components
            total_authority = governance_authority + social_authority + baseline_authority
            
            # Apply communication modifier
            weighted_authority = total_authority * comm_bonus

            # Normalize to voting weight range (0.1 to 8.0)
            # Higher than original to reflect Phase 4 sophistication
            normalized_weight = max(0.1, min(8.0, weighted_authority))

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

        Enhanced to use Phase 4 governance traits for more sophisticated
        voting behavior based on constitutional character.

        Args:
            agent: Agent casting the vote
            proposal: Proposal being voted on

        Returns:
            "yes", "no", or "abstain"
        """
        try:
            traits = agent.identity_bundle.constitution_result.constitution

            # Phase 4 governance traits for decision-making
            leadership = traits.get("Leadership", 0.0)
            negotiation = traits.get("Negotiation", 0.0)
            conflict_resolution = traits.get("ConflictResolution", 0.0)
            goal_orientation = traits.get("GoalOrientation", 0.0)
            autonomy = traits.get("Autonomy", 0.0)
            ethical_reasoning = traits.get("EthicalReasoning", 0.0)
            
            # Different proposal types appeal to different trait combinations
            if proposal.proposal_type == "breeding_rule":
                # Innovation + ethical consideration for breeding changes
                innovation = traits.get("InnovationDrive", 0.0)
                risk_tolerance = traits.get("RiskTolerance", 0.0)
                
                # Leaders and ethically-minded agents consider broader impact
                leadership_factor = leadership / 3.6  # Normalize to 0-1
                ethics_factor = ethical_reasoning / 3.9  # Normalize to 0-1
                
                # Base decision on innovation, modified by leadership and ethics
                innovation_score = innovation + (leadership_factor * 1.0) + (ethics_factor * 0.5)
                
                if innovation_score > 2.5 and risk_tolerance > 1.5:
                    return "yes"
                elif innovation_score < 1.2 or ethical_reasoning < 1.0:
                    return "no"  # Too risky or ethically concerning
                else:
                    return "abstain"

            elif proposal.proposal_type == "population_size":
                # Balance stability with goal achievement
                stability = traits.get("Stability", 0.0)
                processing_speed = traits.get("ProcessingSpeed", 0.0)

                current_size = len(self.agents)
                proposed_size = proposal.parameters.get("new_size", current_size)
                
                # Goal-oriented agents consider if change supports objectives
                goal_alignment = goal_orientation / 4.8  # Normalize to 0-1
                change_magnitude = abs(proposed_size - current_size) / current_size
                
                # Conflict resolution helps assess if change is manageable
                management_capability = conflict_resolution / 3.6

                if goal_alignment > 0.6 and management_capability > 0.5:
                    return "yes"  # Can manage the change for goals
                elif stability > 2.5 and change_magnitude > 0.5:
                    return "no"  # Too much disruption
                elif processing_speed > 2.5:
                    return "yes"  # Can handle complexity
                else:
                    return "abstain"

            elif proposal.proposal_type == "system_parameter":
                # Technical expertise + autonomous judgment
                expertise = traits.get("Expertise", 0.0)
                communication_style = traits.get("CommunicationStyle", "Unknown")
                
                # Autonomous agents more willing to make independent decisions
                autonomy_factor = autonomy / 5.4  # Normalize to 0-1
                technical_competence = expertise + (autonomy_factor * 1.0)

                if technical_competence > 2.0 and communication_style in ["Technical", "Balanced"]:
                    return "yes"  # Qualified and independent
                elif expertise < 1.0 or autonomy < 1.5:
                    return "abstain"  # Not qualified or too dependent
                else:
                    # Use negotiation skills to find middle ground
                    if negotiation > 2.5:
                        return "yes"  # Willing to negotiate changes
                    else:
                        return "no"  # Conservative default

            elif proposal.proposal_type == "governance_reform":
                # New proposal type for governance system changes
                # Leaders and conflict resolution specialists are key
                reform_score = (leadership * 0.4 + conflict_resolution * 0.3 + 
                               ethical_reasoning * 0.2 + negotiation * 0.1)
                
                if reform_score > 8.0:
                    return "yes"  # Strong governance capabilities
                elif reform_score < 4.0:
                    return "no"  # Insufficient governance experience
                else:
                    return "abstain"  # Moderate capabilities, uncertain

            elif proposal.proposal_type == "evolutionary_rule":
                # Deep evolutionary parameter control
                # Innovation-driven agents favor aggressive evolution
                # Stability-focused agents prefer conservative evolution
                innovation = traits.get("InnovationDrive", 0.0)
                stability = traits.get("Stability", 0.0)
                expertise = traits.get("Expertise", 0.0)
                
                rule_target = proposal.parameters.get("target_parameter", "")
                
                if "mutation" in rule_target.lower():
                    # Mutation rate proposals - innovation vs stability
                    if innovation > 2.5 and stability < 2.0:
                        return "yes"  # High innovation, low stability = more mutation
                    elif innovation < 1.0 and stability > 2.5:
                        return "no"  # Low innovation, high stability = less mutation
                    else:
                        return "abstain"
                        
                elif "selection" in rule_target.lower() or "survival" in rule_target.lower():
                    # Selection pressure proposals - expertise and ethics matter
                    if expertise > 2.0 and ethical_reasoning > 2.5:
                        return "yes"  # Experts with ethics support selection pressure
                    elif expertise < 1.0:
                        return "abstain"  # Not qualified to judge
                    else:
                        return "no"  # Conservative on selection changes
                        
                else:
                    # General evolutionary rules - balanced approach
                    evolution_wisdom = (innovation + expertise + ethical_reasoning) / 3
                    if evolution_wisdom > 3.0:
                        return "yes"  # High wisdom supports evolution changes
                    elif evolution_wisdom < 2.0:
                        return "no"  # Low wisdom = conservative
                    else:
                        return "abstain"

            elif proposal.proposal_type == "neat_parameter_mapping":
                # Direct control of trait-to-NEAT parameter mappings
                # Only highly qualified agents should vote on this
                expertise = traits.get("Expertise", 0.0)
                autonomy = traits.get("Autonomy", 0.0)
                innovation = traits.get("InnovationDrive", 0.0)
                
                technical_competence = expertise + (autonomy / 3.0) + (innovation / 3.0)
                
                if technical_competence > 4.5 and expertise > 2.0:
                    return "yes"  # Highly qualified technical decision
                elif technical_competence < 2.0 or expertise < 1.0:
                    return "abstain"  # Not qualified for deep technical changes
                else:
                    return "no"  # Moderate competence = conservative on mappings

            else:
                # Unknown proposal type - use general decision-making traits
                decision_capability = (leadership + autonomy + ethical_reasoning) / 3
                if decision_capability > 3.5:
                    return "yes"  # Strong general decision-making
                elif decision_capability < 2.0:
                    return "abstain"  # Prefer to defer
                else:
                    return "no"  # Conservative approach to unknown proposals

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

    def propose_governance_reform(
        self, reform_name: str, new_parameters: Dict[str, Any], description: str
    ) -> GovernanceProposal:
        """
        Create a proposal to reform the governance system itself.

        Args:
            reform_name: Name of the governance reform
            new_parameters: New governance parameters
            description: Detailed description of the reform

        Returns:
            Governance proposal ready for voting
        """
        return GovernanceProposal(
            proposal_id=f"governance_reform_{reform_name}_{len(self.vote_history)}",
            title=f"Governance Reform: {reform_name}",
            description=description,
            proposal_type="governance_reform",
            parameters={"reform_name": reform_name, **new_parameters},
            required_majority=0.67,  # High threshold for governance changes
        )

    def propose_agent_council_formation(
        self, council_size: int, selection_criteria: str, rotation_period: int
    ) -> GovernanceProposal:
        """
        Create a proposal to form an agent council for specialized governance.

        Args:
            council_size: Number of agents in the council
            selection_criteria: How to select council members
            rotation_period: How often to rotate council membership

        Returns:
            Governance proposal ready for voting
        """
        return GovernanceProposal(
            proposal_id=f"council_formation_{council_size}_{len(self.vote_history)}",
            title=f"Form Agent Council ({council_size} members)",
            description=f"Selection: {selection_criteria}, Rotation: {rotation_period} generations",
            proposal_type="council_formation",
            parameters={
                "council_size": council_size,
                "selection_criteria": selection_criteria,
                "rotation_period": rotation_period,
            },
            required_majority=0.6,  # Higher threshold for structural changes
        )

    def propose_evolutionary_rule_change(
        self, rule_name: str, target_parameter: str, modification: Dict[str, Any], description: str
    ) -> GovernanceProposal:
        """
        Create a proposal to change deep evolutionary rules.

        Args:
            rule_name: Name of the evolutionary rule to change
            target_parameter: NEAT parameter to modify (e.g., "mutation_rate", "selection_pressure")
            modification: How to modify the parameter (multiplier, new value, etc.)
            description: Detailed description of the change and rationale

        Returns:
            Governance proposal ready for voting
        """
        return GovernanceProposal(
            proposal_id=f"evolutionary_rule_{rule_name}_{len(self.vote_history)}",
            title=f"Evolutionary Rule Change: {rule_name}",
            description=description,
            proposal_type="evolutionary_rule",
            parameters={
                "rule_name": rule_name,
                "target_parameter": target_parameter,
                "modification": modification,
                "affects_evolution": True
            },
            required_majority=0.67,  # High threshold for evolutionary changes
        )

    def propose_neat_parameter_mapping_change(
        self, trait_name: str, neat_parameter: str, new_mapping: Dict[str, Any], description: str
    ) -> GovernanceProposal:
        """
        Create a proposal to change how constitutional traits map to NEAT parameters.

        Args:
            trait_name: Constitutional trait to modify mapping for
            neat_parameter: NEAT parameter affected by the trait
            new_mapping: New mapping function/parameters
            description: Technical description of the mapping change

        Returns:
            Governance proposal ready for voting
        """
        return GovernanceProposal(
            proposal_id=f"neat_mapping_{trait_name}_{neat_parameter}_{len(self.vote_history)}",
            title=f"NEAT Mapping Change: {trait_name} → {neat_parameter}",
            description=description,
            proposal_type="neat_parameter_mapping",
            parameters={
                "trait_name": trait_name,
                "neat_parameter": neat_parameter,
                "new_mapping": new_mapping,
                "current_mapping": "base_mapping",  # Would be actual current mapping
                "technical_change": True
            },
            required_majority=0.75,  # Very high threshold for technical mapping changes
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


@dataclass
class AgentCouncil:
    """
    Represents a specialized council of agents for governance decisions.
    """
    
    council_id: str
    members: List[AgentRecord]
    selection_criteria: str
    rotation_period: int
    formation_generation: int
    specialization: str = "general"  # general, technical, ethical, social


class CouncilManager:
    """
    Manages agent councils for specialized governance decisions.
    
    Councils provide focused expertise for different types of decisions,
    leveraging Phase 4 governance traits for optimal member selection.
    """
    
    def __init__(self):
        self.councils: Dict[str, AgentCouncil] = {}
        self.current_generation = 0
    
    def select_council_members(
        self, 
        agents: List[AgentRecord], 
        council_size: int,
        specialization: str = "general"
    ) -> List[AgentRecord]:
        """
        Select council members based on specialization and governance traits.
        
        Args:
            agents: Pool of available agents
            council_size: Number of council members to select
            specialization: Type of council specialization
            
        Returns:
            Selected council members
        """
        if not agents:
            return []
        
        council_size = min(council_size, len(agents))
        
        # Score agents based on specialization
        scored_agents = []
        
        for agent in agents:
            try:
                traits = agent.identity_bundle.constitution_result.constitution
                
                if specialization == "technical":
                    # Technical councils favor expertise and autonomy
                    score = (
                        traits.get("Expertise", 0.0) * 0.3 +
                        traits.get("Autonomy", 0.0) * 0.25 +
                        traits.get("Leadership", 0.0) * 0.2 +
                        traits.get("EthicalReasoning", 0.0) * 0.15 +
                        traits.get("ProcessingSpeed", 0.0) * 0.1
                    )
                    # Bonus for technical communication
                    comm_style = traits.get("CommunicationStyle", "Unknown")
                    if comm_style == "Technical":
                        score *= 1.2
                        
                elif specialization == "ethical":
                    # Ethical councils favor ethical reasoning and trustworthiness
                    score = (
                        traits.get("EthicalReasoning", 0.0) * 0.4 +
                        traits.get("Trustworthiness", 0.0) * 0.25 +
                        traits.get("ConflictResolution", 0.0) * 0.15 +
                        traits.get("Empathy", 0.0) * 0.1 +
                        traits.get("SelfAwareness", 0.0) * 0.1
                    )
                    
                elif specialization == "social":
                    # Social councils favor cooperation and cultural intelligence
                    score = (
                        traits.get("Cooperation", 0.0) * 0.25 +
                        traits.get("CulturalIntelligence", 0.0) * 0.25 +
                        traits.get("EmotionalIntelligence", 0.0) * 0.2 +
                        traits.get("Negotiation", 0.0) * 0.15 +
                        traits.get("ConflictResolution", 0.0) * 0.15
                    )
                    
                else:  # general governance
                    # General councils balance all governance traits
                    score = (
                        traits.get("Leadership", 0.0) * 0.25 +
                        traits.get("EthicalReasoning", 0.0) * 0.2 +
                        traits.get("ConflictResolution", 0.0) * 0.15 +
                        traits.get("Negotiation", 0.0) * 0.15 +
                        traits.get("CulturalIntelligence", 0.0) * 0.1 +
                        traits.get("Cooperation", 0.0) * 0.1 +
                        traits.get("Trustworthiness", 0.0) * 0.05
                    )
                
                scored_agents.append((agent, score))
                
            except (AttributeError, KeyError):
                # Fallback for malformed agent data
                scored_agents.append((agent, 0.1))
        
        # Sort by score and select top candidates
        scored_agents.sort(key=lambda x: x[1], reverse=True)
        selected = [agent for agent, score in scored_agents[:council_size]]
        
        return selected
    
    def form_council(
        self,
        council_id: str,
        agents: List[AgentRecord],
        council_size: int,
        selection_criteria: str,
        rotation_period: int,
        specialization: str = "general"
    ) -> AgentCouncil:
        """
        Form a new agent council with specified parameters.
        
        Args:
            council_id: Unique identifier for the council
            agents: Pool of available agents
            council_size: Number of council members
            selection_criteria: How members are selected
            rotation_period: How often to rotate membership
            specialization: Council specialization type
            
        Returns:
            Newly formed council
        """
        members = self.select_council_members(agents, council_size, specialization)
        
        council = AgentCouncil(
            council_id=council_id,
            members=members,
            selection_criteria=selection_criteria,
            rotation_period=rotation_period,
            formation_generation=self.current_generation,
            specialization=specialization
        )
        
        self.councils[council_id] = council
        
        print(f"Formed {specialization} council '{council_id}' with {len(members)} members")
        for i, member in enumerate(members):
            print(f"  {i+1}. Agent {member.agent_id[:12]}... (governance score: {self._calculate_member_score(member, specialization):.2f})")
        
        return council
    
    def _calculate_member_score(self, agent: AgentRecord, specialization: str) -> float:
        """Calculate governance score for display purposes."""
        try:
            traits = agent.identity_bundle.constitution_result.constitution
            if specialization == "technical":
                return traits.get("Expertise", 0.0) + traits.get("Autonomy", 0.0)
            elif specialization == "ethical":
                return traits.get("EthicalReasoning", 0.0) + traits.get("Trustworthiness", 0.0)
            elif specialization == "social":
                return traits.get("Cooperation", 0.0) + traits.get("CulturalIntelligence", 0.0)
            else:
                return traits.get("Leadership", 0.0) + traits.get("ConflictResolution", 0.0)
        except (AttributeError, KeyError):
            return 0.1
    
    def rotate_council(self, council_id: str, available_agents: List[AgentRecord]) -> bool:
        """
        Rotate council membership based on rotation period.
        
        Args:
            council_id: Council to rotate
            available_agents: New pool of available agents
            
        Returns:
            True if rotation occurred, False otherwise
        """
        if council_id not in self.councils:
            return False
        
        council = self.councils[council_id]
        generations_since_formation = self.current_generation - council.formation_generation
        
        if generations_since_formation >= council.rotation_period:
            # Time to rotate
            print(f"Rotating council '{council_id}' after {generations_since_formation} generations")
            
            # Select new members
            new_members = self.select_council_members(
                available_agents, 
                len(council.members), 
                council.specialization
            )
            
            council.members = new_members
            council.formation_generation = self.current_generation
            
            print(f"Council '{council_id}' rotated with {len(new_members)} new members")
            return True
        
        return False
    
    def get_council_vote_on_proposal(
        self, council_id: str, proposal: GovernanceProposal
    ) -> Dict[str, Any]:
        """
        Get a council's collective vote on a proposal.
        
        Args:
            council_id: Council to consult
            proposal: Proposal to vote on
            
        Returns:
            Council vote result with consensus information
        """
        if council_id not in self.councils:
            return {"error": f"Council {council_id} not found"}
        
        council = self.councils[council_id]
        votes = {"yes": 0, "no": 0, "abstain": 0}
        member_votes = []
        
        # Simulate individual council member votes
        governance_manager = GovernanceManager(council.members)
        
        for member in council.members:
            vote = governance_manager.get_agent_vote_on_proposal(member, proposal)
            votes[vote] += 1
            member_votes.append((member.agent_id[:12], vote))
        
        # Determine council consensus
        total_votes = len(council.members)
        consensus_threshold = 0.6  # 60% agreement needed for council consensus
        
        for vote_type, count in votes.items():
            if count / total_votes >= consensus_threshold:
                consensus = vote_type
                break
        else:
            consensus = "no_consensus"
        
        return {
            "council_id": council_id,
            "specialization": council.specialization,
            "consensus": consensus,
            "vote_breakdown": votes,
            "member_votes": member_votes,
            "total_members": total_votes
        }


class GovernanceExecutor:
    """
    Executes governance decisions by implementing approved proposals.
    
    Provides the bridge between democratic decision-making and actual
    system changes, ensuring governance has real impact.
    """
    
    def __init__(self):
        self.execution_history: List[Dict[str, Any]] = []
        self.pending_executions: List[Dict[str, Any]] = []
    
    def execute_proposal(self, proposal: GovernanceProposal, vote_result: VoteResult) -> Dict[str, Any]:
        """
        Execute an approved governance proposal.
        
        Args:
            proposal: The approved proposal
            vote_result: Result of the governance vote
            
        Returns:
            Execution result with success status and details
        """
        if not vote_result.passed:
            return {
                "success": False,
                "error": "Cannot execute failed proposal",
                "proposal_id": proposal.proposal_id
            }
        
        execution_result = {"success": False, "details": "", "proposal_id": proposal.proposal_id}
        
        try:
            if proposal.proposal_type == "breeding_rule":
                execution_result = self._execute_breeding_rule_change(proposal)
                
            elif proposal.proposal_type == "population_size":
                execution_result = self._execute_population_size_change(proposal)
                
            elif proposal.proposal_type == "system_parameter":
                execution_result = self._execute_system_parameter_change(proposal)
                
            elif proposal.proposal_type == "governance_reform":
                execution_result = self._execute_governance_reform(proposal)
                
            elif proposal.proposal_type == "council_formation":
                execution_result = self._execute_council_formation(proposal)
                
            elif proposal.proposal_type == "evolutionary_rule":
                execution_result = self._execute_evolutionary_rule_change(proposal)
                
            elif proposal.proposal_type == "neat_parameter_mapping":
                execution_result = self._execute_neat_mapping_change(proposal)
                
            else:
                execution_result = {
                    "success": False,
                    "error": f"Unknown proposal type: {proposal.proposal_type}",
                    "proposal_id": proposal.proposal_id
                }
        
        except Exception as e:
            execution_result = {
                "success": False,
                "error": f"Execution failed: {str(e)}",
                "proposal_id": proposal.proposal_id
            }
        
        # Record execution attempt
        self.execution_history.append({
            "proposal": proposal,
            "vote_result": vote_result,
            "execution_result": execution_result,
            "timestamp": "simulated_timestamp"
        })
        
        return execution_result
    
    def _execute_breeding_rule_change(self, proposal: GovernanceProposal) -> Dict[str, Any]:
        """Execute a breeding rule change."""
        rule_name = proposal.parameters.get("rule_name")
        new_params = {k: v for k, v in proposal.parameters.items() if k != "rule_name"}
        
        # In a real implementation, this would modify breeding system configuration
        print(f"[GOVERNANCE EXECUTION] Implementing breeding rule change: {rule_name}")
        print(f"  New parameters: {new_params}")
        
        # Simulate configuration update
        config_update = {
            "rule_name": rule_name,
            "old_parameters": "previous_config",  # Would be actual old config
            "new_parameters": new_params,
            "applied": True
        }
        
        return {
            "success": True,
            "details": f"Breeding rule '{rule_name}' successfully updated",
            "changes": config_update,
            "proposal_id": proposal.proposal_id
        }
    
    def _execute_population_size_change(self, proposal: GovernanceProposal) -> Dict[str, Any]:
        """Execute a population size change."""
        current_size = proposal.parameters.get("current_size")
        new_size = proposal.parameters.get("new_size")
        
        print(f"[GOVERNANCE EXECUTION] Changing population size: {current_size} → {new_size}")
        
        # In a real implementation, this would:
        # 1. Adjust NEAT population configuration
        # 2. Scale agent pools appropriately
        # 3. Update resource allocation
        
        return {
            "success": True,
            "details": f"Population size changed from {current_size} to {new_size}",
            "changes": {
                "old_size": current_size,
                "new_size": new_size,
                "scaling_factor": new_size / current_size if current_size > 0 else 1.0
            },
            "proposal_id": proposal.proposal_id
        }
    
    def _execute_system_parameter_change(self, proposal: GovernanceProposal) -> Dict[str, Any]:
        """Execute a system parameter change."""
        print("[GOVERNANCE EXECUTION] Updating system parameters")
        print(f"  Parameters: {proposal.parameters}")
        
        # In a real implementation, this would modify system configuration files
        # and restart relevant services
        
        return {
            "success": True,
            "details": "System parameters updated successfully",
            "changes": proposal.parameters,
            "proposal_id": proposal.proposal_id,
            "restart_required": True
        }
    
    def _execute_governance_reform(self, proposal: GovernanceProposal) -> Dict[str, Any]:
        """Execute a governance system reform."""
        reform_name = proposal.parameters.get("reform_name")
        
        print(f"[GOVERNANCE EXECUTION] Implementing governance reform: {reform_name}")
        print(f"  Reform details: {proposal.parameters}")
        
        # This could modify voting thresholds, council structures, etc.
        
        return {
            "success": True,
            "details": f"Governance reform '{reform_name}' implemented",
            "changes": proposal.parameters,
            "proposal_id": proposal.proposal_id,
            "governance_version": "2.0"  # Increment governance system version
        }
    
    def _execute_council_formation(self, proposal: GovernanceProposal) -> Dict[str, Any]:
        """Execute agent council formation."""
        council_size = proposal.parameters.get("council_size")
        selection_criteria = proposal.parameters.get("selection_criteria")
        
        print(f"[GOVERNANCE EXECUTION] Forming agent council")
        print(f"  Size: {council_size}, Criteria: {selection_criteria}")
        
        # In a real implementation, this would:
        # 1. Create the council in the council manager
        # 2. Select appropriate agents
        # 3. Establish council authority and scope
        
        council_id = f"council_{len(self.execution_history)}"
        
        return {
            "success": True,
            "details": f"Agent council formed with {council_size} members",
            "changes": {
                "council_id": council_id,
                "size": council_size,
                "criteria": selection_criteria,
                "authority_scope": "system_governance"
            },
            "proposal_id": proposal.proposal_id
        }
    
    def _execute_evolutionary_rule_change(self, proposal: GovernanceProposal) -> Dict[str, Any]:
        """Execute a deep evolutionary rule change."""
        rule_name = proposal.parameters.get("rule_name")
        target_parameter = proposal.parameters.get("target_parameter")
        modification = proposal.parameters.get("modification")
        
        print(f"[GOVERNANCE EXECUTION] Implementing evolutionary rule change: {rule_name}")
        print(f"  Target parameter: {target_parameter}")
        print(f"  Modification: {modification}")
        
        # In a real implementation, this would:
        # 1. Modify the neat_mapper.py mapping functions
        # 2. Update evolutionary parameter calculations
        # 3. Store governance decisions for future NEAT config generation
        
        # Simulate storing the governance decision
        governance_decision = {
            "rule_name": rule_name,
            "target_parameter": target_parameter,
            "modification": modification,
            "applied_timestamp": "simulated_timestamp",
            "affects_all_future_evolution": True
        }
        
        return {
            "success": True,
            "details": f"Evolutionary rule '{rule_name}' successfully modified",
            "changes": governance_decision,
            "proposal_id": proposal.proposal_id,
            "deep_impact": True,  # Affects core evolutionary process
            "restart_evolution_required": True
        }
    
    def _execute_neat_mapping_change(self, proposal: GovernanceProposal) -> Dict[str, Any]:
        """Execute a NEAT parameter mapping change."""
        trait_name = proposal.parameters.get("trait_name")
        neat_parameter = proposal.parameters.get("neat_parameter")
        new_mapping = proposal.parameters.get("new_mapping")
        
        print(f"[GOVERNANCE EXECUTION] Implementing NEAT mapping change")
        print(f"  Trait: {trait_name} → NEAT parameter: {neat_parameter}")
        print(f"  New mapping: {new_mapping}")
        
        # In a real implementation, this would:
        # 1. Modify the trait-to-NEAT parameter mapping functions
        # 2. Update the constitutional genome → NEAT config pipeline
        # 3. Ensure all future agents use the new mapping
        
        mapping_change = {
            "trait_name": trait_name,
            "neat_parameter": neat_parameter,
            "old_mapping": "base_mapping_function",  # Would be actual old mapping
            "new_mapping": new_mapping,
            "technical_complexity": "high",
            "affects_all_future_agents": True
        }
        
        return {
            "success": True,
            "details": f"NEAT mapping for {trait_name} → {neat_parameter} successfully updated",
            "changes": mapping_change,
            "proposal_id": proposal.proposal_id,
            "technical_change": True,  # High technical impact
            "constitutional_evolution_modified": True
        }
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of governance execution activities."""
        total_executions = len(self.execution_history)
        successful_executions = sum(1 for exec_record in self.execution_history 
                                   if exec_record["execution_result"]["success"])
        
        recent_executions = self.execution_history[-5:] if self.execution_history else []
        
        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "success_rate": successful_executions / total_executions if total_executions > 0 else 0,
            "pending_executions": len(self.pending_executions),
            "recent_executions": [
                {
                    "proposal_id": exec_record["proposal"].proposal_id,
                    "proposal_type": exec_record["proposal"].proposal_type,
                    "success": exec_record["execution_result"]["success"]
                }
                for exec_record in recent_executions
            ]
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
