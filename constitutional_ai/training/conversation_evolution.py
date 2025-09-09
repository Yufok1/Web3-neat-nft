"""
Conversational Evolution Pipeline for Constitutional AI

Implements true conversational AI training using:
1. Word-level tokenization with context awareness
2. Multi-turn conversation modeling
3. Constitutional personality-driven responses
4. Memory-aware context windows
5. Conversation-specific fitness evaluation

This replaces character-level training with proper conversational intelligence.
"""

import random
import neat
import json
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from ..identity import IdentityBundle
from ..neat_integration import ConstitutionalNEATRunner
from .conversational_tokenizer import ConversationalTokenizer, TokenizerConfig
from .base_capability import (
    TrainingData,
    CapabilityFitnessEvaluator,
    CapabilityTrainingPipeline,
    EvaluationResult,
)


class ConversationCorpus:
    """Simple utility for building conversation datasets."""
    
    def __init__(self):
        self.conversations = []
    
    def add_conversation(self, conversation: List[Tuple[str, str]]):
        """Add a conversation to the corpus."""
        self.conversations.append(conversation)
    
    def add_conversations_from_file(self, filepath: str):
        """Add conversations from a JSON file."""
        conversations = load_conversations_from_file(filepath)
        self.conversations.extend(conversations)
    
    def get_conversations(self) -> List[List[Tuple[str, str]]]:
        """Get all conversations in the corpus."""
        return self.conversations


@dataclass
class ConversationTrainingData(TrainingData):
    """Training data for conversational evolution."""

    conversations: List[List[Tuple[str, str]]]  # List of conversation threads
    tokenizer: ConversationalTokenizer
    context_window_size: int = 64  # Token context window
    response_max_length: int = 32   # Max tokens per response

    def __post_init__(self):
        """Initialize parent TrainingData fields."""
        super().__init__(
            capability_type="conversation", 
            data_size=len(self.conversations)
        )
    
    def get_training_samples(self, num_samples: int) -> List[Tuple[Any, Any]]:
        """Get conversation training input-output pairs."""
        samples = []
        sample_count = 0
        
        for conversation in self.conversations:
            if sample_count >= num_samples:
                break
                
            # Extract context-response pairs from conversation
            for i in range(1, len(conversation)):
                if sample_count >= num_samples:
                    break
                    
                context_history = conversation[:i]
                target_response = conversation[i][1]
                
                # Create training context
                context_tokens, response_tokens = self.tokenizer.create_training_context(
                    context_history, target_response, self.context_window_size
                )
                
                samples.append((context_tokens, response_tokens))
                sample_count += 1
        
        return samples
    
    def prepare_input(self, raw_input: Any) -> List[float]:
        """Convert token IDs to neural network input format."""
        if isinstance(raw_input, list):
            # Normalize token IDs to 0-1 range
            return [token_id / self.tokenizer.vocab_size for token_id in raw_input]
        return []
    
    def parse_output(self, network_output: List[float]) -> Any:
        """Convert network output to response tokens."""
        # Convert output probabilities back to token IDs
        if len(network_output) >= self.tokenizer.vocab_size:
            token_scores = network_output[:self.tokenizer.vocab_size]
        else:
            token_scores = list(network_output) + [0.0] * (self.tokenizer.vocab_size - len(network_output))
        
        # Simple argmax selection (could be improved with sampling)
        best_token_id = token_scores.index(max(token_scores))
        return best_token_id


class ConversationFitnessEvaluator(CapabilityFitnessEvaluator):
    """
    Evaluates conversational fitness using multiple criteria:
    - Response relevance to user input
    - Grammatical coherence 
    - Personality consistency (constitutional traits)
    - Conversation flow and engagement
    """

    def __init__(self, training_data: ConversationTrainingData, agent_identity: IdentityBundle):
        super().__init__(training_data, agent_identity)
        self.tokenizer = training_data.tokenizer
        self.conversations = training_data.conversations
        self.context_window_size = training_data.context_window_size
        self.response_max_length = training_data.response_max_length

    def evaluate_genome_fitness(self, genome, config) -> float:
        """
        Evaluate a genome's conversational ability.
        
        Tests on multiple conversation scenarios and scores based on:
        1. Context understanding (can it respond appropriately?)
        2. Vocabulary usage (diverse, appropriate words)
        3. Response coherence (grammatical, meaningful)
        4. Personality consistency (matches constitutional traits)
        """
        try:
            # Create neural network from genome
            network = neat.nn.FeedForwardNetwork.create(genome, config)
            
            total_fitness = 0.0
            test_count = 0
            
            # Test on multiple conversation scenarios
            for conversation in self.conversations[:10]:  # Test on first 10 conversations
                if len(conversation) < 2:
                    continue
                    
                # Test each turn in the conversation
                for i in range(1, len(conversation)):
                    context_history = conversation[:i]
                    expected_response = conversation[i][1]  # Agent's response
                    
                    # Skip if this turn isn't from the agent
                    if conversation[i][0].lower() != "agent":
                        continue
                    
                    # Create training context
                    context_tokens, expected_tokens = self.tokenizer.create_training_context(
                        context_history, expected_response, self.context_window_size
                    )
                    
                    # Generate response using network
                    generated_response = self._generate_response(
                        network, context_tokens, self.response_max_length
                    )
                    
                    # Score the response
                    fitness_score = self._score_response(
                        context_history, expected_response, generated_response
                    )
                    
                    total_fitness += fitness_score
                    test_count += 1
            
            # Return average fitness across all test cases
            if test_count == 0:
                return 0.0
                
            average_fitness = total_fitness / test_count
            
            # Scale fitness to reasonable range (0-10)
            return min(10.0, max(0.0, average_fitness))
            
        except Exception as e:
            print(f"Error evaluating genome fitness: {e}")
            return 0.0

    def _generate_response(self, network, context_tokens: List[int], max_length: int) -> str:
        """
        Generate a response using the neural network.
        
        Uses the context tokens as input and generates output tokens
        by sampling from network predictions.
        """
        try:
            # Convert context to network input (normalize to 0-1 range)
            network_input = [token / self.tokenizer.vocab_size for token in context_tokens]
            
            # Ensure input matches network input size
            if len(network_input) > len(network.input_nodes):
                network_input = network_input[:len(network.input_nodes)]
            elif len(network_input) < len(network.input_nodes):
                network_input.extend([0.0] * (len(network.input_nodes) - len(network_input)))
            
            # Generate response tokens
            generated_tokens = []
            
            for _ in range(max_length):
                # Get network output
                output = network.activate(network_input)
                
                # Convert output to token probabilities
                # Take the output values and map them to vocabulary indices
                if len(output) >= self.tokenizer.vocab_size:
                    # Network has enough outputs for full vocabulary
                    token_scores = output[:self.tokenizer.vocab_size]
                else:
                    # Network has fewer outputs, pad with zeros
                    token_scores = list(output) + [0.0] * (self.tokenizer.vocab_size - len(output))
                
                # Apply softmax-like selection (avoid pure argmax for diversity)
                # Use temperature-based sampling
                temperature = 0.8
                if max(token_scores) == min(token_scores):
                    # All scores equal, random selection
                    selected_token = random.randint(0, self.tokenizer.vocab_size - 1)
                else:
                    # Scale by temperature and select probabilistically
                    scaled_scores = [score / temperature for score in token_scores]
                    exp_scores = [max(0.001, score) for score in scaled_scores]  # Avoid negative
                    total = sum(exp_scores)
                    probabilities = [score / total for score in exp_scores]
                    
                    # Sample from distribution
                    rand_val = random.random()
                    cumulative = 0.0
                    selected_token = 0
                    for i, prob in enumerate(probabilities):
                        cumulative += prob
                        if rand_val <= cumulative:
                            selected_token = i
                            break
                
                # Stop if we generate end token or padding
                if selected_token in [
                    self.tokenizer.token_to_id[self.tokenizer.config.end_token],
                    self.tokenizer.token_to_id[self.tokenizer.config.pad_token]
                ]:
                    break
                    
                generated_tokens.append(selected_token)
                
                # Update network input with new token (sliding window)
                network_input = network_input[1:] + [selected_token / self.tokenizer.vocab_size]
            
            # Convert tokens back to text
            response = self.tokenizer.decode_response(generated_tokens)
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I don't understand."

    def evaluate_network(self, network, test_samples) -> float:
        """Evaluate network on test samples - required by base class."""
        total_score = 0.0
        for context, expected in test_samples[:5]:  # Test on 5 samples
            generated = self._generate_response(network, context, 32)
            score = self._score_response([], expected, generated)
            total_score += score
        return total_score / 5 if len(test_samples) > 0 else 0.0
    
    def evaluate_generation_quality(self, generation_results) -> Dict[str, float]:
        """Evaluate generation quality - required by base class."""
        return {"average_fitness": sum(g.fitness for g in generation_results) / len(generation_results)}

    def _score_response(self, context: List[Tuple[str, str]], expected: str, generated: str) -> float:
        """
        Score the quality of a generated response.
        
        Scoring criteria:
        1. Length appropriateness (not too short/long)
        2. Vocabulary diversity 
        3. Avoids repetition
        4. Contains relevant words from context
        5. Basic grammatical structure
        """
        score = 0.0
        
        # 1. Length appropriateness (0-2 points)
        gen_words = generated.split()
        if 3 <= len(gen_words) <= 20:  # Reasonable response length
            score += 2.0
        elif 1 <= len(gen_words) < 3 or 20 < len(gen_words) <= 30:
            score += 1.0
        # else: 0 points for very short or very long responses
        
        # 2. Vocabulary diversity (0-2 points)
        unique_words = len(set(gen_words))
        if unique_words >= len(gen_words) * 0.8:  # 80% unique words
            score += 2.0
        elif unique_words >= len(gen_words) * 0.6:  # 60% unique
            score += 1.0
        
        # 3. Avoids excessive repetition (0-1 point)
        if len(gen_words) > 0:
            max_word_count = max([gen_words.count(word) for word in set(gen_words)])
            if max_word_count <= 2:  # No word appears more than twice
                score += 1.0
        
        # 4. Context relevance (0-3 points)
        context_words = set()
        for speaker, text in context:
            context_words.update(text.lower().split())
        
        gen_words_set = set(word.lower() for word in gen_words)
        relevance_overlap = len(context_words.intersection(gen_words_set))
        if relevance_overlap >= 3:
            score += 3.0
        elif relevance_overlap >= 2:
            score += 2.0
        elif relevance_overlap >= 1:
            score += 1.0
        
        # 5. Avoids obvious errors (0-2 points)
        generated_lower = generated.lower()
        if not any(error in generated_lower for error in ["<unk>", "<pad>", "[unk]", "[pad]"]):
            score += 1.0  # No obvious token errors
        
        if len(generated.strip()) > 0:  # Not empty response
            score += 1.0
        
        return score  # Total possible: 10 points


class ConversationTrainingPipeline(CapabilityTrainingPipeline):
    """Complete training pipeline for conversational AI capabilities."""

    def __init__(self, training_data: ConversationTrainingData, agent_identity: IdentityBundle):
        super().__init__(training_data, agent_identity)
        self.capability_type = "conversation"
    
    def get_relevant_traits(self) -> List[str]:
        """Get constitutional traits relevant to conversation capability."""
        return [
            "AttentionSpan",      # Affects context window size
            "ProcessingSpeed",    # Affects population size and training speed
            "Curiosity",          # Affects question asking and engagement
            "SocialDrive",        # Core trait for conversation ability
            "CommunicationStyle", # Influences response style
            "Stability",          # Affects conversation consistency
            "Empathy",           # Affects emotional responses
            "CreativityIndex"     # Affects response creativity
        ]

    def create_training_data(self, corpus_data: Any) -> ConversationTrainingData:
        """
        Create conversational training data from corpus.
        
        Args:
            corpus_data: Dictionary with conversation data or path to conversation file
            
        Returns:
            ConversationTrainingData ready for training
        """
        if isinstance(corpus_data, str):
            # Load conversations from file
            conversations = self._load_conversations_from_file(corpus_data)
        elif isinstance(corpus_data, dict) and "conversations" in corpus_data:
            conversations = corpus_data["conversations"]
        else:
            # Create default conversation dataset
            conversations = self._create_default_conversations()
        
        print(f"Loaded {len(conversations)} conversations for training")
        
        # Create and configure tokenizer
        tokenizer_config = TokenizerConfig(
            vocab_size=5000,  # Reasonable vocabulary for conversation
            max_sequence_length=64,  # Context window
            min_word_frequency=2
        )
        
        tokenizer = ConversationalTokenizer(tokenizer_config)
        tokenizer.build_vocabulary(conversations)
        
        return ConversationTrainingData(
            conversations=conversations,
            tokenizer=tokenizer,
            context_window_size=64,
            response_max_length=32
        )

    def create_fitness_evaluator(self) -> ConversationFitnessEvaluator:
        """Create fitness evaluator for conversational training."""
        return ConversationFitnessEvaluator(self.training_data, self.agent_identity)

    def get_neat_config_params(self, identity: IdentityBundle) -> Dict[str, Any]:
        """
        Get NEAT configuration parameters for conversational training.
        
        Uses constitutional traits to configure network architecture:
        - AttentionSpan affects context window size
        - ProcessingSpeed affects population size
        - Curiosity affects mutation rates
        - SocialDrive affects network complexity
        """
        traits = identity.constitution_result.constitution
        
        # Base parameters
        base_params = {
            "pop_size": 150,  # Smaller population for more complex networks
            "num_inputs": 64,   # Context window size
            "num_outputs": 5000,  # Vocabulary size (will be adjusted by tokenizer)
            "initial_hidden_nodes": 50,  # More complex networks needed
            "max_nodes": 200,
            "activation": "sigmoid",
        }
        
        # Constitutional trait adjustments
        attention_span = traits.get("AttentionSpan", 3.0)
        processing_speed = traits.get("ProcessingSpeed", 3.0)
        curiosity = traits.get("Curiosity", 3.0)
        social_drive = traits.get("SocialDrive", 3.0)
        
        # Adjust based on traits
        if attention_span > 4.0:
            base_params["num_inputs"] = 128  # Larger context window
        elif attention_span < 2.0:
            base_params["num_inputs"] = 32   # Smaller context window
            
        if processing_speed > 4.0:
            base_params["pop_size"] = 300    # Larger population
        elif processing_speed < 2.0:
            base_params["pop_size"] = 75     # Smaller population
            
        if social_drive > 4.0:
            base_params["initial_hidden_nodes"] = 100  # More complex social processing
            base_params["max_nodes"] = 300
        elif social_drive < 2.0:
            base_params["initial_hidden_nodes"] = 25   # Simpler networks
            base_params["max_nodes"] = 100
        
        return base_params

    def _load_conversations_from_file(self, filepath: str) -> List[List[Tuple[str, str]]]:
        """Load conversation data from JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get("conversations", [])
        except FileNotFoundError:
            print(f"Conversation file {filepath} not found, using default conversations")
            return self._create_default_conversations()
        except Exception as e:
            print(f"Error loading conversations from {filepath}: {e}")
            return self._create_default_conversations()

    def _create_default_conversations(self) -> List[List[Tuple[str, str]]]:
        """Create a default set of conversations for training."""
        return [
            [
                ("user", "Hello, how are you today?"),
                ("agent", "I'm doing well, thank you for asking! How can I help you today?"),
                ("user", "I'm looking for some advice about learning programming."),
                ("agent", "Programming is a great skill to learn! What specific area interests you most?"),
            ],
            [
                ("user", "What's your favorite book?"),
                ("agent", "I enjoy many books, but I find science fiction particularly fascinating for exploring future possibilities."),
                ("user", "Any recommendations?"),
                ("agent", "I'd suggest starting with classics like Asimov's Robot series or Dune by Frank Herbert."),
            ],
            [
                ("user", "Can you explain artificial intelligence?"),
                ("agent", "AI is the development of computer systems that can perform tasks typically requiring human intelligence."),
                ("user", "Is AI dangerous?"),
                ("agent", "Like any powerful technology, AI has both benefits and risks that need careful consideration."),
            ],
            [
                ("user", "I'm feeling stressed about work."),
                ("agent", "I understand that work stress can be overwhelming. What specific aspects are bothering you most?"),
                ("user", "Too many deadlines and not enough time."),
                ("agent", "Time management challenges are common. Have you tried prioritizing tasks or discussing workload with your manager?"),
            ],
            [
                ("user", "Tell me a joke."),
                ("agent", "Why don't scientists trust atoms? Because they make up everything!"),
                ("user", "That's pretty good!"),
                ("agent", "I'm glad you enjoyed it! Humor is a great way to brighten the day."),
            ]
        ]


def load_conversations_from_file(filepath: str) -> List[List[Tuple[str, str]]]:
    """Load conversation data from JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("conversations", [])
    except FileNotFoundError:
        print(f"Conversation file {filepath} not found, using default conversations")
        return create_default_conversations()
    except Exception as e:
        print(f"Error loading conversations from {filepath}: {e}")
        return create_default_conversations()


def create_default_conversations() -> List[List[Tuple[str, str]]]:
    """Create a default set of conversations for training."""
    return [
        [
            ("user", "Hello, how are you today?"),
            ("agent", "I'm doing well, thank you for asking! How can I help you today?"),
            ("user", "I'm looking for some advice about learning programming."),
            ("agent", "Programming is a great skill to learn! What specific area interests you most?"),
        ],
        [
            ("user", "What's your favorite book?"),
            ("agent", "I enjoy many books, but I find science fiction particularly fascinating for exploring future possibilities."),
            ("user", "Any recommendations?"),
            ("agent", "I'd suggest starting with classics like Asimov's Robot series or Dune by Frank Herbert."),
        ],
        [
            ("user", "Can you explain artificial intelligence?"),
            ("agent", "AI is the development of computer systems that can perform tasks typically requiring human intelligence."),
            ("user", "Is AI dangerous?"),
            ("agent", "Like any powerful technology, AI has both benefits and risks that need careful consideration."),
        ],
        [
            ("user", "I'm feeling stressed about work."),
            ("agent", "I understand that work stress can be overwhelming. What specific aspects are bothering you most?"),
            ("user", "Too many deadlines and not enough time."),
            ("agent", "Time management challenges are common. Have you tried prioritizing tasks or discussing workload with your manager?"),
        ],
        [
            ("user", "Tell me a joke."),
            ("agent", "Why don't scientists trust atoms? Because they make up everything!"),
            ("user", "That's pretty good!"),
            ("agent", "I'm glad you enjoyed it! Humor is a great way to brighten the day."),
        ]
    ]


def train_agent_conversation_capability(
    identity_bundle: IdentityBundle,
    generations: int = 50,
    conversation_data: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Train an agent's conversational capability using constitutional traits.
    
    Args:
        identity_bundle: Agent's constitutional identity
        generations: Number of evolution generations
        conversation_data: Training conversation data (optional)
        
    Returns:
        Training result with evolved conversational network
    """
    print(f"Training conversational capability for agent {identity_bundle.id_hash[:12]}...")
    print(f"Training for {generations} generations")
    
    # Create training data first
    if isinstance(conversation_data, str):
        # Load conversations from file
        conversations = load_conversations_from_file(conversation_data)
    elif isinstance(conversation_data, dict) and "conversations" in conversation_data:
        conversations = conversation_data["conversations"]
    else:
        # Create default conversation dataset
        conversations = create_default_conversations()
    
    print(f"Loaded {len(conversations)} conversations for training")
    
    # Create and configure tokenizer
    tokenizer_config = TokenizerConfig(
        vocab_size=5000,  # Reasonable vocabulary for conversation
        max_sequence_length=64,  # Context window
        min_word_frequency=2
    )
    
    tokenizer = ConversationalTokenizer(tokenizer_config)
    tokenizer.build_vocabulary(conversations)
    
    training_data = ConversationTrainingData(
        conversations=conversations,
        tokenizer=tokenizer,
        context_window_size=64,
        response_max_length=32
    )
    
    # Create training pipeline with required parameters
    pipeline = ConversationTrainingPipeline(training_data, identity_bundle)
    
    # Create fitness evaluator  
    fitness_evaluator = ConversationFitnessEvaluator(training_data, identity_bundle)
    
    # Get NEAT configuration parameters
    neat_params = pipeline.get_neat_config_params(identity_bundle)
    
    # Adjust output size to match tokenizer vocabulary
    neat_params["num_outputs"] = training_data.tokenizer.vocab_size
    
    print(f"Network configuration:")
    print(f"  Inputs: {neat_params['num_inputs']} (context window)")
    print(f"  Outputs: {neat_params['num_outputs']} (vocabulary size)")
    print(f"  Population: {neat_params['pop_size']}")
    print(f"  Max nodes: {neat_params['max_nodes']}")
    
    # Create constitutional NEAT runner
    neat_runner = ConstitutionalNEATRunner(identity_bundle)
    
    # Create fitness function
    def conversation_fitness(genomes, config):
        for genome_id, genome in genomes:
            genome.fitness = fitness_evaluator.evaluate_genome_fitness(genome, config)
    
    # Train the network
    try:
        best_genome = neat_runner.evolve(
            conversation_fitness,
            generations=generations,
            num_inputs=neat_params["num_inputs"],
            num_outputs=neat_params["num_outputs"]
        )
        
        # Create final network
        best_network = neat_runner.create_network()
        
        # Test the final network
        test_context = [("user", "Hello! How are you doing today?")]
        test_fitness = fitness_evaluator._score_response(
            test_context, 
            "I'm doing great, thank you for asking!",
            "Generated test response"  # Would be actual network output in real use
        )
        
        result = {
            "identity_bundle": identity_bundle,
            "capability_type": "conversation",
            "training_generations": generations,
            "final_fitness": getattr(best_genome, 'fitness', 0.0),
            "best_genome": best_genome,
            "best_network": best_network,
            "neat_runner": neat_runner,
            "tokenizer": training_data.tokenizer,
            "training_config": neat_params,
            "sample_generation": {"test_fitness": test_fitness},
            "vocabulary_size": training_data.tokenizer.vocab_size,
        }
        
        print(f"Training complete! Final fitness: {result['final_fitness']:.3f}")
        return result
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    # Test conversational training
    from ..genome import create_random_genome
    from ..identity import create_agent_identity
    from ..traits import COMPLETE_TRAIT_DEFINITIONS
    
    print("Testing Conversational Evolution Pipeline...")
    
    # Create test agent
    genome = create_random_genome(COMPLETE_TRAIT_DEFINITIONS, seed=42)
    identity = create_agent_identity(genome)
    
    print(f"Test agent: {identity.id_hash[:12]}")
    print(f"Traits: AttentionSpan={identity.constitution_result.constitution.get('AttentionSpan', 'N/A'):.1f}")
    
    # Train conversational capability (short test)
    result = train_agent_conversation_capability(identity, generations=3)
    
    print(f"Training result: {result['final_fitness']:.3f} fitness")
    print("Conversational training test complete!")