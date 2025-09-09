"""
Conversational Tokenization System for Constitutional AI

Implements word-level tokenization with special tokens for true conversation:
- Context awareness with <START>, <END>, <USER>, <AGENT> markers
- Vocabulary building from conversational datasets
- Constitutional trait-aware encoding
"""

import re
import json
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from collections import Counter


@dataclass
class TokenizerConfig:
    """Configuration for conversational tokenizer."""
    
    vocab_size: int = 10000  # Target vocabulary size
    max_sequence_length: int = 128  # Maximum tokens per sequence
    min_word_frequency: int = 2  # Minimum frequency to include word
    include_subword_tokens: bool = False  # For rare words
    
    # Special tokens
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"
    start_token: str = "<START>"
    end_token: str = "<END>"
    user_token: str = "<USER>"
    agent_token: str = "<AGENT>"
    context_separator: str = "<SEP>"


class ConversationalTokenizer:
    """
    Word-level tokenizer designed for conversational AI training.
    
    Creates vocabulary from conversation datasets and provides encoding/decoding
    for NEAT neural network training with constitutional personality awareness.
    """
    
    def __init__(self, config: Optional[TokenizerConfig] = None):
        """Initialize tokenizer with configuration."""
        self.config = config or TokenizerConfig()
        
        # Vocabulary mappings
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special token IDs (reserved first slots)
        self.special_tokens = [
            self.config.pad_token,
            self.config.unk_token,
            self.config.start_token,
            self.config.end_token,
            self.config.user_token,
            self.config.agent_token,
            self.config.context_separator,
        ]
        
        # Initialize with special tokens
        self._initialize_special_tokens()
        
    def _initialize_special_tokens(self):
        """Initialize special tokens in vocabulary."""
        for i, token in enumerate(self.special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
        self.vocab_size = len(self.special_tokens)
        
    def build_vocabulary(self, conversations: List[List[Tuple[str, str]]]):
        """
        Build vocabulary from conversation datasets.
        
        Args:
            conversations: List of conversations, where each conversation is
                          a list of (speaker, text) tuples.
                          Example: [("user", "Hello"), ("agent", "Hi there!")]
        """
        print(f"Building vocabulary from {len(conversations)} conversations...")
        
        # Collect all words
        word_counts = Counter()
        
        for conversation in conversations:
            for speaker, text in conversation:
                words = self._tokenize_text(text)
                word_counts.update(words)
        
        print(f"Found {len(word_counts)} unique words")
        
        # Filter by frequency and take top words
        frequent_words = [
            word for word, count in word_counts.most_common()
            if count >= self.config.min_word_frequency
        ]
        
        # Take top words to reach target vocab size
        available_slots = self.config.vocab_size - len(self.special_tokens)
        vocabulary_words = frequent_words[:available_slots]
        
        # Add to vocabulary
        for word in vocabulary_words:
            if word not in self.token_to_id:
                self.token_to_id[word] = self.vocab_size
                self.id_to_token[self.vocab_size] = word
                self.vocab_size += 1
        
        print(f"Final vocabulary size: {self.vocab_size} tokens")
        print(f"Coverage: {len(vocabulary_words)} content words + {len(self.special_tokens)} special tokens")
        
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Raw text string
            
        Returns:
            List of word tokens
        """
        # Simple regex tokenization (could be enhanced with spaCy/NLTK)
        # Handles punctuation, contractions, etc.
        words = re.findall(r"\w+|[^\w\s]", text.lower())
        return words
        
    def encode_conversation(
        self, 
        conversation: List[Tuple[str, str]], 
        max_length: Optional[int] = None
    ) -> List[int]:
        """
        Encode a conversation into token IDs.
        
        Args:
            conversation: List of (speaker, text) tuples
            max_length: Maximum sequence length (uses config default if None)
            
        Returns:
            List of token IDs representing the conversation
        """
        max_len = max_length or self.config.max_sequence_length
        tokens = [self.token_to_id[self.config.start_token]]
        
        for speaker, text in conversation:
            # Add speaker marker
            speaker_token = (
                self.config.user_token if speaker.lower() == "user" 
                else self.config.agent_token
            )
            tokens.append(self.token_to_id[speaker_token])
            
            # Add text tokens
            words = self._tokenize_text(text)
            for word in words:
                token_id = self.token_to_id.get(word, self.token_to_id[self.config.unk_token])
                tokens.append(token_id)
                
                if len(tokens) >= max_len - 1:  # Reserve space for END token
                    break
            
            # Add separator between turns
            if len(tokens) < max_len - 1:
                tokens.append(self.token_to_id[self.config.context_separator])
        
        # Add end token
        tokens.append(self.token_to_id[self.config.end_token])
        
        # Pad or truncate to max_length
        if len(tokens) < max_len:
            tokens.extend([self.token_to_id[self.config.pad_token]] * (max_len - len(tokens)))
        else:
            tokens = tokens[:max_len]
            tokens[-1] = self.token_to_id[self.config.end_token]  # Ensure proper ending
            
        return tokens
    
    def decode_response(self, token_ids: List[int], skip_special: bool = True) -> str:
        """
        Decode token IDs back into text.
        
        Args:
            token_ids: List of token IDs
            skip_special: Whether to skip special tokens in output
            
        Returns:
            Decoded text string
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                
                if skip_special and token in self.special_tokens:
                    if token in [self.config.user_token, self.config.agent_token]:
                        tokens.append(f"[{token}]")  # Keep speaker markers for readability
                    continue
                    
                tokens.append(token)
        
        # Simple reconstruction (could be enhanced with proper detokenization)
        text = " ".join(tokens)
        
        # Basic punctuation cleanup
        text = re.sub(r' ([.!?,:;])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r' \' ', r"' ", text)  # Fix contractions
        
        return text.strip()
    
    def create_training_context(
        self, 
        conversation_history: List[Tuple[str, str]], 
        target_response: str,
        context_length: int = 64
    ) -> Tuple[List[int], List[int]]:
        """
        Create training input/output pair for NEAT networks.
        
        Args:
            conversation_history: Previous conversation turns
            target_response: Expected agent response
            context_length: Length of context window for input
            
        Returns:
            Tuple of (input_context_ids, target_output_ids)
        """
        # Encode conversation history as context
        context_tokens = self.encode_conversation(
            conversation_history, max_length=context_length
        )
        
        # Encode target response
        target_words = self._tokenize_text(target_response)
        target_tokens = []
        for word in target_words:
            token_id = self.token_to_id.get(word, self.token_to_id[self.config.unk_token])
            target_tokens.append(token_id)
        
        # Pad target to reasonable length
        max_response_length = 32
        if len(target_tokens) < max_response_length:
            target_tokens.extend(
                [self.token_to_id[self.config.pad_token]] * 
                (max_response_length - len(target_tokens))
            )
        else:
            target_tokens = target_tokens[:max_response_length]
            
        return context_tokens, target_tokens
    
    def save_tokenizer(self, filepath: str):
        """Save tokenizer configuration and vocabulary to file."""
        data = {
            "config": {
                "vocab_size": self.config.vocab_size,
                "max_sequence_length": self.config.max_sequence_length,
                "min_word_frequency": self.config.min_word_frequency,
                "special_tokens": {
                    "pad_token": self.config.pad_token,
                    "unk_token": self.config.unk_token,
                    "start_token": self.config.start_token,
                    "end_token": self.config.end_token,
                    "user_token": self.config.user_token,
                    "agent_token": self.config.agent_token,
                    "context_separator": self.config.context_separator,
                }
            },
            "vocabulary": {
                "token_to_id": self.token_to_id,
                "id_to_token": {str(k): v for k, v in self.id_to_token.items()},  # JSON needs string keys
                "vocab_size": self.vocab_size
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Tokenizer saved to {filepath}")
    
    @classmethod
    def load_tokenizer(cls, filepath: str) -> 'ConversationalTokenizer':
        """Load tokenizer from saved file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Reconstruct config
        config_data = data["config"]
        config = TokenizerConfig(
            vocab_size=config_data["vocab_size"],
            max_sequence_length=config_data["max_sequence_length"],
            min_word_frequency=config_data["min_word_frequency"],
            pad_token=config_data["special_tokens"]["pad_token"],
            unk_token=config_data["special_tokens"]["unk_token"],
            start_token=config_data["special_tokens"]["start_token"],
            end_token=config_data["special_tokens"]["end_token"],
            user_token=config_data["special_tokens"]["user_token"],
            agent_token=config_data["special_tokens"]["agent_token"],
            context_separator=config_data["special_tokens"]["context_separator"],
        )
        
        # Create tokenizer and load vocabulary
        tokenizer = cls(config)
        tokenizer.token_to_id = data["vocabulary"]["token_to_id"]
        tokenizer.id_to_token = {int(k): v for k, v in data["vocabulary"]["id_to_token"].items()}
        tokenizer.vocab_size = data["vocabulary"]["vocab_size"]
        
        print(f"Tokenizer loaded from {filepath} with vocabulary size {tokenizer.vocab_size}")
        
        return tokenizer


# Example conversation data structure for training
def create_example_conversations() -> List[List[Tuple[str, str]]]:
    """Create example conversation data for testing."""
    return [
        [
            ("user", "Hello, how are you?"),
            ("agent", "I'm doing well, thank you! How can I help you today?"),
            ("user", "What's your favorite color?"),
            ("agent", "I find blue fascinating - it represents both calm and depth.")
        ],
        [
            ("user", "Can you explain quantum physics?"),
            ("agent", "Quantum physics studies matter and energy at the smallest scales, where particles can exist in multiple states simultaneously."),
            ("user", "That sounds complex!"),
            ("agent", "It is! But that's what makes it so interesting to explore.")
        ],
        [
            ("user", "Tell me a joke"),
            ("agent", "Why don't scientists trust atoms? Because they make up everything!"),
            ("user", "That's funny!"),
            ("agent", "I'm glad you enjoyed it! Humor is a great way to connect.")
        ]
    ]


if __name__ == "__main__":
    # Test the tokenizer
    print("Testing Conversational Tokenizer...")
    
    # Create tokenizer
    tokenizer = ConversationalTokenizer()
    
    # Build vocabulary from example data
    conversations = create_example_conversations()
    tokenizer.build_vocabulary(conversations)
    
    # Test encoding/decoding
    test_conversation = [
        ("user", "Hello there!"),
        ("agent", "Hi! How can I help you?")
    ]
    
    encoded = tokenizer.encode_conversation(test_conversation)
    decoded = tokenizer.decode_response(encoded)
    
    print(f"Original: {test_conversation}")
    print(f"Encoded: {encoded[:20]}...")  # Show first 20 tokens
    print(f"Decoded: {decoded}")
    
    # Test training context creation
    history = [("user", "What's the weather like?")]
    target = "I don't have access to current weather data, but I can help you find weather information online."
    
    context, target_tokens = tokenizer.create_training_context(history, target)
    print(f"Training context length: {len(context)}")
    print(f"Target response length: {len(target_tokens)}")
    
    print("Tokenizer test complete!")