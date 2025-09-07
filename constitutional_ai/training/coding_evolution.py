"""
Coding Evolution Pipeline for Constitutional AI Agents

This module implements evolutionary programming capability learning:
1. Basic syntax pattern recognition
2. Function structure understanding
3. Logic implementation (if/else, loops)
4. Algorithm development (sorting, searching)
5. Code completion and generation

Constitutional traits influence:
- Problem-solving approach (analytical vs creative)
- Code style and verbosity preferences
- Learning rate and pattern recognition
- Innovation vs convention balance
"""

import random
import re
import neat
from typing import Dict, List, Tuple, Any, Optional

from ..identity import IdentityBundle
from .base_capability import (
    TrainingData,
    CapabilityFitnessEvaluator,
    CapabilityTrainingPipeline,
    EvaluationResult,
)


class CodingTrainingData(TrainingData):
    """Training data for coding evolution."""

    def __init__(
        self,
        code_samples: List[Dict[str, str]],
        patterns: List[str],
        keywords: List[str],
    ):
        super().__init__(capability_type="coding", data_size=len(code_samples))
        self.code_samples = code_samples
        self.patterns = patterns
        self.keywords = keywords

        # Create vocabulary from code samples
        self.char_to_int, self.int_to_char, self.vocab_size = self._build_vocabulary()

    def _build_vocabulary(self):
        """Build character vocabulary from code samples."""
        all_chars = set()

        for sample in self.code_samples:
            all_chars.update(sample["input"])
            all_chars.update(sample["output"])

        # Add common programming characters
        programming_chars = {
            "{",
            "}",
            "[",
            "]",
            "(",
            ")",
            ";",
            ":",
            "=",
            "+",
            "-",
            "*",
            "/",
            "<",
            ">",
            "!",
            "&",
            "|",
            "^",
            "%",
            "?",
            "~",
            "\\",
            '"',
            "'",
            "`",
        }
        all_chars.update(programming_chars)

        chars = sorted(list(all_chars))
        char_to_int = {char: i for i, char in enumerate(chars)}
        int_to_char = {i: char for i, char in enumerate(chars)}

        return char_to_int, int_to_char, len(chars)

    def get_training_samples(
        self, num_samples: int
    ) -> List[Tuple[List[int], List[int]]]:
        """Get training input-output pairs for coding tasks."""
        samples = []

        for _ in range(num_samples):
            # Pick random code sample
            sample = random.choice(self.code_samples)

            # Convert prompt and code to character indices
            prompt_chars = [
                self.char_to_int.get(char, 0) for char in sample["input"][:20]
            ]  # Limit length
            code_chars = [
                self.char_to_int.get(char, 0) for char in sample["output"][:40]
            ]  # Limit length

            # Pad sequences to consistent length
            prompt_chars = (prompt_chars + [0] * 20)[:20]
            code_chars = (code_chars + [0] * 40)[:40]

            samples.append((prompt_chars, code_chars))

        return samples

    def prepare_input(self, raw_input: List[int]) -> List[float]:
        """Convert character sequence to neural network input."""
        return [x / self.vocab_size for x in raw_input]

    def parse_output(self, network_output: List[float]) -> List[int]:
        """Convert network output to character indices."""
        return [
            max(0, min(int(x * self.vocab_size), self.vocab_size - 1))
            for x in network_output
        ]

    @classmethod
    def from_samples(cls, code_samples: List[Dict[str, str]]):
        """Create training data from code samples."""

        # Extract patterns and keywords
        patterns = []
        keywords = set()

        for sample in code_samples:
            code = sample["output"]

            # Extract function definitions
            if "def " in code:
                patterns.append("function_definition")

            # Extract control structures
            if any(kw in code for kw in ["if ", "for ", "while "]):
                patterns.append("control_structure")

            # Extract common keywords
            code_keywords = re.findall(
                r"\\b(def|if|else|elif|for|while|return|import|from|class|try|except)\\b",
                code,
            )
            keywords.update(code_keywords)

        return cls(
            code_samples=code_samples,
            patterns=list(set(patterns)),
            keywords=list(keywords),
        )


class CodingEvolutionFitness(CapabilityFitnessEvaluator):
    """
    Fitness evaluator for coding evolution.

    Measures programming capability through:
    - Syntax correctness
    - Pattern recognition
    - Code completion accuracy
    - Problem-solving logic
    """

    def setup_trait_weights(self):
        """Setup evaluation weights based on constitutional traits."""
        self.logic_weight = self._normalize_trait("ProcessingSpeed", 0.2, 0.5)
        self.creativity_weight = self._normalize_trait("InnovationDrive", 0.1, 0.3)
        self.accuracy_weight = self._normalize_trait("Expertise", 0.2, 0.4)
        self.pattern_weight = self._normalize_trait("MetaLearning", 0.1, 0.3)

    def evaluate_network(
        self, network: neat.nn.FeedForwardNetwork, num_tests: int = 50
    ) -> float:
        """
        Evaluate a neural network's coding capability.

        Returns fitness score based on:
        - Code completion accuracy
        - Syntax pattern recognition
        - Problem-solving logic
        """
        total_fitness = 0.0
        test_samples = self.training_data.get_training_samples(num_tests)

        for prompt_chars, expected_code_chars in test_samples:
            # Prepare network input (first 4 chars of prompt)
            input_chars = prompt_chars[:4]
            normalized_input = self.training_data.prepare_input(input_chars)

            # Get network prediction
            output = network.activate(normalized_input)

            # Convert output to character indices (first 4 outputs)
            predicted_chars = self.training_data.parse_output(output[:4])
            expected_chars = expected_code_chars[:4]

            # Calculate accuracy
            accuracy = (
                sum(1 for p, e in zip(predicted_chars, expected_chars) if p == e) / 4.0
            )

            # Pattern recognition bonus
            pattern_score = self._evaluate_pattern_recognition(
                predicted_chars, expected_chars
            )

            # Combine scores with trait weights
            sample_fitness = (
                accuracy * self.accuracy_weight + pattern_score * self.pattern_weight
            )

            total_fitness += sample_fitness

        return total_fitness / num_tests if num_tests > 0 else 0.0

    def _evaluate_pattern_recognition(
        self, predicted: List[int], expected: List[int]
    ) -> float:
        """Evaluate how well the network recognizes coding patterns."""
        # Convert back to characters for pattern analysis
        predicted_chars = [self.training_data.int_to_char.get(i, "") for i in predicted]
        expected_chars = [self.training_data.int_to_char.get(i, "") for i in expected]

        predicted_str = "".join(predicted_chars)
        expected_str = "".join(expected_chars)

        # Check for programming patterns
        pattern_score = 0.0

        # Parentheses matching
        if "(" in expected_str and ")" in predicted_str:
            pattern_score += 0.1

        # Common operators
        for op in ["=", "+", "-", "*"]:
            if op in expected_str and op in predicted_str:
                pattern_score += 0.05

        # Keyword recognition
        for keyword in ["def", "if", "for"]:
            if keyword in expected_str.lower() and keyword in predicted_str.lower():
                pattern_score += 0.1

        return min(pattern_score, 1.0)

    def evaluate_generation_quality(
        self,
        network: neat.nn.FeedForwardNetwork,
        prompt: str = "def hello",
        max_length: int = 60,
    ) -> EvaluationResult:
        """
        Evaluate the quality of generated code.

        Returns detailed evaluation with code samples and metrics.
        """
        # Convert prompt to character indices
        prompt_chars = [
            self.training_data.char_to_int.get(char, 0) for char in prompt[:4]
        ]

        # Generate code
        generated_code = prompt
        current_input = prompt_chars

        for _ in range(max_length - len(prompt)):
            # Normalize input
            normalized_input = self.training_data.prepare_input(current_input)

            # Predict next character
            output = network.activate(normalized_input)
            next_char_idx = self.training_data.parse_output(output[:1])[0]
            next_char = self.training_data.int_to_char.get(next_char_idx, "")

            generated_code += next_char

            # Update input for next prediction
            current_input = current_input[1:] + [next_char_idx]

        # Analyze generated code
        syntax_score = self._analyze_syntax(generated_code)
        pattern_score = self._analyze_patterns(generated_code)
        fitness_score = self.evaluate_network(network, num_tests=10)

        return EvaluationResult(
            fitness_score=fitness_score,
            sample_outputs={"generated_code": generated_code, "prompt": prompt},
            performance_metrics={
                "length": len(generated_code),
                "syntax_score": syntax_score,
                "pattern_score": pattern_score,
                "unique_chars": len(set(generated_code)),
            },
            evaluation_metadata={"max_length": max_length, "language": "python"},
        )

    def _analyze_syntax(self, code: str) -> float:
        """Analyze basic syntax correctness."""
        score = 0.0

        # Balanced parentheses
        paren_balance = 0
        for char in code:
            if char == "(":
                paren_balance += 1
            elif char == ")":
                paren_balance -= 1

        if paren_balance == 0:
            score += 0.3

        # Contains valid Python keywords
        python_keywords = ["def", "if", "else", "for", "while", "return", "import"]
        for keyword in python_keywords:
            if keyword in code:
                score += 0.1
                break

        # Basic structure indicators
        if ":" in code:  # Likely has proper indentation structure
            score += 0.2

        if any(op in code for op in ["=", "+", "-", "*"]):  # Has operators
            score += 0.2

        return min(score, 1.0)

    def _analyze_patterns(self, code: str) -> float:
        """Analyze coding pattern recognition."""
        score = 0.0

        # Function definition pattern
        if "def " in code and "(" in code and ":" in code:
            score += 0.3

        # Control flow pattern
        if any(pattern in code for pattern in ["if ", "for ", "while "]):
            score += 0.2

        # Variable assignment pattern
        if "=" in code and "==" not in code:
            score += 0.2

        # Import pattern
        if "import" in code or "from" in code:
            score += 0.2

        return min(score, 1.0)


class CodingTrainingPipeline(CapabilityTrainingPipeline):
    """
    Complete pipeline for evolving coding capabilities in constitutional agents.

    Training progression:
    1. Syntax pattern recognition
    2. Code completion
    3. Function generation
    4. Problem solving
    """

    def create_fitness_evaluator(self) -> CodingEvolutionFitness:
        """Create the fitness evaluator for coding capability."""
        return CodingEvolutionFitness(self.training_data, self.agent_identity)

    def get_neat_config_params(self) -> Dict[str, Any]:
        """Get coding-specific NEAT configuration parameters."""
        return {
            "num_inputs": 4,  # 4 character sequence input
            "num_outputs": 4,  # 4 character prediction output
            "num_hidden": 8,  # More hidden nodes for complex pattern recognition
            "max_nodes": 50,  # Allow more complex networks for coding
            "activation_function": "tanh",  # Good for sequence prediction
        }

    def get_relevant_traits(self) -> List[str]:
        """Get constitutional traits most relevant to coding capability."""
        return [
            "ProcessingSpeed",  # Logical thinking speed
            "InnovationDrive",  # Creative problem solving
            "Expertise",  # Technical accuracy
            "MetaLearning",  # Pattern recognition
            "AttentionSpan",  # Focus on complex problems
            "LearningRate",  # Skill acquisition speed
        ]


def create_coding_training_corpus() -> CodingTrainingData:
    """
    Create a substantial training corpus for coding evolution using real datasets.

    This now loads actual coding examples instead of tiny hardcoded samples.
    """
    try:
        from ..corpus_loader import get_coding_corpus

        print("Loading substantial coding corpus for training...")
        # Get real code samples for training
        code_samples = get_coding_corpus(500_000)  # 500KB corpus

        print(f"Loaded coding corpus: {len(code_samples)} samples")
        return CodingTrainingData.from_samples(code_samples)

    except ImportError:
        print("Corpus loader not available, using fallback")
        return _get_fallback_coding_corpus()


def _get_fallback_coding_corpus() -> CodingTrainingData:
    """Fallback coding corpus if corpus_loader isn't available."""
    code_samples = [
        # Basic function definitions
        {
            "input": "create hello function",
            "output": "def hello():\\n    print('Hello, World!')\\n    return True",
            "language": "python",
        },
        {
            "input": "add two numbers",
            "output": "def add(a, b):\\n    result = a + b\\n    return result",
            "language": "python",
        },
        {
            "input": "check if even",
            "output": "def is_even(n):\\n    if n % 2 == 0:\\n        return True\\n    return False",
            "language": "python",
        },
        # Control structures
        {
            "input": "loop through list",
            "output": "for item in items:\\n    print(item)\\n    process(item)",
            "language": "python",
        },
        {
            "input": "conditional check",
            "output": "if condition:\\n    do_something()\\nelse:\\n    do_other()",
            "language": "python",
        },
        {
            "input": "while loop count",
            "output": "count = 0\\nwhile count < 10:\\n    count += 1\\n    print(count)",
            "language": "python",
        },
        # Data structures
        {
            "input": "create list",
            "output": "my_list = [1, 2, 3, 4, 5]\\nfor i in my_list:\\n    print(i)",
            "language": "python",
        },
        {
            "input": "dictionary access",
            "output": "data = {'key': 'value'}\\nresult = data.get('key')\\nprint(result)",
            "language": "python",
        },
        # Error handling
        {
            "input": "try catch block",
            "output": "try:\\n    risky_operation()\\nexcept Exception as e:\\n    handle_error(e)",
            "language": "python",
        },
        # Class definition
        {
            "input": "simple class",
            "output": "class MyClass:\\n    def __init__(self):\\n        self.value = 0\\n    def get(self):\\n        return self.value",
            "language": "python",
        },
    ]

    # Multiply samples for more training data
    extended_samples = code_samples * 20  # More repetitions for better training

    return CodingTrainingData.from_samples(extended_samples)


# Factory function for easy coding capability training
def train_agent_coding_capability(
    agent_identity: IdentityBundle,
    training_data: Optional[CodingTrainingData] = None,
    generations: int = 30,
) -> Dict[str, Any]:
    """
    Convenience function to train an agent's coding capability.

    Args:
        agent_identity: Constitutional agent to train
        training_data: Coding training data (uses default if None)
        generations: Training generations

    Returns:
        Training results
    """
    if training_data is None:
        training_data = create_coding_training_corpus()

    pipeline = CodingTrainingPipeline(training_data, agent_identity)

    # Use CPU training (optimal for NEAT evolution)
    print("🧠 Using CPU training (optimal for NEAT)...")
    return pipeline.train_capability(generations)
