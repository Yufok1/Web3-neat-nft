"""
Corpus Loading System for Constitutional AI Training

This module provides comprehensive corpus loading from various sources:
- HuggingFace datasets for language training
- GitHub code datasets for programming training
- Custom corpus files
- Web scraping and preprocessing

Supports multi-megabyte training corpora for serious AI evolution.
"""

import os
import json
from typing import Dict, List
from dataclasses import dataclass

try:
    from datasets import load_dataset

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


@dataclass
class CorpusConfig:
    """Configuration for corpus loading."""

    language_corpus_size: int = 10_000_000  # 10MB of text
    coding_corpus_size: int = 5_000_000  # 5MB of code
    cache_dir: str = "corpus_cache"
    min_text_length: int = 100
    max_text_length: int = 2000


class LanguageCorpusLoader:
    """Load substantial language training corpora."""

    def __init__(self, config: CorpusConfig = None):
        self.config = config or CorpusConfig()
        os.makedirs(self.config.cache_dir, exist_ok=True)

    def load_wikipedia_corpus(self, language: str = "en") -> str:
        """Load Wikipedia articles for language training."""
        cache_file = os.path.join(self.config.cache_dir, f"wikipedia_{language}.txt")

        if os.path.exists(cache_file):
            print(f"Loading cached Wikipedia corpus from {cache_file}")
            with open(cache_file, "r", encoding="utf-8") as f:
                return f.read()

        if not HF_AVAILABLE:
            print("HuggingFace datasets not available, using fallback corpus")
            return self._get_fallback_language_corpus()

        print("Downloading text corpus from HuggingFace...")
        try:
            # Try modern datasets that work
            datasets_to_try = [
                ("wikitext", "wikitext-103-raw-v1"),  # Wikitext dataset
                ("c4", "en"),  # Common Crawl dataset
                ("pile", None),  # The Pile dataset
                ("bookcorpus", None),  # BookCorpus
                ("pg19", None),  # Project Gutenberg
                ("tiny_shakespeare", None),  # Shakespeare texts
                ("wikipedia", "20220301.en"),  # Wikipedia dataset
            ]

            for dataset_name, subset in datasets_to_try:
                try:
                    print(f"Trying {dataset_name}...")

                    if dataset_name == "wikitext" and subset:
                        dataset = load_dataset(
                            "wikitext",
                            subset,
                            split="train",
                            streaming=True,
                        )
                    elif dataset_name == "c4" and subset:
                        dataset = load_dataset(
                            "allenai/c4",
                            subset,
                            split="train",
                            streaming=True,
                            trust_remote_code=True,
                        )
                    elif dataset_name == "pile":
                        dataset = load_dataset(
                            "monology/pile-uncopyrighted", split="train", streaming=True
                        )
                    elif dataset_name == "bookcorpus":
                        dataset = load_dataset(
                            "bookcorpus", split="train", streaming=True
                        )
                    elif dataset_name == "pg19":
                        dataset = load_dataset("pg19", split="train", streaming=True)
                    elif dataset_name == "tiny_shakespeare":
                        dataset = load_dataset(
                            "tiny_shakespeare", split="train", streaming=True
                        )
                    elif dataset_name == "wikipedia" and subset:
                        dataset = load_dataset(
                            "wikipedia", subset, split="train", streaming=True
                        )
                    else:
                        continue

                    corpus_text = ""
                    doc_count = 0

                    for doc in dataset:
                        text = doc.get("text", "")
                        if len(text) >= self.config.min_text_length:
                            # Truncate very long documents
                            if len(text) > self.config.max_text_length:
                                text = text[: self.config.max_text_length]

                            corpus_text += text + "\n\n"
                            doc_count += 1

                            # Stop when we have enough text
                            if len(corpus_text) >= self.config.language_corpus_size:
                                break

                            if doc_count % 50 == 0:
                                print(
                                    f"Processed {doc_count} documents, {
                                        len(corpus_text):,} characters"
                                )

                    if len(corpus_text) > 10000:  # Got substantial data
                        # Cache the corpus
                        with open(cache_file, "w", encoding="utf-8") as f:
                            f.write(corpus_text)

                        print(
                            f"{dataset_name} corpus loaded: {
                                len(corpus_text):,} characters, {doc_count} documents"
                        )
                        return corpus_text

                except Exception as dataset_error:
                    error_msg = str(dataset_error).lower()
                    if "script" in error_msg and (
                        "no longer supported" in error_msg or "deprecated" in error_msg
                    ):
                        print(
                            f"Dataset {dataset_name} has deprecated scripts, skipping..."
                        )
                    else:
                        print(f"Failed to load {dataset_name}: {dataset_error}")
                    continue

            # If all datasets failed, use fallback
            print("All HuggingFace datasets failed, using fallback")
            return self._get_fallback_language_corpus()

        except Exception as e:
            print(f"Error loading corpus: {e}")
            return self._get_fallback_language_corpus()

    def load_openwebtext_corpus(self) -> str:
        """Load OpenWebText corpus for diverse language training."""
        cache_file = os.path.join(self.config.cache_dir, "openwebtext.txt")

        if os.path.exists(cache_file):
            print(f"Loading cached OpenWebText corpus from {cache_file}")
            with open(cache_file, "r", encoding="utf-8") as f:
                return f.read()

        if not HF_AVAILABLE:
            return self._get_fallback_language_corpus()

        print("Downloading OpenWebText corpus...")
        try:
            # Try multiple dataset options
            datasets_to_try = [
                ("Skylion007/openwebtext", None),  # Alternative hosting
                ("openwebtext", None),  # Original but may be deprecated
            ]

            for dataset_name, subset in datasets_to_try:
                try:
                    print(f"Trying {dataset_name}...")

                    if subset:
                        dataset = load_dataset(
                            dataset_name, subset, split="train", streaming=True
                        )
                    else:
                        dataset = load_dataset(
                            dataset_name, split="train", streaming=True
                        )

                    corpus_text = ""
                    doc_count = 0

                    for doc in dataset:
                        text = doc.get("text", "")
                        if len(text) >= self.config.min_text_length:
                            # Truncate very long documents
                            if len(text) > self.config.max_text_length:
                                text = text[: self.config.max_text_length]

                            corpus_text += text + "\n\n"
                            doc_count += 1

                            if len(corpus_text) >= self.config.language_corpus_size:
                                break

                        if doc_count % 50 == 0:
                            print(
                                f"Processed {doc_count} documents, {len(corpus_text):,} characters"
                            )

                    if len(corpus_text) > 10000:  # Got substantial data
                        # Cache the corpus
                        with open(cache_file, "w", encoding="utf-8") as f:
                            f.write(corpus_text)

                        print(
                            f"OpenWebText corpus loaded: {
                                len(corpus_text):,} characters, {doc_count} documents"
                        )
                        return corpus_text

                except Exception as dataset_error:
                    error_msg = str(dataset_error).lower()
                    if "script" in error_msg and (
                        "no longer supported" in error_msg or "deprecated" in error_msg
                    ):
                        print(
                            f"Dataset {dataset_name} has deprecated scripts, trying alternatives..."
                        )
                    else:
                        print(f"Failed to load {dataset_name}: {dataset_error}")
                    continue

            # If all datasets failed, use fallback
            print("All OpenWebText datasets failed, using fallback corpus")
            return self._get_fallback_language_corpus()

        except Exception as e:
            print(f"Error loading OpenWebText: {e}")
            return self._get_fallback_language_corpus()

    def load_books_corpus(self) -> str:
        """Load BookCorpus or similar for literature-style training."""
        cache_file = os.path.join(self.config.cache_dir, "books.txt")

        if os.path.exists(cache_file):
            print(f"Loading cached books corpus from {cache_file}")
            with open(cache_file, "r", encoding="utf-8") as f:
                return f.read()

        if not HF_AVAILABLE:
            return self._get_fallback_language_corpus()

        print("Downloading books corpus...")
        try:
            # Try different book datasets
            datasets_to_try = [
                ("bookcorpus", None),
                ("bookcorpusopen", None),
                ("the_pile_books3", None),
            ]

            for dataset_name, subset in datasets_to_try:
                try:
                    if subset:
                        dataset = load_dataset(
                            dataset_name, subset, split="train", streaming=True
                        )
                    else:
                        dataset = load_dataset(
                            dataset_name, split="train", streaming=True
                        )

                    corpus_text = ""
                    book_count = 0

                    for book in dataset:
                        text = book.get("text", "")
                        if len(text) >= self.config.min_text_length:
                            corpus_text += text + "\n\n"
                            book_count += 1

                            if len(corpus_text) >= self.config.language_corpus_size:
                                break

                            if book_count % 10 == 0:
                                print(
                                    f"Processed {book_count} books, {len(corpus_text):,} characters"
                                )

                    if len(corpus_text) > 1000:  # Got some data
                        # Cache the corpus
                        with open(cache_file, "w", encoding="utf-8") as f:
                            f.write(corpus_text)

                        print(
                            f"Books corpus loaded from {dataset_name}: {
                                len(corpus_text):,} characters"
                        )
                        return corpus_text

                except Exception as dataset_error:
                    print(f"Failed to load {dataset_name}: {dataset_error}")
                    continue

            # If all datasets failed
            return self._get_fallback_language_corpus()

        except Exception as e:
            print(f"Error loading books corpus: {e}")
            return self._get_fallback_language_corpus()

    def _get_fallback_language_corpus(self) -> str:
        """Fallback corpus when HuggingFace is unavailable."""
        base_texts = [
            """
            The art of language emerges through practice and evolution. Each word carries meaning, each sentence builds understanding.
            Language learning requires exposure to diverse texts, from literature to science, from poetry to technical documentation.
            Through repetition and variation, patterns emerge that enable communication and expression.

            Constitutional AI agents develop language capabilities through evolutionary pressure, selecting for coherence, creativity, and precision.
            Each agent's traits influence their learning style - some favor technical precision, others creative expression.

            The relationship between mind and language reflects deeper patterns of intelligence, whether biological or artificial.
            Through training and selection, simple networks learn complex linguistic behaviors.
            """,
            """
            Words flow like water, finding their path through the landscape of meaning. Grammar provides structure while creativity provides life.
            The dance between rules and exceptions creates the rich tapestry of human communication.

            In the realm of artificial minds, language becomes a bridge between internal states and external expression.
            Each generation builds upon the last, refining capability through constitutional evolution.

            Text generation requires understanding not just words but contexts, implications, and subtle meanings.
            The best language models capture not just patterns but the essence of communication itself.
            """,
            """
            Through diverse training data, agents learn to navigate the complexity of human language with increasing sophistication.
            From simple character prediction to complex reasoning, the journey of language learning mirrors the evolution of intelligence.

            Neural networks evolve through generations, each iteration improving upon the last through careful selection and breeding.
            The constitutional framework provides the genetic foundation for this evolution, encoding traits that shape learning behavior.

            Machine learning combines statistical pattern recognition with evolutionary principles to create intelligent systems.
            Each training example contributes to the network's understanding, building knowledge incrementally over time.
            """,
            """
            Programming languages provide the syntax and semantics for instructing computers to perform complex tasks.
            From assembly language to high-level abstractions, each generation of programming tools builds upon previous innovations.

            Software engineering principles guide the development of robust, maintainable, and efficient computer programs.
            Design patterns capture proven solutions to common programming problems, enabling reuse and consistency.

            Algorithms form the foundation of computer science, providing systematic methods for solving computational problems.
            Data structures organize information in ways that enable efficient access, manipulation, and storage.
            """,
            """
            Scientific inquiry follows systematic methods to understand natural phenomena and develop technological innovations.
            The scientific method involves observation, hypothesis formation, experimentation, and theory development.

            Mathematics provides the language of science, offering precise tools for modeling physical reality.
            Statistical analysis enables researchers to draw meaningful conclusions from experimental data.

            Technology advances through iterative improvement, with each innovation building upon previous discoveries.
            Interdisciplinary collaboration accelerates progress by combining insights from diverse fields of study.
            """,
        ]

        # Create substantial corpus by combining and repeating base texts
        substantial_corpus = ""
        for i in range(200):  # Create 200 variations for more substantial training
            text_index = i % len(base_texts)
            variation = base_texts[text_index]

            # Add some variation by inserting numbers and technical terms
            if i % 3 == 0:
                variation = variation.replace("language", f"language {i}")
            if i % 5 == 0:
                variation = variation.replace("evolution", f"evolution {i}")
            if i % 7 == 0:
                variation = variation.replace("training", f"training {i}")
            if i % 11 == 0:
                variation = variation.replace("learning", f"learning {i}")
            if i % 13 == 0:
                variation = variation.replace("intelligence", f"intelligence {i}")

            substantial_corpus += variation + "\n\n"

        return substantial_corpus

    def get_mixed_corpus(self) -> str:
        """Get a mixed corpus combining multiple sources."""
        print("Loading mixed language corpus...")

        corpora = []

        # Try to load from multiple sources
        wiki_corpus = self.load_wikipedia_corpus()
        if len(wiki_corpus) > 10000:
            corpora.append(wiki_corpus[: self.config.language_corpus_size // 3])

        try:
            web_corpus = self.load_openwebtext_corpus()
            if len(web_corpus) > 10000:
                corpora.append(web_corpus[: self.config.language_corpus_size // 3])
        except BaseException:
            pass

        try:
            books_corpus = self.load_books_corpus()
            if len(books_corpus) > 10000:
                corpora.append(books_corpus[: self.config.language_corpus_size // 3])
        except BaseException:
            pass

        # Combine all corpora
        if corpora:
            mixed_corpus = "\n\n".join(corpora)
            print(
                f"Mixed corpus created: {
                    len(mixed_corpus):,} characters from {
                    len(corpora)} sources"
            )
            return mixed_corpus
        else:
            return self._get_fallback_language_corpus()


class CodingCorpusLoader:
    """Load substantial coding training corpora."""

    def __init__(self, config: CorpusConfig = None):
        self.config = config or CorpusConfig()
        os.makedirs(self.config.cache_dir, exist_ok=True)

    def load_python_code_corpus(self) -> List[Dict[str, str]]:
        """Load Python code examples from modern code datasets."""
        cache_file = os.path.join(self.config.cache_dir, "python_code.json")

        if os.path.exists(cache_file):
            print(f"Loading cached Python code corpus from {cache_file}")
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)

        if not HF_AVAILABLE:
            return self._get_fallback_coding_corpus()

        print("Downloading Python code corpus...")

        # Try modern working datasets
        datasets_to_try = [
            ("bigcode/the-stack", "python"),
            ("codeparrot/github-code", "python"),
            ("huggingface/CodeBERTa-language-id", None),
        ]

        for dataset_name, subset in datasets_to_try:
            try:
                print(f"Trying {dataset_name}...")

                if subset:
                    dataset = load_dataset(
                        dataset_name,
                        data_files=f"data/{subset}/*",
                        split="train",
                        streaming=True,
                        trust_remote_code=True,
                    )
                else:
                    dataset = load_dataset(
                        dataset_name,
                        split="train",
                        streaming=True,
                        trust_remote_code=True,
                    )

                code_samples = []
                sample_count = 0
                total_size = 0

                for sample in dataset:
                    # Handle different dataset formats
                    code = ""
                    if "content" in sample:
                        code = sample["content"]
                    elif "code" in sample:
                        code = sample["code"]
                    elif "text" in sample:
                        code = sample["text"]

                    if len(code) >= 50 and len(code) <= 2000:  # Reasonable code length
                        # Extract function name or create generic input
                        input_desc = "function"
                        if "def " in code:
                            try:
                                func_line = [
                                    line
                                    for line in code.split("\n")
                                    if line.strip().startswith("def ")
                                ][0]
                                func_name = (
                                    func_line.split("(")[0].replace("def ", "").strip()
                                )
                                input_desc = f"function {func_name}"
                            except BaseException:
                                input_desc = "python function"

                        code_sample = {
                            "input": input_desc,
                            "output": code,
                            "language": "python",
                        }

                        code_samples.append(code_sample)
                        total_size += len(code)
                        sample_count += 1

                        if total_size >= self.config.coding_corpus_size:
                            break

                        if sample_count % 100 == 0:
                            print(
                                f"Collected {sample_count} Python functions, {
                                    total_size:,} characters"
                            )

                if len(code_samples) > 10:  # Got some data
                    # Cache the corpus
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(code_samples, f)

                    print(
                        f"Python code corpus loaded from {dataset_name}: {sample_count} functions, {
                            total_size:,} characters"
                    )
                    return code_samples

            except Exception as e:
                print(f"Failed to load {dataset_name}: {e}")
                continue

        # If all failed, use fallback
        print("All code datasets failed, using fallback")
        return self._get_fallback_coding_corpus()

    def load_multi_language_code_corpus(self) -> List[Dict[str, str]]:
        """Load code examples from multiple programming languages."""
        cache_file = os.path.join(self.config.cache_dir, "multi_code.json")

        if os.path.exists(cache_file):
            print(f"Loading cached multi-language code corpus from {cache_file}")
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)

        if not HF_AVAILABLE:
            return self._get_fallback_coding_corpus()

        print("Downloading multi-language code corpus...")
        languages = ["python", "javascript", "java", "go", "php", "ruby"]
        all_samples = []

        for lang in languages:
            try:
                dataset = load_dataset(
                    "code_search_net", lang, split="train", streaming=True
                )
                lang_samples = 0

                for sample in dataset:
                    if lang_samples >= 500:  # Limit per language
                        break

                    code = sample.get("func_code_string", "")
                    docstring = sample.get("func_documentation_string", "")

                    if len(code) >= 30 and len(code) <= 1500:
                        code_sample = {
                            "input": (
                                docstring[:80] if docstring else f"{lang} function"
                            ),
                            "output": code,
                            "language": lang,
                        }

                        all_samples.append(code_sample)
                        lang_samples += 1

                print(f"Collected {lang_samples} {lang} samples")

            except Exception as e:
                print(f"Error loading {lang}: {e}")
                continue

        if all_samples:
            # Cache the corpus
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(all_samples, f)

            total_size = sum(len(s["output"]) for s in all_samples)
            print(
                f"Multi-language code corpus: {len(all_samples)} samples, {total_size:,} characters"
            )
            return all_samples
        else:
            return self._get_fallback_coding_corpus()

    def _get_fallback_coding_corpus(self) -> List[Dict[str, str]]:
        """Fallback coding corpus when datasets are unavailable."""
        return [
            {
                "input": "hello function",
                "output": "def hello():\n    print('Hello, World!')\n    return True",
                "language": "python",
            },
            {
                "input": "add numbers",
                "output": "def add(a, b):\n    result = a + b\n    return result",
                "language": "python",
            },
            {
                "input": "check even",
                "output": "def is_even(n):\n    return n % 2 == 0",
                "language": "python",
            },
            {
                "input": "factorial",
                "output": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
                "language": "python",
            },
            {
                "input": "sort list",
                "output": "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr",
                "language": "python",
            },
            {
                "input": "find max",
                "output": "def find_max(numbers):\n    if not numbers:\n        return None\n    max_val = numbers[0]\n    for num in numbers[1:]:\n        if num > max_val:\n            max_val = num\n    return max_val",
                "language": "python",
            },
            {
                "input": "reverse string",
                "output": "def reverse_string(s):\n    return s[::-1]",
                "language": "python",
            },
            {
                "input": "fibonacci",
                "output": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
                "language": "python",
            },
            {
                "input": "prime check",
                "output": "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
                "language": "python",
            },
            {
                "input": "count words",
                "output": "def count_words(text):\n    words = text.split()\n    return len(words)",
                "language": "python",
            },
        ] * 50  # Repeat for more training data


# Factory functions for easy usage
def get_language_corpus(corpus_size: int = 10_000_000) -> str:
    """Get a substantial language training corpus."""
    config = CorpusConfig(language_corpus_size=corpus_size)
    loader = LanguageCorpusLoader(config)
    return loader.get_mixed_corpus()


def get_coding_corpus(corpus_size: int = 5_000_000) -> List[Dict[str, str]]:
    """Get a substantial coding training corpus."""
    config = CorpusConfig(coding_corpus_size=corpus_size)
    loader = CodingCorpusLoader(config)
    return loader.load_multi_language_code_corpus()


if __name__ == "__main__":
    # Test corpus loading
    print("Testing corpus loading...")

    # Test language corpus
    lang_corpus = get_language_corpus(1_000_000)  # 1MB test
    print(f"Language corpus: {len(lang_corpus):,} characters")
    print("Sample:", lang_corpus[:200])

    # Test coding corpus
    code_corpus = get_coding_corpus(500_000)  # 500KB test
    print(f"Coding corpus: {len(code_corpus)} samples")
    if code_corpus:
        print("Sample:", code_corpus[0])
