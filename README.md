## System Overview

A platform for evolving AI agents using a diploid genome system, neural evolution (NEAT), and blockchain-based NFT integration. Agents develop skills in language and coding through evolutionary processes and can be bred across generations to combine traits.

---

## Core Components

### Constitutional AI
- **Diploid Genome System**: Agents have maternal and paternal alleles following Mendelian genetics.
- **Stabilization Types**: Six behaviors (Static, Progressive, Oscillatory, Degenerative, Chaotic, Multi-attractor) define agent personality and trait evolution.
- **Trait Resolution**: Uses recursive fixed-point processes for emergent trait computation.
- **AI Traits**: 15 traits covering cognitive (Perception, Working Memory, Expertise), learning (Learning Rate, Transfer Learning, Meta-Learning), behavioral (Attention Span, Risk Tolerance, Innovation Drive), performance (Processing Speed, Stability, Energy Efficiency), and social (Social Drive, Communication Style, Curiosity) aspects.
- **Identity System**: Cryptographically verified agent authenticity using SHA-256 hashing.

### Language Corpus System
- **Data Sources**: Integrates HuggingFace datasets (WikiText-103, OpenWebText, BookCorpus, C4).
- **Corpus Size**: Scalable from 50KB (testing) to 10MB+ (production).
- **Caching**: Downloads data once, stores locally for reuse.
- **Fallback**: Automatically switches between data sources if one fails.

### CPU-Optimized Training
- **NEAT-Optimized**: CPU training specifically designed for NeuroEvolution of Augmenting Topologies.
- **Flexible Architecture**: Each neural network can have different topologies (not possible with GPU batching).
- **Sequential Processing**: Evolutionary algorithms work best with CPU's flexible processing.
- **Memory Efficient**: No GPU memory transfer overhead for diverse network structures.

### Logic Test Battery
- **Test Levels**: Six levels of increasing difficulty:
  - Level 1: XOR (4 patterns)
  - Level 2: AND/OR Gates (8 patterns)
  - Level 3: NAND Gate (4 patterns)
  - Level 4: 3-Input XOR (8 patterns)
  - Level 5: Majority Gate (8 patterns)
  - Level 6: 4-Input Parity (16 patterns)
- **Purpose**: Benchmarks agent learning, validates learning vs. memorization, and analyzes trait impact on performance.
- **Processing**: CPU-optimized sequential processing for diverse network topologies.

### Mathematical Properties
- **Fixed Points**: Trait convergence guaranteed via Kleene fixed points.
- **Monotone Mapping**: Higher trait values do not decrease neural parameters.
- **Reproduction**: Deterministic, producing identical agents for identical inputs.
- **Hashing**: SHA-256 for agent authenticity verification.

### Visual Identity
- **Color Generation**: HSV/RGB colors computed from constitutional traits.
- **NFT Artwork**: Unique visual signatures for each agent.
- **Descriptions**: Human-readable color-based personality indicators.

---

## System Architecture

```
Constitutional AI → NEAT Evolution → Blockchain NFTs
- Diploid Genomes         - Neural Networks        - ERC721 Tokens
- 6 Stabilization Types   - Trait-Configured       - Visual Identity
- Trait Resolution        - Learning Capable       - Provenance Chain
- Identity Creation       - Problem Solving        - Immutable Storage
```

### Directory Structure

```
web3-neat-nft/
├── constitutional_ai/
│   ├── genome.py                  # Diploid genome with stabilization
│   ├── traits.py                  # 15 AI traits
│   ├── emergent_constitution.py   # Fixed-point trait resolution
│   ├── breeder.py                 # Mendelian breeding
│   ├── identity.py                # Agent identity creation
│   ├── neat_mapper.py             # Trait-to-NEAT mapping
│   ├── neat_integration.py        # NEAT-python integration
│   ├── color_mapping_simple.py    # Visual DNA generation
│   ├── persistence.py             # Agent save/load
│   ├── corpus_loader.py           # Language corpus integration
│   └── training/                  # CPU-optimized training modules
│       ├── base_capability.py     # Capability framework
│       ├── language_evolution.py  # Language learning
│       └── coding_evolution.py    # Coding skill development
├── agents/                        # Trained agent storage
├── logic_tests.py                 # Logic test battery
├── agent_browser.py               # CLI for agent management
├── quick_test.py                  # System verification
├── contracts/                     # NFT smart contracts
├── web3/                         # Blockchain integration
├── tests/                         # Test suite
└── constitutional_neat/          # Legacy directory
```

---

## Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- Web3 stack (Hardhat, Ethers.js)
- `neat-python` library
- CPU-optimized for NEAT evolution

### Steps

```bash
git clone https://github.com/yourusername/web3-neat-nft.git
cd web3-neat-nft
pip install -r requirements.txt
npm install
```

---

## Usage

### Agent Browser CLI

```bash
# Verify system
python quick_test.py

# List agents with traits
python agent_browser.py list

# Show agent details
python agent_browser.py show 9488b0a1

# Train agent
python agent_browser.py train 9488b0a1 language --generations 5
python agent_browser.py train 9488b0a1 coding --generations 3

# Test agent
python agent_browser.py test 9488b0a1 "Hello world"
python agent_browser.py test-coding 9488b0a1 "def hello"

# Breed agents
python agent_browser.py breed 9488b0a1 725af723 --count 2 --generations 3

# Thoroughbred breeding
python agent_browser.py breed 2212c880 31fdce95 --count 5 --generations 8

# Tournament breeding
python agent_browser.py progressive-breed 2212c880 31fdce95 --rounds 3 --offspring-per-round 4

# Chat with agent
python agent_browser.py chat 9488b0a1

# Run logic tests
python logic_tests.py --all

# Show top agents
python top_agents.py
```

### Create and Train Agent

```python
from constitutional_ai import create_random_genome, COMPLETE_TRAIT_DEFINITIONS, create_agent_identity
from constitutional_ai.training.language_evolution import train_agent_language_capability
from constitutional_ai.persistence import save_training_result

# Create genome
genome = create_random_genome(COMPLETE_TRAIT_DEFINITIONS, seed=42)

# Create identity
identity = create_agent_identity(genome, seed_closure=123)

# Train in language
result = train_agent_language_capability(identity, generations=5)

# Save agent
result['identity_bundle'] = identity
agent_id = save_training_result(result, 'language')
```

---

## System Status (January 2025)

- **Agents**: 8+ trained agents available.
- **Breeding**: Fully functional with trait inheritance.
- **Top Agents** (example):
  - Agent 2212c880: AttentionSpan=8.07, Perception=6.67, InnovationDrive=2.95
  - Agent 31fdce95: AttentionSpan=7.44, Perception=5.99, Fitness=0.896
  - Agent f7c98a03: High-fitness thoroughbred offspring
- **Command**: `python agent_browser.py breed 2212c880 31fdce95 --count 3 --generations 6`

---

## Multi-Capability Training

### Language Training

```python
from constitutional_ai.corpus_loader import get_language_corpus
from constitutional_ai.training.language_evolution import train_agent_language_capability
from constitutional_ai.persistence import load_agent

# Load 1MB Wikipedia text
corpus = get_language_corpus(1_000_000)

# Train agent
result = train_agent_language_capability(agent_identity, generations=10)

# Test agent
record = load_agent(result['agent_id'])
```

### Coding Training

```python
from constitutional_ai.training.coding_evolution import train_agent_coding_capability

# Train agent
result = train_agent_coding_capability(agent_identity, generations=8)
```

### Cross-Capability Breeding

```python
from constitutional_ai.persistence import load_agent

# Load specialized agents
language_expert = load_agent("9488b0a1")
coding_expert = load_agent("725af723")

# Breed
# Command: python agent_browser.py breed 9488b0a1 725af723
```

---

## Logic Test Results (Example)

```
Agent 31fdce95 (High LearningRate):
├── Level 1 (XOR): PASS (4/4)
├── Level 2 (AND/OR): PASS (8/8)
├── Level 3 (NAND): PASS (4/4)
├── Level 4 (3-XOR): PASS (8/8)

Agent 53bac160 (High Stability):
├── Level 1 (XOR): PASS (4/4)
├── Level 2 (AND/OR): PASS (8/8)
├── Level 3 (NAND): FAIL (2/4)
```

---

## NEAT Integration

- **Population Size**: 50–2000 agents, based on traits.
- **Mutation Rates**: Controlled by Innovation Drive and Risk Tolerance.
- **Network Architecture**: Derived from cognitive/performance traits.
- **Learning Parameters**: Set by Learning Rate and Meta-Learning.
- **Selection Pressure**: Influenced by Stability and Social traits.

---

## Visual DNA

```python
from constitutional_ai import traits_to_hsv_simple, traits_to_simple_color, get_simple_color_description

traits = identity.constitution_result.constitution
hsv_color = traits_to_hsv_simple(traits)
hex_color = traits_to_simple_color(traits)  # e.g., "#A040F4"
description = get_simple_color_description(hex_color)  # e.g., "Blue-dominant"
```

---

## Testing

### Test Commands

```bash
# Constitutional system
python -c "from constitutional_ai import *; genome = create_random_genome(COMPLETE_TRAIT_DEFINITIONS, seed=42); identity = create_agent_identity(genome); print(f'Constitutional system: {identity.id_hash[:12]}')"

# Breeding system
python -c "from constitutional_ai import ConstitutionalBreeder, create_random_genome, COMPLETE_TRAIT_DEFINITIONS; breeder = ConstitutionalBreeder(); p1 = create_random_genome(COMPLETE_TRAIT_DEFINITIONS, seed=1); p2 = create_random_genome(COMPLETE_TRAIT_DEFINITIONS, seed=2); result = breeder.breed_agents(p1, p2, seed=3); print(f'Breeding system: {result.offspring.compute_genome_hash()[:12]}')"

# NEAT integration
python -c "from constitutional_ai.neat_integration import evolve_constitutional_agent; from constitutional_ai import create_random_genome, COMPLETE_TRAIT_DEFINITIONS; import neat; def simple_fitness(genomes, config): [setattr(g, 'fitness', 1.0) for _, g in genomes]; genome = create_random_genome(COMPLETE_TRAIT_DEFINITIONS, seed=42); result = evolve_constitutional_agent(genome, simple_fitness, generations=2); print(f'NEAT integration: {result['final_fitness']}')"
```

---

## Performance Metrics

- **Capabilities**: Language and coding skill evolution, cross-generation breeding, agent persistence, population scaling (50–2000 agents), trait diversity, identity verification, visual DNA.
- **Fitness Improvement**: Offspring fitness from 0.037 to 0.043.
- **Scalability**:
  - Generation time: ~2 seconds for 1500+ agents.
  - Trait resolution: Converges in <50 iterations.
  - Memory: Efficient genome storage.
- **Neural Evolution**: Unlimited complexity via NEAT-python.

---

## Deployment

```bash
# Compile contracts
npm run compile

# Deploy to testnet
npm run deploy

# Verify contract
npx hardhat verify --network goerli DEPLOYED_CONTRACT_ADDRESS
```

---

## Applications

- AI agent NFTs
- Evolving game NPCs
- AI research tools
- Educational platform for genetics/AI/blockchain
- Art generation with unique visuals

---

## License

MIT License (see LICENSE file).

---

## Resources

- NEAT Algorithm: [http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
- Constitutional AI: [docs/](docs/)
- Web3 Integration: [contracts/README.md](contracts/README.md)

---
