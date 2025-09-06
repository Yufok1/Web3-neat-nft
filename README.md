# Web3 NEAT NFT - AI Evolution on the Blockchain

A revolutionary Web3 system that breeds artificial intelligence neural networks using NEAT (NeuroEvolution of Augmenting Topologies) algorithm and mints the evolved AI systems as NFTs on the blockchain.

## 🧬 What is Web3 NEAT NFT?

This project combines cutting-edge artificial intelligence evolution with blockchain technology to create unique, collectible AI genomes. Each NFT represents a neural network that has been evolved through the NEAT algorithm, creating truly one-of-a-kind artificial intelligence systems that can be owned, traded, and utilized.

### Key Features

- **🤖 AI Evolution**: Uses NEAT algorithm to evolve neural network topologies
- **🌊 Blockchain Integration**: Mints evolved AI systems as NFTs on Ethereum
- **🗂️ IPFS Storage**: Stores genome data and metadata on decentralized storage
- **🔍 Species Tracking**: Maintains genetic lineage and species information
- **📊 Fitness Scoring**: Tracks performance metrics of each AI genome
- **🎨 Unique Visualization**: Each genome generates unique visual representations

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   NEAT Engine   │────│  Blockchain      │────│     IPFS        │
│                 │    │  Smart Contract  │    │   Metadata      │
│ • Population    │    │                  │    │   Storage       │
│ • Species       │    │ • NFT Minting    │    │                 │
│ • Evolution     │    │ • Lineage Track  │    │ • Genome Data   │
│ • Fitness Eval  │    │ • Ownership      │    │ • Visualizations│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Directory Structure

```
web3-neat-nft/
├── neat/                    # NEAT algorithm implementation
│   ├── core/               # Core NEAT classes
│   │   ├── genome.py       # Neural network genome representation
│   │   ├── population.py   # Population management
│   │   └── species.py      # Species classification
│   ├── genetics/           # Genetic operators
│   ├── neural_network/     # Network execution
│   ├── evolution/          # Evolution strategies
│   └── visualization/      # Genome visualization
├── contracts/              # Smart contracts
│   └── NEATNFT.sol        # Main NFT contract
├── web3/                   # Blockchain integration
│   └── blockchain_manager.py
├── scripts/                # Deployment and utility scripts
│   └── deploy.js          # Contract deployment
├── tests/                  # Test files
├── requirements.txt        # Python dependencies
└── package.json           # Node.js dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- IPFS node (optional, for metadata storage)
- Ethereum wallet with testnet ETH

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/web3-neat-nft.git
   cd web3-neat-nft
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Node.js dependencies**
   ```bash
   npm install
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Environment Configuration

Create a `.env` file with the following variables:

```env
# Blockchain Configuration
PROVIDER_URL=https://goerli.infura.io/v3/your-project-id
PRIVATE_KEY=your-private-key-here
CONTRACT_ADDRESS=deployed-contract-address

# IPFS Configuration  
IPFS_URL=/ip4/127.0.0.1/tcp/5001

# NEAT Algorithm Parameters
POPULATION_SIZE=150
MUTATION_RATE=0.1
COMPATIBILITY_THRESHOLD=3.0
```

## 💡 Usage Examples

### Basic AI Evolution

```python
from neat import Population, NEATAlgorithm
from web3.blockchain_manager import BlockchainManager

# Initialize NEAT population
population = Population(size=150, input_size=4, output_size=2)

# Define fitness function for your task
def fitness_function(genome):
    # Your AI task evaluation logic here
    return calculated_fitness_score

# Evolve for multiple generations
for generation in range(100):
    population.evaluate_fitness(fitness_function)
    population.evolve_generation()
    
    # Get best genome from this generation
    champion = population.get_best_genome()
    print(f"Generation {generation}: Best fitness = {champion.fitness}")
```

### Minting AI NFTs

```python
from web3.blockchain_manager import BlockchainManager

# Initialize blockchain manager
blockchain = BlockchainManager(
    provider_url=os.getenv('PROVIDER_URL'),
    private_key=os.getenv('PRIVATE_KEY'),
    contract_address=os.getenv('CONTRACT_ADDRESS')
)

# Mint champion genome as NFT
champion = population.get_best_genome()
result = blockchain.mint_genome_nft(
    genome=champion,
    generation=generation,
    recipient_address="0x..." # Your wallet address
)

if result:
    token_id, tx_hash = result
    print(f"NFT minted! Token ID: {token_id}, TX: {tx_hash}")
```

### Smart Contract Deployment

```bash
# Compile contracts
npm run compile

# Deploy to testnet
npx hardhat run scripts/deploy.js --network goerli

# Verify contract (optional)
npx hardhat verify --network goerli DEPLOYED_CONTRACT_ADDRESS
```

## 🧪 NEAT Algorithm Details

### Genome Structure

Each genome represents a neural network with:
- **Nodes**: Input, hidden, and output neurons
- **Connections**: Weighted links between nodes  
- **Topology**: Dynamic network structure
- **Innovation Numbers**: Historical tracking for crossover

### Evolution Process

1. **Initialization**: Create random population of simple networks
2. **Evaluation**: Test each genome on the fitness function
3. **Speciation**: Group similar genomes into species
4. **Selection**: Choose parents based on adjusted fitness
5. **Reproduction**: Create offspring through crossover and mutation
6. **Mutation**: Add nodes, add connections, modify weights

### NFT Integration

High-performing genomes are minted as NFTs with:
- **Unique Hash**: SHA-256 of genome structure
- **Metadata**: Fitness, generation, species information
- **Visualization**: Auto-generated network diagram
- **Lineage**: Parent-child relationships on-chain

## 🎨 NFT Metadata Structure

Each AI genome NFT includes rich metadata:

```json
{
  "name": "NEAT AI Genome #a1b2c3d4",
  "description": "An evolved AI neural network from generation 42",
  "image": "ipfs://QmGenomeVisualization...",
  "attributes": [
    {"trait_type": "Generation", "value": 42},
    {"trait_type": "Fitness Score", "value": 0.95},
    {"trait_type": "Species ID", "value": 3},
    {"trait_type": "Node Count", "value": 12},
    {"trait_type": "Connection Count", "value": 18}
  ],
  "genome_data": { /* Full genome structure */ },
  "genome_hash": "a1b2c3d4e5f6...",
  "created_at": 1694073600
}
```

## 🔧 Configuration & Parameters

### NEAT Parameters

- **Population Size**: Number of genomes per generation
- **Compatibility Threshold**: Species boundary distance
- **Mutation Rates**: Probabilities for different mutations
- **Survival Rate**: Percentage of each species that survives

### Blockchain Parameters

- **Gas Limit**: Maximum gas per transaction
- **Gas Price**: Wei per gas unit
- **Confirmation Blocks**: Blocks to wait for finality

## 🧪 Testing

Run the test suite:

```bash
# Python tests
pytest tests/

# Smart contract tests  
npm test

# Integration tests
python -m pytest tests/integration/
```

## 📈 Performance & Scaling

### Optimization Strategies

- **Parallel Fitness Evaluation**: Utilize multiple CPU cores
- **Batch NFT Minting**: Group multiple genomes per transaction
- **IPFS Pinning**: Ensure metadata persistence
- **Gas Optimization**: Efficient smart contract operations

### Scaling Considerations

- **Layer 2 Solutions**: Deploy on Polygon, Arbitrum, or Optimism
- **Sidechains**: Use dedicated blockchain for AI evolution
- **Hybrid Approach**: Evolution off-chain, results on-chain

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Install dependencies
4. Run tests to ensure everything works
5. Make your changes
6. Add tests for new functionality
7. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🔗 Links & Resources

- **Documentation**: [Full docs](https://web3-neat-nft.readthedocs.io)
- **Discord**: [Join our community](https://discord.gg/web3-neat-nft)
- **Twitter**: [@Web3NEATNFT](https://twitter.com/Web3NEATNFT)
- **NEAT Paper**: [Original Research](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)

## 🚨 Disclaimer

This is experimental software. Use at your own risk. Always test thoroughly on testnets before mainnet deployment.

---

**Built with ❤️ for the intersection of AI and Web3**
