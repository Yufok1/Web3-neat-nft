"""
Blockchain manager for Web3 NEAT NFT system.
Handles interactions with smart contracts and IPFS.
"""

import json
import hashlib
from typing import Dict, List, Optional, Tuple
from web3 import Web3
from eth_account import Account
import ipfshttpclient
from neat.core.genome import Genome


class BlockchainManager:
    """
    Manages blockchain interactions for the NEAT NFT system.

    Handles smart contract deployment, genome minting, and IPFS storage.
    """

    def __init__(
        self,
        provider_url: str,
        private_key: str,
        contract_address: Optional[str] = None,
        ipfs_url: str = "/ip4/127.0.0.1/tcp/5001",
    ):
        """
        Initialize blockchain manager.

        Args:
            provider_url: Web3 provider URL (e.g., Infura, Alchemy)
            private_key: Private key for transaction signing
            contract_address: Address of deployed NEAT NFT contract
            ipfs_url: IPFS node URL
        """
        self.w3 = Web3(Web3.HTTPProvider(provider_url))
        self.account = Account.from_key(private_key)
        self.contract_address = contract_address
        self.contract = None

        # Initialize IPFS client
        try:
            self.ipfs = ipfshttpclient.connect(ipfs_url)
        except Exception as e:
            print(f"Warning: Could not connect to IPFS: {e}")
            self.ipfs = None

        # Load contract ABI if contract address provided
        if contract_address:
            self._load_contract()

    def _load_contract(self):
        """Load the NEAT NFT contract."""
        # In production, this would load from a compiled artifacts file
        # For now, we'll define a minimal ABI
        contract_abi = [
            {
                "inputs": [
                    {"name": "to", "type": "address"},
                    {"name": "genomeHash", "type": "bytes32"},
                    {"name": "fitness", "type": "uint256"},
                    {"name": "generation", "type": "uint256"},
                    {"name": "speciesId", "type": "uint256"},
                    {"name": "tokenURI", "type": "string"},
                ],
                "name": "mintGenome",
                "outputs": [{"name": "", "type": "uint256"}],
                "type": "function",
            },
            {
                "inputs": [{"name": "tokenId", "type": "uint256"}],
                "name": "getGenomeInfo",
                "outputs": [
                    {"name": "genomeHash", "type": "bytes32"},
                    {"name": "fitness", "type": "uint256"},
                    {"name": "generation", "type": "uint256"},
                    {"name": "speciesId", "type": "uint256"},
                ],
                "type": "function",
            },
            {
                "inputs": [{"name": "genomeHash", "type": "bytes32"}],
                "name": "isGenomeMinted",
                "outputs": [{"name": "", "type": "bool"}],
                "type": "function",
            },
        ]

        self.contract = self.w3.eth.contract(
            address=self.contract_address, abi=contract_abi
        )

    def upload_to_ipfs(self, data: dict) -> Optional[str]:
        """
        Upload data to IPFS and return the hash.

        Args:
            data: Dictionary data to upload

        Returns:
            IPFS hash string or None if upload failed
        """
        if not self.ipfs:
            print("IPFS not available")
            return None

        try:
            result = self.ipfs.add_json(data)
            return result
        except Exception as e:
            print(f"Failed to upload to IPFS: {e}")
            return None

    def create_genome_metadata(self, genome: Genome, generation: int) -> dict:
        """
        Create NFT metadata for a genome.

        Args:
            genome: NEAT genome instance
            generation: Generation number

        Returns:
            NFT metadata dictionary
        """
        genome_data = genome.to_dict()

        # Calculate genome hash
        genome_hash = self._calculate_genome_hash(genome_data)

        # Create NFT metadata following OpenSea standard
        metadata = {
            "name": f"NEAT AI Genome #{genome_hash[:8]}",
            "description": f"An evolved artificial intelligence neural network from generation {generation}",
            "image": "https://via.placeholder.com/400x400/1a1a1a/ffffff?"
            "text=AI+Genome",  # Placeholder
            "attributes": [
                {"trait_type": "Generation", "value": generation},
                {"trait_type": "Fitness Score", "value": genome.fitness},
                {"trait_type": "Species ID", "value": genome.species_id},
                {"trait_type": "Node Count", "value": len(genome.nodes)},
                {"trait_type": "Connection Count", "value": len(genome.connections)},
                {"trait_type": "Input Size", "value": genome.input_size},
                {"trait_type": "Output Size", "value": genome.output_size},
            ],
            "genome_data": genome_data,
            "genome_hash": genome_hash,
            "created_at": int(self.w3.eth.get_block("latest")["timestamp"]),
        }

        return metadata

    def _calculate_genome_hash(self, genome_data: dict) -> str:
        """Calculate a unique hash for genome data."""
        # Create a deterministic string representation
        genome_str = json.dumps(genome_data, sort_keys=True)
        return hashlib.sha256(genome_str.encode()).hexdigest()

    def mint_genome_nft(
        self, genome: Genome, generation: int, recipient_address: str
    ) -> Optional[Tuple[int, str]]:
        """
        Mint an NFT for a genome.

        Args:
            genome: NEAT genome to mint
            generation: Generation number
            recipient_address: Address to receive the NFT

        Returns:
            Tuple of (token_id, transaction_hash) or None if failed
        """
        if not self.contract:
            print("Contract not loaded")
            return None

        try:
            # Create and upload metadata
            metadata = self.create_genome_metadata(genome, generation)
            ipfs_hash = self.upload_to_ipfs(metadata)

            if not ipfs_hash:
                print("Failed to upload metadata to IPFS")
                return None

            # Prepare contract call
            genome_hash_bytes = bytes.fromhex(metadata["genome_hash"])
            token_uri = f"ipfs://{ipfs_hash}"

            # Check if genome already minted
            if self.contract.functions.isGenomeMinted(genome_hash_bytes).call():
                print("Genome already minted")
                return None

            # Build transaction
            function_call = self.contract.functions.mintGenome(
                recipient_address,
                genome_hash_bytes,
                int(genome.fitness * 1000),  # Scale fitness to avoid decimals
                generation,
                genome.species_id,
                token_uri,
            )

            # Estimate gas
            gas_estimate = function_call.estimateGas({"from": self.account.address})

            # Build transaction
            transaction = function_call.buildTransaction(
                {
                    "from": self.account.address,
                    "gas": gas_estimate + 10000,  # Add buffer
                    "gasPrice": self.w3.eth.gas_price,
                    "nonce": self.w3.eth.get_transaction_count(self.account.address),
                }
            )

            # Sign and send transaction
            signed_txn = self.w3.eth.account.sign_transaction(
                transaction, self.account.privateKey
            )
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)

            # Wait for transaction receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            if receipt.status == 1:
                # Extract token ID from logs (simplified)
                token_id = len(
                    self.get_owned_tokens(recipient_address)
                )  # Approximation
                return (token_id, tx_hash.hex())
            else:
                print("Transaction failed")
                return None

        except Exception as e:
            print(f"Failed to mint NFT: {e}")
            return None

    def get_genome_info(self, token_id: int) -> Optional[dict]:
        """
        Get genome information from blockchain.

        Args:
            token_id: NFT token ID

        Returns:
            Dictionary with genome info or None if failed
        """
        if not self.contract:
            return None

        try:
            genome_hash, fitness, generation, species_id = (
                self.contract.functions.getGenomeInfo(token_id).call()
            )

            return {
                "genome_hash": genome_hash.hex(),
                "fitness": fitness / 1000.0,  # Unscale fitness
                "generation": generation,
                "species_id": species_id,
            }
        except Exception as e:
            print(f"Failed to get genome info: {e}")
            return None

    def get_owned_tokens(self, address: str) -> List[int]:
        """
        Get token IDs owned by an address.

        Args:
            address: Wallet address

        Returns:
            List of token IDs
        """
        # In a full implementation, this would query Transfer events
        # or use a more efficient method
        # For now, returning empty list as placeholder
        return []

    def is_genome_minted(self, genome: Genome) -> bool:
        """
        Check if a genome has already been minted.

        Args:
            genome: Genome to check

        Returns:
            True if already minted
        """
        if not self.contract:
            return False

        try:
            genome_data = genome.to_dict()
            genome_hash = self._calculate_genome_hash(genome_data)
            genome_hash_bytes = bytes.fromhex(genome_hash)

            return self.contract.functions.isGenomeMinted(genome_hash_bytes).call()
        except Exception as e:
            print(f"Failed to check if genome is minted: {e}")
            return False
