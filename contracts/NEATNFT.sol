// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/Counters.sol";

/**
 * @title NEATNFT
 * @dev NFT contract for minting evolved AI neural networks as NFTs
 * Each NFT represents a unique neural network genome evolved through NEAT algorithm
 */
contract NEATNFT is ERC721, ERC721URIStorage, Ownable {
    using Counters for Counters.Counter;
    
    Counters.Counter private _tokenIdCounter;
    
    // Mapping from token ID to genome data hash
    mapping(uint256 => bytes32) public genomeHashes;
    
    // Mapping from token ID to fitness score
    mapping(uint256 => uint256) public fitnessScores;
    
    // Mapping from token ID to generation number
    mapping(uint256 => uint256) public generations;
    
    // Mapping from token ID to species ID
    mapping(uint256 => uint256) public speciesIds;
    
    // Mapping to track if a genome hash has been minted
    mapping(bytes32 => bool) public genomeMinted;
    
    // Events
    event GenomeMinted(
        uint256 indexed tokenId,
        address indexed owner,
        bytes32 genomeHash,
        uint256 fitness,
        uint256 generation,
        uint256 speciesId
    );
    
    event GenomeEvolved(
        uint256 indexed parentId,
        uint256 indexed childId,
        bytes32 childGenomeHash
    );
    
    constructor() ERC721("NEAT AI Genome", "NEAT") {}
    
    /**
     * @dev Mint a new NFT representing an evolved AI genome
     * @param to Address to mint the NFT to
     * @param genomeHash Hash of the genome data stored on IPFS
     * @param fitness Fitness score of the genome
     * @param generation Generation number when genome was created
     * @param speciesId Species ID the genome belongs to
     * @param tokenURI URI pointing to NFT metadata on IPFS
     */
    function mintGenome(
        address to,
        bytes32 genomeHash,
        uint256 fitness,
        uint256 generation,
        uint256 speciesId,
        string memory tokenURI
    ) public onlyOwner returns (uint256) {
        require(!genomeMinted[genomeHash], "Genome already minted");
        
        uint256 tokenId = _tokenIdCounter.current();
        _tokenIdCounter.increment();
        
        _safeMint(to, tokenId);
        _setTokenURI(tokenId, tokenURI);
        
        genomeHashes[tokenId] = genomeHash;
        fitnessScores[tokenId] = fitness;
        generations[tokenId] = generation;
        speciesIds[tokenId] = speciesId;
        genomeMinted[genomeHash] = true;
        
        emit GenomeMinted(tokenId, to, genomeHash, fitness, generation, speciesId);
        
        return tokenId;
    }
    
    /**
     * @dev Mint a child genome NFT from parent genome(s)
     * @param to Address to mint the NFT to
     * @param parentId Token ID of parent genome
     * @param genomeHash Hash of the child genome data
     * @param fitness Fitness score of the child genome
     * @param generation Generation number of the child
     * @param speciesId Species ID the child belongs to
     * @param tokenURI URI pointing to child NFT metadata
     */
    function mintChildGenome(
        address to,
        uint256 parentId,
        bytes32 genomeHash,
        uint256 fitness,
        uint256 generation,
        uint256 speciesId,
        string memory tokenURI
    ) public onlyOwner returns (uint256) {
        require(_exists(parentId), "Parent genome does not exist");
        require(!genomeMinted[genomeHash], "Child genome already minted");
        
        uint256 childId = mintGenome(to, genomeHash, fitness, generation, speciesId, tokenURI);
        
        emit GenomeEvolved(parentId, childId, genomeHash);
        
        return childId;
    }
    
    /**
     * @dev Get genome information for a token
     * @param tokenId Token ID to query
     * @return genomeHash Hash of the genome data
     * @return fitness Fitness score
     * @return generation Generation number
     * @return speciesId Species ID
     */
    function getGenomeInfo(uint256 tokenId) 
        public 
        view 
        returns (
            bytes32 genomeHash,
            uint256 fitness,
            uint256 generation,
            uint256 speciesId
        ) 
    {
        require(_exists(tokenId), "Token does not exist");
        
        return (
            genomeHashes[tokenId],
            fitnessScores[tokenId],
            generations[tokenId],
            speciesIds[tokenId]
        );
    }
    
    /**
     * @dev Check if a genome has been minted
     * @param genomeHash Hash of the genome to check
     * @return True if genome has been minted
     */
    function isGenomeMinted(bytes32 genomeHash) public view returns (bool) {
        return genomeMinted[genomeHash];
    }
    
    /**
     * @dev Get total number of minted genomes
     * @return Total number of tokens minted
     */
    function totalSupply() public view returns (uint256) {
        return _tokenIdCounter.current();
    }
    
    // Override required functions
    function _burn(uint256 tokenId) internal override(ERC721, ERC721URIStorage) {
        super._burn(tokenId);
    }
    
    function tokenURI(uint256 tokenId)
        public
        view
        override(ERC721, ERC721URIStorage)
        returns (string memory)
    {
        return super.tokenURI(tokenId);
    }
    
    function supportsInterface(bytes4 interfaceId)
        public
        view
        override(ERC721, ERC721URIStorage)
        returns (bool)
    {
        return super.supportsInterface(interfaceId);
    }
}
