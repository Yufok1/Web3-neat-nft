/**
 * Deployment script for NEAT NFT smart contract
 * 
 * This script deploys the NEATNFT contract to the specified network
 * and saves the deployment information for later use.
 */

const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  console.log("Starting NEAT NFT contract deployment...");

  // Get the contract factory
  const NEATNFT = await hre.ethers.getContractFactory("NEATNFT");

  // Deploy the contract
  console.log("Deploying NEATNFT contract...");
  const neatNFT = await NEATNFT.deploy();

  // Wait for deployment to complete
  await neatNFT.deployed();

  console.log("NEATNFT deployed to:", neatNFT.address);

  // Save deployment information
  const deploymentInfo = {
    network: hre.network.name,
    contractAddress: neatNFT.address,
    deployer: (await hre.ethers.getSigners())[0].address,
    deploymentTime: new Date().toISOString(),
    blockNumber: await hre.ethers.provider.getBlockNumber(),
    gasUsed: (await neatNFT.deployTransaction.wait()).gasUsed.toString()
  };

  // Create deployments directory if it doesn't exist
  const deploymentsDir = path.join(__dirname, "..", "deployments");
  if (!fs.existsSync(deploymentsDir)) {
    fs.mkdirSync(deploymentsDir, { recursive: true });
  }

  // Save deployment info to file
  const deploymentPath = path.join(deploymentsDir, `${hre.network.name}.json`);
  fs.writeFileSync(deploymentPath, JSON.stringify(deploymentInfo, null, 2));

  console.log("Deployment info saved to:", deploymentPath);

  // Verify contract on Etherscan (if not local network)
  if (hre.network.name !== "hardhat" && hre.network.name !== "localhost") {
    console.log("Waiting for block confirmations...");
    await neatNFT.deployTransaction.wait(5);

    try {
      await hre.run("verify:verify", {
        address: neatNFT.address,
        constructorArguments: [],
      });
      console.log("Contract verified on Etherscan");
    } catch (error) {
      console.log("Error verifying contract:", error);
    }
  }

  console.log("Deployment completed successfully!");
}

// Handle errors
main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
