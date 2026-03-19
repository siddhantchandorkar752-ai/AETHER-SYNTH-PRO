import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from loguru import logger
import asyncio

from config import cfg
from src.utils.hardware_util import get_precision_device, log_vram_usage
from src.generator.dcgan_engine import Generator, Discriminator
from src.agents.critic_agent import CriticAgent

async def train_evolutionary_gan():
    device = get_precision_device()
    logger.info(f"🚀 Starting AETHER-SYNTH Training on {device}")

    # 1. Initialize Neural Networks
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    
    # 2. Initialize Cognitive Agent
    agent = CriticAgent()

    # 3. Optimizers & Loss Function
    criterion = nn.BCELoss()
    optimizerG = optim.Adam(netG.parameters(), lr=cfg.LEARNING_RATE, betas=(cfg.BETA1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=cfg.LEARNING_RATE, betas=(cfg.BETA1, 0.999))

    # 4. Fixed Noise for Visualization (to see evolution)
    fixed_noise = torch.randn(16, cfg.LATENT_DIM).to(device)

    logger.info("⚡ Entering Evolutionary Loop (Epoch 1 to 5 Simulation)...")

    for epoch in range(1, 6):  # Simulating 5 Epochs for the demo
        # --- SIMULATED TRAINING STEP ---
        # In a real scenario, you'd loop through a dataset here
        noise = torch.randn(cfg.BATCH_SIZE, cfg.LATENT_DIM).to(device)
        
        # Generate Fake Images
        fake = netG(noise)
        
        # Dummy Losses for demonstration (In real: calculate via Backprop)
        loss_g = 0.7 + (1.0 / epoch) 
        loss_d = 0.3 - (0.05 / epoch)

        # 5. AGENT AUDIT (The Elite Part)
        if epoch % 2 == 0:  # Audit every 2nd epoch to save API tokens
            logger.info(f"🔍 Epoch {epoch}: Requesting Agent Audit...")
            feedback = await agent.audit_generation(epoch, loss_g, loss_d)
            # Future: Logic to auto-adjust Learning Rate based on feedback
        
        # 6. SAVE RESULTS
        with torch.no_grad():
            gen_images = netG(fixed_noise)
            save_path = cfg.OUTPUT_DIR / f"synthetic_epoch_{epoch}.png"
            save_image(gen_images, save_path, normalize=True)
            logger.success(f"🖼️ Synthetic Data Saved: {save_path}")

        log_vram_usage(device)

    logger.success("🏆 AETHER-SYNTH: Training Cycle Complete. Portfolio Asset Ready.")

if __name__ == "__main__":
    try:
        asyncio.run(train_evolutionary_gan())
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
    except Exception as e:
        logger.critical(f"System Crash: {str(e)}")
