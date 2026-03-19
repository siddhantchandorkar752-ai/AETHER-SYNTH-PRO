import torch
from pathlib import Path
from loguru import logger

def save_checkpoint(state, filename="checkpoint.pth"):
    """Saves model state for future recovery."""
    path = Path("output") / filename
    torch.save(state, path)
    logger.info(f"💾 Checkpoint saved at: {path}")

def load_checkpoint(filename, generator, discriminator):
    """Loads model state to resume training."""
    path = Path("output") / filename
    if path.exists():
        checkpoint = torch.load(path)
        generator.load_state_dict(checkpoint['gen_state_dict'])
        discriminator.load_state_dict(checkpoint['disc_state_dict'])
        logger.success(f"📂 Checkpoint loaded from: {path}")
        return checkpoint['epoch']
    return 0
