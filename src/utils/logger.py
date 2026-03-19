import sys
from loguru import logger
from pathlib import Path

def setup_logger():
    """
    Enterprise-grade logging setup.
    Color-coded console output + Compressed file logs.
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Remove default handler
    logger.remove()

    # Console Handler (High Visibility)
    logger.add(
        sys.stderr,
        format="<white>{time:HH:mm:ss}</white> | <level>{level: <8}</level> | <cyan>{module}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO"
    )

    # File Handler (Permanent Audit Trail)
    logger.add(
        "logs/aether_core.log",
        rotation="10 MB",
        retention="10 days",
        level="DEBUG",
        compression="zip"
    )
    
    logger.info("🚀 AETHER-SYNTH: Logging Engine Active.")

# Initialize immediately on import
setup_logger()
