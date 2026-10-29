"""
ROMP Main Entry Script
----------------------
This script serves as the primary execution point for the Rainy season Onset
Metrics Package (ROMP). It orchestrates the calculation of skill scores
or the generation of spatial performance maps.

Usage:
    momp-run
    or
    python -m momp.driver
"""

import logging
import sys
import traceback

# Local Package Imports
from momp.lib.loader import get_cfg, get_setting
from momp.lib.assertion import ROMPConfigError
from momp.app.bin_skill_score import skill_score_in_bins
from momp.app.spatial_far_mr_mae import spatial_far_mr_mae_map
from momp.lib.onset_config import onset_setup
from momp.utils.printing import print_momp_banner


log_dir = "logs"

# Configure logging for professional status updates
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler()                        # Prints to terminal
    ]
)

logger = logging.getLogger(__name__)


def run_momp(cfg=None, setting=None):
    """
    Executes the standard ROMP evaluation workflow.

    This function triggers the bin-based skill score calculations
    or generates spatial maps for False Alarm Rate (FAR),
    Miss Rate (MR), and Mean Absolute Error (MAE).

    Args:
        cfg: The loaded configuration dictionary/SimpleNameSpace object.
        setting: The environment and directory settings.
    """

    try:
        if cfg is None:
            cfg = get_cfg()
        if setting is None:
            setting = get_setting()
    except (ROMPConfigError, FileNotFoundError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    print_momp_banner(cfg)

    print("Onset settings:")
    for line in onset_setup(vars(cfg)).lines():
        print(f"  {line}")
    for line in getattr(cfg, "onset_lead_lines", ()):
        print(f"  {line}")

    logger.info("Starting ROMP Workflow...")

    try:
        # 1. Calculate and save Skill Scores in defined day bins
        skill_score_in_bins()

        # 2. Generate spatial metrics and maps
        spatial_far_mr_mae_map()

        logger.info("ROMP Workflow completed successfully!")

    except Exception as e:
        logger.error(f"ROMP failed during execution: {e}")
        traceback.print_exc()
        sys.exit(1)

# ------------------------------------------------------------------------------
# EXECUTION BLOCK
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    run_momp()
