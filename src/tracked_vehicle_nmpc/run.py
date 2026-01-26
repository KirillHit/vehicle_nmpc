"""..."""

import logging

import hydra
from omegaconf import DictConfig

from tracked_vehicle_nmpc.scripts.test import main as test_main

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    test_main()


if __name__ == "__main__":
    main()
