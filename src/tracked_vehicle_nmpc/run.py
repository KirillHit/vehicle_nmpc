"""Entrypoint for the tracked vehicle NMPC application."""

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from tracked_vehicle_nmpc.utils import Runner
from tracked_vehicle_nmpc.utils.config import BaseConfig, register_configs

register_configs()

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run the application using the provided Hydra configuration."""
    cfg_obj = OmegaConf.to_object(cfg)
    if not isinstance(cfg_obj, BaseConfig):
        log.error("Invalid configuration type: %s.", type(cfg_obj).__name__)
        return

    runner = Runner()
    try:
        runner.run(cfg_obj)
    except Exception:
        log.exception("Experiment failed!")


if __name__ == "__main__":
    main()
