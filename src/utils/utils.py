import re
import warnings
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Tuple

from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

from src.utils import pylogger, rich_utils


def _get_num_devices(devices: Any) -> Optional[int]:
    if isinstance(devices, int):
        return devices
    if isinstance(devices, (list, tuple, ListConfig)):
        return len(devices)
    return None

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def normalize_trainer_devices(cfg: DictConfig) -> None:
    """Normalizes Hydra/CLI device overrides into Lightning-compatible values.

    Common CLI inputs in this project include:
    - `trainer.devices=0`   -> single GPU index 0
    - `trainer.devices=0,1` -> multi-GPU indices [0, 1]
    - `trainer.devices=[0]` -> already explicit

    Lightning interprets bare integer `0` as "use zero devices", which is invalid for CUDA.
    We rewrite index-like inputs into explicit GPU index lists.

    Additionally, if the final configuration resolves to a single GPU, we disable DDP-specific
    strategy settings to avoid unnecessary subprocess launch and terminal/progress-bar issues.
    """
    if not cfg.get("trainer") or "devices" not in cfg.trainer:
        return

    devices = cfg.trainer.devices
    accelerator = cfg.trainer.get("accelerator")

    if accelerator not in {"gpu", "cuda"}:
        return

    if isinstance(devices, int):
        if devices == 0:
            OmegaConf.update(cfg, "trainer.devices", [0], merge=False)
    elif isinstance(devices, ListConfig):
        OmegaConf.update(cfg, "trainer.devices", list(devices), merge=False)
    elif isinstance(devices, str):
        stripped = devices.strip()

        if stripped == "0":
            OmegaConf.update(cfg, "trainer.devices", [0], merge=False)
        else:
            bracket_match = re.fullmatch(r"\[(\d+(?:\s*,\s*\d+)*)\]", stripped)
            if bracket_match:
                parsed = [int(part.strip()) for part in bracket_match.group(1).split(",")]
                OmegaConf.update(cfg, "trainer.devices", parsed, merge=False)
            else:
                csv_match = re.fullmatch(r"\d+(?:\s*,\s*\d+)+", stripped)
                if csv_match:
                    parsed = [int(part.strip()) for part in stripped.split(",")]
                    OmegaConf.update(cfg, "trainer.devices", parsed, merge=False)

    resolved_devices = cfg.trainer.devices
    num_devices = _get_num_devices(resolved_devices)
    if num_devices == 1:
        with open_dict(cfg.trainer):
            if "strategy" in cfg.trainer:
                del cfg.trainer["strategy"]
        if "sync_batchnorm" in cfg.trainer:
            OmegaConf.update(cfg, "trainer.sync_batchnorm", False, merge=False)



def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value
