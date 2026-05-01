import copy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import rootutils
import numpy as np
import torch
import torch.nn.functional as F
import lightning.pytorch as L
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from transformers import AutoModel, AutoTokenizer

import json
import os
import yaml

from rich import get_console
from rich.table import Table

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    normalize_trainer_devices,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)

THRESHOLD = 0.95
GUO_BATCH_SIZE = 32
SENTENCE_EMBEDDER = "sentence-transformers/all-mpnet-base-v2"
TMR_PREFIX = "TMR-"
EVT_PREFIX = "EVT-"


def save_metrics_yaml(path: str, metrics: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(metrics, f, sort_keys=False, allow_unicode=True)



def resolve_eval_save_dir(cfg: DictConfig) -> str:
    ckpt_path = cfg.ckpt_path
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.abspath(os.path.join(cfg.paths.root_dir, ckpt_path))

    ckpt_path = os.path.abspath(ckpt_path)
    ckpt_dir = ckpt_path if os.path.isdir(ckpt_path) else os.path.dirname(ckpt_path)
    return os.path.join(ckpt_dir, "eval")


def print_table(title, metrics):
    table = Table(title=title)
    table.add_column("Metrics", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    for key, value in metrics.items():
        table.add_row(key, str(value))
    console = get_console()
    console.print(table, justify="center")


def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]
    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = torch.norm(activation[:, first_dices] - activation[:, second_dices], p=2, dim=2)
    return dist.mean()


def _to_builtin(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _round_metrics(metrics, rounding=2):
    return {k: round(float(v), rounding) for k, v in metrics.items()}


class SentenceEmbedder:
    def __init__(self, model_name: str = SENTENCE_EMBEDDER, device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts, batch_size: int = 256):
        embeddings = []
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded_inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            encoded_inputs = {k: v.to(self.device) for k, v in encoded_inputs.items()}
            output = self.model(**encoded_inputs)
            attention_mask = encoded_inputs["attention_mask"]
            token_embeddings = output.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            embeddings.append(sentence_embeddings.cpu())
        return torch.cat(embeddings, dim=0).numpy()


def transpose(x):
    return x.T


def get_sim_matrix(x, y):
    x_logits = x / np.linalg.norm(x, axis=-1, keepdims=True)
    y_logits = y / np.linalg.norm(y, axis=-1, keepdims=True)
    return x_logits @ transpose(y_logits)


def break_ties_average(sorted_dists, gt_dists):
    locs = np.argwhere((sorted_dists - gt_dists) == 0)
    steps = np.diff(locs[:, 0])
    splits = np.nonzero(steps)[0] + 1
    splits = np.insert(splits, 0, 0)
    summed_cols = np.add.reduceat(locs[:, 1], splits)
    counts = np.diff(np.append(splits, locs.shape[0]))
    avg_cols = summed_cols / counts
    return avg_cols


def break_ties_optimistically(sorted_dists, gt_dists):
    rows, cols = np.where((sorted_dists - gt_dists) == 0)
    _, idx = np.unique(rows, return_index=True)
    cols = cols[idx]
    return cols


def cols2metrics(cols, num_queries, rounding=2):
    metrics = {}
    vals = [str(x).zfill(2) for x in [1, 2, 3, 5, 10]]
    for val in vals:
        metrics[f"R{val}"] = 100 * float(np.sum(cols < int(val))) / num_queries

    metrics["MedR"] = float(np.median(cols) + 1)

    if rounding is not None:
        for key in metrics:
            metrics[key] = round(metrics[key], rounding)
    return metrics


def contrastive_metrics(
    sims,
    text_selfsim=None,
    threshold=None,
    return_cols=False,
    rounding=2,
    break_ties="averaging",
):
    n, m = sims.shape
    assert n == m
    num_queries = n

    dists = -sims
    sorted_dists = np.sort(dists, axis=1)
    gt_dists = np.diag(dists)[:, None]

    if text_selfsim is not None and threshold is not None:
        real_threshold = 2 * threshold - 1
        idx = np.argwhere(text_selfsim > real_threshold)
        partition = np.unique(idx[:, 0], return_index=True)[1]
        gt_dists = np.minimum.reduceat(dists[tuple(idx.T)], partition)
        gt_dists = gt_dists[:, None]

    rows, cols = np.where((sorted_dists - gt_dists) == 0)

    if rows.size > num_queries:
        assert np.unique(rows).size == num_queries, "issue in metric evaluation"
        if break_ties == "optimistically":
            cols = break_ties_optimistically(sorted_dists, gt_dists)
        elif break_ties == "averaging":
            cols = break_ties_average(sorted_dists, gt_dists)

    assert cols.size == num_queries, f"expected ranks to match queries ({cols.size} vs {num_queries})"

    if return_cols:
        return cols2metrics(cols, num_queries, rounding=rounding), cols
    return cols2metrics(cols, num_queries, rounding=rounding)


def all_contrastive_metrics(sims, emb=None, threshold=None, rounding=2, return_cols=False):
    text_selfsim = None
    if emb is not None:
        text_selfsim = emb @ emb.T

    t2m_m, t2m_cols = contrastive_metrics(
        sims, text_selfsim, threshold, return_cols=True, rounding=rounding
    )
    m2t_m, m2t_cols = contrastive_metrics(
        sims.T, text_selfsim, threshold, return_cols=True, rounding=rounding
    )

    all_m = {}
    for key in t2m_m:
        all_m[f"t2m/{key}"] = t2m_m[key]
        all_m[f"m2t/{key}"] = m2t_m[key]

    all_m["t2m/len"] = float(len(sims))
    all_m["m2t/len"] = float(len(sims[0]))
    if return_cols:
        return all_m, t2m_cols, m2t_cols
    return all_m


def compute_guo_metrics(sim_matrix):
    num_samples = len(sim_matrix)
    idx = np.arange(num_samples)
    np.random.seed(0)
    np.random.shuffle(idx)
    idx_batches = [idx[GUO_BATCH_SIZE * i : GUO_BATCH_SIZE * (i + 1)] for i in range(num_samples // GUO_BATCH_SIZE)]

    all_metrics = []
    for idx_batch in idx_batches:
        batch_sim_matrix = sim_matrix[np.ix_(idx_batch, idx_batch)]
        all_metrics.append(all_contrastive_metrics(batch_sim_matrix, rounding=None))

    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = round(float(np.mean([metrics[key] for metrics in all_metrics])), 2)
    return avg_metrics


def _get_tmr_repo_root() -> str:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(repo_root, "third_packages", "TMR")


def _get_tmr_dataset_name(dataset_name: str) -> str:
    if dataset_name == "hml3d":
        return "humanml3d"
    if dataset_name == "kit":
        return "kitml"
    raise ValueError(f"Unsupported dataset for TMR-aligned retrieval export: {dataset_name}")


def _get_tmr_dataset_paths(dataset_name: str) -> Dict[str, str]:
    tmr_root = _get_tmr_repo_root()
    tmr_dataset_name = _get_tmr_dataset_name(dataset_name)
    annotations_root = os.path.join(tmr_root, "datasets", "annotations", tmr_dataset_name)
    return {
        "tmr_root": tmr_root,
        "annotations_root": annotations_root,
        "annotations_path": os.path.join(annotations_root, "annotations.json"),
        "test_split": os.path.join(annotations_root, "splits", "test.txt"),
        "nsim_split": os.path.join(annotations_root, "splits", "nsim_test.txt"),
    }


def _load_tmr_keyids(split_path: str) -> List[str]:
    with open(split_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _load_tmr_annotations(annotations_path: str) -> Dict[str, Dict[str, Any]]:
    with open(annotations_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tmr_aligned_protocol_data(cfg: DictConfig, protocol: str) -> List[Dict[str, Any]]:
    dataset_name = cfg.data.dataset_name
    paths = _get_tmr_dataset_paths(dataset_name)
    split_path = paths["nsim_split"] if protocol == "nsim" else paths["test_split"]
    keyids = _load_tmr_keyids(split_path)
    annotations = _load_tmr_annotations(paths["annotations_path"])

    samples = []
    for keyid in keyids:
        sample = annotations.get(keyid)
        if sample is None:
            continue
        annotations_list = sample.get("annotations", [])
        if not annotations_list:
            continue
        first_annotation = annotations_list[0]
        samples.append(
            {
                "keyid": keyid,
                "motion_path": sample["path"],
                "start": float(first_annotation["start"]),
                "end": float(first_annotation["end"]),
                "caption": first_annotation["text"],
                "captions": [annotation["text"] for annotation in annotations_list],
            }
        )
    return samples


def _get_tmr_retrieval_encoder(model: LightningModule):
    text_encoder = getattr(model, "text_encoder", None)
    retrieval_encoder = getattr(text_encoder, "tmr", None)
    if retrieval_encoder is None:
        return None
    if not hasattr(retrieval_encoder, "encode_text") or not hasattr(retrieval_encoder, "encode_motion"):
        return None
    return retrieval_encoder


@torch.no_grad()
def collect_retrieval_embeddings_from_tmr_data(
    model: LightningModule,
    protocol_samples: List[Dict[str, Any]],
    protocol: str,
    nsim_text_mode: str = "average_all",
):
    retrieval_encoder = _get_tmr_retrieval_encoder(model)
    if retrieval_encoder is None:
        raise RuntimeError("TMR-aligned retrieval export requires a TMR-backed text encoder with motion/text encoders.")

    text_encoder_module = getattr(model, "text_encoder", None)
    tmr_wrapper = getattr(text_encoder_module, "tmr", None)
    if tmr_wrapper is None:
        raise RuntimeError("TMR-aligned retrieval export requires TMRWrapperEncoder.")

    from hydra.utils import instantiate

    tmr_root = _get_tmr_repo_root()
    tmr_cfg = copy.deepcopy(tmr_wrapper.base_cfg)
    tmr_cfg.data.path = str((Path(tmr_root) / tmr_cfg.data.path).resolve())
    tmr_cfg.data.text_to_token_emb.path = str((Path(tmr_root) / tmr_cfg.data.text_to_token_emb.path).resolve())
    if "text_to_sent_emb" in tmr_cfg.data:
        tmr_cfg.data.text_to_sent_emb.path = str((Path(tmr_root) / tmr_cfg.data.text_to_sent_emb.path).resolve())
    tmr_cfg.data.motion_loader.base_dir = str((Path(tmr_root) / tmr_cfg.data.motion_loader.base_dir).resolve())
    if "normalizer" in tmr_cfg.data.motion_loader:
        tmr_cfg.data.motion_loader.normalizer.base_dir = str(
            (Path(tmr_root) / tmr_cfg.data.motion_loader.normalizer.base_dir).resolve()
        )

    motion_loader = instantiate(tmr_cfg.data.motion_loader)
    device = next(retrieval_encoder.parameters()).device

    text_embs = []
    motion_embs = []
    captions = []
    motion_cache = {}

    for sample in protocol_samples:
        motion_key = (sample["motion_path"], sample["start"], sample["end"])
        if motion_key not in motion_cache:
            motion_x_dict = motion_loader(
                path=sample["motion_path"],
                start=sample["start"],
                end=sample["end"],
            )
            motion_cache[motion_key] = motion_x_dict["x"]
        motion_tensor = motion_cache[motion_key].to(device)
        motion_emb = retrieval_encoder.encode_motion([motion_tensor])[0]

        if protocol == "nsim":
            caption_candidates = sample["captions"] or [sample["caption"]]
            if nsim_text_mode == "average_all":
                text_batch = retrieval_encoder.encode_text(caption_candidates)
                text_emb = text_batch.mean(dim=0)
                captions.append(caption_candidates[0])
            elif nsim_text_mode == "first_caption":
                text_emb = retrieval_encoder.encode_text([sample["caption"]])[0]
                captions.append(sample["caption"])
            else:
                raise ValueError(f"Unsupported nsim_text_mode: {nsim_text_mode}")
        else:
            text_emb = retrieval_encoder.encode_text([sample["caption"]])[0]
            captions.append(sample["caption"])

        text_embs.append(torch.flatten(text_emb, start_dim=0).cpu().numpy())
        motion_embs.append(torch.flatten(motion_emb, start_dim=0).cpu().numpy())

    return np.asarray(text_embs), np.asarray(motion_embs), captions


def compute_retrieval_protocol_metrics(
    cfg: DictConfig,
    model: LightningModule,
    nsim_text_mode: str,
) -> Dict[str, Dict[str, float]]:
    protocol_metrics = {}
    sentence_embedder = SentenceEmbedder(device=str(model.device))

    normal_samples = load_tmr_aligned_protocol_data(cfg, protocol="normal")
    normal_text_embs, normal_motion_embs, normal_captions = collect_retrieval_embeddings_from_tmr_data(
        model,
        normal_samples,
        protocol="normal",
        nsim_text_mode=nsim_text_mode,
    )
    normal_sim = get_sim_matrix(normal_text_embs, normal_motion_embs)
    normal_sent_embs = sentence_embedder.encode(normal_captions)

    protocol_metrics["normal"] = all_contrastive_metrics(normal_sim)
    protocol_metrics[f"threshold_{THRESHOLD}"] = all_contrastive_metrics(
        normal_sim, normal_sent_embs, threshold=THRESHOLD
    )
    sorted_idx = np.argsort(np.asarray([sample["keyid"] for sample in normal_samples]))
    protocol_metrics["guo"] = compute_guo_metrics(normal_sim[np.ix_(sorted_idx, sorted_idx)])

    nsim_samples = load_tmr_aligned_protocol_data(cfg, protocol="nsim")
    nsim_text_embs, nsim_motion_embs, _nsim_captions = collect_retrieval_embeddings_from_tmr_data(
        model,
        nsim_samples,
        protocol="nsim",
        nsim_text_mode=nsim_text_mode,
    )
    nsim_sim = get_sim_matrix(nsim_text_embs, nsim_motion_embs)
    protocol_metrics["nsim"] = all_contrastive_metrics(nsim_sim)
    return protocol_metrics


def save_prefixed_retrieval_metrics(
    save_dir: str,
    protocol_metrics: Dict[str, Dict[str, float]],
    prefix: str,
) -> None:
    for protocol_name, metrics in protocol_metrics.items():
        metric_yaml = os.path.join(save_dir, f"{prefix}{protocol_name}.yaml")
        save_metrics_yaml(metric_yaml, _to_builtin(metrics))
        log.info(f"Retrieval YAML metrics saved to {metric_yaml}")


def export_retrieval_protocol_metrics(cfg: DictConfig, model: LightningModule) -> Dict[str, Dict[str, Dict[str, float]]]:
    tmr_protocol_metrics = compute_retrieval_protocol_metrics(
        cfg,
        model,
        nsim_text_mode="first_caption",
    )
    evt_protocol_metrics = compute_retrieval_protocol_metrics(
        cfg,
        model,
        nsim_text_mode="average_all",
    )

    save_dir = resolve_eval_save_dir(cfg)
    os.makedirs(save_dir, exist_ok=True)

    save_prefixed_retrieval_metrics(save_dir, tmr_protocol_metrics, prefix=TMR_PREFIX)
    save_prefixed_retrieval_metrics(save_dir, evt_protocol_metrics, prefix=EVT_PREFIX)

    for protocol_name, metrics in evt_protocol_metrics.items():
        metric_yaml = os.path.join(save_dir, f"{protocol_name}.yaml")
        save_metrics_yaml(metric_yaml, _to_builtin(metrics))
        log.info(f"EventT2M-native retrieval YAML metrics saved to {metric_yaml}")

    return {"TMR": tmr_protocol_metrics, "EVT": evt_protocol_metrics}


def export_event_native_metrics_yaml(save_dir: str, metrics: Dict[str, Any], prefix: str = "") -> str:
    metric_yaml = os.path.join(save_dir, f"{prefix}native_normal.yaml")
    save_metrics_yaml(metric_yaml, {_k: _to_builtin(_v) for _k, _v in metrics.items()})
    return metric_yaml


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    assert cfg.ckpt_path

    torch.set_float32_matmul_precision('high')

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.ckpt_path is None or cfg.ckpt_path == "none":
        print("No ckpt!")
        exit()
    else:
        state_dict = torch.load(cfg.ckpt_path, map_location="cpu")["state_dict"]
        keys_list = list(state_dict.keys())
        for key in keys_list:
            if 'orig_mod.' in key:
                deal_key = key.replace('_orig_mod.', '')
                state_dict[deal_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict, strict=False)

    num_parameters = sum([x.numel() for x in model.denoiser.parameters() if x.requires_grad])
    log.info("Total parameters: %.3fM" % (num_parameters / 1000_000))

    save_dir = resolve_eval_save_dir(cfg)
    os.makedirs(save_dir, exist_ok=True)

    all_metrics_new = {}
    if not cfg.get("retrieval_only", False):
        log.info("Starting testing!")
        all_metrics = {}
        replication_times = cfg.model.metrics.replicate_times

        for i in range(replication_times):
            log.info(f"Evaluating Model - Replication {i}")
            metrics = trainer.test(model, datamodule=datamodule)[0]
            if cfg.model.metrics.enable_mm_metric:
                log.info(f"Evaluating MultiModality - Replication {i}")
                datamodule.mm_mode(True, cfg.model.metrics.mm_num_samples)
                mm_metrics = trainer.test(model, datamodule=datamodule)[0]
                metrics.update(mm_metrics)
                datamodule.mm_mode(False)

            for key, item in metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = [item]
                else:
                    all_metrics[key] += [item]

        for key, item in all_metrics.items():
            mean, conf_interval = get_metric_statistics(np.array(item), replication_times)
            all_metrics_new[key + "/mean"] = mean
            all_metrics_new[key + "/conf_interval"] = conf_interval
        print_table(f"Mean Metrics", all_metrics_new)
        all_metrics_new.update(all_metrics)

        metric_file = os.path.join(cfg.paths.output_dir, "metrics.json")
        with open(metric_file, "w", encoding="utf-8") as f:
            json.dump({_k: _to_builtin(_v) for _k, _v in all_metrics_new.items()}, f, indent=4)

        native_metric_yaml = export_event_native_metrics_yaml(save_dir, all_metrics_new, prefix="")
        native_metric_yaml_evt = export_event_native_metrics_yaml(save_dir, all_metrics_new, prefix=EVT_PREFIX)

        log.info(f"Testing done, the metrics are saved to {str(metric_file)}")
        log.info(f"Event native YAML metrics saved to {str(native_metric_yaml)}")
        log.info(f"Prefixed event native YAML metrics saved to {str(native_metric_yaml_evt)}")
    else:
        log.info("Skipping native diffusion evaluation and exporting retrieval metrics only.")

    retrieval_protocol_metrics = export_retrieval_protocol_metrics(cfg, model)
    for variant_name, variant_metrics in retrieval_protocol_metrics.items():
        log.info(f"{variant_name}-style protocol metrics exported: {list(variant_metrics.keys())}")

    return all_metrics_new, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    normalize_trainer_devices(cfg)
    extras(cfg)
    evaluate(cfg)


if __name__ == "__main__":
    main()
