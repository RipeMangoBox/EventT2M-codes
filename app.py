from functools import lru_cache
import argparse
import codecs as cs
import json
import math
import re
import inspect
from pathlib import Path
from typing import Dict, List, Tuple

import gradio as gr
import numpy as np
import torch
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import RedirectResponse

from TMR_model_wrapper import TMR_Wrapper


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TEXT = "A person is "
EMPTY_DECOMP_MARKDOWN = "### Event-Level Decomposition\n_No decomposition yet._"

WEBSITE = """
<div class="embed_hidden">
<h1 style='text-align: center'>Event-T2M: Text-to-Motion Retrieval + Event-Level Decomposition Demo</h1>

<h2 style='text-align: center'>
<nobr>ICLR 2026</nobr>
</h2>

<p style='text-align: center'>
This demo follows the same retrieval visualization setup as TMR/MotionPatches and additionally shows Event-T2M event-level decomposition.
</p>
</div>
"""

EXAMPLES = [
    "A person walks forward and then sits down.",
    "A person runs in a circle and waves both arms.",
    "A person bends down to pick something up and stands back up.",
    "A person steps backward, pauses, and kicks with the right leg.",
    "A person turns left, jumps, and lands with both feet.",
    "A person raises both arms while walking sideways.",
]

CSS = """
.video-card-markdown {
    min-height: 138px;
}

.decomposition-markdown {
    min-height: 110px;
}
"""


def patch_starlette_template_response_for_legacy_gradio():
    """Compat shim for Gradio<4.44 with Starlette>=1.0 TemplateResponse signature."""
    try:
        from starlette.templating import Jinja2Templates
    except Exception:
        return

    template_response = Jinja2Templates.TemplateResponse
    params = list(inspect.signature(template_response).parameters.values())
    if len(params) < 3 or params[1].name != "request":
        return

    if getattr(Jinja2Templates.TemplateResponse, "_tmr_compat_patched", False):
        return

    original_template_response = template_response

    def compat_template_response(self, *args, **kwargs):
        # Old gradio call style: TemplateResponse(name, context, ...)
        if args and isinstance(args[0], str):
            name = args[0]
            context = args[1] if len(args) > 1 else kwargs.get("context")
            request = kwargs.get("request")
            if request is None and isinstance(context, dict):
                request = context.get("request")
            if request is None:
                raise RuntimeError(
                    "TemplateResponse compatibility patch could not locate 'request' in context."
                )
            remaining_args = args[2:] if len(args) > 2 else ()
            return original_template_response(
                self, request, name, context, *remaining_args, **kwargs
            )
        return original_template_response(self, *args, **kwargs)

    compat_template_response._tmr_compat_patched = True  # type: ignore[attr-defined]
    Jinja2Templates.TemplateResponse = compat_template_response


patch_starlette_template_response_for_legacy_gradio()


class LegacyPredictPathMiddleware(BaseHTTPMiddleware):
    """Route compatibility for frontends still calling /api/predict/*."""

    async def dispatch(self, request, call_next):
        path = request.scope.get("path", "")
        if path == "/api/predict":
            new_path = "/run/predict"
            request.scope["path"] = new_path
            request.scope["raw_path"] = new_path.encode("utf-8")
        elif path.startswith("/api/predict/"):
            suffix = path[len("/api/predict") :]
            new_path = f"/run{suffix}"
            request.scope["path"] = new_path
            request.scope["raw_path"] = new_path.encode("utf-8")
        return await call_next(request)


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(value: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(numeric):
        return 0.0
    return numeric


def load_split_ids(data_root: Path, split: str) -> List[str]:
    split_candidates = [
        data_root / f"{split}.txt",
        PROJECT_ROOT / "dataset" / "annotations" / "humanml3d" / "splits" / f"{split}.txt",
        PROJECT_ROOT / "third_packages" / "TMR" / "datasets" / "annotations" / "humanml3d" / "splits" / f"{split}.txt",
    ]
    for split_path in split_candidates:
        if split_path.exists():
            with cs.open(split_path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
    raise FileNotFoundError(f"Could not find split file for '{split}'.")


def parse_text_entries(text_path: Path) -> List[Tuple[str, float, float]]:
    if not text_path.exists():
        return []

    parsed_lines: List[Tuple[str, float, float]] = []
    with cs.open(text_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split("#")
            caption = parts[0].strip() if parts else ""
            f_tag = _safe_float(parts[2]) if len(parts) >= 3 else 0.0
            to_tag = _safe_float(parts[3]) if len(parts) >= 4 else 0.0
            parsed_lines.append((caption, f_tag, to_tag))
    return parsed_lines


def parse_decomposed_blocks(text_decomposed_path: Path) -> List[List[str]]:
    if not text_decomposed_path.exists():
        return []
    raw = text_decomposed_path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    block_strs = [block for block in re.split(r"\n\s*\n", raw) if block.strip()]
    blocks: List[List[str]] = []
    for block in block_strs:
        events = []
        for line in block.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            events.append(stripped.split("#")[0].strip())
        blocks.append(events)
    return blocks


def choose_preferred_caption_index(entries: List[Tuple[str, float, float]]) -> int:
    for idx, (_caption, f_tag, to_tag) in enumerate(entries):
        if f_tag == 0.0 and to_tag == 0.0:
            return idx
    return 0


def clean_event_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text.rstrip(" ,;")


def parse_manual_decomposition(raw_text: str) -> List[str]:
    if raw_text is None:
        return []
    lines = [clean_event_text(line) for line in raw_text.splitlines()]
    return [line for line in lines if line]


def auto_decompose_text(text: str) -> List[str]:
    if text is None:
        return []
    cleaned = clean_event_text(text)
    if not cleaned:
        return []

    chunks = re.split(r"\b(?:and then|then|after that|afterwards|finally)\b|[.;]", cleaned, flags=re.IGNORECASE)
    events = [clean_event_text(chunk) for chunk in chunks]
    events = [event for event in events if event]
    if not events:
        return [cleaned]
    return events


def format_events_markdown(events: List[str], source_label: str) -> str:
    if not events:
        return EMPTY_DECOMP_MARKDOWN
    lines = [f"### Event-Level Decomposition ({source_label})"]
    lines.extend([f"{idx}. {event}" for idx, event in enumerate(events, start=1)])
    return "\n".join(lines)


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tmr_model_dir", default="third_packages/TMR/models/tmr_humanml3d_guoh3dfeats")
    parser.add_argument("--latents_dir", default=None)
    parser.add_argument("--data_root", default="./dataset/HumanML3D")
    parser.add_argument("--device", default=None)
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--share", action="store_true")
    return parser


args = build_argparser().parse_args()

device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
tmr_model_dir = Path(args.tmr_model_dir)
latents_dir = resolve_path(args.latents_dir) if args.latents_dir else resolve_path(args.tmr_model_dir) / "latents"
data_root = resolve_path(args.data_root)
text_dir = data_root / "texts"
texts_decomposed_dir = data_root / "texts_decomposed"
animations_dir = data_root / "animations"

unit_emb_path = latents_dir / "humanml3d_all_unit.npy"
keyids_index_path = latents_dir / "humanml3d_keyids_index_all.json"

if not unit_emb_path.exists():
    raise FileNotFoundError(f"Missing precomputed gallery embeddings: {unit_emb_path}")
if not keyids_index_path.exists():
    raise FileNotFoundError(f"Missing key-id index mapping: {keyids_index_path}")

unit_motion_embs = torch.from_numpy(np.load(unit_emb_path)).float().to(device)
keyids_index: Dict[str, int] = load_json(keyids_index_path)
all_keyids = {
    "all": load_split_ids(data_root, "all"),
    "test": load_split_ids(data_root, "test"),
}

model = TMR_Wrapper(tmr_model_dir).to(device)
model.eval()


@lru_cache(maxsize=32768)
def load_motion_metadata(keyid: str) -> Dict:
    video_path = animations_dir / f"{keyid}.mp4"
    parsed_entries = parse_text_entries(text_dir / f"{keyid}.txt")

    caption = ""
    f_tag = 0.0
    to_tag = 0.0
    entry_index = 0
    if parsed_entries:
        entry_index = choose_preferred_caption_index(parsed_entries)
        caption, f_tag, to_tag = parsed_entries[entry_index]

    blocks = parse_decomposed_blocks(texts_decomposed_dir / f"{keyid}.txt")
    events = blocks[entry_index] if entry_index < len(blocks) else []

    return {
        "video_path": str(video_path.resolve()) if video_path.exists() else None,
        "caption": caption,
        "f_tag": f_tag,
        "to_tag": to_tag,
        "events": events,
    }


def retrieve(*, text: str, split: str = "test", nmax: int = 8) -> List[Dict]:
    candidate_keyids = [keyid for keyid in all_keyids[split] if keyid in keyids_index]
    if not candidate_keyids:
        return []

    index = [keyids_index[keyid] for keyid in candidate_keyids]
    gallery_embs = unit_motion_embs[index]

    with torch.inference_mode():
        text_emb = model.encode_text([text])[0]
        text_emb = text_emb / (text_emb.norm(p=2) + 1e-8)
        text_emb = text_emb.to(gallery_embs.device)
        scores = ((gallery_embs @ text_emb).detach().cpu().numpy() / 2.0) + 0.5

    candidate_keyids = np.asarray(candidate_keyids)
    sorted_idxs = np.argsort(-scores)

    results = []
    for idx in sorted_idxs:
        keyid = candidate_keyids[idx]
        metadata = load_motion_metadata(str(keyid))
        if metadata["video_path"] is None:
            continue
        result = {
            **metadata,
            "keyid": str(keyid),
            "score": float(scores[idx]),
        }
        results.append(result)
        if len(results) >= nmax:
            break
    return results


def format_result_markdown(data: Dict) -> str:
    if data["f_tag"] == 0.0 and data["to_tag"] == 0.0:
        segment = "full clip"
    else:
        segment = f"{data['f_tag']:.1f}s -> {data['to_tag']:.1f}s"

    lines = [
        f"**Score**: {data['score']:.3f}",
        f"**Motion ID**: `{data['keyid']}`",
        f"**Segment**: {segment}",
        f"**Caption**: {data['caption']}",
    ]
    if data["events"]:
        lines.append("**Events**:")
        lines.extend([f"{idx}. {event}" for idx, event in enumerate(data["events"], start=1)])
    return "  \n".join(lines)


def retrieve_component(text, decomposed_text, splits_choice, nvids, n_component=24):
    if text == DEFAULT_TEXT or text == "" or text is None:
        outputs = [EMPTY_DECOMP_MARKDOWN]
        for _ in range(n_component):
            outputs.extend([None, ""])
        return outputs

    manual_events = parse_manual_decomposition(decomposed_text)
    if manual_events:
        query_decomp_markdown = format_events_markdown(manual_events, source_label="manual")
    else:
        auto_events = auto_decompose_text(text)
        query_decomp_markdown = format_events_markdown(auto_events, source_label="auto")

    split = "test" if "Unseen" in splits_choice else "all"
    datas = retrieve(text=text, split=split, nmax=min(int(nvids), n_component))

    outputs = [query_decomp_markdown]
    for data in datas:
        outputs.extend([data["video_path"], format_result_markdown(data)])

    while len(outputs) < 1 + n_component * 2:
        outputs.extend([None, ""])
    return outputs


theme = gr.themes.Default(primary_hue="blue", secondary_hue="gray")

with gr.Blocks(css=CSS, theme=theme) as demo:
    gr.Markdown(WEBSITE)
    videos = []
    infos = []

    with gr.Row():
        with gr.Column(scale=3):
            text = gr.Textbox(
                placeholder="Type the motion you want to search with a sentence",
                show_label=True,
                label="Text prompt",
                value=DEFAULT_TEXT,
            )
            decomposed_text = gr.Textbox(
                show_label=True,
                label="Event decomposition (optional, one event per line)",
                placeholder="Leave empty to auto-decompose the prompt.",
                lines=5,
                value="",
            )
            with gr.Row():
                btn = gr.Button("Retrieve", variant="primary")
                clear = gr.Button("Clear", variant="secondary")
            with gr.Row():
                splits_choice = gr.Radio(
                    ["All motions", "Unseen motions"],
                    label="Gallery of motion",
                    value="All motions",
                    info="The motion gallery is coming from HumanML3D",
                )
                nvideo_slider = gr.Radio(
                    [4, 8, 12, 16, 24],
                    label="Videos",
                    value=8,
                    info="Number of videos to display",
                )
        with gr.Column(scale=2):
            examples = gr.Examples(
                examples=[[x, ""] for x in EXAMPLES],
                inputs=[text, decomposed_text],
                examples_per_page=12,
                run_on_click=False,
                cache_examples=False,
            )

    query_events = gr.Markdown(value=EMPTY_DECOMP_MARKDOWN, elem_classes=["decomposition-markdown"])

    for _ in range(6):
        with gr.Row():
            for _ in range(4):
                with gr.Column():
                    video = gr.Video(interactive=False, show_label=False, label=None, height=220)
                    info = gr.Markdown(elem_classes=["video-card-markdown"])
                    videos.append(video)
                    infos.append(info)

    outputs = [query_events] + [item for pair in zip(videos, infos) for item in pair]

    btn.click(
        fn=retrieve_component,
        inputs=[text, decomposed_text, splits_choice, nvideo_slider],
        outputs=outputs,
        api_name=False,
    )
    text.submit(
        fn=retrieve_component,
        inputs=[text, decomposed_text, splits_choice, nvideo_slider],
        outputs=outputs,
        api_name=False,
    )
    decomposed_text.submit(
        fn=retrieve_component,
        inputs=[text, decomposed_text, splits_choice, nvideo_slider],
        outputs=outputs,
        api_name=False,
    )
    splits_choice.change(
        fn=retrieve_component,
        inputs=[text, decomposed_text, splits_choice, nvideo_slider],
        outputs=outputs,
        api_name=False,
    )
    nvideo_slider.change(
        fn=retrieve_component,
        inputs=[text, decomposed_text, splits_choice, nvideo_slider],
        outputs=outputs,
        api_name=False,
    )

    def clear_outputs():
        cleared = [EMPTY_DECOMP_MARKDOWN]
        for _ in range(24):
            cleared.extend([None, ""])
        return cleared + [DEFAULT_TEXT, ""]

    clear.click(fn=clear_outputs, outputs=outputs + [text, decomposed_text], api_name=False)

allowed_paths = [str(animations_dir.resolve())] if animations_dir.exists() else None
demo.app.add_middleware(LegacyPredictPathMiddleware)


async def legacy_predict_route():
    return RedirectResponse(url="/run/predict", status_code=307)


async def legacy_predict_named_route(api_name: str):
    return RedirectResponse(url=f"/run/{api_name}", status_code=307)


demo.app.add_api_route("/api/predict", legacy_predict_route, methods=["POST"], include_in_schema=False)
demo.app.add_api_route("/api/predict/{api_name:path}", legacy_predict_named_route, methods=["POST"], include_in_schema=False)

launch_kwargs = {
    "allowed_paths": allowed_paths,
    "share": args.share,
    "server_port": args.port,
}

try:
    demo.launch(**launch_kwargs)
except ValueError as exc:
    if "localhost is not accessible" in str(exc) and not args.share:
        print("Localhost is not accessible in this environment; retrying with share=True.")
        launch_kwargs["share"] = True
        demo.launch(**launch_kwargs)
    else:
        raise
