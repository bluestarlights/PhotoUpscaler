import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import zipfile

import gradio as gr
import numpy as np
from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass
class UpscaleConfig:
    output_dir: Path
    target_long_edge: int = 3840
    model_id: str = "caidas/swin2SR-classical-sr-x2-64"


def discover_images(root: Path) -> List[Path]:
    return sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    )


def parse_uploaded_files(uploaded_files) -> List[Path]:
    if not uploaded_files:
        return []
    paths: List[Path] = []
    for file_item in uploaded_files:
        raw = getattr(file_item, "name", file_item)
        p = Path(raw)
        if p.suffix.lower() in SUPPORTED_EXTENSIONS and p.exists():
            paths.append(p)
    return paths


def gather_input_images(input_dir: str, uploaded_files) -> Tuple[List[Path], Optional[Path], str]:
    folder = Path(input_dir).expanduser().resolve()
    from_folder = discover_images(folder) if folder.exists() else []
    from_upload = parse_uploaded_files(uploaded_files)

    merged: List[Path] = []
    seen = set()
    for p in [*from_folder, *from_upload]:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            merged.append(p)

    if not merged:
        return [], folder, "스캔 이미지가 없습니다. 폴더 경로나 업로드 파일을 확인해주세요."

    msg = f"총 {len(merged)}장 준비 완료 (폴더 {len(from_folder)}장 + 업로드 {len(from_upload)}장)."
    return merged, folder, msg


def build_upscaler(model_id: str):
    import torch
    from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = Swin2SRForImageSuperResolution.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return processor, model, device


def upscale_with_swin2sr(image: Image.Image, processor, model, device: str, target_long_edge: int) -> Image.Image:
    import torch

    rgb = image.convert("RGB")
    np_img = np.array(rgb)
    inputs = processor(images=np_img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.inference_mode():
        outputs = model(pixel_values)

    out = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    out = (out * 255.0).round().astype(np.uint8)
    out = np.transpose(out, (1, 2, 0))
    upscaled = Image.fromarray(out)

    current_long = max(upscaled.size)
    if current_long < target_long_edge:
        scale = target_long_edge / current_long
        new_size = (max(1, int(upscaled.width * scale)), max(1, int(upscaled.height * scale)))
        upscaled = upscaled.resize(new_size, Image.Resampling.LANCZOS)

    return upscaled


def ensure_unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    idx = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{idx}{path.suffix}")
        if not candidate.exists():
            return candidate
        idx += 1


def build_output_path(img_path: Path, output_dir: Path, folder_root: Optional[Path]) -> Path:
    if folder_root and folder_root in img_path.parents:
        return output_dir / img_path.relative_to(folder_root)
    return ensure_unique_path(output_dir / img_path.name)


def make_zip(output_dir: Path) -> Path:
    zip_path = output_dir / "upscaled_results.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in output_dir.rglob("*"):
            if p.is_file() and p != zip_path:
                zf.write(p, p.relative_to(output_dir))
    return zip_path


def run_batch(input_dir, uploaded_files, output_dir, target_long_edge):
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    images, folder_root, prep_message = gather_input_images(input_dir, uploaded_files)
    if not images:
        yield prep_message, [], None
        return

    config = UpscaleConfig(output_dir=out_dir, target_long_edge=int(target_long_edge))
    processor, model, device = build_upscaler(config.model_id)

    gallery = []
    for idx, img_path in enumerate(images, start=1):
        out_path = build_output_path(img_path, config.output_dir, folder_root)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with Image.open(img_path) as img:
                result = upscale_with_swin2sr(img, processor, model, device, config.target_long_edge)
                result.save(out_path)
            gallery.append((str(out_path), f"완료: {img_path.name}"))
            yield f"{prep_message}\n[{idx}/{len(images)}] 업스케일 완료: {img_path.name}", gallery, None
        except Exception as exc:
            yield f"[{idx}/{len(images)}] 실패: {img_path.name} ({exc})", gallery, None

    zip_file = make_zip(config.output_dir)
    yield f"총 {len(gallery)}장 완료. 결과 폴더: {config.output_dir}", gallery, str(zip_file)


def build_ui() -> gr.Blocks:
    default_in = str((Path.cwd() / "scan").resolve())
    default_out = str((Path.cwd() / "output_4k").resolve())

    with gr.Blocks(title="PhotoUpscaler 4K (Python 최신버전 호환)") as demo:
        gr.Markdown(
            """
            # 📸 PhotoUpscaler (4K)
            - Python 최신 버전 호환 (basicsr/realesrgan 의존성 제거)
            - 폴더 스캔 + 다중 파일 업로드 배치 처리
            - Swin2SR 기반 업스케일 + ZIP 다운로드
            """
        )

        with gr.Row():
            input_dir = gr.Textbox(label="입력 폴더(재귀 검색)", value=default_in)
            output_dir = gr.Textbox(label="출력 폴더", value=default_out)

        upload_files = gr.Files(label="추가 업로드(여러 장 선택 가능)", file_count="multiple")
        target_long_edge = gr.Slider(2160, 6144, value=3840, step=64, label="목표 긴 변 해상도")

        run_btn = gr.Button("여러 사진 업스케일 시작", variant="primary")
        log = gr.Textbox(label="진행 로그", lines=4)
        gallery = gr.Gallery(label="결과 미리보기", columns=4, height="auto")
        zip_file = gr.File(label="결과 ZIP 다운로드")

        run_btn.click(
            fn=run_batch,
            inputs=[input_dir, upload_files, output_dir, target_long_edge],
            outputs=[log, gallery, zip_file],
        )

    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="PhotoUpscaler 4K UI")
    parser.add_argument("--host", default="127.0.0.1", help="Gradio bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="Gradio bind port")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    parser.add_argument("--inbrowser", action="store_true", help="Open UI in browser automatically")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ui = build_ui()
    ui.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=args.inbrowser,
    )
