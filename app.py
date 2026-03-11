from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import zipfile

import cv2
import gradio as gr
import numpy as np

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass
class UpscaleConfig:
    output_dir: Path
    target_long_edge: int = 3840
    denoise_strength: float = 0.5
    tile: int = 512
    tile_pad: int = 10
    pre_pad: int = 0
    face_enhance: bool = True


def discover_images(root: Path) -> List[Path]:
    return sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    )


def parse_uploaded_files(uploaded_files) -> List[Path]:
    if not uploaded_files:
        return []

    paths = []
    for f in uploaded_files:
        raw = getattr(f, "name", f)
        p = Path(raw)
        if p.suffix.lower() in SUPPORTED_EXTENSIONS and p.exists():
            paths.append(p)
    return paths


def gather_input_images(input_dir: str, uploaded_files) -> Tuple[List[Path], Optional[Path], str]:
    folder = Path(input_dir).expanduser().resolve()
    from_folder: List[Path] = []
    if folder.exists():
        from_folder = discover_images(folder)

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

    msg = (
        f"총 {len(merged)}장 준비 완료 (폴더 {len(from_folder)}장 + 업로드 {len(from_upload)}장)."
    )
    return merged, folder, msg


def build_realesrganer(config: UpscaleConfig):
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    )

    half = torch.cuda.is_available()
    if not half:
        print("[WARN] CUDA GPU가 감지되지 않아 CPU 모드로 동작합니다. 속도가 매우 느릴 수 있습니다.")

    upsampler = RealESRGANer(
        scale=4,
        model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        model=model,
        tile=config.tile,
        tile_pad=config.tile_pad,
        pre_pad=config.pre_pad,
        half=half,
    )

    face_enhancer = None
    if config.face_enhance:
        try:
            from gfpgan import GFPGANer

            face_enhancer = GFPGANer(
                model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.4/GFPGANv1.4.pth",
                upscale=2,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=upsampler,
            )
        except Exception as exc:
            print(f"[WARN] GFPGAN 로드 실패. 얼굴 보정 없이 진행합니다: {exc}")
    return upsampler, face_enhancer


def upscale_to_4k(
    image_bgr: np.ndarray,
    upsampler,
    face_enhancer,
    target_long_edge: int,
    denoise_strength: float,
) -> np.ndarray:
    if face_enhancer is not None:
        _, _, enhanced = face_enhancer.enhance(
            image_bgr,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=float(np.clip(denoise_strength, 0.0, 1.0)),
        )
    else:
        enhanced, _ = upsampler.enhance(image_bgr, outscale=4)

    now_long_edge = max(enhanced.shape[:2])
    if now_long_edge >= target_long_edge:
        return enhanced

    scale = target_long_edge / now_long_edge
    new_w = max(1, int(round(enhanced.shape[1] * scale)))
    new_h = max(1, int(round(enhanced.shape[0] * scale)))
    return cv2.resize(enhanced, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


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
        rel = img_path.relative_to(folder_root)
        return output_dir / rel
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


def run_batch(input_dir, uploaded_files, output_dir, target_long_edge, denoise_strength, tile, face_enhance):
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    images, folder_root, prep_message = gather_input_images(input_dir, uploaded_files)
    if not images:
        yield prep_message, [], None
        return

    config = UpscaleConfig(
        output_dir=output_path,
        target_long_edge=int(target_long_edge),
        denoise_strength=float(denoise_strength),
        tile=int(tile),
        face_enhance=bool(face_enhance),
    )

    upsampler, face_enhancer = build_realesrganer(config)

    gallery = []
    for idx, img_path in enumerate(images, start=1):
        out_path = build_output_path(img_path, config.output_dir, folder_root)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            yield f"[{idx}/{len(images)}] 읽기 실패: {img_path}", gallery, None
            continue

        result = upscale_to_4k(
            image_bgr=image_bgr,
            upsampler=upsampler,
            face_enhancer=face_enhancer,
            target_long_edge=config.target_long_edge,
            denoise_strength=config.denoise_strength,
        )
        cv2.imwrite(str(out_path), result)
        gallery.append((str(out_path), f"완료: {img_path.name}"))

        yield (
            f"{prep_message}\n[{idx}/{len(images)}] 업스케일 완료: {img_path.name}",
            gallery,
            None,
        )

    zip_file = make_zip(config.output_dir)
    yield (
        f"총 {len(gallery)}장 완료. 결과 폴더: {config.output_dir}\nZIP 다운로드: {zip_file.name}",
        gallery,
        str(zip_file),
    )


def build_ui() -> gr.Blocks:
    default_in = str((Path.cwd() / "scan").resolve())
    default_out = str((Path.cwd() / "output_4k").resolve())

    with gr.Blocks(title="PhotoUpscaler 4K") as demo:
        gr.Markdown(
            """
            # 📸 PhotoUpscaler (4K)
            폴더 스캔 + 다중 파일 업로드를 동시에 지원합니다.
            여러 사진을 한 번에 큐로 처리하고 결과를 ZIP으로 다운로드할 수 있습니다.
            """
        )

        with gr.Row():
            input_dir = gr.Textbox(label="입력 폴더(재귀 검색)", value=default_in)
            output_dir = gr.Textbox(label="출력 폴더", value=default_out)

        upload_files = gr.Files(label="추가 업로드(여러 장 선택 가능)", file_count="multiple")

        with gr.Row():
            target_long_edge = gr.Slider(2160, 6144, value=3840, step=64, label="목표 긴 변 해상도")
            denoise_strength = gr.Slider(0, 1, value=0.5, step=0.05, label="얼굴 보정 강도")
            tile = gr.Dropdown(choices=[0, 256, 512, 1024], value=512, label="타일 크기")
            face_enhance = gr.Checkbox(value=True, label="얼굴 보정(GFPGAN)")

        run_btn = gr.Button("여러 사진 업스케일 시작", variant="primary")
        log = gr.Textbox(label="진행 로그", lines=4)
        gallery = gr.Gallery(label="결과 미리보기", columns=4, height="auto")
        zip_file = gr.File(label="결과 ZIP 다운로드")

        run_btn.click(
            fn=run_batch,
            inputs=[input_dir, upload_files, output_dir, target_long_edge, denoise_strength, tile, face_enhance],
            outputs=[log, gallery, zip_file],
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=7860)
