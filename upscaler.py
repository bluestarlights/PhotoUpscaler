#!/usr/bin/env python3
"""4K video upscaler CLI.

기본 모드는 ffmpeg lanczos 스케일링으로 동작하고,
realesrgan 실행 파일이 있으면 AI 프레임 업스케일링도 사용할 수 있습니다.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd))
    return subprocess.run(cmd, check=check)


def require_bin(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"필수 실행 파일이 없습니다: {name}")


def probe_video(path: Path) -> dict:
    require_bin("ffprobe")
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)


def ffmpeg_lanczos(input_path: Path, output_path: Path, width: int, height: int, crf: int, preset: str) -> None:
    require_bin("ffmpeg")
    vf = (
        f"scale=w={width}:h={height}:force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-c:a",
        "copy",
        str(output_path),
    ]
    run(cmd)


def realesrgan_pipeline(
    input_path: Path,
    output_path: Path,
    width: int,
    height: int,
    crf: int,
    preset: str,
    realesrgan_bin: str,
    model: str,
) -> None:
    require_bin("ffmpeg")
    with tempfile.TemporaryDirectory(prefix="upscale_") as d:
        tmp = Path(d)
        frames_in = tmp / "frames_in"
        frames_out = tmp / "frames_out"
        audio_file = tmp / "audio.m4a"
        frames_in.mkdir()
        frames_out.mkdir()

        run(["ffmpeg", "-y", "-i", str(input_path), str(frames_in / "%08d.png")])
        run(["ffmpeg", "-y", "-i", str(input_path), "-vn", "-c:a", "copy", str(audio_file)], check=False)

        run(
            [
                realesrgan_bin,
                "-i",
                str(frames_in),
                "-o",
                str(frames_out),
                "-n",
                model,
            ]
        )

        vf = (
            f"scale=w={width}:h={height}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            "30",
            "-i",
            str(frames_out / "%08d.png"),
        ]
        if audio_file.exists() and audio_file.stat().st_size > 0:
            cmd += ["-i", str(audio_file)]
        cmd += [
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
        ]
        if audio_file.exists() and audio_file.stat().st_size > 0:
            cmd += ["-c:a", "aac", "-shortest"]
        cmd += [str(output_path)]
        run(cmd)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="블로그 내용을 참고한 4K 비디오 업스케일러")
    p.add_argument("input", type=Path, help="입력 비디오 파일")
    p.add_argument("output", type=Path, help="출력 비디오 파일")
    p.add_argument("--target-width", type=int, default=3840)
    p.add_argument("--target-height", type=int, default=2160)
    p.add_argument("--method", choices=["lanczos", "realesrgan"], default="lanczos")
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--preset", default="slow")
    p.add_argument("--realesrgan-bin", default="realesrgan-ncnn-vulkan")
    p.add_argument("--realesrgan-model", default="realesr-animevideov3")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"입력 파일이 없습니다: {args.input}")

    meta = probe_video(args.input)
    v_streams = [s for s in meta.get("streams", []) if s.get("codec_type") == "video"]
    if not v_streams:
        raise SystemExit("비디오 스트림을 찾지 못했습니다.")

    print("입력 비디오 정보:")
    print(json.dumps({"video_stream": v_streams[0]}, ensure_ascii=False, indent=2))

    if args.method == "lanczos":
        ffmpeg_lanczos(args.input, args.output, args.target_width, args.target_height, args.crf, args.preset)
    else:
        realesrgan_pipeline(
            args.input,
            args.output,
            args.target_width,
            args.target_height,
            args.crf,
            args.preset,
            args.realesrgan_bin,
            args.realesrgan_model,
        )

    print(f"완료: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"명령 실행 실패: {e}", file=sys.stderr)
        sys.exit(e.returncode)
