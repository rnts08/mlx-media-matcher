import shutil
import subprocess

import pytest

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
requires_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True)


@pytest.fixture(scope="session")
def make_video(tmp_path_factory):
    """Generate a short deterministic test clip with ffmpeg's testsrc."""

    def _make(name, seconds=2.0, size="320x240", fps=10, pattern="testsrc"):
        path = tmp_path_factory.mktemp("videos") / name
        _run([
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i",
            f"{pattern}=size={size}:rate={fps}:duration={seconds}",
            "-pix_fmt", "yuv420p", str(path),
        ])
        return str(path)

    return _make


@pytest.fixture(scope="session")
def reencode(tmp_path_factory):
    """Re-encode/rescale a clip to a new resolution (a re-encode duplicate)."""

    def _reencode(src, name, size):
        path = tmp_path_factory.mktemp("variants") / name
        _run([
            "ffmpeg", "-v", "error", "-y", "-i", src,
            "-vf", f"scale={size}", "-c:v", "libx264", "-crf", "28",
            "-pix_fmt", "yuv420p", str(path),
        ])
        return str(path)

    return _reencode
