"""Piper text-to-speech run in this process on the CPU.

A voice is a pair of files (`<name>.onnx`, `<name>.onnx.json`) in the Hugging Face repository `rhasspy/piper-voices`.
The service serves the voices listed in its `voices` setting; each one loads from the local cache and stays loaded.
One generation runs at a time per replica; scale with replicas.

shortcut: the update check compares the whole voices repository, so it also reports changes to voices nobody uses."""

from __future__ import annotations

import asyncio
import io
import logging
import os
import re
import threading
import uuid
import wave
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Iterator

_log = logging.getLogger(__name__)

DEFAULT_REPO = "rhasspy/piper-voices"
DEFAULT_VOICES = "en_US-lessac-medium"
VOICE_PATTERN = r"^[a-z]{2,3}_[A-Z]{2}-[a-z0-9_]+-(x_low|low|medium|high)$"


class EngineError(RuntimeError):
    """The text could not be spoken. `status` is the HTTP status the service maps it to."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def voice_file(name: str) -> str:
    """Path of a voice inside the repository, without extension: `en_US-lessac-medium` -> `en/en_US/lessac/medium/en_US-lessac-medium`."""
    locale, speaker, quality = name.split("-", 2)
    return f"{locale.split('_')[0]}/{locale}/{speaker}/{quality}/{name}"


class Voice:
    """A loaded voice that turns text into a WAV file."""

    def __init__(self, voice: Any) -> None:
        self._voice = voice
        self.rate: int = voice.config.sample_rate

    def pieces(self, text: str, speed: float) -> Iterator[bytes]:
        """16-bit mono audio, one piece per sentence, made as the caller asks for the next one."""
        from piper import SynthesisConfig

        for chunk in self._voice.synthesize(text, syn_config=SynthesisConfig(length_scale=1 / speed)):
            if chunk.audio_int16_bytes:
                yield chunk.audio_int16_bytes

    def speak(self, text: str, speed: float) -> tuple[bytes, float]:
        from piper import SynthesisConfig

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            self._voice.synthesize_wav(text, wav, syn_config=SynthesisConfig(length_scale=1 / speed))
            seconds = wav.getnframes() / wav.getframerate()
        return buf.getvalue(), seconds


def load(repo: str, name: str, threads: int | None) -> Voice:
    from huggingface_hub import hf_hub_download
    from piper import PiperVoice

    base = voice_file(name)
    onnx = hf_hub_download(repo, base + ".onnx", local_files_only=True)
    config = hf_hub_download(repo, base + ".onnx.json", local_files_only=True)
    voice = PiperVoice.load(onnx, config_path=config)
    if threads:  # Piper does not expose the thread count, so the session is rebuilt with it
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = threads
        voice.session = onnxruntime.InferenceSession(onnx, options, providers=["CPUExecutionProvider"])
    return Voice(voice)


def _remote_sha(repo: str) -> str:
    from huggingface_hub import HfApi

    return HfApi().model_info(repo, token=False).sha


def _cached_sha(repo: str, voice: str) -> str | None:
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(repo, voice_file(voice) + ".onnx")
    return Path(path).parts[Path(path).parts.index("snapshots") + 1] if isinstance(path, str) else None  # the folder is named by commit


def _download(repo: str, voices: list[str]) -> None:
    from huggingface_hub import hf_hub_download

    revision = _remote_sha(repo)
    for name in voices:
        for suffix in (".onnx", ".onnx.json"):
            hf_hub_download(repo, voice_file(name) + suffix, revision=revision, token=False)


def default_marker() -> Path:
    """File in the shared models volume whose content changes when one replica downloads new weights."""
    configured = os.environ.get("SERVICE_WEIGHTS_MARKER")
    return Path(configured) if configured else Path(os.environ.get("HF_HOME") or "~/.cache/huggingface").expanduser() / ".ttspiper-weights-version"


def parse_voices(text: str) -> list[str]:
    """The voices of a comma separated list; raises ValueError for a name that is not a Piper voice name."""
    names = [item.strip() for item in text.split(",") if item.strip()]
    for name in names:
        if not re.fullmatch(VOICE_PATTERN, name):
            raise ValueError(f"{name!r} is not a Piper voice name, such as en_US-lessac-medium")
    if not names:
        raise ValueError("At least one voice is needed")
    return names


class PiperEngine:
    """The `loader`, `remote_sha`, `cached_sha` and `download` callables are injectable for tests."""

    def __init__(self, repo: str = DEFAULT_REPO, voices: str = DEFAULT_VOICES, *, timeout: float = 120.0,
                 marker: Path | None = None, loader: Callable[[str, str, int | None], Voice] = load,
                 remote_sha: Callable[[str], str] = _remote_sha, cached_sha: Callable[[str, str], str | None] = _cached_sha,
                 download: Callable[[str, list[str]], None] = _download) -> None:
        self.model = repo
        self.voice_names = parse_voices(voices)
        self._threads: int | None = None
        self._timeout = timeout
        self._marker = marker
        self._seen = self._read_marker()
        self._loader = loader
        self._remote_sha_of, self._cached_sha_of, self._download_of = remote_sha, cached_sha, download
        self._loaded: dict[str, Voice] = {}
        self._load_lock = threading.Lock()
        self._gate = asyncio.Semaphore(1)
        self.generation = 0  # part of the answer cache key; changes whenever the loaded voices are dropped

    @property
    def voices(self) -> list[str]:
        return list(self.voice_names)

    @voices.setter
    def voices(self, text: str) -> None:
        self.voice_names = parse_voices(text)

    @property
    def default_voice(self) -> str:
        return self.voice_names[0]

    @property
    def num_thread(self) -> int | None:
        return self._threads

    @num_thread.setter
    def num_thread(self, count: int) -> None:
        if count != self._threads:
            self._threads = count
            self._forget()  # loaded voices keep the thread count they were built with

    @property
    def ready(self) -> bool:
        return self.default_voice in self._loaded

    async def close(self) -> None:
        self._loaded = {}

    async def warm(self) -> None:
        """Load the default voice now so the first request does not pay for it."""
        await asyncio.to_thread(self._voice, self.default_voice)

    def _voice(self, name: str) -> Voice:
        with self._load_lock:
            if name not in self._loaded:
                try:
                    self._loaded[name] = self._loader(self.model, name, self._threads)
                except OSError:
                    raise EngineError("Model weights are not downloaded. Ask an admin to press Download on the Updates card.", 503) from None
            return self._loaded[name]

    def _name(self, voice: str | None) -> str:
        name = voice or self.default_voice
        if name not in self.voice_names:
            raise EngineError(f"Voice {name!r} is not served here; the voices are {', '.join(self.voice_names)}", 404)
        return name

    async def synthesize(self, text: str, voice: str | None, speed: float) -> dict[str, Any]:
        name = self._name(voice)
        async with self._gate:
            wav, seconds = await asyncio.to_thread(self._speak, text, name, speed)
        return {"wav": wav, "seconds": seconds, "voice": name}

    async def stream(self, text: str, voice: str | None, speed: float) -> AsyncIterator[tuple[int, bytes]]:
        """The audio as it is made: (sample rate, 16-bit mono PCM) per sentence. Holds the generation slot until
        the last piece, or until the caller closes the iterator."""
        name = self._name(voice)
        async with self._gate:
            loaded = await asyncio.to_thread(self._voice, name)
            pieces = loaded.pieces(text, speed)
            made = False
            while (piece := await asyncio.to_thread(next, pieces, None)) is not None:
                made = True
                yield loaded.rate, piece
        if not made:
            raise EngineError("No speech could be made from that text.", 422)

    def _speak(self, text: str, name: str, speed: float) -> tuple[bytes, float]:
        return self._voice(name).speak(text, speed)

    # Updates: the Hub revision of the voices repository, shared by all replicas through one models volume.

    async def cached_sha(self) -> str | None:
        return await asyncio.to_thread(self._cached_sha_of, self.model, self.default_voice)

    async def remote_sha(self) -> str:
        return await asyncio.to_thread(self._remote_sha_of, self.model)

    async def pull(self) -> None:
        await asyncio.to_thread(self._download_of, self.model, self.voices)

    async def unload(self) -> None:
        """Forget the loaded voices here and tell the other replicas to do the same; the next call loads the new files."""
        if self._marker is not None:
            self._marker.parent.mkdir(parents=True, exist_ok=True)
            self._seen = uuid.uuid4().hex
            self._marker.write_text(self._seen, "utf-8")
        self._forget()

    def sync(self) -> None:
        """Called every few seconds: drop the loaded voices when another replica downloaded new files."""
        current = self._read_marker()
        if current != self._seen:
            self._seen = current
            self._forget()

    def _forget(self) -> None:
        with self._load_lock:
            self._loaded = {}
            self.generation += 1  # kept answers are from the old files or threads

    def _read_marker(self) -> str | None:
        try:
            return self._marker.read_text("utf-8") if self._marker else None
        except OSError:
            return None
