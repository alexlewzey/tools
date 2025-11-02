import functools
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import fire
import pandas as pd
import torch
import transformers
from pydub import AudioSegment, effects, silence
from tqdm.auto import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def normalise(*paths: Path | str) -> None:
    """Normalise an arbitrary number of audio file."""
    for path in paths:
        path = Path(path)
        suffix = path.suffix.strip(".")
        clip = effects.normalize(AudioSegment.from_file(path.as_posix(), suffix))
        clip.export(path.as_posix(), format=suffix)
        print(f"normalised: {path.as_posix()}")


def normaliser(path: Path | str, fmt: str) -> None:
    """Normalise every file in a directory of a particular audio format."""
    for p in Path(path).iterdir():
        if p.is_file() and p.suffix.strip(".") == fmt:
            clip = effects.normalize(AudioSegment.from_file(p.as_posix(), fmt))
            clip.export(p.as_posix(), format=fmt)
            print(f"normalised: {p.as_posix()}")


def fmt2fmt(
    path: str | Path, in_fmt: str, out_fmt: str, norm: bool = True, unlink: bool = True
) -> None:
    """Covert every file in a directory of a particular format to another format and
    remove the original.

    Args:
        path: directory containing audio files
        in_fmt: input file extension eg m4a
        out_fmt: output file extension eg wav
        norm: bool if True normalise the audio once converted to new format
        unlink: bool if True delete the input file
    """
    for in_path in Path(path).iterdir():
        if in_path.is_file() and in_path.suffix.strip(".") == in_fmt:
            clip = AudioSegment.from_file(in_path.as_posix(), format=in_fmt)
            out_path = in_path.parent / f"{in_path.stem}.{out_fmt}"
            clip.export(out_path.as_posix(), format=out_fmt)
            print(f"coverted: {in_path.name} > {out_path.as_posix()}")
            if unlink:
                in_path.unlink()
            if norm:
                normalise(out_path)


def m4a2wav(path: str | Path, **kwargs) -> None:
    fmt2fmt(path, in_fmt="m4a", out_fmt="wav", **kwargs)


def mp32wav(path: str | Path) -> None:
    fmt2fmt(path, in_fmt="mp3", out_fmt="wav")


def prune(path: str | Path, fmt: str) -> None:
    """Remove every file in the path that does is not a wav, py or asd file type."""
    for f in Path(path).iterdir():
        if f.suffix.strip(".") == fmt:
            print(f"removing file: {f.name}")
            f.unlink()


def create_pipe() -> transformers.Pipeline:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )
    return pipe


def get_pipe():
    pipe = functools.partial(
        create_pipe(), generate_kwargs={"language": "english", "task": "translate"}
    )
    return pipe


def concat_text(df: pd.DataFrame) -> str:
    return " ".join(df.text.tolist()).strip()


def get_datetime() -> str:
    return datetime.now().replace(microsecond=0).isoformat().replace(":", "")


def speech2text(
    path: str | Path,
    min_silence_len: int = 1000,
    silence_thresh: int = -30,
    keep_silence: int = 1000,
) -> None:
    """For a given directory of m4a|wav audio files convert the audio speech into text
    and save in a text file."""
    pipe = get_pipe()
    path = Path(path)
    m4a2wav(path)
    dir_wavs = sorted(path.glob("*.wav"))
    path_tmp = Path("tmp") / "chunk.wav"
    path_tmp.parent.mkdir(exist_ok=True)
    timestamp = get_datetime()
    path_parquet = path / f"translation_{timestamp}.parquet"
    path_txt = path / f"translation_{timestamp}.txt"
    rows = []
    for path in tqdm(dir_wavs, desc="file"):
        wav = AudioSegment.from_wav(path)
        chunks = silence.split_on_silence(
            wav,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=keep_silence,
        )
        for i, chunk in tqdm(enumerate(chunks), desc="chunk", total=len(chunks)):
            chunk.export(path_tmp.as_posix(), format="wav")
            pred = pipe(path_tmp.as_posix())
            rows.append((path, i, pred["text"], path.stat().st_ctime))
    shutil.rmtree(path_tmp.parent)
    df = pd.DataFrame(rows, columns=["path", "chunk", "text", "ctime"]).sort_values(
        ["ctime", "chunk"]
    )
    df.to_parquet(path_parquet)
    with path_txt.open("w") as f:
        text = concat_text(df)
        f.write(text)


if sys.platform == "darwin":
    try:
        commands = ["ffmpeg", "-version"]
        subprocess.run(commands, check=True, stdout=subprocess.PIPE)  # noqa: S603
    except subprocess.CalledProcessError:
        print(
            "ffmpeg is required to run this app on mac. Please install ffmpeg: brew "
            "install ffmpeg"
        )
        sys.exit(1)


def main():
    fire.Fire(
        {
            "normalise": normalise,
            "normaliser": normaliser,
            "fmt2fmt": fmt2fmt,
            "m4a2wav": m4a2wav,
            "mp32wav": mp32wav,
            "prune": prune,
            "speech2text": speech2text,
        }
    )
