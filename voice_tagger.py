#!/usr/bin/env python3
"""
Voice-Tagger — Extract comprehensive vocal tags from audio files.

Analyzes voice recordings using OpenSMILE (eGeMAPSv02), librosa pitch tracking,
and SpeechBrain emotion recognition to produce human-readable tags and raw features.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import librosa
import numpy as np
import opensmile
import torch

# ---------------------------------------------------------------------------
# Thresholds for tag derivation
# Sources:
#   Jitter/Shimmer/HNR: Teixeira et al. (2013), "Vocal Acoustic Analysis"
#   Pitch ranges: Titze (1994), "Principles of Voice Production"
#   Alpha Ratio / Hammarberg: Eyben et al. (2016), GeMAPS paper
#   Loudness dynamics: Schuller et al. (2013), INTERSPEECH ComParE
# ---------------------------------------------------------------------------
THRESHOLDS = {
    # Voice quality
    "jitter_rough": 0.025,          # above → Rough/Gravelly
    "jitter_creaky": 0.04,          # above → Creaky
    "shimmer_breathy": 1.0,         # dB above → Breathy
    "hnr_hoarse": 5.0,              # below → Hoarse
    "hnr_clean": 15.0,              # above → Clean

    # Pitch (semitones re 27.5 Hz)  — eGeMAPSv02 uses semitone scale
    "pitch_low_st": 30.0,           # ~165 Hz — below → Low Pitch
    "pitch_high_st": 42.0,          # ~440 Hz — above → High Pitch
    "pitch_male_max_st": 33.0,      # ~196 Hz
    "pitch_female_min_st": 36.0,    # ~262 Hz
    "pitch_cv_monotone": 0.10,      # stddevNorm below → Monotone
    "pitch_cv_expressive": 0.25,    # stddevNorm above → Expressive

    # Brightness / warmth
    "alpha_ratio_bright": 0.0,      # above → Bright/Cutting (more HF energy)
    "alpha_ratio_warm": -12.0,      # below → Warm/Mellow
    "hammarberg_rich": 25.0,        # above → Rich/Full
    "h1h2_pressed": 2.0,            # logRelF0-H1-H2 below → Pressed/Tense
    "h1h2_breathy": 10.0,           # above → Breathy (open quotient)

    # Energy / dynamics
    "loudness_cv_steady": 0.15,     # below → Steady
    "loudness_cv_dynamic": 0.35,    # above → Dynamic
    "peaks_per_sec_fast": 3.0,      # above → Fast Pacing
    "peaks_per_sec_slow": 1.0,      # below → Slow Pacing

    # Formant resonance (Hz)
    "f1_open": 700.0,               # above → Open/Projected
    "f3_ringy": 3000.0,             # above → Ringy/Resonant
    "f1_nasal": 350.0,              # below → Nasal Hint

    # Voicing patterns (seconds)
    "voiced_seg_long": 0.4,         # above → Long Sustained
    "voiced_seg_short": 0.15,       # below → Short Choppy
    "unvoiced_seg_pauses": 0.3,     # above → Deliberate Pauses
}

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}


# ---------------------------------------------------------------------------
# Audio format helpers
# ---------------------------------------------------------------------------

def ensure_wav(file_path: str) -> tuple[str, bool]:
    """Return (wav_path, is_temp). Converts non-WAV to a temp WAV via ffmpeg."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".wav":
        return file_path, False

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", file_path, "-ar", "16000", "-ac", "1", tmp.name],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        os.unlink(tmp.name)
        raise RuntimeError(
            "ffmpeg not found. Install it: https://ffmpeg.org/download.html"
        )
    except subprocess.CalledProcessError as exc:
        os.unlink(tmp.name)
        raise RuntimeError(f"ffmpeg conversion failed for {file_path}") from exc

    return tmp.name, True


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_opensmile(smile, wav_path: str) -> dict:
    """Extract all 88 eGeMAPSv02 functionals as a flat dict."""
    df = smile.process_file(wav_path)
    return {col: round(float(df[col].values[0]), 6) for col in df.columns}


def extract_pitch_librosa(wav_path: str) -> dict:
    """Extract pitch features with librosa pyin."""
    y, sr = librosa.load(wav_path, sr=16000)
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
    )
    f0_voiced = f0[~np.isnan(f0)] if f0 is not None else np.array([])

    if len(f0_voiced) == 0:
        return {
            "f0_mean": 0.0,
            "f0_std": 0.0,
            "f0_min": 0.0,
            "f0_max": 0.0,
            "voiced_fraction": 0.0,
        }

    total_frames = len(f0) if f0 is not None else 1
    return {
        "f0_mean": round(float(np.mean(f0_voiced)), 2),
        "f0_std": round(float(np.std(f0_voiced)), 2),
        "f0_min": round(float(np.min(f0_voiced)), 2),
        "f0_max": round(float(np.max(f0_voiced)), 2),
        "voiced_fraction": round(float(len(f0_voiced) / total_frames), 4),
    }


def extract_emotion(emotion_classifier, wav_path: str) -> dict:
    """Classify emotion using SpeechBrain wav2vec2-IEMOCAP."""
    out_prob, score, index, text_lab = emotion_classifier.classify_file(wav_path)
    probs = out_prob.squeeze().tolist()
    labels = ["ang", "hap", "sad", "neu"]
    return {
        "label": text_lab[0],
        "scores": {lab: round(p, 4) for lab, p in zip(labels, probs)},
    }


# ---------------------------------------------------------------------------
# Tag derivation
# ---------------------------------------------------------------------------

def derive_tags(osm: dict, pitch: dict) -> dict:
    """Map raw features to human-readable tags organised by category."""
    T = THRESHOLDS
    tags = {
        "voice_quality": [],
        "pitch": [],
        "brightness_warmth": [],
        "energy_dynamics": [],
        "formant_resonance": [],
        "voicing_patterns": [],
    }

    # --- Voice quality ---
    jitter = osm.get("jitterLocal_sma3nz_amean", 0)
    shimmer = osm.get("shimmerLocaldB_sma3nz_amean", 0)
    hnr = osm.get("HNRdBACF_sma3nz_amean", 0)

    if jitter > T["jitter_creaky"]:
        tags["voice_quality"].append("Creaky")
    elif jitter > T["jitter_rough"]:
        tags["voice_quality"].append("Rough/Gravelly")
    if shimmer > T["shimmer_breathy"]:
        tags["voice_quality"].append("Breathy")
    if hnr < T["hnr_hoarse"]:
        tags["voice_quality"].append("Hoarse")
    if hnr > T["hnr_clean"]:
        tags["voice_quality"].append("Clean")

    # --- Pitch ---
    f0_st = osm.get("F0semitoneFrom27.5Hz_sma3nz_amean", 0)
    f0_cv = osm.get("F0semitoneFrom27.5Hz_sma3nz_stddevNorm", 0)

    if f0_st < T["pitch_low_st"]:
        tags["pitch"].append("Low Pitch")
    elif f0_st > T["pitch_high_st"]:
        tags["pitch"].append("High Pitch")
    else:
        tags["pitch"].append("Mid Pitch")

    if f0_st <= T["pitch_male_max_st"]:
        tags["pitch"].append("Male Range")
    elif f0_st >= T["pitch_female_min_st"]:
        tags["pitch"].append("Female Range")

    if f0_cv < T["pitch_cv_monotone"]:
        tags["pitch"].append("Monotone")
    elif f0_cv > T["pitch_cv_expressive"]:
        tags["pitch"].append("Expressive")

    # --- Brightness / warmth ---
    alpha = osm.get("alphaRatioV_sma3nz_amean", -6)
    hammarberg = osm.get("hammarbergIndexV_sma3nz_amean", 20)
    h1h2 = osm.get("logRelF0-H1-H2_sma3nz_amean", 5)

    if alpha > T["alpha_ratio_bright"]:
        tags["brightness_warmth"].append("Bright/Cutting")
    elif alpha < T["alpha_ratio_warm"]:
        tags["brightness_warmth"].append("Warm/Mellow")

    if hammarberg > T["hammarberg_rich"]:
        tags["brightness_warmth"].append("Rich/Full")

    if h1h2 < T["h1h2_pressed"]:
        tags["brightness_warmth"].append("Pressed/Tense")
    elif h1h2 > T["h1h2_breathy"]:
        tags["brightness_warmth"].append("Breathy Tone")

    # --- Energy / dynamics ---
    loud_cv = osm.get("loudness_sma3_stddevNorm", 0.2)
    peaks = osm.get("loudnessPeaksPerSec", 2)
    flux = osm.get("spectralFlux_sma3_amean", 0)

    if loud_cv < T["loudness_cv_steady"]:
        tags["energy_dynamics"].append("Steady")
    elif loud_cv > T["loudness_cv_dynamic"]:
        tags["energy_dynamics"].append("Dynamic")

    if peaks > T["peaks_per_sec_fast"]:
        tags["energy_dynamics"].append("Fast Pacing")
    elif peaks < T["peaks_per_sec_slow"]:
        tags["energy_dynamics"].append("Slow Pacing")

    # --- Formant resonance ---
    f1 = osm.get("F1frequency_sma3nz_amean", 500)
    f3 = osm.get("F3frequency_sma3nz_amean", 2500)

    if f1 < T["f1_nasal"]:
        tags["formant_resonance"].append("Nasal Hint")
    if f1 > T["f1_open"]:
        tags["formant_resonance"].append("Open/Projected")
    if f3 > T["f3_ringy"]:
        tags["formant_resonance"].append("Ringy/Resonant")

    # --- Voicing patterns ---
    voiced_len = osm.get("MeanVoicedSegmentLengthSec", 0.25)
    unvoiced_len = osm.get("MeanUnvoicedSegmentLength", 0.15)

    if voiced_len > T["voiced_seg_long"]:
        tags["voicing_patterns"].append("Long Sustained")
    elif voiced_len < T["voiced_seg_short"]:
        tags["voicing_patterns"].append("Short Choppy")

    if unvoiced_len > T["unvoiced_seg_pauses"]:
        tags["voicing_patterns"].append("Deliberate Pauses")

    return tags


# ---------------------------------------------------------------------------
# Process a single file
# ---------------------------------------------------------------------------

def process_file(
    file_path: str,
    smile,
    emotion_classifier=None,
    verbose: bool = False,
) -> dict:
    """Analyse one audio file and return structured results."""
    wav_path, is_temp = ensure_wav(file_path)
    try:
        osm_features = extract_opensmile(smile, wav_path)
        pitch_features = extract_pitch_librosa(wav_path)

        tags = derive_tags(osm_features, pitch_features)

        emotion = None
        if emotion_classifier is not None:
            emotion = extract_emotion(emotion_classifier, wav_path)
            tags["emotion"] = [
                f"{emotion['label']} ({emotion['scores'][emotion['label']]:.0%})"
            ]

        flat_tags = [tag for cat_tags in tags.values() for tag in cat_tags]

        result = {
            "filename": os.path.basename(file_path),
            "status": "ok",
            "tags": tags,
            "flat_tags": flat_tags,
            "raw_features": {
                "opensmile_egemaps": osm_features,
                "pitch_librosa": pitch_features,
            },
        }
        if emotion is not None:
            result["raw_features"]["emotion_speechbrain"] = emotion

        if verbose:
            print(f"  OK  {os.path.basename(file_path)}: {flat_tags}")

        return result

    finally:
        if is_temp and os.path.exists(wav_path):
            os.unlink(wav_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def collect_audio_files(folder: str) -> list[str]:
    """Recursively collect supported audio files from folder."""
    files = []
    for root, _, names in os.walk(folder):
        for name in names:
            if os.path.splitext(name)[1].lower() in SUPPORTED_EXTENSIONS:
                files.append(os.path.join(root, name))
    files.sort()
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Voice-Tagger: extract vocal tags from audio files.",
    )
    parser.add_argument(
        "folder",
        help="Path to folder containing audio files (.wav, .mp3, .flac, .ogg)",
    )
    parser.add_argument(
        "-o", "--output",
        default="voice_tags.json",
        help="Output JSON path (default: voice_tags.json)",
    )
    parser.add_argument(
        "--skip-emotion",
        action="store_true",
        help="Skip SpeechBrain emotion model (faster, no GPU needed)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Force device: cuda or cpu (default: auto-detect)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print per-file progress",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"Error: '{args.folder}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load models ---
    print("Loading OpenSMILE eGeMAPSv02...")
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    emotion_classifier = None
    if not args.skip_emotion:
        print("Loading SpeechBrain emotion model (wav2vec2-IEMOCAP)...")
        from speechbrain.inference.interfaces import foreign_class

        emotion_classifier = foreign_class(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2Vec2Classifier",
            run_opts={"device": device},
        )
    else:
        print("Emotion model skipped (--skip-emotion)")

    # --- Collect files ---
    files = collect_audio_files(args.folder)
    if not files:
        print(f"No supported audio files found in '{args.folder}'.")
        sys.exit(0)

    print(f"Found {len(files)} audio file(s). Analysing...")
    t0 = time.time()

    # --- Process ---
    results = []
    success = 0
    failed = 0
    for fpath in files:
        try:
            result = process_file(
                fpath,
                smile,
                emotion_classifier=emotion_classifier,
                verbose=args.verbose,
            )
            results.append(result)
            success += 1
        except Exception as exc:
            failed += 1
            results.append({
                "filename": os.path.basename(fpath),
                "status": "error",
                "error": str(exc),
            })
            if args.verbose:
                print(f"  FAIL {os.path.basename(fpath)}: {exc}")

    elapsed = round(time.time() - t0, 1)

    # --- Write output ---
    output = {
        "metadata": {
            "total_files": len(files),
            "successful": success,
            "failed": failed,
            "elapsed_seconds": elapsed,
        },
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Done — {success} ok, {failed} failed, {elapsed}s elapsed.")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
