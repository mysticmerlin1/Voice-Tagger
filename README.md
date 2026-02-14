# Voice-Tagger

Standalone CLI tool that extracts comprehensive vocal tags from audio files using acoustic analysis and AI emotion detection.

## Features

- **88 eGeMAPSv02 features** via OpenSMILE (voice quality, spectral, formant, temporal)
- **Pitch tracking** via librosa pyin (f0 mean/std/min/max, voiced fraction)
- **Emotion classification** via SpeechBrain wav2vec2-IEMOCAP (ang/hap/sad/neu)
- **Human-readable tags** derived from acoustic features (Breathy, Clean, Warm, Dynamic, etc.)
- **Multi-format**: `.wav`, `.mp3`, `.flac`, `.ogg`
- **Per-file error handling**: bad files are reported, batch continues

## Requirements

- Python 3.9+
- [ffmpeg](https://ffmpeg.org/download.html) (required for non-WAV formats)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Basic — analyse all audio in a folder
python voice_tagger.py ./voiceovers

# Custom output path
python voice_tagger.py ./voiceovers -o results.json

# Skip emotion model (faster, no GPU needed)
python voice_tagger.py ./voiceovers --skip-emotion

# Force GPU
python voice_tagger.py ./voiceovers --device cuda

# Verbose per-file progress
python voice_tagger.py ./voiceovers -v
```

## Output

JSON file with this structure:

```json
{
  "metadata": { "total_files": 5, "successful": 4, "failed": 1, "elapsed_seconds": 12.3 },
  "results": [
    {
      "filename": "narrator_01.wav",
      "status": "ok",
      "tags": {
        "voice_quality": ["Clean"],
        "pitch": ["Mid Pitch", "Male Range"],
        "brightness_warmth": ["Warm/Mellow"],
        "energy_dynamics": ["Steady"],
        "formant_resonance": ["Ringy/Resonant"],
        "voicing_patterns": ["Long Sustained"],
        "emotion": ["neu (85%)"]
      },
      "flat_tags": ["Clean", "Mid Pitch", "Male Range", "Warm/Mellow", "Steady", "Ringy/Resonant", "Long Sustained", "neu (85%)"],
      "raw_features": {
        "opensmile_egemaps": { "...all 88 features..." },
        "pitch_librosa": { "f0_mean": 142.5, "f0_std": 18.3, "..." },
        "emotion_speechbrain": { "label": "neu", "scores": { "ang": 0.02, "hap": 0.05, "sad": 0.08, "neu": 0.85 } }
      }
    }
  ]
}
```

## Tag Categories

| Category | Example Tags |
|----------|-------------|
| Voice quality | Clean, Rough/Gravelly, Breathy, Hoarse, Creaky |
| Pitch | Low/Mid/High Pitch, Male/Female Range, Monotone, Expressive |
| Brightness/warmth | Warm/Mellow, Bright/Cutting, Rich/Full, Pressed/Tense |
| Energy/dynamics | Steady, Dynamic, Fast/Slow Pacing |
| Formant resonance | Nasal Hint, Open/Projected, Ringy/Resonant |
| Voicing patterns | Long Sustained, Short Choppy, Deliberate Pauses |
| Emotion | ang, hap, sad, neu (with probability) |
