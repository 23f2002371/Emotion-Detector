---
title: Speech Emotion Recognition
emoji: 🎙️
colorFrom: indigo
colorTo: pink
sdk: streamlit
sdk_version: 1.49.1
python_version: "3.11"
app_file: app.py
pinned: false
---

# Speech Emotion Recognition

Run locally with:

```bash
streamlit run app.py
```

Upload or record a short speech clip and the model predicts one of six emotions:
angry, disgust, fear, happy, neutral or sad.

## How it works

1. The clip is resampled to 16 kHz mono, trimmed of silence, peak-normalised and cropped to at most 4 s.
2. WavLM base-plus (a general speech model, not trained on emotion labels) turns it into features:
   mean and standard deviation over time of layers 5 to 7, averaged over those layers.
3. A standard scaler and an RBF-kernel SVM pick the emotion.

## Accuracy

Macro F1 of about 0.73 on 1,429 clips from 19 speakers never seen in training (chance is 0.17).
Disgust is the hardest emotion. The model uses no speaker normalisation, so a single clip is enough.

## Limitations

- Trained on four acted-speech datasets (CREMA-D, RAVDESS, TESS, SAVEE), all English.
  Real-world, noisy, spontaneous or non-English speech will probably be less accurate.
- Clips longer than 4 s are cropped to the middle 4 s.
- Guessing emotion from voice is uncertain. Do not use this for decisions about people.
