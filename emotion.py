import os

import joblib
import librosa
import numpy as np
import torch
from transformers import AutoModel

HERE = os.path.dirname(os.path.abspath(__file__))
_bundle = joblib.load(os.path.join(HERE, "emotion_svm.pkl"))
CFG = _bundle["config"]
SCALER = _bundle["scaler"]
SVM = _bundle["svm"]
NAMES = _bundle["label_names"]

# WavLM, cut after the last layer we use (the layers above it are never needed)
_ssl = AutoModel.from_pretrained(CFG["ssl_model"]).eval()
_ssl.encoder.layers = _ssl.encoder.layers[: max(CFG["layers"])]


def prep_audio(path):
    sr = CFG["sample_rate"]
    y, _ = librosa.load(path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=CFG["trim_top_db"])
    max_len = CFG["max_seconds"] * sr
    if len(y) > max_len:                      # keep the middle of long clips
        s = (len(y) - max_len) // 2
        y = y[s:s + max_len]
    if len(y) < int(0.3 * sr):
        raise ValueError("Audio is too short: need at least 0.3 s of speech.")
    y = y / (np.abs(y).max() + 1e-8)          # peak-normalise, as in training
    return y.astype(np.float16).astype(np.float32)


def embed(y):
    x = torch.from_numpy(y)[None]
    with torch.no_grad():
        hs = _ssl(x, output_hidden_states=True).hidden_states
    stats = [torch.cat([hs[L][0].mean(0), hs[L][0].std(0)]) for L in CFG["layers"]]
    return torch.stack(stats).mean(0).numpy()   # 1536 numbers, layers averaged


def predict(path):
    z = SCALER.transform(embed(prep_audio(path))[None])
    scores = SVM.decision_function(z)[0]
    ranking = [(NAMES[i], float(scores[i])) for i in np.argsort(-scores)]
    return NAMES[int(SVM.predict(z)[0])], ranking
