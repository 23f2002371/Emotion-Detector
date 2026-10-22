import os
import tempfile

import streamlit as st

from emotion import predict

EMOJI = {"angry": "😠", "disgust": "🤢", "fear": "😨",
         "happy": "😊", "neutral": "😐", "sad": "😢"}

st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️",
                   layout="centered")

st.title("🎙️ Speech Emotion Recognition")
st.write("Record or upload a short speech clip and the model will estimate its dominant emotion.")
st.caption("Best results come from 1-4 seconds of clear English speech.")

if "audio_reset" not in st.session_state:
    st.session_state.audio_reset = 0

audio = st.audio_input("Record a speech clip", key=f"audio_input_{st.session_state.audio_reset}")
uploaded = st.file_uploader(
    "Or upload an audio file",
    type=["wav", "mp3", "m4a", "ogg", "flac"],
    key=f"audio_upload_{st.session_state.audio_reset}",
)
audio_file = audio or uploaded

if audio_file is not None:
    st.audio(audio_file)
    if st.button("Clear audio", use_container_width=True):
        st.session_state.audio_reset += 1
        st.rerun()

    if st.button("Analyze emotion", type="primary", use_container_width=True):
        suffix = os.path.splitext(getattr(audio_file, "name", "audio.wav"))[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(audio_file.getvalue())
            temp_path = temp_file.name

        try:
            with st.spinner("Extracting speech features..."):
                label, ranking = predict(temp_path)
            st.success(f"Predicted emotion: {EMOJI.get(label, '')} {label.title()}")
            st.subheader("Model scores")
            st.bar_chart({emotion: score for emotion, score in ranking}, horizontal=True)
            st.caption("Scores are raw SVM scores, not probabilities. Higher means the model favours that emotion more.")
        except Exception as error:
            st.error(f"Could not process this audio: {error}")
        finally:
            os.unlink(temp_path)
else:
    st.info("Record or upload a short speech clip to begin.")

with st.expander("About this model"):
    st.write("This model predicts angry, disgust, fear, happy, neutral, or sad speech. "
             "It was trained on acted English speech and achieved about 0.73 macro-F1 "
             "on speakers it had not seen during training. Real-world audio may be less reliable.")
