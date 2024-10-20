import time, os
import logging
import streamlit as st
import kapre
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
from PIL import Image
from settings import IMAGE_DIR, DURATION, WAVE_OUTPUT_FILE
from src.sound import sound
from src.model import CNN
from keras.models import load_model
from setup_logging import setup_logging

setup_logging()
logger = logging.getLogger('app')

class CustomSTFT(kapre.time_frequency.STFT):
    def call(self, inputs):
        # Apply the standard STFT
        stft_output = super().call(inputs)
        # Convert complex64 to magnitude (float32)
        return tf.abs(stft_output)

def init_model():
    cnn = load_model('models/final_best_model.h5', custom_objects={'STFT': kapre.time_frequency.STFT,
                                                            'Magnitude': kapre.time_frequency.Magnitude,
                                                            'ApplyFilterbank': kapre.time_frequency.ApplyFilterbank,
                                                            'MagnitudeToDecibel': kapre.time_frequency.MagnitudeToDecibel})
    return cnn

def get_spectrogram(type='mel'):
    logger.info("Extracting spectrogram")
    y, sr = librosa.load(WAVE_OUTPUT_FILE, duration=DURATION)
    ps = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    logger.info("Spectrogram Extracted")
    format = '%+2.0f'
    if type == 'DB':
        ps = librosa.power_to_db(ps, ref=np.max)
        format = ''.join[format, 'DB']
        logger.info("Converted to DB scale")
    return ps, format

def display(spectrogram, format):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, y_axis='mel', x_axis='time')
    plt.title('Mel-Spectrogram')
    plt.colorbar(format=format)
    plt.tight_layout()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(clear_figure=False)

def process_wav_file(y, sr):
    cnn = init_model()
    audio_data_processed = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    audio_data_processed = librosa.util.fix_length(audio_data_processed, size=5300, axis=1)
    audio_data_processed = audio_data_processed.transpose(1, 0)
    audio_data_processed = np.expand_dims(audio_data_processed, 0)
    result = cnn.predict(audio_data_processed)
    return result

def main():
    title = "Detecting Depression from Audio Samples"
    st.title(title)

    if st.button('Record'):
        with st.spinner(f'Recording for {DURATION} seconds ....'):
            sound.record()
        st.success("Recording completed")

    if st.button('Play'):
        # sound.play()
        try:
            audio_file = open(WAVE_OUTPUT_FILE, 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav')
        except:
            st.write("Please record sound first")

    if st.button('Classify'):
        cnn = init_model()
        with st.spinner("Classifying the audio recording"):
            y, sr = librosa.load(WAVE_OUTPUT_FILE)
            result = process_wav_file(y, sr)
        st.success("Classification completed")
        if (result >= 0.55):
            st.write(result)
            st.write('Depressed')
        else:
            st.write('Not Depressed')

    # Add a placeholder
    if st.button('Display Spectrogram'):
        # type = st.radio("Scale of spectrogram:",
        #                 ('mel', 'DB'))
        if os.path.exists(WAVE_OUTPUT_FILE):
            spectrogram, format = get_spectrogram(type='mel')
            display(spectrogram, format)
        else:
            st.write("Please record sound first")

if __name__ == '__main__':
    main()
    # for i in range(100):
    #   # Update the progress bar with each iteration.
    #   latest_iteration.text(f'Iteration {i+1}')
    #   bar.progress(i + 1)
    #   time.sleep(0.1)
