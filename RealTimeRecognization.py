import os
from Extract import SpeechFeatureExtractor
import sounddevice as sd
import numpy as np
import wave
import tempfile
import matplotlib.pyplot as plt
from keras.models import load_model
import librosa
from sklearn.preprocessing import LabelEncoder
import time

# Function to record audio and save it to a file
def record_audio_to_file(output_file, seconds, rate=16000, channels=1):
    audio_format = np.int16 # Use 16-bit integer format for audio

    print("Recording...")

    # Record audio using SoundDevice
    recorded_audio = sd.rec(int(seconds * rate), samplerate=rate, channels=channels, dtype=audio_format)
    sd.wait()  # Wait for the recording to finish

    print("Finished recording")

    # Save the recorded audio to a file in WAV format
    wf = wave.open(output_file, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(np.dtype(audio_format).itemsize)
    wf.setframerate(rate)
    wf.writeframes(recorded_audio.tobytes())
    wf.close()

# Initialize the feature extractor

model_path = 'emotion_classifier3.h5'


# Record audio for a fixed duration and save it to a temporary file
record_duration = 5  # seconds
'''
with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav_file:
    record_audio_to_file(temp_wav_file.name, record_duration)

    # Use SpeechFeatureExtractor to get the feature vector as a string
    speech_feature_extractor = SpeechFeatureExtractor(temp_wav_file.name)
    features = speech_feature_extractor.features
    features = np.expand_dims(features, axis=-1)
    print(features)

    model = load_model('emotion_classifier.h5')

    label_encoder = LabelEncoder()
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    label_encoder.fit(emotion_labels)

    predictions = model.predict(np.array([features]))

    predicted_class = label_encoder.inverse_transform(np.argmax(predictions, axis=1))[0]
    print(f"Predicted emotion: {predicted_class}")

    # Clean up the temporary file
    os.unlink(temp_wav_file.name)
'''

label_encoder = LabelEncoder()
emotion_labels = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']
label_encoder.fit(emotion_labels)
model = load_model('emotion_classifier1.h5')

# Continuously record audio, predict emotion, and print the result
while True:

    # Create a temporary file to store the recorded audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav_file:
        # Record audio and save it to the temporary file
        record_audio_to_file(temp_wav_file.name, record_duration)
        speech_feature_extractor = SpeechFeatureExtractor(temp_wav_file.name)

        ############################## Demostration of audio file
        y, sr = librosa.load(temp_wav_file.name, sr=22050, duration=2.5, offset=0.5)

        plt.figure(figsize=(14, 5))
        plt.plot(y)
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.title('Waveform')
        plt.show()

        # Define analysis parameters
        n_fft = 2048
        hop_length = int ( sr * 0.01 ) # 10ms hop

        # Compute STFT
        D = librosa . stft ( y , n_fft = n_fft , hop_length = hop_length , window = 'hamming' )
        S_db = librosa . amplitude_to_db ( np . abs ( D ), ref = np . max )

        # Display spectrogram
        librosa . display . specshow ( S_db , sr = sr , hop_length = hop_length , x_axis = 'time' , y_axis = 'hz')
        plt.colorbar ( format = ' %+2.0f dB' )
        plt.title ( 'STFT spectrogram' )
        plt.tight_layout()
        plt.show()
        ##################################

        features = speech_feature_extractor.features
        features = np.expand_dims(features, axis=-1)

        predictions = model.predict(np.array([features]))

        predicted_class = label_encoder.inverse_transform(np.argmax(predictions, axis=1))[0]

        print(f'Predicted emotion: {predicted_class}')

        # Delete the temporary file
        os.unlink(temp_wav_file.name)

    # Add a 1-second delay between recordings and predictions
    time.sleep(1)
