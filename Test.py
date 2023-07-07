import numpy as np
from Extract import SpeechFeatureExtractor
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder


model = load_model('emotion_classifier.h5')

label_encoder = LabelEncoder()
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
label_encoder.fit(emotion_labels)

file = '/Users/hujiawei/Desktop/Final_Project/OriginalDataset/neutral/neutral262.wav'
speech_extractor = SpeechFeatureExtractor(audio_file=file)
features = speech_extractor.features
features = np.expand_dims(features, axis=-1)

predictions = model.predict(np.array([features]))

predicted_class = label_encoder.inverse_transform(np.argmax(predictions, axis=1))[0]
print(f"Predicted emotion: {predicted_class}")
