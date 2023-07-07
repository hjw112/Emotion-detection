import openai

openai.api_key = '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'
audio_file= open("/Users/hujiawei/Downloads/AudioWAV/1002_IEO_FEA_HI.wav", "rb")
transcript = openai.Audio.transcribe("whisper-1", audio_file)
print(transcript)
#openai.Model.list()
