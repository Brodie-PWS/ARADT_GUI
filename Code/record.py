import pyaudio
import wave
import sys
import os
import time
from tkinter import messagebox

FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
WAVE_OUTPUT_FILENAME = "Samples/Recorded_Sample_{}.wav"

global filename_counter
filename_counter = 0

def record_audio(duration):
    audio = pyaudio.PyAudio()
    # Start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
    frames = []
    for i in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    # Stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Reformat Filename for Uniqueness
    timestr = time.strftime("%d-%m-%Y_(%H%M%S)")
    UNIQUE_OUTPUT_FILENAME = WAVE_OUTPUT_FILENAME.format(timestr)

    waveFile = wave.open(UNIQUE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

    print(f'Recording Saved To {UNIQUE_OUTPUT_FILENAME}')
