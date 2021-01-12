import pyaudio
import wave
import sys
import os
import time
import tkinter as tk

FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "Samples/Recorded_Sample_{}.wav"

global filename_counter
filename_counter = 0

def record_audio(amount):
    if not amount:
        amount = 1

    for x in range(amount):
        audio = pyaudio.PyAudio()
        # Start Recording
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
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

# def play_audio(fname):
#     wf = wave.open(fname, 'rb')
#     p = pyaudio.PyAudio()
#
#     # Open stream (2)
#     stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
#                     channels=wf.getnchannels(),
#                     rate=wf.getframerate(),
#                     output=True)
#
#     # Read data
#     data = wf.readframes(CHUNK)
#
#     # Play stream (3)
#     while len(data) > 0:
#         stream.write(data)
#         data = wf.readframes(CHUNK)
#
#     # Stop stream (4)
#     stream.stop_stream()
#     stream.close()
#
#     # Close PyAudio (5)
#     p.terminate()
