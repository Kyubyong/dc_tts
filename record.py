import pyaudio
import wave
import keyboard

# Press enter to start the recording, then press the keyboard to stop it

chunk = 1024  
sample_format = pyaudio.paInt16  
channels = 2
fs = 44100  
seconds = 11 # max duration

# read the file and cycle for each sentence
with open("transcript.csv", "r") as f:
    for line in f:
        tokens = line.split("|")
        index = tokens[0]
        sentence = tokens[1]
        p = pyaudio.PyAudio() 
        input(f"Next: {sentence}")
        print('Recording...')

        stream = p.open(format=sample_format,
                        channels=channels,
                        rate=fs,
                        frames_per_buffer=chunk,
                        input=True)

        frames = []  

        # Store data in chunks for 10 seconds
        for i in range(0, int(fs / chunk * seconds)):
            if (keyboard.is_pressed(' ')):
                break
            data = stream.read(chunk)
            frames.append(data)
 
        stream.stop_stream()
        stream.close()
        p.terminate()

        print('Finished recording')
        
        # Save the recorded data as a WAV file
        wf = wave.open(f"{index}.wav", 'wb')

        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))
        wf.close()