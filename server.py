import socket
import os
from playsound import playsound
from pydub import AudioSegment


def sendToClient(msg):
    msg = msg.decode('utf-8')
    lang = msg[:3] # ITA or ENG
    msg = msg[3:] # actual message
    words = msg.split(" ")
    if len(words) > 18:
        sentences = []
        sentence = ""
        for i in range(len(words)):
            sentence += words[i] + " "
            if i%12 == 0 and i != 0:
                sentences.append(sentence)
                sentence = ""
            elif i == len(words)-1:
                sentences.append(sentence)

        with open('harvard_sentences.txt','w') as f:
            first = True
            i = 1
            for sentence in sentences:
                if first:
                    f.write("first line\n1. "+str(sentence)+"\n")
                    first = False
                else:
                    f.write(f"{i}. {str(sentence)}\n")
                i += 1
        num_sentences = len(sentences)
    else:
        with open('harvard_sentences.txt','w') as f:
            f.write("first line\n1. "+str(msg)+"\n")
        num_sentences = 1

    if (lang == 'ITA'):
        os.system('python synthesize.py ITA')
    else:
        os.system('python synthesize.py ENG')

    sounds = 0
    for i in range(0, num_sentences):
        sounds += AudioSegment.from_wav(f"samples/{i+1}.wav")
    # increase volume by 10dB
    sounds += 10
    sounds.export("backup/final.wav", format="wav") 
    f.close()
    with open('backup/final.wav', 'rb') as f:
        audiob = f.read()
    clientsocket.send(audiob)
    clientsocket.close()
    f.close()
    

if __name__ == '__main__':
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("0.0.0.0", 1234))
    s.listen(5)

    while True:
        print("Waiting for connection...")
        clientsocket, address = s.accept()
        print(f"Connection from {address} has been established")
        msg = clientsocket.recv(2048)
        print(msg)
        sendToClient(msg)

