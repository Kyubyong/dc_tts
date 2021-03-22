# A guide to clone anyone's voice and use it as a text-to-speech with android 

<img align="right" src="https://github.com/simsax/Voice_cloner/blob/myChanges/demo.gif" width="216" height="384" />

<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#introduction">Introduction</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#data-preparation">Data preparation</a></li>
      </ul>
    </li>
    <li><a href="#training">Training</a></li>
    <li><a href="#testing">Testing</a></li>
    <li><a href="#creating-the-android-app">Creating the android app</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#notes">Notes</a></li>
  </ol>
</details>

<br />

---

## Introduction
This is a fun little project I made out of boredom. After seeing [Kyubyong's] text-to-speech model, I decided to create an android application that can read what I write with my own voice. If you copy the code and follow my steps, you'll be able to do the same.

[Kyubyong's]: https://github.com/Kyubyong/dc_tts
<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

This is the most important part. If you want this to work, make sure you have the following things installed. Since the model uses an old version of python and tensorflow, I suggest creating a virtual environment and install everything there.

* []() Python 3.6
* []() Tensorflow 1.15.0
* []() librosa
* []() tqdm
* []() matplotlib
* []() scipy
* []() Android Studio
* []() An android phone (?)
* []() Some time to lose

### Data preparation

In order to clone your voice you need around 200 samples of your voice, each one between 2-10 seconds. This means that you can clone anyone's voice with only 15-20 minutes of audio, thanks to transfer learning.
1. First, you need to download the [pretrained model] if you want to make an english voice. Otherwise, find an online text-to-speech dataset of the desired language and train the model from scratch. For example, I made an italian version of my voice, starting from [this] dataset. 
[Here] you can download the italian pre-trained model I generated. 
Make sure to put the pretrained model inside the 'logdir' directory.
2. Inside LJSpeech-1.1 you have to edit the transcript.csv file to match your audio samples. Each line must have this format: <audioFileName|original sentence|normalized sentence>, where the audio name is without the extension and the normalized sentence contains the conversion from numbers to words. Take a look at the original transcript.csv and you'll understand it easily. Then, copy your audio samples inside the wavs folder. If you want to make the data generation process less painful, I suggest writing the transcript file first, then record the sentences using record.py.

[pretrained model]: https://www.dropbox.com/s/1oyipstjxh2n5wo/LJ_logdir.tar?dl=0
[this]: https://www.caito.de/2019/01/the-m-ailabs-speech-dataset/
[here]: https://www.dropbox.com/s/y38m6hrucah3ua7/logdir.rar?dl=0

## Training

If you want to understand how the model works, you should read [this paper]. Otherwise, treat it as a black box and mechanically follow my steps.

1. Edit hyperparams.py and make sure that prepro is set to True. Also, edit the data path to match the correct location inside your local pc. Set the batch size to 16 or 32 depending on your ram. You can also tune max_N and max_T.
2. Run prepo.py only one time. After this step you should see two new folders, 'megs' and 'mals'. If you change dataset, then delete megs and mals and run the prepo.py again.
3. Run 'python train.py 1'. This is going to take a different amount of steps for each voice, but usually after 10k steps the result should already be decent.
4. Run 'python train.py 2'. You have to train it at least 2k steps, otherwise the voice will not sound human.

[this paper]: https://arxiv.org/abs/1710.08969

## Testing

Open harvard_sentences.txt and edit the lines as you desire. Then, run 'python synthesize.py'. If everything is correct, a 'samples' directory should appear. 

## Creating the android app

As you can see, it's not very comfortable to generate the sentences. That's why I decided to make this process more user-friendly.
The android app is basically just a wrapper that let you generate the audios, save them locally on the phone and share them.
When you write something and press the play button in the app, the message is sent to the server.py, that launches synthesize.py and then sends the audio back to the android application.
If you want to use the application outside your local network, make sure to set up the port forwarding, opening the access to the port written in the server.py. The default port is '1234'. You can change it if you want, but remember to change also the port in the MainActivity.java. You also have to set your ip address in the same file.
By default the model only computes sentences shorter than 10 seconds, but in the server.py I worked around this problem by splitting the input message into small sentences, then running the synthesize on every sentence and merging the resulting audios.

## Usage

Run 'python server.py' on your local pc. Then leave it on for as long as you need.

## Notes
* []() In case something is not clear or you bump into some weird error, don't be afraid to ask.
* []() This is my first android project, I had no prior experience on mobile development. So the code is probably not optimal, but it works.
