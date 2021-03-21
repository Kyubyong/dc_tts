# A guide to clone anyone's voice and use it as a text-to-speech with android 

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
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>




## Introduction
This is a fun little project I made out of boredom. After seeing [Kyubyong's] text-to-speech model, I decided to create an android application that can read what I write with my own voice. If you copy the code and follow my steps, you'll be able to do the same.

[Kyubyong's]: https://github.com/Kyubyong/dc_tts
<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

This is the most important part. If you want this to work, make sure you have the following things installed. Since the model uses and old version of python and tensorflow, I suggest creating a virtual environment and install everything there.

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
1. First, you need to download the [pretrained model] if you want to make an english voice. Otherwise, find an online text-to-speech dataset of the desired language and train the model from scratch. For example, I made an italian version of my voice, starting from [this] dataset. [Here] you can download the italian pre-trained model I generated. 
Make sure to put the pretrained model inside the 'logdir' directory.
2. Inside LJSpeech-1.1 you have to edit the transcript.csv file to match your audio samples. Each line must have this format: <audioFileName|original sentence|normalized sentence>, where the audio name is without the extension and the normalized sentence contains the conversion from numbers to words. Take a look at the original transcript.csv and you'll understand it easily. Then, copy your audio samples inside the wavs folder. If you want to make the data generation process less painful, I suggest writing the transcript file first, then record the sentences using record.py.

[pretrained model]: https://www.dropbox.com/s/1oyipstjxh2n5wo/LJ_logdir.tar?dl=0
[this]: https://www.caito.de/2019/01/the-m-ailabs-speech-dataset/
[here]:

## Training

If you want to understand how the model works, you should read [this]. Otherwise, treat it as a black box and mechanically follow my steps.

1. Edit hyperparams.py and make sure that prepro is set to True. Also, edit the data path to match the correct location inside your local pc.
2. Run prepo.py
3. Run 'python train.py 1'. This is going to take a different amount of steps for each voice, but usually after 10k steps the result should already be decent.
4. Run 'python train.py 2'. You have to train it at least 2k steps, otherwise the voice will not sound human.

[this]: https://arxiv.org/abs/1710.08969

## Testing

Open harvard_sentences.txt and edit the lines as you desire. Then, run 'python synthesize.py'. If everything is correct, a 'samples' directory should appear. 

## Creating the android app

As you can see, it's not very comfortable to generate the sentences. That's why I decided to make this process more user-friendly.
The android app is basically just a wrapper that let you generate the audios, save them locally on the phone and share them.
When you write something and press the play button in the app, the message is sent to the server.py, that launches synthesize.py and then sends the audio back to the android application.
If you want to use the application outside your local network, make sure to set up the port forwarding, opening the access to the port written in the server.py. The default port is '1234'. You can change it if you want, but remember to change also the port in the MainActivity.java

## Usage

Run 'python server.py' on your local pc. Then leave it on for as long as you need.

I have to add images and/or gifs now



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email

Project Link: [https://github.com/github_username/repo_name](https://github.com/github_username/repo_name)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* []()
* []()
* []()





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/github_username
