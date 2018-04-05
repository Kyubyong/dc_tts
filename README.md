# A TensorFlow Implementation of DC-TTS: yet another text-to-speech model

I implement yet another text-to-speech model, dc-tts, introduced in [Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention](https://arxiv.org/abs/1710.08969). My goal, however, is not just replicating the paper. Rather, I'd like to gain insights about various sound projects.

## Requirements
  * NumPy >= 1.11.1
  * TensorFlow >= 1.3 (Note that the API of `tf.contrib.layers.layer_norm` has changed since 1.3)
  * librosa
  * tqdm
  * matplotlib
  * scipy

## Data

<img src="https://image.shutterstock.com/z/stock-vector-korean-alphabet-korean-hangul-pattern-693680611.jpg" height="200" align="right">
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/9c/Kate_Winslet_March_18%2C_2014_%28headshot%29.jpg/890px-Kate_Winslet_March_18%2C_2014_%28headshot%29.jpg" height="200" align="right">
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f6/Nick_Offerman_at_UMBC_%28cropped%29.jpg/440px-Nick_Offerman_at_UMBC_%28cropped%29.jpg" height="200" align="right">
<img src="https://image.shutterstock.com/z/stock-vector-lj-letters-four-colors-in-abstract-background-logo-design-identity-in-circle-alphabet-letter-418687846.jpg" height="200" align="right">

I train English models and an Korean model on four different speech datasets. <p> 1. [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/) <br/> 2. [Nick Offerman's Audiobooks](https://www.audible.com.au/search?searchNarrator=Nick+Offerman) <br/> 3. [Kate Winslet's Audiobook](https://www.audible.com.au/pd/Classics/Therese-Raquin-Audiobook/B00FF0SLW4/ref=a_search_c4_1_3_srTtl?qid=1516854754&sr=1-3) <br/> 4. [KSS Dataset](https://kaggle.com/bryanpark/korean-single-speaker-speech-dataset)

LJ Speech Dataset is recently widely used as a benchmark dataset in the TTS task because it is publicly available, and it has 24 hours of reasonable quality samples.
Nick's and Kate's audiobooks are additionally used to see if the model can learn even with less data, variable speech samples. They are 18 hours and 5 hours long, respectively. Finally, KSS Dataset is a Korean single speaker speech dataset that lasts more than 12 hours.


## Training
  * STEP 0. Download [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/) or prepare your own data.
  * STEP 1. Adjust hyper parameters in `hyperparams.py`. (If you want to do preprocessing, set prepro True`.
  * STEP 2. Run `python train.py 1` for training Text2Mel. (If you set prepro True, run python prepro.py first)
  * STEP 3. Run `python train.py 2` for training SSRN.

You can do STEP 2 and 3 at the same time, if you have more than one gpu card.

## Training Curves

<img src="fig/training_curves.png">

## Attention Plot
<img src="fig/attention.gif">

## Sample Synthesis
I generate speech samples based on [Harvard Sentences](http://www.cs.columbia.edu/~hgs/audio/harvard.html) as the original paper does. It is already included in the repo.

  * Run `synthesize.py` and check the files in `samples`.

## Generated Samples

| Dataset       | Samples |
| :----- |:-------------|
| LJ      | [50k](https://soundcloud.com/kyubyong-park/sets/dc_tts) [200k](https://soundcloud.com/kyubyong-park/sets/dc_tts_lj_200k) [310k](https://soundcloud.com/kyubyong-park/sets/dc_tts_lj_310k) [800k](https://soundcloud.com/kyubyong-park/sets/dc_tts_lj_800k)|
| Nick      | [40k](https://soundcloud.com/kyubyong-park/sets/dc_tts_nick_40k) [170k](https://soundcloud.com/kyubyong-park/sets/dc_tts_nick_170k) [300k](https://soundcloud.com/kyubyong-park/sets/dc_tts_nick_300k) [800k](https://soundcloud.com/kyubyong-park/sets/dc_tts_nick_800k)|
| Kate| [40k](https://soundcloud.com/kyubyong-park/sets/dc_tts_kate_40k) [160k](https://soundcloud.com/kyubyong-park/sets/dc_tts_kate_160k) [300k](https://soundcloud.com/kyubyong-park/sets/dc_tts_kate_300k) [800k](https://soundcloud.com/kyubyong-park/sets/dc_tts_kate_800k) |
| KSS| [400k](https://soundcloud.com/kyubyong-park/sets/dc_tts_ko_400k) |

## Pretrained Model for LJ

Download [this](https://www.dropbox.com/s/1oyipstjxh2n5wo/LJ_logdir.tar?dl=0).

## Notes

  * The paper didn't mention normalization, but without normalization I couldn't get it to work. So I added layer normalization.
  * The paper fixed the learning rate to 0.001, but it didn't work for me. So I decayed it.
  * I tried to train Text2Mel and SSRN simultaneously, but it didn't work. I guess separating those two networks mitigates the burden of training.
  * The authors claimed that the model can be trained within a day, but unfortunately the luck was not mine. However obviously this is much fater than Tacotron as it uses only convolution layers.
  * Thanks to the guided attention, the attention plot looks monotonic almost from the beginning. I guess this seems to hold the aligment tight so it won't lose track.
  * The paper didn't mention dropouts. I applied them as I believe it helps for regularization.
  * Check also other TTS models such as [Tacotron](https://github.com/kyubyong/tacotron) and [Deep Voice 3](https://github.com/kyubyong/deepvoice3).
  
