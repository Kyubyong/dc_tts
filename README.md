# A TensorFlow Implementation of DC-TTS: yet another text-to-speech model

I implement yet another text-to-speech model, dc-tts, introduced in [Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention](https://arxiv.org/abs/1710.08969). My goal, however, is not just replicating the paper. Rather, I'd like to gain insights about various sound projects.

## Requirements
  * NumPy >= 1.11.1
  * TensorFlow == 1.3 (Note that the API of `tf.contrib.layers.layer_norm` has changed since 1.3)
  * librosa
  * tqdm
  * matplotlib
  * scipy

## Data

I test the model on three different speech datasets.
  1. [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/)
  2. [Nick Offerman's Audiobooks](https://www.audible.com.au/search?searchNarrator=Nick+Offerman)
  3. [Kate Winslet's Audiobook](https://www.audible.com.au/pd/Classics/Therese-Raquin-Audiobook/B00FF0SLW4/ref=a_search_c4_1_3_srTtl?qid=1516854754&sr=1-3)

LJ Speech Dataset is recently widely used as a benchmark dataset in the TTS task because it is publicly available, and it has 24 hours of reasonable quality samples.
Nick's and Kate's audiobooks are additionally used to see if the model can learn even with less data, variable speech samples. They are 18 hours and 5 hours long, respectively.


## Training
  * STEP 0. Download [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/) or prepare your own data.
  * STEP 1. Adjust hyper parameters in `hyperparams.py`.
  * STEP 2. Run `python train.py 1` for training Text2Mel.
  * STEP 3. Run `python train.py 2` for training SSRN.

You can do STEP 2 and 3 at the same time, if you have more than one gpu card.

## Sample Synthesis
I generate speech samples based on [Harvard Sentences](http://www.cs.columbia.edu/~hgs/audio/harvard.html) as the original paper does. It is already included in the repo.

  * Run `synthesize.py` and check the files in `samples`.

## Generated Samples

| Dataset       | Sample location (recorded global step) |
| :----- |:-------------|
| LJ      | [50k](https://soundcloud.com/kyubyong-park/sets/dc_tts) |
| Nick      | [40k](https://soundcloud.com/kyubyong-park/sets/dc_tts_nick_40k) |
| Kate| [40k](https://soundcloud.com/kyubyong-park/sets/dc_tts_kate_40k)      |



