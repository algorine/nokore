# Nokore - A Collection of Scripts for Detecting Spammy, Fake or Otherwise Dangerous Communications Online

This repository applies SIMOn -- a Character-level CNN + bidirectional LSTM modeling library for text classification -- to email spam classification, and other forms of social media "dangerous/fake/spammy" communication detection. 

The repository achieves this via a collection of scripts, with the eventual goal of comparing the SIMOn-based-model to emerging giants BERT and/or ELMo. 

The name "Nokore" is a Twi word, by the Akan people of Ghana, for "Truth".

Architecture is described at https://arxiv.org/abs/1901.08456

Review https://github.com/algorine/simon

Also See Texas AI Summit Talk video: https://youtu.be/SmIsWF1xBeI

# Getting Started

To get started, make sure you are using python v3.5+ and pip install via

`pip3 install git+https://github.com/algorine/simon`

Then, install `keras-bert` using `pip install -q keras-bert`.

Then, study the scripts and pretrained models included in the Nokore/scripts directory.

Rendered Jupyter notebooks are also provided in the Nokore/scripts directory, and they are meant to be self-explanatory.

