# AmbianceNet

**A system for matching text with emotionally resonant music**

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/yeununchoo/AmbianceNet/main)

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/yeununchoo/AmbianceNet)

Team B612: Yeunun Choo, Theodora Kunicki, Jin Hyeok Noh

[CS 2470](https://brown-deep-learning.github.io/dl-website-2022/index.html), Group Project

Spring 2022

Brown University

## Overview

The purpose of this project is to build a system to match text with emotionally resonant music. To do this, we have utilized a shared emotional embedding space to "translate" emotions from a text context to music.

## Data

The text data is the [GoEmotions](https://ai.googleblog.com/2021/10/goemotions-dataset-for-fine-grained.html) dataset created by Google, distributed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

The music data is the [Google AudioSet](https://research.google.com/audioset/) dataset, distributed under the [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/).

We have also used the [NRC Valence, Arousal, and Dominance (NRC-VAD) Lexicon](https://saifmohammad.com/WebPages/nrc-vad.html).

## Environment

The Python version is 3.9.9, and the Pip version is 21.2.4. Here are the versions of the major packages used. 

```
numpy: 1.21.6
pandas: 1.4.2
tensorflow: 2.7.0
matplotlib: 3.5.1
jupyterlab: 3.3.4
```

See the [requirements.txt](https://github.com/yeununchoo/AmbianceNet/blob/main/requirements.txt) file for more.

## License

[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under the
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg


