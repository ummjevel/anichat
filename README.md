# Project anichat

welcome to project anichat.



# Concept

캐릭터 음성챗봇 개발

## Member
* [전민정](https://github.com/ummjevel)
* [이현중](https://github.com/slaustld)
* [이연교](https://github.com/Lyeon1)
* [박영준](https://github.com/jooney-ai)
* [곽형민](https://github.com/Zeropo22)


## Time Table

날짜 | 계획 | 세부사항
---|:---:|---:
01/03 ~ 01/15 | 음성데이터 수집 | 
01/04 ~ 01/13 | 레퍼런스 공부 |
01/09 | 레퍼런스 설명회 |
01/11 ~ 01/27 | 음성모델 1차 학습 |
01/16 | 레퍼런스 설명회 |
01/19 ~ 01/28 | 챗봇 데이터 생성 |
01/30 ~ 02/04 | 음성 모델 성능 평가 및 개선 |
01/30 ~ 02/05 | 챗봇 학습 |
02/06 ~ 02/12 | 최종 성능 개선 및 서빙 |
02/13 | 모델 서빙 완료 |
02/13 ~ 02/17 | ppt 및 대본 작성 |
02/17 | 발표 |

# Data

We selected one animation, Detective Conan.
and chose 2 characters. Conan and Mori Cogoro.

we collect our data, conan and mori voices and script.

## TTS Data
first training data amount is shown in the following table.

second training data amount is shown in the following table.

## Chatbot Data

## STT Data

we didn't training STT. we just use whisper. added it to our pipeline.

# Model

## TTS
[VITS](https://github.com/jaywalnut310/vits)

### modification
because vits source code is for english so we add korean cleaner from [here](https://github.com/jaywalnut310/vits/pull/104/commits/4ebc313eecdfed04016618c3dfad055f95e2ac15) and add symbols. we had to modify vits/text/symbols.py because we use korean and [IPA](https://en.wikipedia.org/wiki/International_Phonetic_Alphabet) so there is several things to add alphabet.

## Chatbot
[Poly-encoder](https://github.com/chijames/Poly-Encoder)

## STT
[Whisper](https://github.com/openai/whisper)






## utils
* **split-video-with-srt.py** : using [pysrt](https://github.com/byroot/pysrt) library, spliting one video to several videos using srt file's timestamps.
```python
  python split-video-with-srt.py -i [mp4 file] -if [srt file] -o [save folder]
```
* **make-dataset.py** : this file makes text file for vits model training. you can make text file for single or multispeakers. but you need to use vits/preprocessing.py to make .txt.cleaned file. that is real text file used by model.

* **text-augmentation.py** : for chatbot model, we generated texts but we think it is not enough for model. so we find some text augmentation techniques from [here](https://amitness.com/2020/05/data-augmentation-for-nlp/). and we want to use the RD from [this paper](https://arxiv.org/abs/1901.11196). but we think random deletion is kind of risky in our data so we made adv deletion. because our language is korean so we use [konlpy](https://konlpy.org/en/latest/) [mecab](https://pypi.org/project/mecab-python3/) for pos tagging.

* **text2speech2text.py** : because we have to load several models and use them, we have to test. it is test python file for using 3 models in same environment.

# Environment
