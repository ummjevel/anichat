import os
import argparse
from tqdm import tqdm
import random
import re
from random import shuffle
random.seed(1)


def korean_cleaner():
    pass


def random_deletion(words, p=0.1):

    # obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return words

	# randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

	# if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]

    return new_words


if __name__ == "__main__":
    # argparse
    sentence, p_rd=0.1,
    parser = argparse.ArgumentParser(description='Sentence Deletion')
    # sentence
    parser.add_argument('--sentence', '-s', type=str, help='sentence', default='안녕하세요? 반갑습니다. 저는 누굴까요?')
    # p_rd
    parser.add_argument('--p_rd', '-rd', type=str, help='sentence deletion p_rd', default='0.1')
    args = parser.parse_args()

    # sentence delete
    words = sentence.split(' ')

    print(random_deletion(words, p_rd))
