import argparse
from tqdm import tqdm
import random
from konlpy.tag import Mecab
from random import shuffle

mecab = Mecab()

'''
pip install konlpy
sudo apt-get install g++ openjdk-8-jdk python3-dev python3-pip curl
'''

# delete MAG(일반부사)
def adv_deletion(sentence):
    pos_values = mecab.pos(sentence)
    print(pos_values)
    pos_advs = [text for text, po in pos_values if po == 'MAG'] 
    print(pos_advs)
    # if MAG is several, delete random one.
    if len(pos_advs) >= 1:
        sentence = sentence.replace(random.choice(pos_advs), '')
    else:
        sentence = ''
    return sentence


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
    parser = argparse.ArgumentParser(description='Sentence Deletion')
    # sentence
    parser.add_argument('--sentence', '-s', type=str, help='sentence', default='쿠레노 지로가 어떻게 죽었는지 알려줘')
    # p_rd
    parser.add_argument('--p_rd', '-rd', type=str, help='sentence deletion p_rd', default='0.1')
    args = parser.parse_args()

    # sentence delete
    words = args.sentence.split(' ')
    # print(random_deletion(words, p_rd))
    print(adv_deletion(args.sentence))
