import gensim.downloader as dl
import numpy as np
from matplotlib import pyplot as plt


# Q 1
def gen_lists_of_the_most_similar_words(model, words, topn):
    """
    this function prints the topn most similar words for each word in the list of words
    :param model: model loaded from gensim
    :param words: list of words
    :param topn: number of similar words to print for each word in the list of words
    :return: None
    """
    for word in words:
        print(f"Most similar words of {word}: {[w for w, _ in model.most_similar(word, topn=topn)]} \\")
        print("------------------------------------------------------------------------------")


def polysemous_words(model, first_group, second_group):
    """
    this function prints the 10 most similar words for each word in the first group and the second group
    :param model: model loaded from gensim
    :param first_group: first group of words
    :param second_group: second group of words
    :return: None
    """
    # get the most similar words for each word in the group
    print(f'First group: {first_group}')
    for word in first_group:
        print(f'Neighbours for {word}:')
        print(model.most_similar(word, topn=10))

    print(f'Second group: {second_group}')
    for word in second_group:
        print(f'Neighbours for {word}:')
        print(model.most_similar(word, topn=10))


def sim_models(modelA, modelB, word):
    """
    :param modelA: model loaded from gensim
    :param modelB: model loaded from gensim
    :param word: A string
    :return: amount of words that are in the top 10 most similar words in both models divided by the amount of words
    """
    set1 = {word for word, sim in modelA.most_similar(word)}
    set2 = {word for word, sim in modelB.most_similar(word)}
    return len(list(set1.intersection(set2))) / len(list(set1))


def ap(judged):
    """

    :param judged: list of 0's and 1's where 1 means relevant and 0 means not relevant
    :return: ap value
    """
    ap_value = 0
    prec = 0
    count = 0
    for i in range(len(judged)):
        count += 1
        prec += (judged[i] == 1)
        ap_value += (prec / count) * judged[i]
    return ap_value / prec


def get_words(words, judgment):
    """

    :param words: list of words
    :param judgment: list of 0's and 1's where 1 means relevant and 0 means not relevant
    :return: the list of words that are relevant according to the judgment list
    """
    lst = []
    for i, w in enumerate(words):
        if judgment[i] == 1:
            lst.append(w)
    return lst



