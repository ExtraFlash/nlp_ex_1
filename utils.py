import gensim.downloader as dl
import numpy as np
from matplotlib import pyplot as plt


# Q 1
def gen_lists_of_the_most_similar_words(model, words, topn):
    for word in words:
        print(f"Most similar words of {word}: {[w for w, _ in model.most_similar(word, topn=topn)]} \\")
        print("------------------------------------------------------------------------------")


def polysemous_words(model, first_group, second_group):
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
    set1 = {word for word, sim in modelA.most_similar(word)}
    set2 = {word for word, sim in modelB.most_similar(word)}
    return len(list(set1.intersection(set2))) / len(list(set1))


def ap(judged):
    ap_value = 0
    prec = 0
    count = 0
    for i in range(len(judged)):
        count += 1
        prec += (judged[i] == 1)
        ap_value += (prec / count) * judged[i]
    return ap_value / prec


def get_words(words, judgment):
    lst = []
    for i, w in enumerate(words):
        if judgment[i] == 1:
            lst.append(w)
    return lst


# if __name__ == '__main__':
# model = dl.load("word2vec-google-news-300")
# # this will take a while on first load as it downloads a 1.6G file.
# # later calls will be cached.
# # You can now use various methods of the “model“ object.
# # you can access the vocabulary like so:
# vocab = model.index_to_key
# polysemous_words()

# print("\\\")
if __name__ == '__main__':
    print(ap([0, 0, 1, 1, 0, 1, 1, 0, 1]))
    # z = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # lst = [1,1,0]
    # lst_np = np.array(lst)
    # z_ing = z[lst_np == 1]
    # z_ed = z[lst_np == 1]
    # plt.scatter(z_ing[:, 0], z_ing[:, 1], c='r', label='ing')
    # plt.scatter(z_ed[:, 0], z_ed[:, 1], c='b', label='ed')
    # plt.legend()
    # plt.show()
