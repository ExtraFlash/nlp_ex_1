import gensim.downloader as dl
import numpy as np
from matplotlib import pyplot as plt
from sklearn import decomposition
import utils


def q1(model):
    words = ["word", "cat", "pen", "chat", "door"]
    utils.gen_lists_of_the_most_similar_words(model, words, topn=20)


def q2(model):
    first_group = ['paper', 'buck', 'tie']
    second_group = ['clearly', 'court', 'orange']
    utils.polysemous_words(model, first_group, second_group)


def q2_1(model):
    first_group = ['paper', 'buck', 'tie']
    utils.gen_lists_of_the_most_similar_words(model, first_group, topn=10)


def q3(model):
    w1 = 'cold'
    w2 = 'frozen'
    w3 = 'hot'
    print(model.similarity(w1, w3) > model.similarity(w1, w2))


def q4(model_wiki, model_twitter):
    list_similar = ["dog", "red", "game", "east", "cake"]
    list_differ = ["the", "we", "a", "are", "is"]
    print('For the first group:')
    for word in list_similar:
        print("word:" + word, ",model similarity: " + str(utils.sim_models(model_wiki, model_twitter, word)))
    print('For the second group:')
    for word in list_differ:
        print("word:" + word, ",model similarity: " + str(utils.sim_models(model_wiki, model_twitter, word)))


def q5(model):
    # get the 5000 most frequent words
    vocab5k = model.index_to_key[1:5000]
    # the list of the embedding of each word
    list_embed = []
    # the list of the words
    id2word = []
    # list of 1's and 0's where 1 means the word ends with "ing" and 0 means the word ends with "ed"
    is_ing_list = []

    # go over the 5000 most frequent words and add the embedding of the words that end with "ing" or "ed"
    for word in vocab5k:
        if word.endswith("ing") or word.endswith("ed"):
            list_embed.append(model[word])
            id2word.append(word)
            if word.endswith("ing"):
                is_ing_list.append(1)
            else:
                is_ing_list.append(0)

    # convert the lists to numpy arrays
    list_embed = np.array(list_embed)
    is_ing_list = np.array(is_ing_list)

    # reduce the dimension of the embeddings to 2
    pca = decomposition.PCA(n_components=2)
    pca.fit(list_embed)  # use a set of vectors to learn the PCA transformation
    Z = pca.transform(list_embed)  # transform a set of vectors to reduce their dim

    # plot the embeddings
    Z_ing = Z[is_ing_list == 1]
    Z_ed = Z[is_ing_list == 0]
    plt.scatter(Z_ing[:, 0], Z_ing[:, 1], c="b", label='ing')
    plt.scatter(Z_ed[:, 0], Z_ed[:, 1], c="g", label='ed')
    plt.legend()
    plt.show()


def q9():
    semantic_juddgements_chat_gpt_cat = ([1] * 16) + ([0] * 4)
    semantic_juddgements_chat_gpt_pen = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0,  # 'penmanship'
                                         1, 1, 1, 1, 0, 0, 1, 1, 1, 1]

    topic_juddgements_chat_gpt_cat = [1] * 20
    topic_juddgements_chat_gpt_pen = [1] * 20

    semantic_juddgements_word2vec_cat = [1, 0, 1, 1, 0, 0, 0, 1, 1, 0,  # 'chiwawa'
                                         0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
    semantic_juddgements_word2vec_pen = [1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]

    topic_juddgements_word2vec_cat = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # 'chiwawa'
                                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    topic_juddgements_word2vec_pen = [1, 1, 1, 1, 0, 1, 0, 1, 1, 1,  # quill pen
                                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    ap_semantic_chat_gpt_cat = utils.ap(semantic_juddgements_chat_gpt_cat)
    ap_semantic_chat_gpt_pen = utils.ap(semantic_juddgements_chat_gpt_pen)

    map_semantic_chat_gpt = (ap_semantic_chat_gpt_cat + ap_semantic_chat_gpt_pen) / 2

    ap_topic_chat_gpt_cat = utils.ap(topic_juddgements_chat_gpt_cat)
    ap_topic_chat_gpt_pen = utils.ap(topic_juddgements_chat_gpt_pen)
    map_topic_chat_gpt = (ap_topic_chat_gpt_cat + ap_topic_chat_gpt_pen) / 2

    ap_semantic_word2vec_cat = utils.ap(semantic_juddgements_word2vec_cat)
    ap_semantic_word2vec_pen = utils.ap(semantic_juddgements_word2vec_pen)
    map_semantic_word2vec = (ap_semantic_word2vec_cat + ap_semantic_word2vec_pen) / 2

    ap_topic_word2vec_cat = utils.ap(topic_juddgements_word2vec_cat)
    ap_topic_word2vec_pen = utils.ap(topic_juddgements_word2vec_pen)
    map_topic_word2vec = (ap_topic_word2vec_cat + ap_topic_word2vec_pen) / 2

    print("MAP semantic chat-gpt: " + str(map_semantic_chat_gpt))
    print("MAP semantic word2vec: " + str(map_semantic_word2vec))
    print('---------------------------')
    print("MAP topic chat-gpt: " + str(map_topic_chat_gpt))
    print("MAP topic word2vec: " + str(map_topic_word2vec))

    gpt_cat = ["feline", "kitty", "kitten", "tabby", "moggy", "tomcat", "pussycat",
               "meow", "purr", "whiskers", "domesticated", "claws", "tail", "pet",
               "domestic cat", "Siamese", "lion", "tiger", "leopard", "panther"]
    gpt_pen = ["writing instrument", "pencil", "ink"
        , "ballpoint"
        , "quill"
        , "fountain pen"
        , "marker"
        , "biro"
        , "stylus"
        , "penmanship"
        , "writing"
        , "notepad"
        , "notebook"
        , "jotter"
        , "stationery"
        , "parchment"
        , "scribe"
        , "scribble"
        , "scrawl"
        , "doodle"]
    word2vec_cat = ['cats', 'dog', 'kitten', 'feline', 'beagle', 'puppy', 'pup', 'pet', 'felines', 'chihuahua', 'pooch',
                    'kitties', 'dachshund', 'poodle', 'stray_cat', 'Shih_Tzu', 'tabby', 'basset_hound',
                    'golden_retriever', 'Siamese_cat']
    word2vec_pen = ['pens', 'pencil', 'quill', 'ballpoint', 'prefilled_disposable_insulin', 'ballpoint_pen',
                    'Vera_Ramone', 'feather_quill', 'notepad', 'quill_pen', 'biro', 'fountain_pen',
                    'Anoto_functionality', 'Logitech_io2', 'io2', 'ballpoint_ink', 'scribble', 'ink', 'stubby_pencil',
                    'Wacom_Penabled']

    print("For the word 'cat':")

    print('Neighbours according to ChatGPT:')
    print(gpt_cat)
    print("-----------------------------------------------------------")
    print('Neighbours according to Word2Vec:')
    print(word2vec_cat)
    print("-----------------------------------------------------------")
    print('Our semantic judgment for ChatGPT:')
    print(utils.get_words(gpt_cat, semantic_juddgements_chat_gpt_cat))
    print("-----------------------------------------------------------")
    print('Our topical judgment for ChatGPT:')
    print(utils.get_words(gpt_cat, topic_juddgements_chat_gpt_cat))
    print("-----------------------------------------------------------")
    print('Our semantic judgment for Word2Vec:')
    print(utils.get_words(word2vec_cat, semantic_juddgements_word2vec_cat))

    print('Our topical judgment for Word2Vec:')
    print(utils.get_words(word2vec_cat, topic_juddgements_word2vec_cat))
    print()
    print("-----------------------------------------------------------")
    print()
    print("For the word 'pen':")

    print('Neighbours according to ChatGPT:')
    print(gpt_pen)
    print("-----------------------------------------------------------")
    print('Neighbours according to Word2Vec:')
    print(word2vec_pen)
    print("-----------------------------------------------------------")
    print('Our semantic judgment for ChatGPT:')
    print(utils.get_words(gpt_pen, semantic_juddgements_chat_gpt_pen))
    print("-----------------------------------------------------------")
    print('Our topical judgment for ChatGPT:')
    print(utils.get_words(gpt_pen, topic_juddgements_chat_gpt_pen))
    print("-----------------------------------------------------------")
    print('Our semantic judgment for Word2Vec:')
    print(utils.get_words(word2vec_pen, semantic_juddgements_word2vec_pen))
    print("-----------------------------------------------------------")
    print('Our topical judgment for Word2Vec:')
    print(utils.get_words(word2vec_pen, topic_juddgements_word2vec_pen))


# if __name__ == '__main__':
#     model = dl.load("word2vec-google-news-300")
#     model_wiki = dl.load("glove-wiki-gigaword-200")
#     model_twitter = dl.load("glove-twitter-200")
    q7()
