
# NLP assignment 1

#### Authors: Nadav Cohen 209356526, Or Shkuri 322549114

Our project contains two main files: `utils.py` and `script.py`.

- `utils.py` contains all the utility functions:
- `script.py` contains all the relevant code for printing and plotting the results for each question.

## utils.py
This file contains five utility functions:
- `gen_lists_of_the_most_similar_words(model, words, topn)`: prints the topn most similar words for each word in the list of words
- `polysemous_words(model, first_group, second_group)`: prints the 10 most similar words for each word in the first group and the second group
- `sim_models(modelA, modelB, word)`: returns the amount of words that are in the top 10 most similar words in both models divided by the amount of words.
- `ap(judged)`: returns the ap value according the the judged list
- `get_words(words, judgment)`: returns the list of words that are relevant according to the judgment list

## script.py
This is the main file in our project.
In order to run the code, you need to import the following packages:
```python
import gensim.downloader as dl
import numpy as np
from matplotlib import pyplot as plt
from sklearn import decomposition
import utils
```

After that, you need to load the models from `gensim`:
```python
if __name__ == '__main__':
    model = dl.load("word2vec-google-news-300")
    model_wiki = dl.load("glove-wiki-gigaword-200")
    model_twitter = dl.load("glove-twitter-200")
```

Finally, let's overview the main functions that are used for printing and plotting our results.
- Generating lists of the most similar words: for this part use `def q1(model)`
- Polysemous words: for this part use `def q2(model)` and `def q2_1(model)`
- Synonyms and Antonyms: for this part use `def q3(model)`
- The Effect of Different Corpora: for this part use `def q4(model_wiki, model_twitter)`
- Plotting words in 2D: for this part use `def q5(model)`
- Mean Average Precision (MAP) evaluation: for this part use `def q9()`