# Language Modelling
Complete work has been done on NLTK Brown Corpus. The models implemented in here are as listed below

### Language Modelling with out smoothing
* Unigram, bigram, trigram models with padding
* proof of Zipf's law
* log-likelihood and perplexity score for a few sentences

### Language Modelling with smoothing
* Laplacian or Additive smoothing with k in [0.0001, 0.001, 0.01, 0.1, 1]
* Good turing smoothing for bigram and trigram models

### Interpolation method
* Interpolation method has been applied to bigram model

#### You can look for code along with results in the ipython notebook
#### You can also build the models and find out the log-likelihood and perplexity score of different sentences by 
`python language_modelling.py <input-text-file-path>` <br>
e.g. `python language_modelling.py input.txt` <br> input.txt should have one sentence in each line


