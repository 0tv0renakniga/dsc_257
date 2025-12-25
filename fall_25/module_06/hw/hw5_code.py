# %% [markdown]
# ## <font color='blue'>Text generation using an n-gram model</font>
# 
# In this problem, we'll train a language model on Shakespeare and use it to generate text. Our model will be bare-bones but has the same underlying ideas as much more advanced models that are in wide use. Simply put, language models assign a likelihood (probability) to a given sentence. For example, the sentence "dog is barking" has a higher likelihood than "bark dog is", which is grammatically incoherent. 
# 
# ### <font color='blue'>Bigram and trigram models</font>
# 
# Suppose we have a sentence composed of words (or tokens) $x_1, x_2, ..., x_n$. We want to be able to compute the probability of this sequence, $P(x_1, x_2, ... x_n)$. The distribution can always be factored as
# 
# $$  P(x_1, x_2, ... x_n)  =  P(x_1) \prod_{i=2}^{n}P(x_i|x_{i-1}, x_{i-2}, ..., x_1) .$$
# 
# However, these conditional probabilities are hard to model for even medium-sized vocabularies. One simplification is to make a <em>first-order Markov assumption</em>, which says that the probability of a word given all previous words is equal to the probability of the word given just the one word before it, that is,
# 
# $$ P(x_i | x_1, x_2, ..., x_{i-1}) = P(x_i | x_{i-1}). $$
# 
# This allows us to factor
# 
# $$ P(x_1, x_2, ... x_n)  = \prod_{i=1}^{n}P(x_i|x_{i-1}) $$
# 
# where we take $x_0$ to be a special "START" symbol.
# 
# The above formulation is called a <em>bigram</em> language model, since it is based on pairs of consecutive words. If we expand the context to include the two previous words, we get a <em>trigram</em> model,
#     
# $$ P(x_1, x_2, ... x_n)  = \prod_{i=1}^{n}P(x_i|x_{i-2}, x_{i-1}) $$
# 
# where $x_{-1}, x_0$ are special start symbols. 
# 
# In the same way, we can define an <em>n-gram</em> model by expanding the context to include the $n-1$ previous words.
# 
# ### <font color='blue'>Learning an $n$-gram model</font>
# 
# In this problem, we will use <em>maximum-likelihood estimation</em> to learn an $n$-gram model. For this, we simply need to count the number of occurrences of each $n$-gram and $n-1$ gram in the corpus.
# 
# For instance, to train a trigram model, let $c(u,v,w)$ be the number of times that trigram $(u,v,w)$ is seen in the training corpus and let $c(u,v)$ be the number of times that bigram $(u,v)$ is seen. The maximum likelihood estimate of $P(w|u,v)$ is then 
# $$ P(w|u,v) = \frac{c(u,v,w)}{c(u,v)} .$$
# Of course, we might want to smooth this using Laplace smoothing or something similar.

# %% [markdown]
# ### <font color='blue'>Reading in the data</font>
# 

# %%
import string
import random

# %% [markdown]
# The provided text file `shakespeare.txt` contains a selection of Shakepeare's sonnets and plays. It also contains stage directions, copyright notices, and various other things that we won't bother removing. You should take a quick look at this file. We will read in a list of all sentences in the file.

# %%
sentences = []
with open("shakespeare.txt", "r", encoding="utf8") as f:
    text = f.read()
    text = text.split(".")
    for sentence in text:
        sentence += "."
        sentences.append(sentence)

# %% [markdown]
# Here is an example of one of the sentences.

# %%
sentences[200]

# %% [markdown]
# Let's set aside the very first sentence (we'll use it later). We'll extract all words from the remainder.

# %%
test_play = sentences[0]
sentences = sentences[1:]

# %%
words = set()
for sent in sentences:
    for word in sent.split():
        words.add(word)
vocab = list(words)

# %% [markdown]
# ### <font color='blue'>Class: NGramModel</font> 
# 
# Next, we define a class that will hold an n-gram model. Go through the methods of this class and make sure you understand what is happening in each method as you'll be asked to use these later. Also take note of the data structures used in various methods.
# 
# Some notes:
# * The `generate_n_grams` method takes a sequence (list) of tokens and return all of its ngrams. For example, for `tokens = ['Unsupervised',  'Learning', 'is', 'fun', '.']`, the trigrams would be `[(['START', 'START'], 'Unsupervised'), (['START', 'Unsupervised'], 'Learning') ...]`.
# * The `get_prob` method takes in a list of context tokens and a target token and returns the probability of the target given the context.
# 

# %%
from typing import List, Dict, Tuple

class NGramModel(object):
    def __init__(self, n: int) -> None:
        self.n = n
        self.n_grams = dict() # Stores unique n-grams
        self.context_count = dict() # Stores count of contexts
        self.ngram_count = dict() # Stores count of ngrams: (context, token)

    def tokenize(self, text: str) -> List[str]:
        # Tokenize the given sentence 'text'
        # Treat punctuation as a separate token.
        # Add space before punctuation.
        # Split using spaces.

        for ch in string.punctuation:
            text = text.replace(ch, " " + ch)
        tokens = text.strip().split()

        return tokens

    def generate_n_grams(self, tokens: List[str]) -> List[Tuple[List[str], str]]:
        # Generate all n-grams from the given list of tokens
        # Insert (n-1) <START> tokens before each sentence
        tokens = (self.n-1)*["<START>"] + tokens
        n_grams = list()

        """
        n_grams is a list where each element is
        ([n-1 context tokens], token)
        """
        for i in range(len(tokens)-self.n+1):
            n_grams.append((tokens[i: i+self.n-1], tokens[i+self.n-1]))

        return n_grams

    def fit(self, text: str) -> None:
        # Takes a sentence 'text'
        # Generates all n-grams in the sentence
        # Then updates counts
        new_n_grams = self.generate_n_grams(self.tokenize(text))
        for context, target in new_n_grams:
            # Add context to context dict and store count 
            if tuple(context) in self.context_count:
                self.context_count[tuple(context)] += 1.0
            else:
                self.context_count[tuple(context)] = 1.0
                
            # Save unique n_grams.
            # This part is used for generation only.
            if tuple(context) in self.n_grams:
                if target not in self.n_grams[tuple(context)]:
                    self.n_grams[tuple(context)].append(target)
            else:
                self.n_grams[tuple(context)] = [target]

            # Store n_gram counts
            new_n_gram = (tuple(context), target)
            if new_n_gram in self.ngram_count:
                self.ngram_count[new_n_gram] += 1.0
            else:
                self.ngram_count[new_n_gram] = 1.0
                
            

    def get_prob(self, context: List[str], target: str) -> float:
        """
        Calculates the probability of the target token
        given the context.
        """
        prob = self.ngram_count[(tuple(context), target)] / self.context_count[tuple(context)]
        return prob
        
        
    
    def predict_token(self, context: List[str]) -> str:
        """
        Predict the next token given context.
        A slight randomness ensures we generate a diverse token
        with the same context.
        """
        r = random.random()
        # store the probability of each token.
        token_probs = dict()
        
        try:
            tokens_of_interest = self.n_grams[tuple(context)]
            for token in tokens_of_interest:
                token_probs[token] = self.get_prob(context, token)
        except KeyError: # similar to Laplace smoothing; returns a random word from the vocab.
            ridx = random.randint(0, len(words))
            return vocab[ridx]
            
        
        sum = 0.0
        for key in sorted(token_probs):
            sum += token_probs[key]
            # When the probability sum is > random number
            # return the current token.
            if sum > r:
                return key


# %% [markdown]
# ### <font color='blue'>Routines for fitting data and generating text</font>

# %% [markdown]
# `create_and_fit_model` defines an n-gram model and fits it to data. Its parameters are `n` (the order of the model) and `sentences` (collection of sentences).  

# %%
def create_and_fit_model(n, sentences):
    """
    This is the key function that defines and fits an n-gram model.
    It takes in n and a list of sentences.
    It creates an n-gram model and then calls the `fit` method on 
    one sentence at a time to generate counts.
    """
    model = NGramModel(n)
    for sent in sentences:
        model.fit(sent)
    return model

# %% [markdown]
# `generate_text` uses an n-gram model to generate text, starting from a prompt (which might be empty). It takes as input the `model`, the number of words to generate (`n_outs`) and the `prompt`.

# %%
def generate_text(model: NGramModel, n_outs: int, prompt=None) -> str:
    """
    Generates n_outs words using the trained
    ngram model.
    """
    n = model.n
    # All sentence are initialized with the <START> token
    
    if prompt is not None:
        prompt_tokens = model.tokenize(prompt)
        context_queue = prompt_tokens[-(n-1):]
    else:
        context_queue = (n-1) * ["<START>"]
    result = list()

    for _ in range(n_outs):
        pred_token = model.predict_token(context_queue)
        result.append(pred_token)

        context_queue.pop(0)

        if pred_token == ".":
            # If sentence done. Start a new sentence.
            context_queue = (n-1) * ["<START>"]
        else:
            context_queue.append(pred_token)

    return " ".join(result)


# %%
def print_generated_text(model):
    """
    Prints a 100-word blurb from the provided model.
    """
    num_gen_words = 100
    print(f'{"="*50}\nGenerated text:')
    print("\n")
    print(generate_text(model, num_gen_words))
    print(f'{"="*50}')

# %% [markdown]
# ### <font color='blue'>Experimenting with text generation</font>
# 
# <font color='magenta'>Part (a): Use `create_and_fit` to create n-gram models for n=2 and n=3 and fit them to the provided text from Shakespeare. Then use `print_generated_text` to generate text from each of the two models. Report the two resulting texts.</font>

# %%
## SOLUTION:
import random
random.seed(42)

# 1. Create and fit the bigram (n=2) model
print("--- Fitting Bigram Model (n=2) ---")
bigram_model = create_and_fit_model(n=2, sentences=sentences)

# 2. Generate text from the bigram model
print_generated_text(bigram_model)

# 3. Create and fit the trigram (n=3) model
print("\n--- Fitting Trigram Model (n=3) ---")
trigram_model = create_and_fit_model(n=3, sentences=sentences)

# 4. Generate text from the trigram model
print_generated_text(trigram_model)

# %% [markdown]
# ### <font color='blue'>Computing likelihoods</font>
# 
# In the part, we'll calculate the log-likelihood of a sentence using the models that we just created. This will give us a concrete indication of why incorporating more context is helpful.
# For a given sentence $s$ with $n$ tokens $(x_1, \ldots, x_n)$, the log-likelihood is defined as
# $$ L(s) = \sum_{i=1}^{n}\log p(x_i|\mbox{context}) $$
# For a unigram model, this maximum-likelihood estimates of $p(x_i)$ would simply be
# $$ p(x_i) = \frac{c(x_i)}{N}$$ 
# where N is the total number of words in the dataset and $c(x_i)$ is the number of occurrences of $x_i$.  
# Similarly, for a bigram model, we have
# $$ p(x_i|\mbox{context}) = \frac{c(x_{i-1}, x_i)}{c(x_{i-1})} .$$
# In our code, these probabilities are provided by `get_prob` in `NGramModel`.
# 
# <font color='magenta'>Part (b): Write a function that takes an `NGramModel` and a sentence and returns the log-likelihood of that sentence. </font>

# %%
def model_ll(model, text):
    """
    Takes in an n-gram model and a sample text.
    Returns the log-likelihood of the text under the given model.
    """
    import math
    ll = 0
    # raise NotImplementedError
    ## SOLUTION:
    # 1. Tokenize the text.
    tokens = model.tokenize(text)
    # 2. Generate ngrams for the tokens.
    ngrams = model.generate_n_grams(tokens)


    # 3. Loop through the ngrams, calculate log prob for each ngram and update ll.
    for context, target in ngrams:
        try:
            prob = model.get_prob(context, target)
        except KeyError:
            # Unseen n-gram => zero prob under MLE, log-likelihood is -inf
            return float("-inf")
        ll += math.log(prob)  # natural log
    return ll

# %% [markdown]
# <font color='magenta'>Part (b) cont'd. Use the sentence "And on a love-book pray for my success?" as your input. Calculate the log-likelihood of this sentence under a unigram model and a bigram model and report those numbers.</font>

# %%
text = "And on a love-book pray for my success?"
## SOLUTION:
print("fitting_unigram_model")
unigram_model = create_and_fit_model(n=1, sentences=sentences)
print("fitting_bigram_model")
bigram_model = create_and_fit_model(n=1, sentences=sentences)
print("fitting_trigram_model")
trigram_model = create_and_fit_model(n=1, sentences=sentences)

unigram_ll = model_ll(unigram_model, text)
bigram_ll = model_ll(bigram_model, text)
trigram_ll = model_ll(trigram_model, text)
print(f'Unigram Model Log-Likelihood: {unigram_ll}')
print(f'Bigram Model Log-Likelihood: {bigram_ll}')
print(f'Trigram Model Log-Likelihood: {trigram_ll}')

# %% [markdown]
# ### <font color='blue'>Text completion</font>
# 
# <font color='magenta'>Part (c): Given a prompt from `test_play` (the held out part of the first sonnet), generate 20 words of text (using the function `generate_text`) from the bigram and the trigram model.</font>

# %%
# QUESTION
print(test_play)
prompt = "From fairest creatures we desire increase"

# %% [markdown]
# There are various tweaks that would improve this model: for instance, careful <em>smoothing</em> and using longer-range dependencies (via variable-length Markov models such as <em>probabilistic suffix trees</em>). The next big boost in performance would come from replacing tabular estimates of conditional probabilities $P(x_i|x_1, ...., x_{i-1})$ by <em>recurrent neural nets</em>.


