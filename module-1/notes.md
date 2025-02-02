# Introduction to AI transformers

# Module 1.1 - advanced

- AI is the entire domain of creating machines that learn.
- ML and DL give specific methods to complete this task.
- One weakness is traditional deep learning models, such as recurrent neural networks, is that they process data sequentially. This is therefore very slow.
- Another weakness is the 'vanishing gradient problem'.
  - Problem during backpropagation, where sequence of gradients of loss functions are multiplied by derivatives of activation layers for each layer. This involves small numbers as they progress, so values go towards 0.
  - For example with a traditional RNN, if they are analyzing a long sentence, they have a short memory. Might forget the start of sentence by the end.
- Transformers solve this by:
  - Introducing self-attention to allow parallelization
  - Allows memory of long-range dependencies as there is no more vanishing gradient problem. Because they use positional encoding and attention weights instead of sequential gradient multiplication.

## Transformer architecture

- Developed at Google, released in 2017 paper 'Attention Is All You Need'

### Overview of transformers

- Big picture of the transformer is that it takes an input sequence, or sequence of text, and makes a prediction for what the next word in the sentence or text in the sequence should be.
- This is a prediction through a probability distribution of words, where higher probability word are more likely to be the 'correct' next word in the sentence.
- Transformers are mainly composed of:
  - tokens
  - embedding
  - positional encoding
  - multi-head self-attention mechanism
- Involves an encoder-decoder structure
  - Encoder: processes the context of the input sequence
  - Decoder: generates the output sequence
- The overall steps the transformer foes through to generate the probability distribution of the next word:
  - Tokens and tokenization
    - units of text used in NLP. Size of token is called granularity
  - Word embedding
    - Tokens converted to vector representations, where semantics/meaning of words are encodings in high dimensional space.
  - Positional encoding
    - Gives information on the order of the tokens in the sequence. Important because transformers process data in parallel, not sequentially
  - Attention mechanism
    - How the meanings of entire sequences are calculated by analyzing relevant parts of the input sequence

## Tokens and tokenization

- Words converted to tokens before the embedding step
- Tokens can be more than words, also linguistic elements like syllables, characters, punctuation marks
- Tokenization allows models to handle large vocabulary, and understand different forms of words even if the model hasn't seen the word before

### Word embeddings

- Each token gets a vector representation
- Can be visualized as coordinates in a high dimensional space
- Tokens with 'similar meanings' are close together

### Word embedding matrix

- Process of linking tokens to vector representations, so they can encode concepts
- Embedding matrix has one column for each token, so # of columns can be thought of as size of vocabulary
- Number of rows can be kind of thought of as the models ability to understand context, semantics, and relationships between words
- A vocab of 50000 tokens and 300 dimensional embedding vector space means 15,000,000 parameters just for the word embedding matrix

## Positional encoding

- Transformers don't inherently know token order (for parallelization reasons)
- Positional encoding allows transformer to take into account order of words
- Done using sine functions
- Without positional encoding, 'the cat ate a fish' and 'the fish ate a cat' would be identical
- Positional encoding attaches a vector to each tokens embedding which represents its position in the sequence

## Attention mechanism

- Key part of the transformer
- Invented in 2014
- Processes/understand the input sequences, determining the relevancy between tokens to build up context
- Determines which words in the sequence are important when processing each specific word
- Types of attention:
  - Global attention: Focuses on all tokens, sequentially
  - Local attention: Focuses on a smaller window of tokens within the input
  - Self-attention: Each token 'sees' each other token and basically combines global + local attention
- A vector is an object
- A matrix is an action

### Input tokens are embedded

- Mapping: input text -> tokens -> token IDs -> embedding matrix -> high dim word embedding vector space
- Positions also encoded in this step.

### Query, key, value vectors

- Attention mechanism transforms each embedding vector into:
  - Query (Q): What is this token looking for
  - Key (K): How relevant is this token
  - Values (V): Information about this token
- Each set of {Q, K, V} comprises one 'head'
- Suppose the sentence 'The cat eats fish'
  - Q acts as search query, could represent how the subject 'cat' is performing the action 'eat' on the object 'fish'
  - K represents the relevance of tokens to other tokens. 'cat' and 'fish' would have strong relevance
  - V represents the information related to the token, related to the word embedding from earlier
- These vectors are calculated by multiplying the embedding matrix (each column is the embedding vector for each token) by the query weight matrix, key weight matrix, and value weight matrix. The matrices are learned through training. So each token has a corresponding query, key, and value vector.

### Attention scores

- Attention scores are numerical values that are calculated from {Q, K, V}, and tell model how relevant the token is to the output
- Calculated by taking the dot product between each of the query and key vectors, then scaling the value vectors by the results of the dot product

### Softmax function

- The results vectors from query dot key, multiplied by value are concatenated into a matrix called the attention score
- Attention scores are normalized to be probabilities, using the softmax function
- Take the exponential of each value and divide that by the exponential of every value summed up.
- They are now called attention weights, which tell the model the importance of each token.
- Attention weights are used to calculate a new representation of each token. A weighted sum of the value vectors is calculated, using attention weight to weigh the sum.
- These vectors describe how much attention each token should give to each other token, called the attention output

### Multi-head attention

- Strength of transformer is from multi-head attention
- Allows for different and more nuanced relationships/contextx between words to be captured
- With single head of attention, you have one copy of the attention between each token and each other token
- Multi-head attention has multiple versions
- what the heads do?
  - Semantic relationships
  - Syntactic structure
  - Local context
  - Long-renge dependencies
  - References to same entity
  - Temporal/Positional info
  - Negation handling
  - Punctuation
  - Hierarchy context
- Adding multi-head attention is done by having multiple sets of {Q, K, V} vectors, where each analyzes a different relationship within the input sequence.
- If we have N attention heads, we have N versions of each vector being calculated in parallel
- For each token in the input, language model has different weight matrices, to capture many relationships between words. Attention is calculated multiple times, with each calculation starting with different Q, K, and V mevtors/matrices. The results in the calculated Q, K, V spaces to be varying, capturing different aspects of the input.
- Attention from each head is concatenated together. Linear transformation is applied to combine the information of the heads. Result has now combined context + relationships from different attention heads.

## Position-wide feed forward network

- Attention is good, but need more
- After multi-headed attention weights are calculated, the new token representations are passed through the FFN.
- If there was no FFN, the results would be overly linear.
- Introduces non-linearity to the output, providing refined context and meaning, but could result in reduced flexibility in token representations, and meanings being shallower
- FFN input is token representations from multi-head attention.. FFN works on each token representation independently and is applied to each token in sequence.
- uses non-linear activation functions. This is a similarity between FFN and DL, which is different from more traditional linear methods.
- FFN also introduces token-wise transformation (multiplication by weights and addition by biases)
- Key diff between attention mechanism and position-wise FFN:
  - former involves tokens interacting with all tokens, latter acts on each token fully separately. Which is why it is called 'position-wise'
- FFN takes 2 linear transformations (fully connected layers in deep learning) which have a non-linear activation function between them.
- Each token embedding is multiplied by the weight matrix of the first linear layer and added to the first bias. This goes through ReLU and then this result is multiplied by the second layer which is made of the second weight matrix and bias.
- The encoded output from FFN is sent to decoder. Decoder contains attention and its own feed-forward network
- Decoder sends encoded sequence to masked multi-head attention mechanism. The 'masked' is in how it guesses which token/word will come next. Then then goes through the same position-wise feed-forward and lastly a softmax to calculate the probability distribution of the next token.

# Understanding large language modules (LLMs)

- LLMs are AI systems designed to press and generate human language using advanced transformer architectures
- GenAI is broader, could be text, image, video, audio, etc

## Traditional Language Models

### N-grams

- Predict next work in sequence based on the preceding n-1 words.
- Idea that probability of a word depends on the context of last few words, with n representing the number of words considered.
- Unigram: Only consider current word, no context
- Bigram: Model considers previous word
- Trigram: Model considers two previous words
- Model predicts next word after 'is' by looking at the probability of each word following 'is' based on the training data
- Limitations:
  - Limited context: Only capture local dependencies, cannot handle long-range dependencies between words
  - Data sparsity: If some n-grams don't appear in training data, the model can't make predictions for them

### Bag-of-Words (BoW)

- Represent text as collection of words, without considering order.
- Breaks text down into individual words, each is assigned a count or weight, treating each word as independent entity
- Representation: Each doc is represented as a vector where each entry corresponds to the presence or frequency of a word in that document
- Applications: Document classification
- Limitations:
  - Loss of context: BoW ignores word order+ syntax, poor performance when word sequence matters.
  - Large vocab size: Dimensionality of model increases, making it less efficient.

### Term frequency-inverse document frequency (TF-IDF)

- Evaluates the importance of a word in a document relative to a collection of documents.
- Accounts for the frequency of a word in a document, and adjusts importance based on the how frequently it appears across all documents in the corpus.
- Representation: Each document is a vector where each entry corresponds to a words TF-IDF score. High TF-IDF indicates the word is important in a specific document, but rare across the entire corpus.
- Applications: text mining, text classification, clustering, info retrieval, keyword extraction
- Limitations:
  - No contextual understanding: TF-IDF treats words as independent units.
  - Overemphasis on rare-terms: Rare term means high IDF score, making it seem important
  - TF-IDF disregards order of words

### Recurrent Neural Networks

- RNNs maintain a hidden state that captures information from all the previous words in the sequence, making them capable of learning long-range dependencies.
- Key feature: hidden state of RNN is updated at each step in the sequence, allowing model to theoretically remember information from earlier words.
- For example, embedding layer is input layer that transforms words into vector representations. LSTM (long short term memory) is hidden layer which processes the sequence of word embeddings, captures dependencies between words and maintains memory of past info, and dense layer which is often used as final layer to make predictions.
- Steps to build RNN:

1. Tokenization: Convert words to tokens, create sequences to train model
1. Padding: All sequences are padded to ensure equal length for compatibility with the RNN
1. Model Architecture: Embedding layer maps words to dense vectors. SimpleRNN layer processes sequential data and maintains a hidden state to capture context. Dense layer outputs probabilities for next word.
1. Training: Train model to predict the next word in the sequence
1. Prediction: Given seed text, generate next few words by iteratively predicting and appending next word to input.

- Limitations:
  - Sequential processing (no parallelism): Inefficient as inputs are processed step by step. Makes training slow, and inefficient on modern hardware.
  - Vanishing gradient: RNNs struggle with learning extreme long-range dependencies due to vanishing gradient problem, where gradients shrunk over time making it hard to propagate information from earlier words.
  - Limited context: Struggle to capture dependencies in very long sequences.

### Large language models

- Based on transformer architecture. Heart of it is the attention mechanism. Allows model to weigh importance of different parts of the input sequence when processing a specific part. Means model can focus on relevant words/phrases regardless of position.
- Use encoder-decoder, encoder, or decoder only structure. Encoder processes input sequence and generates contextual representation. Decoder generates the output sequence based on this representation.
- They can process entire input sequence in parallel, making training much faster.
- Explicit feature extraction: Manual process of identifying specific features from raw data that are relevant to a ML task.
- Parameter size: total # of learnable weights or connections within the model.
- LLMs vs traditional NLP:

| Method       | Parameter Size   | Context Length                                      | Capabilities                                      |
|-------------|----------------|-------------------------------------------------|--------------------------------------------------|
| n-gram      | Small (fixed n) | Limited (n words)                               | Basic sequence prediction                      |
| Bag-of-Words | Small          | No context                                     | Word frequency analysis                        |
| RNN         | Moderate       | Dependent on hidden state, limited by vanishing gradients | Sequential data processing, limited long-range dependencies |
| GPT-3       | 175 billion    | Up to 2048 tokens                               | Long-range context recognition, text generation |
| BERT        | 110 million    | Bidirectional (entire sequence)                 | Contextual understanding, classification      |
