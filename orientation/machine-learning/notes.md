# Introduction to machine learning

- Main conceptual diff between classical programming and ML is that in classical there are defined rules for the program to follow whereas in ML, there is a defined algorithm that allows the computer to discover the patterns/relationships of the data on its own
- The model is the final learned system used to make predictions.
- ChatGPT -> chat\[G\]enerative\[P\]retrained\[T\]ransformer

## Supervised learning

- Uses labeled data, where the end result is known as some truth.
- Patterns in the labeled data are determined such that we can apply it to future unlabeled data where the 'truth' is unknown
- Supervised learning generally breaks down into regression and classification
- Supervised regression algorithms:
  - Linear regression: Linear relationship between dependent/indepenent variables
  - Logistic regression: used for binary classification problems, predicting the probability of an event occurring
  - Polynomial regression: Models a non-linear relationship by fitting a polynomial curve to the data. x
  - Decision tree: Tree-like structure to make decisions and determine the independent variable value
  - Random forest: An ensemble of decision trees to improve accuracy + reduce over-fitting
  - Support Vector Machines (SVMs): Support Vector Regression, finds a function through the data (often hyperplane) that fits the data posts as closely as possible, with some degree of tolerance
- Supervised classification algorithms:
  - Decision Trees: Create tree-like structure to classify objects based on features
  - Random forests: Combines multiple decision trees to improve accuracy and reduce over-fitting. Predicts a label unlike regression where it predicts a continuous value.
  - Support Vector Machines (SVMs): Specifically Support Vector Classification, finds a hyperplane that separates data points into different classes
  - K-Nearest Neighbors (KNN): Classifies new data points based on the majority class of their neighbours

## Unsupervised learning

- Uses unlabeled data, where ground truth is unknown. The algorithm learns hidden patterns, structures, or clustering in the data. Generally done by grouping the data based on similarities in their features.
- Most unsupervised learning falls into clustering or dimensionality reduction:
  - K-Means Clustering: Partitions n observations unto k clusters, where each observation is in the cluster with the closest mean (cluster centroid)
  - Hierarchical Clustering: Groups objects into clusters that follow a hierarchical structure.
- Dimensionality reduction is a method where a new representation of the data is found which is in a lower number of dimensions that the original representation. The new representation tries to still maintain the majority of the variance.
  - This has advantages like reducing computational requirements of analysis, putting data in a better form for visualization, reducing over-fitting, reducing noise.
- Unsupervised dimensionality reduction algorithms:
  - Principal Component Analysis (PCA): Most common algorithm. Projects data into a lower dimensional space while retaining variance. Good for capturing linear relationships, but does not work too well for non-linear ones.

## Reinforcement learning

- Subset of ML where models 'learn' by interacting with data and receiving +/- feedback based on each decision. Different than supervised learning because there are no predefined ground truths, RL learns through experience. Key elements are:
  - Agent: Model (learner, decision-maker)
  - Environment: Data the agent interacts with
  - State: Situation agent is in
  - Action: Moves the agent can make
  - Reward: Feedback the agent receives from environment based on the action.
- RL works to learn optimal strategies based on trial and error.
- Learning process involves:
  - Policy: Strategy by agent to determine next action using current state
  - Reward Function: Function that provides feedback signal based on the current state + action
  - Value Function: Function that calculates expected cumulative reward from a state
  - Environment model: Representation of environment that predicts future states and rewards for strategy planning.
- Reinforcement learning algorithms:
  - Q-learning: Finds the optimal actions for a finite Markov decision process (MDP). An MDP can also be called a stochastic control problem and is a model for sequential decision making for uncertain outcomes.
  - SARSA: State-action-reward-state-action, another algorithm for MDPs.

# Introduction to implementing machine learning

- Overall workflow:
  - import libraries
  - load dataset
    - Split data into inputs (also known as features, usually variable x) and outputs (also known as labels, usually variable y)
  - split data
    - Larger one for training, smaller for testing
  - Define + train model
  - Make predictions
  - Evaluate model
    - use accuracy, precision, recall
  - Improve model
    - Use hyperparameter tuning to improve the models performance on predicting training data.

## Evaluation metrics

- Accuracy
  - Measures overall correctness of a classification model. Ration of correct predictions to total number of predictions.
  - accuracy = (# of correct predictions) / (# of total predictions)
- Precision
  - Proportional of true positives over the total predicted positives.
  - Measure of how many of the predicted positive results were actual positive.
  - Especially important when the cost of false positives is high, like medical diagnosis where false positive means unnecessary treatment.
  - precision = (# of true positives) / (# of true positives + # of false positives)
- Recall
  - Measures the proportion of true positives over the total actual positives
- Confusion Matrix
  - Visualizes performance by counting:
    - True positive: Model predicted correctly. Actual result positive, model predict positive
    - True negative: Model predicted correctly. Actual result negative, model predict negative
    - False positive: Model predicted incorrectly. Actual result negative, model predict positive
    - False negative: Model predicted incorrectly. Actual result positive, model predict negative

# Common ML algorithms

## Regression

- Used to relate a continuous output value to one of, or a set of input values.
- Overall goal is to determine the relation between a dependent variable and a set of independent variables.
- Especially useful when we have a dataset with continuous outcomes and want to predict new values, identify trends, or quantify the strength of relationships between variables.

### Linear regression

- Models relationship between dependent variable and one or more independent variables by fitting a straight line (flat plane in higher dims).
- Line, known as regression line, minimizes the sum of squared differences between actual data points and the predictions.
- Best suited for when there is a linear relationship between variables.
- Often used as a baseline model.
- Ex usage: Predicting house price based on features like square footage, number of bedrooms, age.

### Logistic regression

- Used when target variable is binary or categorical, such as predicting whether a customer will buy or not buy a product.
- Fits an S-shaped curve.
- Output is bounded 0-1.
- The probabilities can be extended to classification based on if the result is greater than or less than 0.5
- Ex usage:  Predicting whether a customer will buy a product based on features like age, income, and past purchase history.

### Polynomial regression

- Extends linear regression by fitting a polynomial equation to the data.
- Allows modeling of non-linear relationships between the dependent and independent variables.
- Useful when relation between variables is not linear, but curved line fits better.
- Model includes higher degree terms of input variables, which makes it more flexible but also more prone to overfitting
- Ex usage: Modeling growth rate of bacteria population over time when growth is non-linear

### Decision trees

- Non-linear regression models that work by splitting data into subsets based on the values of input features
- Each node in the tree is a decision rule based on a feature.
- Each leaf represents a predicted outcome
- Useful for capturing complex, non-linear relationships/interactions between features, so suitable for regression and classification.
- However, prone to overfitting, especially if the tree becomes too deep.
- Ex usage: For regression, predicting sales revenue for a retail store based on seasonal trends, holidays, promotional campaigns.

### Random forests

- Ensemble learning method that builds multiple decision trees and combines their predictions
- For regression, they aggregate the predictions of individual trees (usually by averaging).
- By creating multiple trees based on diff subsets of data + features, they reduce overfitting and improve generalization to new data.
- Ex usage: Estimate air quality index based on weather conditions, traffic levels, and pollutant concentrations

### Support vector regression (SVR)

- Aim to find a hyperplane that best fits data while maintaining some tolerance margin around it
- Doesn't minimize error directly, attempts to find a balance between model complexity and error tolerance by creating some margin around the hyperplane where deviations are ignored.
- Particularly effective when data is high-dimensional or when there is a non-linear relationship that can be transformed with kernel functions
- Ex usage: Predicting future price of stick based on historical price movements and technical indicators

## Classification

- Supervised learning technique that assigns a category (also known as label) to input variables
- Involves training model on a dataset where each point is associated with a known label, allowing the model to learn the patterns that separate input classes
- Ideal for scenarios where there are clear cut categories/classes to distinguish between, and where accurate prediction of the labels is crucial for decision making

### Decision trees

- Versatile classification technique that works by splitting data into subsets based on the values of feature inputs and making a series of decisions at each node.
- Each internal node represents a feature-based decision, each branch represents an outcome of that decision, each leaf node represents a class label.
- Tree is build by choosing features that reduce uncertainty at each step.
- Prone to overfitting
- Ex usage: Could be used to classify loan applicants as low-risk or high-risk based on features like credit score, annual income, and employment status

### Random forests

- Each tree is built on a random subset of the data and random subset of features. Final class prediction is determined by voting among the indivdual trees' predictions.
- Helps reduce overfitting
- Particularly effective when there are complex interactions between features or when dataset is noisy.
- Ex usage: Could be used to classify transactions as fraudulent or not based on features like transaction amount, location, time of day

### Support vector classifiers (SVC)

- Seeks to find a hyperplane that best separates different classes in the feature space while maximizing margin between nearest data points of each class (known as support vectors)
- Effective in high-dimensional spaces and can handle linearly separable and non-linearly separable data using kernel functions to map the data to high-dimensional space
- Can be expensive, may require tuning of parameters
- Ex usage: Classifying images of handwritten digits from MNIST dataset

### K-nearest neighbors (KNN)

- Instance based learning method that classifies a new data point based on majority vote of its k nearest neighbors in the feature space
- Measures distance (usually euclidean) between new data point and existing points in a training set, and assigns class that is most common based on neighbors
- Can be expensive, requires storing all training data and computing distances for each new prediction
- Ex usage: recommending movies to a user based on the preferences of their nearest neighbors (users with similar taste)

# Introduction to natural language processing

- NNs are models which use artificial neurons to mimic the brain's functionality
- Biological neuron:
  - Info enters through dendrites, which receive electrical signals from other neurons
  - These signals travel through cell body, down the axon, to reach axon terminals, where they are transmitted to other neurons
- Artificial neuron:
  - Instead of dendrites, we have inputs, which could be features in data.
  - Each input is connected to the next layer of neurons by a weight which adjusts the strength of the signal
  - Network sums the weighted input
  - Activation function is applied to produce the output. The process of summing the inputs and applying the activation function helps the network learn patterns from data
- To build on the idea of artificial neuron, there is the Multilayer Perceptron (MLP):
  - Each layer in MLP contains multiple neurons that work together to process inputs and make predictions
  - Inputs pass through many hidden layers of interconnected neurons
  - Each applying weights+ activation functions
  - Layered structure allows MLP to capture more complex patterns in data.

## 4 key concepts

### Types of layers

- Input layer: Receives raw data
- Hidden layers: Perform non-linear transformations on data to extract features + patterns. Can be many.
- Output layer: Provides final result, such as classification or prediction.

### Neurons

- Each neuron receives inputs, weighs them, sums them, and applies an activation function
- Activation function introduces non-linearity, enabling the MLP to learn complex relationships

### Connections and weights

- Neurons are connected by weighted connections
- Weights determine the importance of each connection and are adjusted during learning

### Learning

- MLP learns from a training dataset
- The backpropagation algorithm adjusts the connection weights to minimize the error between the model's predictions and actual values

# Introduction to deep learning

- 2 majors factors pushed us into era of deep learning:
  - Increased data
  - Increased processing power

## What is deep learning

- Data-driven approach where the rules are learned from data, rather than being explicitly programmed
- Programmer provides the model architecture, along with a large amount of training data that contains both inputs and the correct outputs for those inputs
- The model is then 'trained' on the training data and learns underlying patterns by adjusting internal parameters in the model architecture
- Done through an iterative process where the model learns the ideal parameters that minimize difference between predictions and actual outputs

## Deep learning process

- First need list of variables: inputs and corresponding outputs. Don't need to know relationship between them, but the more accurately we observe them the better
- Then train a model by showing it dataset + correct outputs. Model keeps taking guesses to find out if right or not
- One thing that separates it from traditional ML is the depth + complexity of networks that are used. Networks can have billions of parameters
- The 'deep' refers to the many layers in a model that learn the rules necessary to complete the task. Often called hidden layers

## Deep learning components

#### Input layer

- First layer of deep learning model
- Each node (neuron) represents a feature of the input data
- Ex, if analyzing home prices and input had square footage, # bedrooms, and # bathrooms, then there would be 3 nodes in input layer
  - If analyzing a 20x40 square image, there would be 800 nodes in input layer
- This layer doesn't perform calculation, just feeds next layer

#### Hidden layers

- Layers between input and output
- Perform the intermediate computation that cause the model to learn
- Each layer contains neurons that:
  - Compute weighted sum of the inputs to that layer
  - Applies an activation function to the weighted sum
  - Sends the result of the weighted sum multiplied by the activation function to the next layer

#### Activation function

- After the hidden layer's weighted sum is determined, the activation function is applied
- Suppose there was not an activation function
  - Then no matter how many layers we pass weighted sums through, adding a series of linear functions will always result in a linear function
- The activation function introduces non-linearity to the model
- Common activation functions:
  - ReLU (Rectified linear unit): Outputs 0 for negative values and input itself for positive values
  - Sigmoid for probabilities (output 0-1)
  - tanh (hyperbolic tangent) for output -1-1

#### Output layer

- Produces final result of model
- For classification, might include softmax function to output the probabilities for each class.
- For regression, could be a linear function to predict continuous values
- Number of nodes in output layer represent the number of output features

#### Loss function

- Determines how well the predictions from the model match with the actual values
- Quantifies the diff between predicted output and true target for each input in the training set into a single value representing error of model
- Goal is to minimize this loss
- Different loss functions such as:
  - Mean square error (MSE): Used for regression, calculates averaged square diff between predicted and actual values. Large errors are heavily penalized
  - Cross-entropy loss (log loss): Commonly used for classification, measures diff between predicted probability distribution and true distribution. Predictions that are far from true label are heavily penalized

#### Backpropagation

- Process used to update the model's weights based on the calculated loss
- Involves:

1. Forward pass: Input data passed though the NN and initial predicted output is calculated
1. Compute loss: Using loss function, loss is calculated quantifying how accurate model was
1. Backward pass: Gradient of the loss with respect to each weight is calculated. This quantifies how much each weight in the model contributed to the overall loss
1. Gradient computation: For each weight in the model, backpropagation computes a partial derivative of the loss with respect to that weight. The gradient tells up how the loss changes with small changes in weight.
1. Gradient update: The optimizer uses these gradients to adjust the weights, and then moves them in a direction which reduces loss

#### Optimizer

- Uses the gradients from the backpropagation to adjust model weights such that loss is minimized.
- Determines how to change the weights.
- In other words, fine-tunes the params in the NN during model training
- Balance must be found between convergence speed and accuracy

#### Gradient Descent

- Finds the parameters that minimize the loss function

| Optimizer                 | Advantage                                          | Disadvantage                                      |
|---------------------------|----------------------------------------------------|--------------------------------------------------|
| Stochastic gradient descent (SGD) | Fast and efficient, good for large datasets          | Convergence to optimal parameters might be noisy |
| Momentum                  | Converges quickly                                  | Might not find most optimal set of parameters    |
| Adagrad                   | Adapts the learning rate, good for sparse data     | Learning rate decays over time                   |
| RMSprop                   | Prevents the rapid learning rate decay             | Requires more fine-tuning                        |
| Adam                      | Converges quickly, adapts the learning rate        | Uses lots of memory and may not find the global minimum |
| AdamW                     | Generalizes to more situations better              | More intensive to compute                        |

- Important parameter in optimization is learning rate. Can be thought of as step size that is taken in each step of gradient descent.
- Small learning rate means accurate results but expensive
- Large learning rate means less accurate results, but optimizer converges faster.
- Good practice to smaller with small rate and adjust

#### Training over many epochs

- Once cycle of the preceding steps is an epoch.
- Advantage of DL is to train over many epochs to find most optimal parameters
- Too few could mean model underfit. Too many, model overfit
- Want to track loss over epochs

#### Hyperparameter tuning

- Thought of as turning the dials that control the learning process itself
- Includes:
  - Learning rate: Adjusting step size of the optimizer to find good trade off between computational price + efficiency
  - Number of epochs: Adjusting number of iterations model goes through to reach best trade off between underfitting, overfitting, and time to compute
  - Optimizer: Adjusting method to one best suited for data and model architecture
  - Model parameters: Adjusting params within the model such as number of layers, neurons per layer, or activation functions
- Methods for hyperparameter tuning:
  - Manual search: select/test values based on prior knowledge and intuition
  - Grid search: Define a range for each hyperparameter to be fine-tuned. Every possible combination is tested. Rigorous, but can be very expensive plus hard to do
  - Random search: Define a range, but randomly sample combinations. May not get most optimal set of hyperparameters
  - Bayesian optimization search: Like random search, but you have some idea of the probability distributions of the most optimal hyperparameters. Can search larger hyperparameter space to efficiently find a good approximation of the optimal ones

# Key points

- ML flips the classical programming paradigm
- Many diff data types that can be used for ML analysis
- Supervised ML uses labelled data
- Unsupervised ML uses unlabelled data
- Reinforcement ML uses unlabelled data and a reward function
- Accuracy metrics are used to evaluate ML models
- Supervised ML is typically for regression, classification
- Natural language processing is based on neurons in brain
- DL also flips traditional paradigm in traditional programming of defining rules. DL learns the rules
