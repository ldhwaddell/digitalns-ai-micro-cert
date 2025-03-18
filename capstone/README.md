Thank you to everyone who made this course possible. 

data.py: Logic for downloading and generating the dataset
eval.py: Logic for evaluation the model based on 'MODEL_PATH'. 'MODEL_ID' is used for the tokenizer, so it stays the base bert. 
train.py: Logic for finetuning the model on the dataset.

Thanks!

Information below was generated throughout the training process, and is mirrored in the PDF

untrained:

Accuracy: 0.1968
Precision: 0.1813
Recall: 0.1968
F1-score: 0.1220

Classification Report:
               precision    recall  f1-score   support

           0       0.21      0.53      0.30      2943
           1       0.13      0.03      0.05      3031
           2       0.23      0.00      0.00      3114
           3       0.19      0.45      0.27      2908
           4       0.14      0.00      0.00      3004

    accuracy                           0.20     15000
   macro avg       0.18      0.20      0.12     15000
weighted avg       0.18      0.20      0.12     15000

finetune:

MODEL_ID = "google-bert/bert-base-uncased"
TRAIN_DATASET_PATH = "train.jsonl"
BATCH_SIZE = 128
TOKENIZED_DATASET_PATH = "tokenized_train_dataset"
OUTPUT_DIR = "finetuned_model"
NUM_EPOCHS = 5
LEARNING_RATE = 5e-5
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
PLOT_OUTPUT_PATH = "training_loss_plot.png"

Accuracy: 0.6507
Precision: 0.6519
Recall: 0.6507
F1-score: 0.6513

Classification Report:
               precision    recall  f1-score   support

           0       0.78      0.75      0.77      2943
           1       0.60      0.60      0.60      3031
           2       0.59      0.60      0.60      3114
           3       0.55      0.56      0.56      2908
           4       0.73      0.74      0.74      3004

    accuracy                           0.65     15000
   macro avg       0.65      0.65      0.65     15000
weighted avg       0.65      0.65      0.65     15000

hyperparam

MODEL_ID = "google-bert/bert-base-uncased"
TRAIN_DATASET_PATH = "train.jsonl"
BATCH_SIZE = 16
TOKENIZED_DATASET_PATH = "tokenized_train_dataset"
OUTPUT_DIR = "hyperparam_finetuned_model"
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
WARMUP_STEPS = 1000
WEIGHT_DECAY = 0.01
PLOT_OUTPUT_PATH = "training_loss_plot.png"

Batch: 117/118
Accuracy: 0.6568
Precision: 0.6595
Recall: 0.6568
F1-score: 0.6579

Classification Report:
               precision    recall  f1-score   support

           0       0.79      0.75      0.77      2943
           1       0.60      0.64      0.62      3031
           2       0.61      0.59      0.60      3114
           3       0.56      0.58      0.57      2908
           4       0.75      0.73      0.74      3004

    accuracy                           0.66     15000
   macro avg       0.66      0.66      0.66     15000
weighted avg       0.66      0.66      0.66     15000
