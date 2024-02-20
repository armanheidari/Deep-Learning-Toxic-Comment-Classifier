# Toxic Comment Classifier

<img src="logo.jpg" style="display: block; margin: auto; height: 500px; width: 500px;">

## About the Project

In the current digital era, social media platforms have become a common ground for communication and interaction. However, they are often marred by the presence of toxic comments, which can create a hostile environment and negatively impact the well-being of users. Addressing this issue is of paramount importance.

This project is a dedicated effort towards mitigating this problem. It focuses on the identification and classification of various types of toxic comments prevalent on social media. The objective is to develop a robust mechanism that can accurately detect these comments, thereby enabling their removal and contributing to a safer and more positive online environment. This endeavor not only enhances the user experience but also fosters a healthier digital community.

- - -
## Data

The dataset used to train the model is sourced from a Kaggle competition focused on toxic comment classification. The dataset is split into three separate files:

- **train.csv:** This file contains the training data that the model learns from. Each row represents a comment along with its associated labels.
    
- **test.csv:** This file contains the test data that the model is evaluated on. It includes comments for which the model needs to predict labels.
    
- **test_labels.csv:** This file provides the true labels for the test data. It is used to evaluate the performance of the model’s predictions.
    

The dataset includes six different labels, each representing a different type of toxicity:

- **Toxic:** General category for toxic comments.
- **Severe Toxic:** Comments that are particularly toxic or harmful.
- **Obscene:** Comments that contain obscene language or content.
- **Threat:** Comments that contain threats towards individuals or groups.
- **Insult:** Comments that are insulting towards individuals or groups.
- **Identity Hate:** Comments that express hate towards specific identities or groups.

Each comment in the dataset can have multiple labels, reflecting the fact that a single comment can exhibit multiple types of toxicity. This makes the task a multi-label classification problem.

- - -
## Preprocessing

The raw data requires some preprocessing before it can be used to train the model. Here are the steps that were taken:

- **Data Cleaning:** Upon initial inspection of the data, it was evident that the text data needed to be cleaned. The Natural Language Toolkit (NLTK) library was used for this purpose. This step involved removing unnecessary characters, converting all text to lowercase, and other such operations to standardize the text data.
    
- **Tokenization:** After cleaning the data, the next step was tokenization. This process involves converting the text data into numerical tokens that the model can understand. The `Tokenizer` class from the Keras library was used for this purpose. The `MAX_FEATURES` parameter, which represents the number of unique words in the tokenized data, was calculated during this step.
    
- **Padding:** The length of the comments varied, with some being very short and others being quite long. To ensure that all input data to the model has the same shape, the comments were padded to a static length. Upon analyzing the distribution of comment lengths, it was observed that the majority of comments were below 150 words in length. Therefore, 150 was chosen as the static length for padding.

- - -
## Model Structure

- **Input Layer:** The starting point of our deep learning model. It receives the processed data and passes it on to the next layer.

- **Embedding Layer:** This layer transforms the sparse input data into dense vectors by mapping each word to a vector in a high-dimensional space. This helps to capture semantic relationships between words and improve the model’s understanding of the text.

- **Bidirectional LSTM Layer:** Long Short-Term Memory (LSTM) is a type of Recurrent Neural Network (RNN) that is well-suited for sequential data like text. The bidirectional variant of LSTM processes the text data in both forward and backward directions, allowing it to capture context from both before and after each word which led to improved model performance.

- **Attention Mechanism:** The attention mechanism allows the model to focus on the most relevant parts of the input when generating the output. It assigns higher weights to the important words in the text, thereby improving the model’s ability to understand and generate meaningful responses.

- **1D Convolution + Max Pooling Layers:** The combination of 1D Convolution and Max Pooling layers is a powerful tool for extracting features from the output of the LSTM layer. The Convolution layer applies a series of filters to the input, while the Max Pooling layer reduces the dimensionality of the output, helping to highlight the most important features and improve the model’s performance.

- **Dense Layer:** The Dense layer transforms the output of the Max Pooling layer into a format that is more suitable for the final output layer.

- **Output Layer:** The final layer of the model which uses a sigmoid activation function to classify the input text, because classes can overlap in our problem.

**Note:** Two callbacks are used to increase speed and improve performance of the model.

- **Early Stopping:** This callback is used to halt the training process if the model’s performance does not improve for a specified number of epochs, in this case, 3. The primary purpose of Early Stopping is to prevent overfitting, which occurs when the model performs well on the training data but poorly on unseen data. By stopping the training early, we can ensure that the model generalizes well and does not simply memorize the training data.

- **ModelCheckpoint:** This callback is used to save the model at the end of each epoch. This allows us to retain and use the best performing version of the model, even if the model’s performance decreases in later epochs. By saving the model periodically, we can also resume training at a later time if needed, without losing progress. This is particularly useful in scenarios where training is computationally expensive or time-consuming.

- - -
## Usage

To access the dataset and the pre-trained model used in this project, please follow the provided [Drive link](https://drive.google.com/drive/folders/1JgmPIph_5QjKR6IuhCjgODtQNOT_FVT8).

Please note that the training of the model was carried out on Google Colab. If you wish to retrain the model, it’s important to be mindful of the file paths and adjust them accordingly to your environment. This is crucial to ensure that the code runs smoothly without any file not found errors.

By providing these resources, the project aims to be transparent and reproducible, allowing others to build upon this work, replicate the results, or apply the model to new data. Enjoy exploring and experimenting! 😊

- - -
## Installation

- All the required modules are listed in the `requirements.txt` file. 

```Shell
pip install -r requirements.txt
```

- - -
## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

- - -
## License

BSD 3-Clause "New" or "Revised" License
