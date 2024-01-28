# RNNCaptioning - COCO

RNNCaptioning - COCO is a deep learning project for generating descriptive captions for images using the COCO dataset.

## Description

This project explores the capabilities of Recurrent Neural Networks (RNNs) and attention mechanisms in the field of image captioning. It uses the COCO dataset to train models that can generate natural language descriptions of images. The performance of the models is quantified using BLEU scores, a common metric for evaluating machine-translated text against reference translations.

## Getting Started

### Prerequisites

Ensure you have Python 3.6+ installed on your system. You can install the required packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Installing

Clone the repository to your local machine:

```bash
git clone https://github.com/ngquangtrung57/RNNCaptioning---COCO.git
cd RNNCaptioning---COCO
```

### Preprocessing

Before training the models, run the vocabulary and image preprocessing scripts:

```bash
python vocab_preprocessing.py
python image_preprocessing.py
```

### Training the Model

To train the model from scratch:

```bash
python train_rnn.py
```

For the attention model, use:

```bash
python train_rnn_with_attention.py
```

### Evaluation
Model performance is evaluated using BLEU scores, a common metric for comparing machine-generated text to reference texts. After running the evaluation scripts, you can check the performance of both models.

Evaluate the trained model using:

```bash
python rnn_evaluate.py
```

For the attention model:

```bash
python rnn_model_with_attention_evaluate.py
```

### Streamlit Application

To run the Streamlit application for interactive image captioning:

```bash
streamlit run streamlit_interface.py
```
or 

```bash
streamlit run attention_streamlit_interface.py
```
then upload your own images.

