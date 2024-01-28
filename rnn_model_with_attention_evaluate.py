import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_validation_loader
from vocab_preprocessing import Vocabulary
from rnn_model_with_attention import EncoderCNN, DecoderRNNWithAttention
from torchvision import transforms
from nltk.translate.bleu_score import corpus_bleu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def main(encoder_path, decoder_path, vocab_path, image_dir, caption_path, image_size, embed_size, encoded_image_size, attention_size, hidden_size, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
        
    data_loader = get_validation_loader(image_dir, caption_path, vocab, transform, batch_size, num_workers=num_workers)
    
    encoder = EncoderCNN(encoded_image_size).eval()  
    decoder = DecoderRNNWithAttention(embed_size, attention_size, hidden_size, len(vocab)).eval()
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))
    
    ground_truth = []
    predicted = []
    for i, (images, captions) in enumerate(data_loader):
        images = images.to(device)
        features = encoder(images)
        sampled_seq = decoder.sample_beam_search(features, vocab, device)
        
        sampled_seq = sampled_seq[0][1:-1]  
        captions = [c[1:-1] for c in captions[0]]  
        
        ground_truth.append(captions)
        predicted.append(sampled_seq)

        if i%100==0:
            print(i)        
            bleu_score_1 = corpus_bleu(ground_truth, predicted, weights=(1, 0, 0, 0))
            bleu_score_2 = corpus_bleu(ground_truth, predicted, weights=(0.5, 0.5, 0, 0))
            bleu_score_3 = corpus_bleu(ground_truth, predicted, weights=(1.0/3.0, 1.0/3.0, 1.0/3.0, 0))
            bleu_score_4 = corpus_bleu(ground_truth, predicted)

            print(f"BLEU-1 score: {bleu_score_1}")
            print(f"BLEU-2 score (bigram): {bleu_score_2}") 
            print(f"BLEU-3 score (trigram): {bleu_score_3}")
            print(f"BLEU-4 score (4-gram): {bleu_score_4}")


if __name__ == '__main__':
    ENCODER_PATH = r'rnn_models_with_attention/encoder-3-9000.ckpt'
    DECODER_PATH = r'rnn_models_with_attention/decoder-3-9000.ckpt'
    IMAGE_SIZE = 224
    VOCAB_PATH = r'vocab.pkl'
    IMAGE_DIR = r"coco\images\val2017\resized2017"
    CAPTION_PATH = r"coco\annotations_trainval2017\annotations\captions_val2017.json"
    EMBED_SIZE = 256
    HIDDEN_SIZE = 384
    ATTENTION_SIZE = 384
    ENCODED_IMAGE_SIZE = 14
    NUM_LAYERS = 1
    BATCH_SIZE = 1
    NUM_WORKERS = 1

    main(ENCODER_PATH, DECODER_PATH, VOCAB_PATH, IMAGE_DIR, CAPTION_PATH, IMAGE_SIZE, EMBED_SIZE, ENCODED_IMAGE_SIZE, ATTENTION_SIZE, HIDDEN_SIZE, BATCH_SIZE, NUM_WORKERS)
