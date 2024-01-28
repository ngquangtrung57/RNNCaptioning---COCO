import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_validation_loader 
from vocab_preprocessing import Vocabulary
from rnn_model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def main(encoder_path, decoder_path, image_size, vocab_path, image_dir, caption_path, embed_size, hidden_size, num_layers, batch_size, num_workers):

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build data loader
    data_loader = get_validation_loader(image_dir, caption_path, vocab, 
                                        transform, batch_size,
                                        num_workers=num_workers)
        
    encoder = EncoderCNN(embed_size).eval()  
    decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers)
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
        if i > 5000: break  
        
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

# Configuration parameters
encoder_path = r'rnn_models/encoder-3.ckpt'
decoder_path = r'rnn_models/decoder-3.ckpt'
image_size = 224
vocab_path = 'vocab.pkl'
image_dir = r"coco\images\val2017\resized2017"
caption_path = r"coco\annotations_trainval2017\annotations\captions_val2017.json"
embed_size = 256
hidden_size = 384
num_layers = 1
batch_size = 1
num_workers = 1

# Run the main function with direct parameters
if __name__ == '__main__':
    main(encoder_path, decoder_path, image_size, vocab_path, image_dir, caption_path, embed_size, hidden_size, num_layers, batch_size, num_workers)
