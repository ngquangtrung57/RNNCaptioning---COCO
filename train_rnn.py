import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_train_loader 
from vocab_preprocessing import Vocabulary
from rnn_model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(model_path, image_size, vocab_path, image_dir, caption_path, log_step, save_step, embed_size, hidden_size, num_layers, num_epochs, batch_size, num_workers, learning_rate):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    transform = transforms.Compose([ 
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    data_loader = get_train_loader(image_dir, caption_path, vocab, 
                                   transform, batch_size,
                                   shuffle=True, num_workers=num_workers) 

    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers).to(device)
    
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    
    total_step = len(data_loader)
    for epoch in range(num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):
            
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            if i%log_step==0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch+1, num_epochs, i, total_step, loss.item(), np.exp(loss.item()))) 
                
        torch.save(decoder.state_dict(), os.path.join(
            model_path, 'decoder-{}.ckpt'.format(epoch+1)))
        torch.save(encoder.state_dict(), os.path.join(
            model_path, 'encoder-{}.ckpt'.format(epoch+1)))


if __name__ == '__main__':
    MODEL_PATH = r'rnn_models'
    CROP_SIZE = 224
    VOCAB_PATH = r'vocab.pkl'
    IMAGE_DIR = r'coco\images\train2017\resized2017'
    CAPTION_PATH = r"coco\annotations_trainval2017\annotations\captions_train2017.json"
    LOG_STEP = 100
    SAVE_STEP = 6000
    EMBED_SIZE = 256
    HIDDEN_SIZE = 384
    NUM_LAYERS = 1
    NUM_EPOCHS = 3
    BATCH_SIZE = 64
    NUM_WORKERS = 1
    LEARNING_RATE = 0.0005

    main(MODEL_PATH, CROP_SIZE, VOCAB_PATH, IMAGE_DIR, CAPTION_PATH, LOG_STEP, SAVE_STEP, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_EPOCHS, BATCH_SIZE, NUM_WORKERS, LEARNING_RATE)
