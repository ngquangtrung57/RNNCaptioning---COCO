import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_train_loader
from vocab_preprocessing import Vocabulary
from rnn_model_with_attention import EncoderCNN, DecoderRNNWithAttention
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(model_path, image_size, vocab_path, image_dir, caption_path, log_step, save_step, embed_size, attention_size, hidden_size, num_epochs, batch_size, num_workers, learning_rate):
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
    
    data_loader = get_train_loader(image_dir, caption_path, vocab, transform, batch_size, shuffle=True, num_workers=num_workers)

    encoder = EncoderCNN(image_size // 32).to(device)
    decoder = DecoderRNNWithAttention(embed_size, attention_size, hidden_size, len(vocab)).to(device)
    
    criterion = nn.CrossEntropyLoss().to(device)
    params = list(decoder.parameters()) + list(encoder.adaptive_pool.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    
    total_step = len(data_loader)
    for epoch in range(num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):
            images = images.to(device)
            captions = captions.to(device)

            features = encoder(images)
            outputs, captions, lengths, _ = decoder(features, captions, lengths, device)
            
            targets = pack_padded_sequence(captions[:, 1:], lengths, batch_first=True)[0]
            outputs = pack_padded_sequence(outputs, lengths, batch_first=True)[0]

            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % log_step == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item()}, Perplexity: {np.exp(loss.item())}')
            
            if (i+1) % save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(model_path, f'decoder-{epoch+1}-{i+1}.ckpt'))
                torch.save(encoder.state_dict(), os.path.join(model_path, f'encoder-{epoch+1}-{i+1}.ckpt'))

if __name__ == '__main__':
    MODEL_PATH = r'rnn_models_with_attention'
    IMAGE_SIZE = 224
    VOCAB_PATH = r'vocab.pkl'
    IMAGE_DIR = r'coco\images\train2017\resized2017'
    CAPTION_PATH = r"coco\annotations_trainval2017\annotations\captions_train2017.json"
    LOG_STEP = 100
    SAVE_STEP = 1000
    EMBED_SIZE = 256
    ATTENTION_SIZE = 384  
    HIDDEN_SIZE = 384
    NUM_EPOCHS = 3
    BATCH_SIZE = 64
    NUM_WORKERS = 1
    LEARNING_RATE = 0.0005

    main(MODEL_PATH, IMAGE_SIZE, VOCAB_PATH, IMAGE_DIR, CAPTION_PATH, LOG_STEP, SAVE_STEP, EMBED_SIZE, ATTENTION_SIZE, HIDDEN_SIZE, NUM_EPOCHS, BATCH_SIZE, NUM_WORKERS, LEARNING_RATE)
