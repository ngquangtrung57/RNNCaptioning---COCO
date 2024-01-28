import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import pickle
from rnn_model import EncoderCNN, DecoderRNN
from vocab_preprocessing import Vocabulary
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def generate_caption(image_path, encoder, decoder, vocab, transform):
    image_tensor = load_image(image_path, transform).to(device)    
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample_beam_search(feature, vocab, device)
    sampled_ids = sampled_ids[0][1:-1]  
    
    sampled_caption = [vocab.idx2word[word_id] for word_id in sampled_ids]
    caption = ' '.join(sampled_caption)
    
    return caption

def main():
    st.markdown("<h1 style='text-align: center; color: blue;'>Image Captioning with RNN Model</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Upload an image and the AI will generate a caption for it.</h4>", unsafe_allow_html=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    encoder = EncoderCNN(256).eval().to(device)
    decoder = DecoderRNN(256, 384, len(vocab), 1).to(device)

    encoder.load_state_dict(torch.load('rnn_models/encoder-3.ckpt', map_location=device))
    decoder.load_state_dict(torch.load('rnn_models/decoder-3.ckpt', map_location=device))

    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.write("<h3 style='color: green;'>Uploaded Image:</h3>", unsafe_allow_html=True)
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        
        with st.spinner('🤖 AI is at work... '):
            time.sleep(2)  
            caption = generate_caption(uploaded_file, encoder, decoder, vocab, transform)
        
        st.markdown(f"<h3 style='color: green;'>Generated Caption:</h3><p style='font-size: large'>{caption}</p>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()





