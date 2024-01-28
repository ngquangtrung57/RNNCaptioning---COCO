import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import pickle
import time
from rnn_model_with_attention import EncoderCNN, DecoderRNNWithAttention
from vocab_preprocessing import Vocabulary

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
    st.title("Image Captioning with Attention Model")
    st.write("Upload an image and the AI will generate a caption for it.")
    
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    encoder = EncoderCNN(encoded_image_size=14).eval().to(device)
    decoder = DecoderRNNWithAttention(embed_size=256, attention_size=384, hidden_size=384, vocab_size=len(vocab)).eval().to(device)

    encoder.load_state_dict(torch.load('rnn_models_with_attention/encoder-3-9000.ckpt', map_location=device))
    decoder.load_state_dict(torch.load('rnn_models_with_attention/decoder-3-9000.ckpt', map_location=device))

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        with st.spinner('🤖 AI is at work... '):
            time.sleep(2) 
            caption = generate_caption(uploaded_file, encoder, decoder, vocab, transform)
            
        st.markdown(f"<h3 style='color: green;'>Generated Caption:</h3><p style='font-size: large'>{caption}</p>", unsafe_allow_html=True)
if __name__ == '__main__':
    main()


