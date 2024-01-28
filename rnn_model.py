import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-1]     
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, states=None):
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)         
            outputs = self.linear(hiddens.squeeze(1))            
            _, predicted = outputs.max(1)                        
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                      
            inputs = inputs.unsqueeze(1)                         
        sampled_ids = torch.stack(sampled_ids, 1)                
        return sampled_ids
    
    def sample_beam_search(self, features, vocab, device, beam_size=4):
        k = beam_size
        vocab_size = len(vocab)
        encoder_size = features.size(-1)
        features = features.view(1, 1, encoder_size)
        inputs = features.expand(k, 1, encoder_size)

        top_k_scores = torch.zeros(k, 1).to(device)
        seqs = torch.zeros(k, 1).long().to(device)
        complete_seqs = list()
        complete_seqs_scores = list()
        
        step = 1
        hidden, cell = None, None

        while True:
            if step == 1:
                outputs, (hidden, cell) = self.lstm(inputs, None)
            else:
                outputs, (hidden, cell) = self.lstm(inputs, (hidden, cell))

            outputs = self.linear(outputs.squeeze(1))
            scores = F.log_softmax(outputs, dim=1)
            scores = top_k_scores.expand_as(scores) + scores

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, dim=0)  
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, dim=0)  

            prev_word_inds = (top_k_words / vocab_size).long()  
            next_word_inds = top_k_words % vocab_size  
            if step==1:
                seqs = next_word_inds.unsqueeze(1)
            else:
                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1) 

            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != vocab('<end>')]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds) 

            if k == 0:
                break

            seqs = seqs[incomplete_inds]
            hidden = hidden[:, prev_word_inds[incomplete_inds]]
            cell = cell[:, prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
            inputs = self.embed(k_prev_words)
            if step > self.max_seg_length:
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        return [seq]
    
        return complete_seqs