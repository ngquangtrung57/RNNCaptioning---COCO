import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class EncoderCNN(nn.Module):
    def __init__(self, encoded_image_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-3]      
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        
    def forward(self, images):
        with torch.no_grad():
            feat_vecs = self.resnet(images)  
        feat_vecs = self.adaptive_pool(feat_vecs)  
        feat_vecs = feat_vecs.permute(0, 2, 3, 1) 
        return feat_vecs


class Attention(nn.Module):
    def __init__(self, encoder_size, hidden_size, attention_size):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_size, attention_size)
        self.hidden_att = nn.Linear(hidden_size, attention_size)
        self.full_att = nn.Linear(attention_size, 1) 
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1) 

    def forward(self, encoder_out, hidden_out):

        att1 = self.encoder_att(encoder_out)  
        att2 = self.hidden_att(hidden_out)  
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  
        alpha = self.softmax(att)  
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1) 
        return attention_weighted_encoding, alpha


class DecoderRNNWithAttention(nn.Module):
    def __init__(self, embed_size, attention_size, hidden_size, vocab_size, encoder_size=1024, max_seg_length=20):

        super(DecoderRNNWithAttention, self).__init__()
        
        self.attention = Attention(encoder_size=encoder_size, hidden_size=hidden_size, attention_size=attention_size)
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstmcell = nn.LSTMCell(embed_size+encoder_size, hidden_size, bias=True)
        
        self.init_hidden = nn.Linear(encoder_size, hidden_size) 
        self.init_cell = nn.Linear(encoder_size, hidden_size)  
        self.f_beta = nn.Linear(hidden_size, encoder_size) 
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(hidden_size, vocab_size) 
        self.init_weights()
        
        self.vocab_size = vocab_size
        self.max_seg_length = max_seg_length

    def init_weights(self):

        self.embed.weight.data.uniform_(-0.1,0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1,0.1)

    def init_hidden_state(self, encoder_out):

        mean_encoder_out = encoder_out.mean(dim=1)
        hidden = self.init_hidden(mean_encoder_out)
        cell = self.init_cell(mean_encoder_out)
        return hidden, cell

    def forward(self, encoder_out, captions, lengths, device):

        batch_size, encoder_size, vocab_size = encoder_out.size(0), encoder_out.size(-1), self.vocab_size
        encoder_out = encoder_out.view(batch_size, -1, encoder_size)
        num_pixels = encoder_out.size(1)
        
        embeddings = self.embed(captions) 
        hidden, cell = self.init_hidden_state(encoder_out) 
        
        lengths = [l - 1 for l in lengths]
        max_length = max(lengths)
        
        predictions = torch.zeros(batch_size, max_length, vocab_size).to(device)
        alphas = torch.zeros(batch_size, max_length, num_pixels).to(device)
        
        for t in range(max_length):
            batch_size_t = sum([l > t for l in lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], hidden[:batch_size_t])
            gate = self.sigmoid(self.f_beta(hidden[:batch_size_t]))  
            attention_weighted_encoding = gate * attention_weighted_encoding
            hidden, cell = self.lstmcell(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (hidden[:batch_size_t], cell[:batch_size_t])) 
            predictions[:batch_size_t, t, :] = self.fc(hidden)  
            alphas[:batch_size_t, t, :] = alpha

        return predictions, captions, lengths, alphas
    

    def sample(self, encoder_out, vocab, device):
        batch_size = encoder_out.size(0)
        encoder_size = encoder_out.size(-1)
        
        encoder_out = encoder_out.view(batch_size, -1, encoder_size)
        
        hidden, cell = self.init_hidden_state(encoder_out)
        inputs = self.embed(torch.tensor([vocab('<start>')]).to(device)).repeat(batch_size, 1)
        
        sampled_ids = []
        for t in range(self.max_seg_length):
            attention_weighted_encoding, alpha = self.attention(encoder_out, hidden)
            gate = self.sigmoid(self.f_beta(hidden))
            attention_weighted_encoding = gate * attention_weighted_encoding
            hidden, cell = self.lstmcell(
                torch.cat([inputs, attention_weighted_encoding], dim=1),
                (hidden, cell))

            _, predicted = self.fc(hidden).max(1)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)

        sampled_ids = torch.stack(sampled_ids, 1).tolist()
        sampled_ids = [[vocab('<start>')]+s for s in sampled_ids]
        return sampled_ids

    
    def sample_beam_search(self, encoder_out, vocab, device, beam_size=6):
        k = beam_size
        vocab_size = len(vocab)
        encoder_size = encoder_out.size(-1)
        encoder_out = encoder_out.view(1, -1, encoder_size)
        num_pixels = encoder_out.size(1)

        encoder_out = encoder_out.expand(k, num_pixels, encoder_size)  
        k_prev_words = torch.LongTensor([[vocab('<start>')]] * k).to(device)  
        seqs = k_prev_words
        top_k_scores = torch.zeros(k, 1).to(device)  
        complete_seqs = list()
        complete_seqs_scores = list()
        
        hidden, cell = self.init_hidden_state(encoder_out)
        step = 1
        while True:
            embeddings = self.embed(k_prev_words).squeeze(1)
            awe, _ = self.attention(encoder_out, hidden)
            gate = self.sigmoid(self.f_beta(hidden))
            awe = gate * awe
            hidden, cell = self.lstmcell(torch.cat([embeddings, awe], dim=1), (hidden, cell))

            scores = self.fc(hidden)
            scores = F.log_softmax(scores, dim=1)
            scores = top_k_scores.expand_as(scores) + scores

            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, dim=0)  
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, dim=0)  
            prev_word_inds = top_k_words / vocab_size 
            next_word_inds = top_k_words % vocab_size  
            prev_word_inds = prev_word_inds.long()
            next_word_inds = next_word_inds.long()

    # Now use them for concatenation
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  


            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != vocab.word2idx['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))


            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  


            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            hidden = hidden[prev_word_inds[incomplete_inds]]
            cell = cell[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            if step > self.max_seg_length:
                break
            step += 1
	
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        return [seq]

        return complete_seqs