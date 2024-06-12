from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import random
import json
from collections import defaultdict
from flask_cors import CORS

# Model Definitions
class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, encoder_hidden_dim, decoder_hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, encoder_hidden_dim, bidirectional=True)
        self.fc = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()
        self.attn_fc = nn.Linear((encoder_hidden_dim * 2) + decoder_hidden_dim, decoder_hidden_dim)
        self.v_fc = nn.Linear(decoder_hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[1]
        src_length = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_length, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn_fc(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v_fc(energy).squeeze(2)
        return torch.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, encoder_hidden_dim, decoder_hidden_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.GRU((encoder_hidden_dim * 2) + embedding_dim, decoder_hidden_dim)
        self.fc_out = nn.Linear((encoder_hidden_dim * 2) + decoder_hidden_dim + embedding_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        return prediction, hidden.squeeze(0), a.squeeze(1)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        trg_length = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_length, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        input = trg[0, :]
        for t in range(1, trg_length):
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs

# Function to load vocabulary
def load_vocab(file_path):
    with open(file_path, encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    stoi = defaultdict(lambda: 0, {token: index for index, token in enumerate(vocab_data)})
    itos = {index: token for token, index in stoi.items()}
    return stoi, itos

# Function to load model and vocabularies
def load_model(model_path, ge_vocab_path, am_vocab_path, device):
    ge_stoi, ge_itos = load_vocab(ge_vocab_path)
    am_stoi, am_itos = load_vocab(am_vocab_path)
    
    input_dim = len(ge_stoi)
    output_dim = len(am_stoi)
    encoder_embedding_dim = 256
    decoder_embedding_dim = 256
    encoder_hidden_dim = 512
    decoder_hidden_dim = 512
    encoder_dropout = 0.5
    decoder_dropout = 0.5

    attention = Attention(encoder_hidden_dim, decoder_hidden_dim)
    encoder = Encoder(input_dim, encoder_embedding_dim, encoder_hidden_dim, decoder_hidden_dim, encoder_dropout)
    decoder = Decoder(output_dim, decoder_embedding_dim, encoder_hidden_dim, decoder_hidden_dim, decoder_dropout, attention)

    model = Seq2Seq(encoder, decoder, device).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, ge_stoi, ge_itos, am_stoi, am_itos

# Function to translate sentences
def translate_sentence(sentence, model, ge_stoi, ge_itos, am_stoi, am_itos, sos_token, eos_token, device, max_output_length=25):
    model.eval()
    with torch.no_grad():
        ge_tokens = sentence.split()
        ge_tokens = [sos_token] + ge_tokens + [eos_token]
        ge_ids = [ge_stoi[token] if token in ge_stoi else ge_stoi['<unk>'] for token in ge_tokens]
        src_tensor = torch.LongTensor(ge_ids).unsqueeze(-1).to(device)
        encoder_outputs, hidden = model.encoder(src_tensor)
        trg_ids = [am_stoi[sos_token]]
        attentions = torch.zeros(max_output_length, 1, len(ge_ids))
        unknown_words = []

        for i in range(max_output_length):
            trg_tensor = torch.LongTensor([trg_ids[-1]]).to(device)
            output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs)
            attentions[i] = attention
            predicted_token = output.argmax(-1).item()
            
            if predicted_token == am_stoi['<unk>']:
                if i < len(ge_tokens):
                    unknown_word = ge_tokens[i]
                    unknown_words.append(unknown_word)
                trg_ids.append(am_stoi['<unk>'])
            else:
                trg_ids.append(predicted_token)
                
            if predicted_token == am_stoi[eos_token]:
                break
        
        translation = []
        for idx, token_id in enumerate(trg_ids):
            if token_id == am_stoi['<unk>']:
                translation.append(ge_tokens[idx] if idx < len(ge_tokens) else '<unk>')
            elif am_itos[token_id] not in [sos_token, eos_token]:  # Remove <bos> and <eos>
                translation.append(am_itos[token_id])

    return translation, unknown_words

# Function to split paragraph into sentences
def split_paragraph(paragraph):
    sentences = paragraph.split('።')
    return [sentence.strip() for sentence in sentences if sentence.strip()]

# Function to translate the whole paragraph
def translate_paragraph(paragraph, model, ge_stoi, ge_itos, am_stoi, am_itos, sos_token, eos_token, device):
    sentences = split_paragraph(paragraph)
    translated_sentences = []
    all_unknown_words = []

    for sentence in sentences:
        translation, unknown_words = translate_sentence(sentence, model, ge_stoi, ge_itos, am_stoi, am_itos, sos_token, eos_token, device)
        translated_sentence = ' '.join(translation) + ' ።'
        translated_sentences.append(translated_sentence)
        all_unknown_words.extend(unknown_words)

    translation = ' '.join(translated_sentences)
    # Remove consecutive '።' punctuations
    translation = translation.replace('። ።', '።')
    return translation, all_unknown_words

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/translate": {"origins": ["http://localhost:5173", "https://geeztoamharic-translator.vercel.app"]}})
# CORS(app, resources={r"/translate": {"origins": ["http://localhost:5173", "https://geeztoamharic-translator.vercel.app"]}})
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, ge_stoi, ge_itos, am_stoi, am_itos = load_model('tut3-model.pt', 'ge_vocab.json', 'am_vocab.json', device)
sos_token = '<bos>'
eos_token = '<eos>'

# Define translation endpoint for paragraphs
@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    paragraph = data['paragraph']
    translation, unknown_words = translate_paragraph(paragraph, model, ge_stoi, ge_itos, am_stoi, am_itos, sos_token, eos_token, device)
    # Remove <bos> and <eos> from the translation
    translation = translation.replace(sos_token, "").replace(eos_token, "").strip()
    return jsonify({'translation': translation, 'unknown_words': unknown_words})

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
