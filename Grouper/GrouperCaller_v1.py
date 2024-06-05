import torch
import torch.nn as nn
import math

import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class PointerNetwork(nn.Module):
    def __init__(self, n_hidden: int):
        super().__init__()
        self.n_hidden = n_hidden
        self.w1 = nn.Linear(n_hidden, n_hidden, bias=False)
        self.w2 = nn.Linear(n_hidden, n_hidden, bias=False)
        self.v = nn.Linear(n_hidden, 1, bias=False)

    def forward(self, x_decoder: torch.Tensor, x_encoder: torch.Tensor, eps: float = 1e-16) -> torch.Tensor:

        # (B, Nd, Ne, C) <- (B, Ne, C)
        encoder_transform = self.w1(x_encoder).unsqueeze(1).expand(
          -1, x_decoder.shape[1], -1, -1)
        # (B, Nd, 1, C) <- (B, Nd, C)
        decoder_transform = self.w2(x_decoder).unsqueeze(2)
        # (B, Nd, Ne) <- (B, Nd, Ne, C), (B, Nd, 1, C)
        prod = self.v(torch.tanh(encoder_transform + decoder_transform)).squeeze(-1)
        # (B, Nd, Ne) <- (B, Nd, Ne)
        log_score = torch.nn.functional.log_softmax(prod, dim=-1)
        return log_score


class GrouperTransformerEncoderDecoderAttentionNoFont(nn.Module):
    def __init__(self, feature_dim=16, hidden_dim=64, nhead=2, num_encoder_layers=1, num_decoder_layers=1):
        super(GrouperTransformerEncoderDecoderAttentionNoFont, self).__init__()

        self.type = 'GrouperTransformerEncoderDecoderAttentionNoFont'
        self.pre_project = nn.Linear(feature_dim+2, hidden_dim, bias=False)  # shared across encoder and decoder
        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.pos_decoder = PositionalEncoding(hidden_dim*2)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead,
                                                        dim_feedforward=hidden_dim*2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        # self.post_project = nn.Linear(hidden_dim, hidden_dim*2)
        self.reduce_decoder_input_dim = nn.Linear(hidden_dim*2, hidden_dim, bias=False)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead,
                                                        dim_feedforward=hidden_dim*2, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)
        self.special_token_id = {'<sot>': 16, '<eot>': 17, '<pad>': 18}
        # self.output_layer = nn.Linear(hidden_dim*2, 18)
        self.pointer = PointerNetwork(hidden_dim)

        print('{} loaded.'.format(self.type))

    def generate_attention_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        # mask = mask.masked_fill(mask == 0, float(0.0))
        return mask

    def generate_padding_mask(self, tgt_ids, tgt_len):
        tgt_key_padding_mask = torch.zeros_like(tgt_ids, dtype=torch.bool)
        for i in range(tgt_len.shape[0]):
            tgt_key_padding_mask[i, tgt_len[i] + 1:] = True
        return tgt_key_padding_mask

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, query_id, tgt_ids=None, tgt_len=None):

        # src: (N,L,f_sz)  src: (N,L+3,f_sz+2)
        # append special tokens features to input, <sot> and <eot> are binary vectors
        src_padded = torch.zeros((src.shape[0], src.shape[1] + 2, src.shape[2] + 2), dtype=torch.float).to(src.device)
        src_padded[:, src.shape[1], src.shape[2]] = 1.0  # add <sot> vector
        src_padded[:, src.shape[1] + 1, src.shape[2] + 1] = 1.0  # add <eot> vector
        src_padded[:, :src.shape[1], :src.shape[2]] = src

        src_padded_emb = self.pre_project(src_padded)
        # src_padded_emb = self.pos_encoder(src_padded_emb)

        # src_padding_mask = torch.zeros((src_padded_emb.shape[0], src_padded_emb.shape[1]), dtype=torch.bool)
        # src_padding_mask[:, -1] = True
        # src_padding_mask = src_padding_mask.to(src.device)
        memory = self.transformer_encoder(src_padded_emb)
        # memory = self.post_project(memory)

        # src_padded: (N,L+3,h_dim)  src,memory: (N,L+2,h_dim)  query_id: (N,1)  tgt_ids: (N,L+2)
        # training by teacher forcing
        if tgt_ids is not None:
            batch_indices = torch.arange(tgt_ids.shape[0]).unsqueeze(1).expand(tgt_ids.shape[0], tgt_ids.shape[1])
            # temporarily append one zero vector to facilitate tensor extraction
            memory_padded = torch.zeros((memory.shape[0], memory.shape[1]+1, memory.shape[2]), dtype=torch.float).to(src.device)
            memory_padded[:, :-1, :] = memory
            tgt_tensors = memory_padded[batch_indices, tgt_ids]
            query_tensors = memory[torch.arange(src.shape[0]), query_id].unsqueeze(1).expand(-1, tgt_tensors.shape[1], -1)
            decoder_input_cat = torch.cat([tgt_tensors, query_tensors], dim=-1)
            decoder_input = self.reduce_decoder_input_dim(decoder_input_cat)
            # decoder_input = self.pos_decoder(decoder_input)

            tgt_key_padding_mask = self.generate_padding_mask(tgt_ids, tgt_len).to(src.device)
            # tgt_mask = self.generate_attention_mask(decoder_input.shape[1]).to(src.device)
            tgt_mask = self._generate_square_subsequent_mask(decoder_input.shape[1]).to(src.device)

            decoder_hidden = self.transformer_decoder(decoder_input,
                                                      memory,
                                                      # memory_key_padding_mask=src_padding_mask,
                                                      tgt_mask=tgt_mask,
                                                      tgt_key_padding_mask=tgt_key_padding_mask)

            # decoder_output = self.output_layer(decoder_hidden)

            log_pointer_scores = self.pointer(decoder_hidden, memory)

        # inference or training without teacher forcing
        else:
            query_tensors = memory[torch.arange(memory.shape[0]), query_id].unsqueeze(1)

            decoder_input_cat = torch.cat([memory[:, 16, :].unsqueeze(1), query_tensors], dim=-1)  # <sot; query>

            decoder_input = self.reduce_decoder_input_dim(decoder_input_cat)
            # decoder_input = self.pos_decoder(decoder_input)

            tgt_mask = self._generate_square_subsequent_mask(decoder_input.shape[1]).to(src.device)

            log_pointer_scores = torch.zeros((memory.shape[0], memory.shape[1], memory.shape[1]), dtype=torch.float).to(src.device)

            for t in range(memory.shape[1]-1):

                decoder_hidden = self.transformer_decoder(decoder_input,
                                                          memory,
                                                          # memory_key_padding_mask=src_padding_mask,
                                                          tgt_mask=tgt_mask)
                log_pointer_score = self.pointer(decoder_hidden, memory)
                log_pointer_scores[:, t, :] = log_pointer_score[:, t, :]

                next_id = torch.argmax(log_pointer_score[:, t, :], dim=-1)
                next_input = memory[torch.arange(memory.shape[0]), next_id]
                if t == memory.shape[1]-2:
                    break
                else:
                    next_input_cat = torch.cat([next_input.unsqueeze(1), query_tensors], dim=-1)
                    decoder_input = torch.cat([decoder_input, self.reduce_decoder_input_dim(next_input_cat)], dim=1)
                    tgt_mask = self._generate_square_subsequent_mask(decoder_input.shape[1]).to(src.device)

        return log_pointer_scores


class GrouperCaller:
    def __init__(self, checkpoint_path, device='cuda'):

        self.device = device

        self.model = torch.load(checkpoint_path)
        self.model = self.model.to(self.device)

    def get_toponym_sequence(self, sample_dict):

        source_bezier_centralized_tensor = torch.tensor(sample_dict['source_bezier_centralized'], dtype=torch.float).to(self.device)
        query_id_in_source_tensor = torch.tensor(sample_dict['query_id_in_source'], dtype=torch.long).to(self.device)

        with torch.no_grad():
            outputs = self.model(source_bezier_centralized_tensor.unsqueeze(0),
                            query_id_in_source_tensor.unsqueeze(0))

            output_id_in_source = torch.argmax(outputs, dim=-1)

            try:
                eot_id = output_id_in_source.flatten().tolist().index(17)
            except:
                # special case: no <eot> generated, take all the predicted labels
                eot_id = len(output_id_in_source.flatten().tolist())

            predicted_word_id_in_source = (output_id_in_source.flatten().cpu()).tolist()[:eot_id]

            return predicted_word_id_in_source
    
    def get_toponym_sequence2(self, source_bezier, query_id = 0):
        sample = {'source_bezier_centralized': source_bezier, 'query_id_in_source': query_id}

        pts = np.array(source_bezier[query_id]).reshape(8, 2)
        center = np.mean(pts, axis=0)
        
        # Replicate center 8 times to get a 8x2 matrix
        center = np.tile(center, (8, 1)).reshape(1, -1)

        sample['source_bezier_centralized'] = np.array(sample['source_bezier_centralized']) - center

        return self.get_toponym_sequence(sample)