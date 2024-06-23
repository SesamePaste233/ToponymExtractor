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

    def forward(self, x_decoder: torch.Tensor, x_encoder: torch.Tensor, mask: torch.Tensor, eps: float = 1e-16) -> torch.Tensor:

        # (B, Nd, Ne, C) <- (B, Ne, C)
        encoder_transform = self.w1(x_encoder).unsqueeze(1).expand(-1, x_decoder.shape[1], -1, -1)
        # (B, Nd, 1, C) <- (B, Nd, C)
        decoder_transform = self.w2(x_decoder).unsqueeze(2)
        # (B, Nd, Ne) <- (B, Nd, Ne, C), (B, Nd, 1, C)
        prod = self.v(torch.tanh(encoder_transform + decoder_transform)).squeeze(-1)
        # Apply mask: positions with True are not allowed to attend
        prod = prod.masked_fill(mask.unsqueeze(1).expand(-1, prod.shape[1], -1), float('-inf'))
        # (B, Nd, Ne) <- (B, Nd, Ne)
        log_score = torch.nn.functional.log_softmax(prod, dim=-1)
        return log_score


class GrouperTransformerEncoderDecoderAttentionNoFontPadded(nn.Module):
    def __init__(self, feature_dim=16, hidden_dim=64, nhead=2, num_encoder_layers=1, num_decoder_layers=1):
        super(GrouperTransformerEncoderDecoderAttentionNoFontPadded, self).__init__()

        self.type = 'GrouperTransformerEncoderDecoderAttentionNoFontPadded'
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
        self.special_token_id = {'<sot>': 0, '<eot>': 1, '<pad>': 18}
        # self.output_layer = nn.Linear(hidden_dim*2, 18)
        self.pointer = PointerNetwork(hidden_dim)

        print('{} loaded.'.format(self.type))

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, query_id, src_len, tgt_ids=None, tgt_len=None):

        # src: (N,L,f_sz)  src: (N,L+3,f_sz+2)  src_len: (N, 1)
        # append special tokens features to input, <sot> and <eot> are binary vectors
        src_padded = torch.zeros((src.shape[0], src.shape[1] + 2, src.shape[2] + 2), dtype=torch.float).to(src.device)
        src_padded[:, 0, src.shape[2]] = 1.0  # add <sot> vector
        src_padded[:, 1, src.shape[2] + 1] = 1.0  # add <eot> vector
        src_padded[:, 2:, :src.shape[2]] = src

        src_padded_emb = self.pre_project(src_padded)
        # src_padded_emb = self.pos_encoder(src_padded_emb)

        src_padding_mask = torch.zeros((src_padded_emb.shape[0], src_padded_emb.shape[1]), dtype=torch.bool)
        for i in range(src_padding_mask.shape[0]):
            src_padding_mask[i, src_len[i] + 2:] = True
        src_padding_mask = src_padding_mask.to(src.device)

        memory = self.transformer_encoder(src_padded_emb,
                                          # src_key_padding_mask=src_padding_mask
                                          )
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

            tgt_key_padding_mask = torch.zeros_like(tgt_ids, dtype=torch.bool)
            for i in range(tgt_len.shape[0]):
                tgt_key_padding_mask[i, tgt_len[i] + 1:] = True
            tgt_key_padding_mask = tgt_key_padding_mask.to(src.device)
            # tgt_mask = self.generate_attention_mask(decoder_input.shape[1]).to(src.device)
            tgt_mask = self._generate_square_subsequent_mask(sz = decoder_input.shape[1]).to(src.device)

            decoder_hidden = self.transformer_decoder(decoder_input,
                                                      memory,
                                                      tgt_mask=tgt_mask,
                                                      tgt_key_padding_mask=tgt_key_padding_mask,
                                                      memory_key_padding_mask=src_padding_mask
                                                      )

            # decoder_output = self.output_layer(decoder_hidden)

            log_pointer_scores = self.pointer(decoder_hidden, memory, src_padding_mask)

        # inference or training without teacher forcing
        else:
            query_tensors = memory[torch.arange(memory.shape[0]), query_id].unsqueeze(1)

            decoder_input_cat = torch.cat([memory[:, 0, :].unsqueeze(1), query_tensors], dim=-1)  # <sot; query>

            decoder_input = self.reduce_decoder_input_dim(decoder_input_cat)
            # decoder_input = self.pos_decoder(decoder_input)

            tgt_mask = self._generate_square_subsequent_mask(sz = decoder_input.shape[1]).to(src.device)

            log_pointer_scores = torch.zeros((memory.shape[0], memory.shape[1], memory.shape[1]), dtype=torch.float).to(src.device)

            for t in range(memory.shape[1]-1):

                decoder_hidden = self.transformer_decoder(decoder_input,
                                                          memory,
                                                          memory_key_padding_mask=src_padding_mask,
                                                          tgt_mask=tgt_mask)
                log_pointer_score = self.pointer(decoder_hidden, memory, src_padding_mask)
                log_pointer_scores[:, t, :] = log_pointer_score[:, t, :]

                next_id = torch.argmax(log_pointer_score[:, t, :], dim=-1)
                next_input = memory[torch.arange(memory.shape[0]), next_id]
                if t == memory.shape[1]-2:
                    break
                else:
                    next_input_cat = torch.cat([next_input.unsqueeze(1), query_tensors], dim=-1)
                    decoder_input = torch.cat([decoder_input, self.reduce_decoder_input_dim(next_input_cat)], dim=1)
                    tgt_mask = self._generate_square_subsequent_mask(sz = decoder_input.shape[1]).to(src.device)

        return log_pointer_scores


class GrouperCaller:
    def __init__(self, checkpoint_path, device='cuda'):

        self.device = device

        self.sot_id = 0
        self.eot_id = 1
        self.max_source_length = 16
        self.max_source_length_include_spec = self.max_source_length + 2

        self.pad_id = self.max_source_length_include_spec

        self.model = torch.load(checkpoint_path)
        self.model = self.model.to(self.device)

    def get_toponym_sequence(self, sample_dict):

        source_bezier_centralized_tensor = torch.tensor(sample_dict['source_bezier_centralized'], dtype=torch.float)

        source_length = source_bezier_centralized_tensor.shape[0]

        if source_length < self.max_source_length:
            num_to_pad = self.max_source_length - source_length
            feature_size = source_bezier_centralized_tensor.shape[1]
            bezier_padding_tensor = torch.zeros((num_to_pad, feature_size), dtype=torch.float)

            source_bezier_centralized_tensor = torch.cat([source_bezier_centralized_tensor,
                                                          bezier_padding_tensor],
                                                         dim=0)

        source_bezier_centralized_tensor = source_bezier_centralized_tensor.to(self.device)
        query_id_in_source_tensor = torch.tensor(sample_dict['query_id_in_source'], dtype=torch.long).to(self.device) + 2
        source_len_tensor = torch.tensor(len(source_bezier_centralized_tensor), dtype=torch.long).to(self.device)

        with torch.no_grad():
            outputs = self.model(source_bezier_centralized_tensor.unsqueeze(0),
                            query_id_in_source_tensor.unsqueeze(0),
                            source_len_tensor.unsqueeze(0))

            output_id_in_source = torch.argmax(outputs, dim=-1)

            try:
                eot_id = output_id_in_source.flatten().tolist().index(1)
            except:
                # special case: no <eot> generated, take all the predicted labels
                eot_id = len(output_id_in_source.flatten().tolist())

            predicted_word_id_in_source = (output_id_in_source.flatten().cpu() - 2).tolist()[:eot_id]

            return predicted_word_id_in_source
        
    def get_toponym_sequence2(self, source_bezier, query_id = 0):
        sample = {'source_bezier_centralized': np.array(source_bezier), 'query_id_in_source': query_id}

        pts = np.array(source_bezier[query_id]).reshape(8, 2)
        center = np.mean(pts, axis=0)
        
        # Replicate center 8 times to get a 8x2 matrix
        center = np.tile(center, (8, 1)).reshape(1, -1)

        sample['source_bezier_centralized'] = np.array(sample['source_bezier_centralized']) - center

        dict_ids = self.get_toponym_sequence(sample)
        
        included = set()
        dict_ids = [i for i in dict_ids if i not in included and not included.add(i)]

        return dict_ids

if __name__ == '__main__':

    sample = {'source_bezier_centralized': [[ 433.6984,  119.8897,  475.7384,  120.0397,  517.7684,  120.1897,
           559.8084,  120.3297,  431.9584,  154.3597,  474.3384,  154.5097,
           516.7084,  154.8397,  559.0884,  155.3297],
         [ 174.1184,  209.0697,  179.9484,  179.7997,  185.7784,  150.5297,
           191.6084,  121.2597,  193.6784,  212.9597,  199.5084,  183.6897,
           205.3384,  154.4297,  211.1684,  125.1597],
         [ -68.8516,  -24.8703,  -22.7916,  -25.7003,   23.2284,  -25.7303,
            69.2884,  -24.9703,  -69.4716,   25.5197,  -23.2716,   26.0597,
            22.8584,   25.5897,   69.0284,   24.0997],
         [ 280.4784,  199.4297,  292.0884,  161.5097,  303.6984,  123.5897,
           315.3084,   85.6697,  320.9584,  211.8197,  332.5684,  173.8997,
           344.1784,  135.9797,  355.7884,   98.0597],
         [ -60.0916,   47.1197,  -14.5516,   46.3297,   30.9184,   46.5997,
            76.4384,   47.9197,  -60.1016,  102.5297,  -14.6016,  101.6497,
            30.8984,  100.6397,   76.3884,   99.5097],
         [  33.9684,  258.9897,   40.4084,  235.6797,   46.3484,  212.2697,
            51.7784,  188.7097,   55.3584,  264.9097,   61.2284,  241.7797,
            67.6984,  218.8497,   74.7584,  196.0497],
         [ 154.1584,  284.1197,  161.0484,  259.5797,  166.6184,  234.8997,
           170.8484,  209.7697,  173.5984,  292.4497,  179.5984,  266.7997,
           186.0384,  241.2797,  192.9284,  215.8497],
         [ 443.0084,  -11.5703,  495.3584,  -16.5703,  547.7284,  -21.4103,
           600.1084,  -26.1203,  445.3284,   45.8897,  498.0084,   40.2797,
           550.7284,   35.2897,  603.5184,   30.9197],
         [-376.8316, -185.1503, -318.2516, -163.0103, -259.5416, -141.2503,
          -200.6916, -119.8703, -394.2216, -142.7003, -334.8116, -120.6903,
          -275.3216,  -98.8903, -215.7516,  -77.2803]],
        'query_id_in_source': 2}

    grouper = GrouperCaller('./checkpoints/grouper_model_epoch3.pth')

    print(grouper.get_toponym_sequence(sample))