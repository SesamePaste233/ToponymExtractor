import torch
import torch.nn as nn
import math

import numpy as np

from tqdm import tqdm

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
            outputs = self.model(source_bezier_centralized_tensor,
                            query_id_in_source_tensor)

            output_id_in_source = torch.argmax(outputs, dim=-1)

            predicted_word_id_in_source = []
            for output_ in output_id_in_source:
                try:
                    eot_id = output_.flatten().tolist().index(17)
                except:
                    # special case: no <eot> generated, take all the predicted labels
                    eot_id = len(output_.flatten().tolist())

                predicted_word_id_in_source.append((output_.flatten().cpu()).tolist()[:eot_id])

            return predicted_word_id_in_source

    def _get_toponym_sequence_batch(self, source_bezier_lst, query_id_lst):
        source_b_matrix = np.array(source_bezier_lst)
        sample = {'source_bezier_centralized': source_b_matrix, 'query_id_in_source': query_id_lst}

        pts = source_b_matrix[np.arange(source_b_matrix.shape[0]), query_id_lst].T.reshape(8, 2, -1)
        center = np.mean(pts, axis=0)
        
        # Replicate center 8 times to get a 8x2 matrix
        tile_center = np.tile(center.T, (1, 8))

        tile_center = tile_center.reshape(len(query_id_lst), 1, -1)

        sample['source_bezier_centralized'] = source_b_matrix - tile_center

        dict_ids = self.get_toponym_sequence(sample)

        for j, d_ids in enumerate(dict_ids):
            included = set()
            dict_ids[j] = [i for i in d_ids if i not in included and not included.add(i)]

        return dict_ids
    
    def get_toponym_sequence_batch(self, source_bezier_lst, query_id_lst = None, batch_size = 128, show_progress = True):
        if query_id_lst is None:
            query_id_lst = [0 for _ in range(len(source_bezier_lst))]

        dict_ids_lst = []
        if show_progress:
            for i in tqdm(range(0, len(source_bezier_lst), batch_size)):
                dict_ids_lst.extend(self._get_toponym_sequence_batch(source_bezier_lst[i:i+batch_size], query_id_lst[i:i+batch_size]))
        else:
            for i in range(0, len(source_bezier_lst), batch_size):
                dict_ids_lst.extend(self._get_toponym_sequence_batch(source_bezier_lst[i:i+batch_size], query_id_lst[i:i+batch_size]))

        return dict_ids_lst

if __name__ == '__main__':
    sample = {'source_bezier_centralized': [[[362.9012, -126.4581, 401.2112, -126.8081, 439.4812, -126.4481,
          477.7812, -125.3981, 362.0612, -80.3681, 400.6612, -78.5581,
          439.1412, -78.2681, 477.7612, -79.4981],
         [226.9712, -310.9381, 279.7212, -294.8781, 332.4612, -278.8181,
          385.2112, -262.7581, 219.0012, -284.7581, 271.7512, -268.6981,
          324.4912, -252.6381, 377.2412, -236.5781],
         [-119.2388, -360.9281, -82.4388, -351.2781, -46.1488, -340.2981,
          -10.1488, -327.9981, -129.7688, -321.8481, -92.8788, -311.2981,
          -55.7988, -301.7981, -18.3688, -293.3581],
         [-426.4988, -474.6781, -267.1588, -433.9981, -107.8088, -393.3181,
          51.5412, -352.6481, -433.4188, -447.5981, -274.8088, -407.1181,
          -116.1988, -366.6281, 42.4012, -326.1381],
         [-231.8588, -589.7181, -221.2788, -588.8881, -210.6888, -588.0580,
          -200.0988, -587.2281, -241.1388, -553.6781, -228.5288, -551.8881,
          -215.9188, -550.1081, -203.3088, -548.3181],
         [-338.9388, -492.0581, -308.2988, -491.8981, -277.6588, -491.9581,
          -247.0288, -492.2381, -339.8288, -455.8981, -308.9088, -456.1981,
          -278.0088, -456.8481, -247.0988, -457.8381],
         [-66.7388, -619.0580, -18.6888, -617.7581, 29.3612, -616.7581,
          77.4312, -616.0580, -66.9188, -577.9681, -19.2288, -577.8781,
          28.4612, -577.8881, 76.1512, -578.0181],
         [-197.2388, -587.9380, -180.9388, -587.1081, -164.6388, -586.2681,
          -148.3488, -585.4380, -207.5888, -548.3181, -190.5788, -545.9380,
          -173.5688, -543.5580, -156.5488, -541.1781],
         [321.4312, 99.7419, 328.0812, 110.8419, 334.7212, 121.9419,
          341.3712, 133.0419, 287.2212, 120.2319, 293.8712, 131.3219,
          300.5112, 142.4219, 307.1612, 153.5219],
         [-100.6288, -658.1880, -32.2988, -658.7381, 35.9712, -658.0181,
          104.2712, -656.0081, -101.7988, -619.1581, -33.3688, -618.3281,
          35.0612, -617.9481, 103.4912, -618.0381],
         [343.3212, 134.2519, 350.3712, 146.0419, 357.4212, 157.8319,
          364.4612, 169.6219, 308.1512, 155.2719, 315.1912, 167.0619,
          322.2412, 178.8519, 329.2812, 190.6419],
         [361.3212, -81.2381, 400.1912, -81.7881, 438.9612, -81.1381,
          477.7912, -79.2981, 360.8512, -34.2881, 399.8412, -34.5481,
          438.8112, -35.2081, 477.7812, -36.2581],
         [-301.3688, -153.8581, -224.7988, -151.5381, -148.3188, -151.1281,
          -71.7288, -152.6181, -303.1288, -103.5381, -225.8188, -103.4381,
          -148.5188, -103.9381, -71.2188, -105.0481],
         [18.7212, -22.7581, 19.2012, -8.0081, 19.6712, 6.7519,
          20.1512, 21.5019, -20.1488, -21.4981, -19.6788, -6.7481,
          -19.1988, 8.0019, -18.7188, 22.7519],
         [20.1512, 21.0219, 19.8312, 35.9419, 19.5212, 50.8519,
          19.2012, 65.7619, -17.9488, 20.2119, -18.2688, 35.1219,
          -18.5888, 50.0419, -18.9088, 64.9519],
         [-238.8088, -392.3081, -203.7288, -383.0481, -168.7788, -373.3681,
          -133.9388, -363.2381, -249.3788, -353.2181, -214.2688, -343.0581,
          -178.9788, -333.8081, -143.3988, -325.4681]],
        [[-3.5858e+01, -2.2608e+01, -1.1798e+01, -2.3008e+01, 1.2202e+01,
          -2.2618e+01, 3.6242e+01, -2.1428e+01, -3.6218e+01, 2.2882e+01,
          -1.2188e+01, 2.2132e+01, 1.1802e+01, 2.2042e+01, 3.5832e+01,
          2.2612e+01],
         [-1.2060e+02, 5.5779e+02, -4.5408e+01, 5.5925e+02, 2.9732e+01,
          5.5940e+02, 1.0493e+02, 5.5822e+02, -1.2298e+02, 6.1211e+02,
          -4.6868e+01, 6.1282e+02, 2.9242e+01, 6.1331e+02, 1.0535e+02,
          6.1357e+02],
         [5.3441e+02, 5.5515e+02, 5.4085e+02, 5.3184e+02, 5.4679e+02,
          5.0843e+02, 5.5222e+02, 4.8487e+02, 5.5580e+02, 5.6107e+02,
          5.6167e+02, 5.3794e+02, 5.6814e+02, 5.1501e+02, 5.7520e+02,
          4.9221e+02],
         [4.2312e+01, -4.6433e+02, 8.7772e+01, -4.6482e+02, 1.3323e+02,
          -4.6491e+02, 1.7870e+02, -4.6462e+02, 4.1632e+01, -4.1018e+02,
          8.7392e+01, -4.0792e+02, 1.3290e+02, -4.0796e+02, 1.7865e+02,
          -4.1030e+02],
         [-6.0518e+01, 4.3762e+01, -2.2277e+00, 6.3972e+01, 5.5432e+01,
          8.5622e+01, 1.1264e+02, 1.0870e+02, -7.9368e+01, 8.6132e+01,
          -2.0788e+01, 1.0625e+02, 3.7402e+01, 1.2733e+02, 9.5302e+01,
          1.4934e+02],
         [-6.3477e+00, 6.1021e+02, 3.1232e+01, 6.1201e+02, 6.8812e+01,
          6.1352e+02, 1.0641e+02, 6.1473e+02, -6.2777e+00, 6.6089e+02,
          3.1872e+01, 6.6107e+02, 7.0012e+01, 6.6113e+02, 1.0816e+02,
          6.6109e+02],
         [4.6962e+01, -2.1798e+01, 8.5102e+01, -2.3028e+01, 1.2325e+02,
          -2.3908e+01, 1.6141e+02, -2.4448e+01, 4.5932e+01, 2.3342e+01,
          8.4272e+01, 2.3282e+01, 1.2260e+02, 2.3032e+01, 1.6093e+02,
          2.2562e+01],
         [4.4035e+02, 3.4328e+02, 4.8589e+02, 3.4249e+02, 5.3136e+02,
          3.4276e+02, 5.7688e+02, 3.4408e+02, 4.4034e+02, 3.9869e+02,
          4.8584e+02, 3.9781e+02, 5.3134e+02, 3.9680e+02, 5.7683e+02,
          3.9567e+02],
         [6.7456e+02, 5.0523e+02, 6.8039e+02, 4.7596e+02, 6.8622e+02,
          4.4669e+02, 6.9205e+02, 4.1742e+02, 6.9412e+02, 5.0912e+02,
          6.9995e+02, 4.7985e+02, 7.0578e+02, 4.5059e+02, 7.1161e+02,
          4.2132e+02],
         [-1.7548e+01, 2.2913e+02, -1.1598e+01, 2.4212e+02, -5.6477e+00,
          2.5511e+02, 3.0230e-01, 2.6809e+02, -5.6208e+01, 2.4103e+02,
          -4.9868e+01, 2.5441e+02, -4.3518e+01, 2.6780e+02, -3.7178e+01,
          2.8118e+02],
         [-3.8528e+01, 1.8915e+02, -3.2658e+01, 2.0216e+02, -2.6788e+01,
          2.1517e+02, -2.0918e+01, 2.2818e+02, -7.8028e+01, 2.0438e+02,
          -7.3108e+01, 2.1676e+02, -6.8188e+01, 2.2913e+02, -6.3268e+01,
          2.4150e+02],
         [5.5098e+02, 4.8007e+02, 5.5697e+02, 4.5910e+02, 5.6234e+02,
          4.3803e+02, 5.6710e+02, 4.1675e+02, 5.7338e+02, 4.8586e+02,
          5.7873e+02, 4.6508e+02, 5.8455e+02, 4.4447e+02, 5.9084e+02,
          4.2396e+02],
         [-1.5948e+02, -1.3723e+02, 1.9228e+02, -1.4000e+02, 5.4404e+02,
          -1.4278e+02, 8.9581e+02, -1.4556e+02, -1.7019e+02, -3.6098e+01,
          1.8356e+02, -3.7688e+01, 5.3730e+02, -3.9268e+01, 8.9105e+02,
          -4.0858e+01],
         [1.2361e+02, 1.1101e+02, 1.8219e+02, 1.3315e+02, 2.4090e+02,
          1.5491e+02, 2.9975e+02, 1.7629e+02, 1.0622e+02, 1.5346e+02,
          1.6563e+02, 1.7547e+02, 2.2512e+02, 1.9727e+02, 2.8469e+02,
          2.1888e+02],
         [-5.1888e+01, -4.6461e+02, -2.2228e+01, -4.6489e+02, 7.4323e+00,
          -4.6499e+02, 3.7092e+01, -4.6492e+02, -5.2948e+01, -4.1479e+02,
          -2.3228e+01, -4.1372e+02, 6.4523e+00, -4.1338e+02, 3.6182e+01,
          -4.1376e+02],
         [4.3159e+02, 2.7129e+02, 4.7765e+02, 2.7046e+02, 5.2367e+02,
          2.7043e+02, 5.6973e+02, 2.7119e+02, 4.3097e+02, 3.2168e+02,
          4.7717e+02, 3.2222e+02, 5.2330e+02, 3.2175e+02, 5.6947e+02,
          3.2026e+02]]],
        'query_id_in_source': [13, 0]}

    grouper = GrouperCaller('./Grouper/grouper_model_v1_epoch2.pth')

    print(grouper.get_toponym_sequence_batch(sample['source_bezier_centralized'], sample['query_id_in_source']))
