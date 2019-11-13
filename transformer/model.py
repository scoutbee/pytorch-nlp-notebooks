import math

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .batch import no_peak_mask, create_masks
from .layer import EncoderLayer, DecoderLayer
from .embed import Embedder, PositionalEncoder
from .sublayer import Norm


def get_layer_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_layer_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    
    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)

    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_layer_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab_size, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab_size)
    
    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output
    
    def pred_init_seq(self, src, src_mask, sos, max_seq_len, beam_size):
        batch_size = len(src)
        encoder_output = self.encoder(src, src_mask)
        targets = torch.LongTensor([[sos] for i in range(batch_size)]).to(src.device)
        trg_mask = no_peak_mask(1).to(src.device)
        
        # 1st output (batch_size, 1, target_vocab)
        output = F.softmax(
            self.out(self.decoder(targets, encoder_output, src_mask, trg_mask)),
            dim=-1,
        )
        
        # top k word predictions
        k_probs, k_idx = output[:, -1].data.topk(beam_size)
        k_log_probs = torch.Tensor([
            math.log(p) for p in k_probs.data.view(-1)
        ]).view(batch_size, -1).to(src.device)
        
        # (batch_size, beam_size, max_seq_len)
        k_outputs = torch.zeros(batch_size, beam_size, max_seq_len).long().to(src.device)
        k_outputs[:, :, 0] = sos
        k_outputs[:, :, 1] = k_idx

        # (batch_size, beam_size, max_seq_len, d_model)
        k_encoder_outputs = torch.stack([
            torch.stack([encoder_output[batch, :, :] for k in range(beam_size)]) \
            for batch in range(batch_size)
        ]).to(src.device)
        
        return k_outputs, k_encoder_outputs, k_log_probs
         
        
    def predict(self, src, sos, src_pad, eos, max_seq_len, beam_size):
        batch_size = src.size(0)
        src_mask = (src != src_pad).unsqueeze(-2)
        
        for i in range(1, max_seq_len):
            if i == 1:
                outputs, encoder_outputs, log_probs = self.pred_init_seq(
                    src, 
                    src_mask, 
                    sos,  
                    max_seq_len,
                    beam_size,
                )
                
                encoder_outputs = encoder_outputs.transpose(0,1)
                
            else:
                outputs = outputs.transpose(0,1)
                trg_mask = no_peak_mask(i).to(src.device)
                
                # output shape: (batch_size, beam_size, max_seq_len, target_vocab)
                output = torch.stack([
                    F.softmax(
                        self.out(self.decoder(
                            outputs[beam, :, :i], 
                            encoder_outputs[beam], 
                            src_mask, 
                            trg_mask,
                        )),
                        dim=-1,
                    ) for beam in range(beam_size)
                ]).transpose(0,1)
                
                # get top k predictions for next word 
                # (batch_size, beam_size, 1 (last pos), k)
                probs, idx = output[:, :, -1, :].data.topk(beam_size) 
                probs.to(src.device)
                idx.to(src.device)
                log_probs = torch.tensor(
                    [math.log(p) for p in probs.data.view(-1)]
                ).view(batch_size, beam_size, -1).to(src.device) + log_probs
                k_probs, k_idx = log_probs.view(batch_size, -1).topk(beam_size)
                
                log_probs = k_probs
                outputs = outputs.transpose(0,1)
                outputs[:, :, :i] = outputs[:, k_idx // beam_size, :i]
                outputs[:, :, i] = torch.stack(
                    [row[k_idx[i]] for i, row in enumerate(idx.view(batch_size, -1).data)]
                ).view(batch_size, beam_size)
                
        eos_bool = (outputs[:, 0, :] == eos)
        output_seqs = []
        for i, row in enumerate(eos_bool):
            if torch.sum(row) > 0:
                row_len = (row == 1).tolist().index(1)
            else: 
                row_len = len(row)
            output_seqs.append(outputs[i, 0, 1:row_len].tolist())
        return torch.tensor(output_seqs).long()
