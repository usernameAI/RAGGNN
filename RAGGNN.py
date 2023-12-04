import math
import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder, MLPLayers
from recbole.model.loss import EmbLoss, BPRLoss
from torch.nn.init import xavier_uniform_, normal_, xavier_normal_, orthogonal_, zeros_
from recbole.utils import InputType
from recbole.model.loss import BPRLoss


class GNN(nn.Module):
    def __init__(self, embedding_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.embedding_size = embedding_size
        self.input_size = embedding_size * 2
        self.gate_size = embedding_size * 3
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.embedding_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))

        self.linear_edge_in = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.linear_edge_out = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.linear_hidden = nn.Linear(self.step * self.embedding_size, self.embedding_size, bias=False)
        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.size(1)], self.linear_edge_in(hidden))
        input_out = torch.matmul(A[:, :, A.size(1):2 * A.size(1)], self.linear_edge_out(hidden))
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        reset_gate = torch.sigmoid(i_r + h_r)
        input_gate = torch.sigmoid(i_i + h_i)
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        hy = (1 - input_gate) * hidden + input_gate * new_gate
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class RAGGNN(SequentialRecommender):
    input_type = InputType.POINTWISE
    # input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(RAGGNN, self).__init__(config, dataset)
        self.hidden_size = config['hidden_size']
        self.inner_size = config['inner_size']
        self.step = config['step']
        self.gnn_attention_layers = config['gnn_attention_layers']
        self.gnn_attention_heads = config['gnn_attention_heads']
        self.gnn_attention_dropout_prob = config['gnn_attention_dropout_prob']
        self.gnn_attention_hidden_dropout_prob = config['gnn_attention_hidden_dropout_prob']
        self.gnn_attention_hidden_act = config['gnn_attention_hidden_act']
        self.rnn_layers = config['rnn_layers']
        self.rnn_attention_layers = config['rnn_attention_layers']
        self.rnn_attention_heads = config['rnn_attention_heads']
        self.rnn_attention_dropout_prob = config['rnn_attention_dropout_prob']
        self.rnn_attention_hidden_dropout_prob = config['rnn_attention_hidden_dropout_prob']
        self.rnn_attention_hidden_act = config['rnn_attention_hidden_act']
        self.dense = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.dense_act = nn.Sigmoid()
        self.layer_norm_eps = config['layer_norm_eps']
        self.device = config['device']
        self.reg_weight = config['reg_weight']
        self.loss_type = config['loss_type']
        self.initializer_range = config['initializer_range']
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.gnn = GNN(self.hidden_size, self.step)
        self.gnn_self_attention = TransformerEncoder(
            n_layers=self.gnn_attention_layers,
            n_heads=self.gnn_attention_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.gnn_attention_hidden_dropout_prob,
            attn_dropout_prob=self.gnn_attention_dropout_prob,
            hidden_act=self.gnn_attention_hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )
        self.rnn_layers = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.inner_size,
            num_layers=self.rnn_layers,
            bias=True,
            batch_first=True,
            bidirectional=True
        )
        self.rnn_linear = nn.Linear(
            in_features=2 * self.inner_size,
            out_features=self.hidden_size,
            bias=False
        )
        self.rnn_self_attention = TransformerEncoder(
            n_layers=self.rnn_attention_layers,
            n_heads=self.rnn_attention_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.rnn_attention_hidden_dropout_prob,
            attn_dropout_prob=self.rnn_attention_dropout_prob,
            hidden_act=self.rnn_attention_hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )
        self.reg_loss = EmbLoss()
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ 初始化策略 """
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is None:
            xavier_normal_(module.weight.data)
        if isinstance(module, nn.Linear) and module.bias is not None:
            xavier_normal_(module.weight.data)
            module.bias.data.zero_()

    def _get_slice(self, item_seq):
        items, n_node, A, alias_inputs = [], [], [], []
        max_n_node = item_seq.size(1)
        item_seq = item_seq.cpu().numpy()

        for u_input in item_seq:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)

            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        alias_inputs = torch.LongTensor(alias_inputs).to(self.device)
        A = torch.FloatTensor(A).to(self.device)
        items = torch.LongTensor(items).to(self.device)

        return alias_inputs, A, items

    def forward(self, item_seq, item_seq_len):
        alias_inputs, A, items = self._get_slice(item_seq)
        hidden = self.item_embedding(items)
        gnn_hidden = self.gnn(A, hidden)
        alias_inputs = alias_inputs.view(-1, alias_inputs.size(1), 1).expand(-1, -1, self.hidden_size)
        seq_hidden = torch.gather(gnn_hidden, dim=1, index=alias_inputs)

        a = seq_hidden
        attention_mask = self.get_attention_mask(item_seq, bidirectional=False)
        outputs = self.gnn_self_attention(a, attention_mask, output_all_encoded_layers=True)
        gnn_output = outputs[-1]
        gnn_at = self.gather_indexes(gnn_output, item_seq_len - 1)

        rnn_hidden, _ = self.rnn_layers(hidden)
        rnn_hidden = self.rnn_linear(rnn_hidden)
        rnn_attention_mask = self.get_attention_mask(item_seq, bidirectional=True)
        rnn_output = self.rnn_self_attention(rnn_hidden, rnn_attention_mask, output_all_encoded_layers=True)
        rnn_output = rnn_output[-1]
        rnn_at = self.gather_indexes(rnn_output, item_seq_len - 1)

        concat = torch.cat([gnn_at, rnn_at], dim=-1)
        linear = self.dense(concat)
        weight = self.dense_act(linear)
        seq_output = weight * gnn_at + (1 - weight) * rnn_at
        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
        else:
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
        reg_loss = self.reg_loss(self.item_embedding.weight)
        total_loss = loss + self.reg_weight * reg_loss
        return total_loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores
