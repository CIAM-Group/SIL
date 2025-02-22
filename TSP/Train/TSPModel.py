import torch
import torch.nn as nn
import torch.nn.functional as F


class TSPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.mode = model_params['mode']
        self.encoder = TSP_Encoder(**model_params)
        self.decoder = TSP_Decoder(**model_params)
        self.encoded_nodes = None
    def forward(self, state, selected_node_list, solution, current_step,repair = False):

        batch_size_V = state.data.size(0)
        pproblem_size = state.data.size(1)
        if self.mode == 'train':

            probs = self.decoder(self.encoder(state.data), state.data, state.first_node,
                                 state.current_node, selected_node_list, current_step,
                                 batch_size_V, pproblem_size)


            selected_student = probs.argmax(dim=1)

            selected_teacher = solution[:, current_step - 1]
            prob = probs[torch.arange(batch_size_V)[:, None], selected_teacher[:, None]].reshape(batch_size_V, 1)


        if self.mode == 'test':
            if repair == False:
                if current_step <= 1:
                    self.encoded_nodes = self.encoder(state.data)

                probs = self.decoder(self.encoded_nodes, state.data,state.first_node, state.current_node, selected_node_list,
                                     current_step, batch_size_V, pproblem_size)

                selected_student = probs.argmax(dim=1)  # shape:  B * k
                selected_teacher = selected_student  # shape:  B * k
                prob = 1

            if repair == True:

                probs = self.decoder(self.encoder(state.data), state.data, state.first_node, state.current_node, selected_node_list,
                                     current_step, batch_size_V, pproblem_size)

                selected_student = probs.argmax(dim=1)

                selected_teacher = selected_student
                prob = probs[torch.arange(batch_size_V)[:, None], selected_teacher[:, None]].reshape(batch_size_V, 1)  # shape: [B, 1]


        return selected_teacher, prob, 1, selected_student


def _get_encoding(encoded_nodes, node_index_to_pick):

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)
    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)

    return picked_nodes


class TSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']

        self.embedding = nn.Linear(2, embedding_dim, bias=True)

    def forward(self, data):

        out = self.embedding(data)

        return out

class TSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        self.embedding_first_node = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.embedding_last_node = nn.Linear(embedding_dim, embedding_dim, bias=True)

        self.embedding_last_node_2 = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.embedding_last_node_3 = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.embedding_last_node_4 = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.embedding_last_node_5 = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.embedding_last_node_6 = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.embedding_last_node_7 = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.embedding_last_node_8 = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.embedding_last_node_9 = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.embedding_last_node_10 = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.embedding_last_node_11 = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.embedding_last_node_12 = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.embedding_last_node_13 = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.embedding_last_node_14 = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.embedding_last_node_15 = nn.Linear(embedding_dim, embedding_dim, bias=True)

        self.layers = nn.ModuleList([DecoderLayer(**model_params) for _ in range(encoder_layer_num)])

        self.Linear_final = nn.Linear(embedding_dim, 1, bias=True)

    def _get_new_data(self, data, selected_node_list, prob_size, B_V):

        list = selected_node_list

        new_list = torch.arange(prob_size)[None, :].repeat(B_V, 1)

        new_list_len = prob_size - list.shape[1]

        index_2 = list.type(torch.long)

        index_1 = torch.arange(B_V, dtype=torch.long)[:, None].expand(B_V, index_2.shape[1])


        new_list[index_1, index_2] = -2

        unselect_list = new_list[torch.gt(new_list, -1)].view(B_V, new_list_len)

        new_data = data

        emb_dim = data.shape[-1]

        new_data_len = new_list_len

        index_2_ = unselect_list.repeat_interleave(repeats=emb_dim, dim=1)

        index_1_ = torch.arange(B_V, dtype=torch.long)[:, None].expand(B_V, index_2_.shape[1])

        index_3_ = torch.arange(emb_dim)[None, :].repeat(repeats=(B_V, new_data_len))

        new_data_ = new_data[index_1_, index_2_, index_3_].view(B_V, new_data_len, emb_dim)

        return new_data_

    def _get_encoding(self,encoded_nodes, node_index_to_pick):

        batch_size = node_index_to_pick.size(0)
        pomo_size = node_index_to_pick.size(1)
        embedding_dim = encoded_nodes.size(2)

        gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)

        picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)

        return picked_nodes

    def forward(self,data, problems, first_node, current_node,
                selected_node_list, current_step,batch_size_V,problem_size):

        batch_size_V = data.shape[0]  # B

        problem_size = data.shape[1]

        new_data = data

        left_encoded_node_2 = self._get_new_data(new_data,selected_node_list, problem_size, batch_size_V)

        knearest_num = self.model_params['k_nearest_num']

        new_data_len = left_encoded_node_2.shape[1]

        if new_data_len > knearest_num:

            if selected_node_list.shape[1] == 0:
                last_node = problems[:, [0], :2]
            else:
                last_node = self._get_encoding(problems, selected_node_list[:, [-1]])

            left_node = self._get_new_data(problems, selected_node_list, problem_size, batch_size_V)

            distance = torch.sum((left_node - last_node) ** 2, dim=2).sqrt()

            sort_value, sort_index = torch.topk(distance, k=knearest_num, dim=1, largest=False)

            k_nearest_point = self._get_encoding(left_encoded_node_2, sort_index)

            left_encoded_node_2 = k_nearest_point.clone().detach()


        first_and_last_node = self._get_encoding(new_data,selected_node_list[:,[0,-1]])
        embedded_first_node_ = first_and_last_node[:,0]
        embedded_last_node_ = first_and_last_node[:,1]


        embedded_first_node_ = self.embedding_first_node(embedded_first_node_)
        embedded_last_no1_node_ = self.embedding_last_node(embedded_last_node_)


        embedded_last_no2_node_ = self.embedding_last_node_2(embedded_last_node_)
        embedded_last_no3_node_ = self.embedding_last_node_3(embedded_last_node_)
        embedded_last_no4_node_ = self.embedding_last_node_4(embedded_last_node_)
        embedded_last_no5_node_ = self.embedding_last_node_5(embedded_last_node_)
        embedded_last_no6_node_ = self.embedding_last_node_6(embedded_last_node_)
        embedded_last_no7_node_ = self.embedding_last_node_7(embedded_last_node_)
        embedded_last_no8_node_ = self.embedding_last_node_8(embedded_last_node_)
        embedded_last_no9_node_ = self.embedding_last_node_9(embedded_last_node_)
        embedded_last_no10_node_ = self.embedding_last_node_10(embedded_last_node_)
        embedded_last_no11_node_ = self.embedding_last_node_11(embedded_last_node_)
        embedded_last_no12_node_ = self.embedding_last_node_12(embedded_last_node_)
        embedded_last_no13_node_ = self.embedding_last_node_13(embedded_last_node_)
        embedded_last_no14_node_ = self.embedding_last_node_14(embedded_last_node_)
        embedded_last_no15_node_ = self.embedding_last_node_15(embedded_last_node_)


        out = torch.cat((embedded_first_node_.unsqueeze(1), embedded_last_no1_node_.unsqueeze(1),
                         embedded_last_no2_node_.unsqueeze(1),
                         embedded_last_no3_node_.unsqueeze(1), embedded_last_no4_node_.unsqueeze(1),
                         embedded_last_no5_node_.unsqueeze(1), embedded_last_no6_node_.unsqueeze(1),
                         embedded_last_no7_node_.unsqueeze(1), embedded_last_no8_node_.unsqueeze(1),
                         embedded_last_no9_node_.unsqueeze(1), embedded_last_no10_node_.unsqueeze(1),
                         embedded_last_no11_node_.unsqueeze(1), embedded_last_no12_node_.unsqueeze(1),
                         embedded_last_no13_node_.unsqueeze(1), embedded_last_no14_node_.unsqueeze(1),
                         embedded_last_no15_node_.unsqueeze(1),  ), dim=1)

        inter = torch.cat((embedded_first_node_.unsqueeze(1), embedded_last_no1_node_.unsqueeze(1),
                           embedded_last_no2_node_.unsqueeze(1),
                           embedded_last_no3_node_.unsqueeze(1), embedded_last_no4_node_.unsqueeze(1),
                           embedded_last_no5_node_.unsqueeze(1), embedded_last_no6_node_.unsqueeze(1),
                           embedded_last_no7_node_.unsqueeze(1), embedded_last_no8_node_.unsqueeze(1),
                           embedded_last_no9_node_.unsqueeze(1), embedded_last_no10_node_.unsqueeze(1),
                           embedded_last_no11_node_.unsqueeze(1), embedded_last_no12_node_.unsqueeze(1),
                           embedded_last_no13_node_.unsqueeze(1), embedded_last_no14_node_.unsqueeze(1),
                           embedded_last_no15_node_.unsqueeze(1),
                           left_encoded_node_2), dim=1)
        layer_count = 0

        for layer in self.layers:
            out, inter = layer(out, inter)
            layer_count += 1

        repeatd_num = out.shape[1]

        out = inter
        out = self.Linear_final(out).squeeze(-1)
        out[:, :repeatd_num] = out[:, :repeatd_num] + float('-inf')

        props = F.softmax(out, dim=-1)
        props = props[:, repeatd_num:]

        if new_data_len > knearest_num:
            new_props = torch.zeros(batch_size_V, new_data_len)

            index_1_ = torch.arange(batch_size_V, dtype=torch.long)[:, None].expand(batch_size_V,
                                                                                    sort_index.shape[
                                                                                        1])
            index_2_ = sort_index.type(torch.long)

            new_props[index_1_, index_2_] = props.reshape(batch_size_V, sort_index.shape[1])

            props = new_props

        index_small = torch.le(props, 1e-5)
        props_clone = props.clone()
        props_clone[index_small] = props_clone[index_small] + torch.tensor(1e-7, dtype=props_clone[index_small].dtype)  # 防止概率过小
        props = props_clone

        new_props = torch.zeros(batch_size_V, problem_size)

        index_1_ = torch.arange(batch_size_V, dtype=torch.long)[:, None].expand(batch_size_V, selected_node_list.shape[1])  # shape: [B*(V-1), n]
        index_2_ = selected_node_list.type(torch.long)
        new_props[index_1_, index_2_] = -2
        index = torch.gt(new_props, -1).view(batch_size_V, -1)

        new_props[index] = props.ravel()

        index_small = torch.le(new_props, 0.0)
        new_props[index_small] = 0.0


        return new_props



class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.feedForward = Feed_Forward_Module(**model_params)

    def forward(self, input1):

        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)

        out_concat = multi_head_attention(q, k, v)

        multi_head_out = self.multi_head_combine(out_concat)

        out1 = input1 + multi_head_out
        out2 = self.feedForward(out1)
        out3 = out1 +  out2
        return out3



class DecoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.Wq_2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk_2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv_2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.multi_head_combine_2 = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.feedForward = Feed_Forward_Module(**model_params)
        self.feedForward_2 = Feed_Forward_Module(**model_params)

    def forward(self, input1, input2):

        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input2), head_num=head_num)
        v = reshape_by_heads(self.Wv(input2), head_num=head_num)

        out_concat = multi_head_attention(q, k, v)

        multi_head_out = self.multi_head_combine(out_concat)

        out1 = input1 + multi_head_out
        out2 = self.feedForward(out1)
        out3 = out1 +  out2

        q_2 = reshape_by_heads(self.Wq_2(input2), head_num=head_num)
        k_2 = reshape_by_heads(self.Wk_2(out3), head_num=head_num)
        v_2 = reshape_by_heads(self.Wv_2(out3), head_num=head_num)

        out_concat_2 = multi_head_attention(q_2, k_2, v_2)
        multi_head_out_2 = self.multi_head_combine_2(out_concat_2)

        out1_2 = input2 + multi_head_out_2
        out2_2 = self.feedForward_2(out1_2)

        out3_2 = out1_2 +  out2_2

        return out3, out3_2



def reshape_by_heads(qkv, head_num):

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)

    q_transposed = q_reshaped.transpose(1, 2)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)


    score = torch.matmul(q, k.transpose(2, 3))

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))

    weights = nn.Softmax(dim=3)(score_scaled)

    out = torch.matmul(weights, v)

    out_transposed = out.transpose(1, 2)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)

    return out_concat



class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):

        return self.W2(F.relu(self.W1(input1)))
