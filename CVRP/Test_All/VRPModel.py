import torch
import torch.nn as nn
import torch.nn.functional as F


class VRPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.mode = model_params['mode']
        self.encoder = CVRP_Encoder(**model_params)
        self.decoder = CVRP_Decoder(**model_params)

        self.encoded_nodes = None

    def forward(self, state, selected_node_list, solution, current_step,raw_data_capacity=None, select_probability_accumulate=None,
                decode_method=None, beam_select_flag=None,beam_width = None):

        self.capacity = raw_data_capacity.ravel()[0].item()
        batch_size = state.problems.shape[0]
        problem_size = state.problems.shape[1]
        split_line = problem_size - 1

        def probs_to_selected_nodes(probs_,split_line_,batch_size_):
            selected_node_student_ = probs_.argmax(dim=1)
            is_via_depot_student_ = selected_node_student_ >= split_line_
            not_via_depot_student_ = selected_node_student_ < split_line_

            selected_flag_student_ = torch.zeros(batch_size_,dtype=torch.int)
            selected_flag_student_[is_via_depot_student_] = 1
            selected_node_student_[is_via_depot_student_] = selected_node_student_[is_via_depot_student_]-split_line_ +1
            selected_flag_student_[not_via_depot_student_] = 0
            selected_node_student_[not_via_depot_student_] = selected_node_student_[not_via_depot_student_]+ 1
            return selected_node_student_, selected_flag_student_

        if self.mode == 'train':
            remaining_capacity = state.problems[:, 1, 3]

            probs = self.decoder(self.encoder(state.problems,self.capacity),state.problems,
                                 selected_node_list, current_step,self.capacity,remaining_capacity)

            selected_node_student, selected_flag_student = probs_to_selected_nodes(probs, split_line, batch_size)

            selected_node_teacher = solution[:, current_step,0]

            selected_flag_teacher = solution[:, current_step, 1]

            is_via_depot = selected_flag_teacher==1
            selected_node_teacher_copy = selected_node_teacher-1
            selected_node_teacher_copy[is_via_depot]+=split_line

            prob_select_node = probs[torch.arange(batch_size)[:, None], selected_node_teacher_copy[:, None]].reshape(batch_size, 1)  # shape: [B, 1]

            loss_node = -prob_select_node.log().mean()

        if self.mode == 'test':

            remaining_capacity = state.problems[:, 1, 3]

            if current_step <= 1:
                self.encoded_nodes = self.encoder(state.problems,self.capacity)

            probs = self.decoder(self.encoded_nodes, state.problems, selected_node_list, current_step,self.capacity,remaining_capacity)

            selected_node_student = probs.argmax(dim=1)
            is_via_depot_student = selected_node_student >= split_line
            not_via_depot_student = selected_node_student < split_line

            selected_flag_student = torch.zeros(batch_size, dtype=torch.int)
            selected_flag_student[is_via_depot_student] = 1
            selected_node_student[is_via_depot_student] = selected_node_student[is_via_depot_student] - split_line + 1
            selected_flag_student[not_via_depot_student] = 0
            selected_node_student[not_via_depot_student] = selected_node_student[not_via_depot_student] + 1

            selected_node_teacher = selected_node_student
            selected_flag_teacher = selected_flag_student

            loss_node = torch.tensor(0)

        return loss_node,selected_node_teacher,  selected_node_student,selected_flag_teacher,selected_flag_student


class CVRP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        self.embedding = nn.Linear(3, embedding_dim, bias=True)

    def forward(self, data_,capacity):

        data = data_.clone().detach()
        data= data[:,:,:3]

        data[:,:,2] = data[:,:,2]/capacity

        out = self.embedding(data)

        return out


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

        out1 = input1 +   multi_head_out
        out2 = self.feedForward(out1)

        out3 = out1 + out2
        return out3


class CVRP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        decoder_layer_num = self.model_params['decoder_layer_num']

        self.embedding_first_node = nn.Linear(embedding_dim+1, embedding_dim, bias=True)
        self.embedding_last_node = nn.Linear(embedding_dim+1, embedding_dim, bias=True)
        self.embedding_last_node_2 = nn.Linear(embedding_dim+1, embedding_dim, bias=True)
        self.embedding_last_node_3 = nn.Linear(embedding_dim+1, embedding_dim, bias=True)
        self.embedding_last_node_4 = nn.Linear(embedding_dim+1, embedding_dim, bias=True)
        self.embedding_last_node_5 = nn.Linear(embedding_dim+1, embedding_dim, bias=True)
        self.embedding_last_node_6 = nn.Linear(embedding_dim+1, embedding_dim, bias=True)
        self.embedding_last_node_7 = nn.Linear(embedding_dim+1, embedding_dim, bias=True)
        self.embedding_last_node_8 = nn.Linear(embedding_dim+1, embedding_dim, bias=True)
        self.embedding_last_node_9 = nn.Linear(embedding_dim+1, embedding_dim, bias=True)
        self.embedding_last_node_10 = nn.Linear(embedding_dim+1, embedding_dim, bias=True)
        self.embedding_last_node_11 = nn.Linear(embedding_dim+1, embedding_dim, bias=True)
        self.embedding_last_node_12 = nn.Linear(embedding_dim+1, embedding_dim, bias=True)
        self.embedding_last_node_13 = nn.Linear(embedding_dim+1, embedding_dim, bias=True)
        self.embedding_last_node_14 = nn.Linear(embedding_dim+1, embedding_dim, bias=True)
        self.embedding_last_node_15 = nn.Linear(embedding_dim+1, embedding_dim, bias=True)

        self.layers = nn.ModuleList([DecoderLayer(**model_params) for _ in range(decoder_layer_num)])

        self.Linear_final = nn.Linear(embedding_dim, 2, bias=True)


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


    def forward(self, data, problems,selected_node_list, current_step,capacity,remaining_capacity):

        data_ = data[:,1:,:].clone().detach()
        selected_node_list_ = selected_node_list.clone().detach() - 1

        batch_size_V = data_.shape[0]

        problem_size = data_.shape[1]
        new_data = data_.clone().detach()

        left_encoded_node_2 = self._get_new_data(new_data,selected_node_list_, problem_size, batch_size_V)


        embedded_first_node = data[:,[0],:]

        if selected_node_list_.shape[1]==0:
            embedded_last_node = data[:,[0],:]
        else:
            embedded_last_node = self._get_encoding(new_data, selected_node_list_[:, [-1]])


        knearest_num = self.model_params['k_nearest_num']
        use_k_nearest = self.model_params['use_k_nearest']
        new_data_len = left_encoded_node_2.shape[1]

        if selected_node_list_.shape[1]==0:
            last_node = problems[:,[0],:2]
        else:
            last_node = self._get_encoding(problems[:,1:,:2], selected_node_list_[:, [-1]])

        left_node = self._get_new_data(problems[:,1:,:2], selected_node_list_, problem_size, batch_size_V)


        if use_k_nearest and new_data_len > knearest_num:
            distance = torch.sum((left_node - last_node) ** 2, dim=2).sqrt()  # [B,new_data_len ]
            sort_value, sort_index = torch.topk(distance, k=knearest_num, dim=1, largest=False)
            k_nearest_point = self._get_encoding(left_encoded_node_2, sort_index)
            left_encoded_node_2 = k_nearest_point.clone().detach()

        remaining_capacity = remaining_capacity.reshape(batch_size_V,1,1)/capacity
        first_node_cat = torch.cat((embedded_first_node,remaining_capacity), dim=2)
        last_node_cat = torch.cat((embedded_last_node,remaining_capacity), dim=2)

        embedded_first_node_ = self.embedding_first_node(first_node_cat)
        embedded_last_node_ = self.embedding_last_node(last_node_cat)

        embedded_last_no2_node_ = self.embedding_last_node_2(last_node_cat)
        embedded_last_no3_node_ = self.embedding_last_node_3(last_node_cat)
        embedded_last_no4_node_ = self.embedding_last_node_4(last_node_cat)
        embedded_last_no5_node_ = self.embedding_last_node_5(last_node_cat)
        embedded_last_no6_node_ = self.embedding_last_node_6(last_node_cat)
        embedded_last_no7_node_ = self.embedding_last_node_7(last_node_cat)
        embedded_last_no8_node_ = self.embedding_last_node_8(last_node_cat)
        embedded_last_no9_node_ = self.embedding_last_node_9(last_node_cat)
        embedded_last_no10_node_ = self.embedding_last_node_10(last_node_cat)
        embedded_last_no11_node_ = self.embedding_last_node_11(last_node_cat)
        embedded_last_no12_node_ = self.embedding_last_node_12(last_node_cat)
        embedded_last_no13_node_ = self.embedding_last_node_13(last_node_cat)
        embedded_last_no14_node_ = self.embedding_last_node_14(last_node_cat)
        embedded_last_no15_node_ = self.embedding_last_node_15(last_node_cat)


        out = torch.cat((embedded_first_node_, embedded_last_no2_node_,
                         embedded_last_no3_node_, embedded_last_no4_node_,
                         embedded_last_no5_node_, embedded_last_no6_node_,
                         embedded_last_no7_node_, embedded_last_no8_node_,
                         embedded_last_no9_node_, embedded_last_no10_node_,
                         embedded_last_no11_node_, embedded_last_no12_node_,
                         embedded_last_no13_node_, embedded_last_no14_node_,
                         embedded_last_no15_node_,
                         embedded_last_node_), dim=1)

        inter = torch.cat((embedded_first_node_, embedded_last_no2_node_,
                           embedded_last_no3_node_, embedded_last_no4_node_,
                           embedded_last_no5_node_, embedded_last_no6_node_,
                           embedded_last_no7_node_, embedded_last_no8_node_,
                           embedded_last_no9_node_, embedded_last_no10_node_,
                           embedded_last_no11_node_, embedded_last_no12_node_,
                           embedded_last_no13_node_, embedded_last_no14_node_,
                           embedded_last_no15_node_,
                           embedded_last_node_, left_encoded_node_2), dim=1)
        layer_count = 0

        for layer in self.layers:
            out, inter = layer(out, inter)

            layer_count += 1

        repeatd_num = 16

        out = inter
        out = self.Linear_final(out).squeeze(-1)
        out[:, 0:repeatd_num] =  out[:,  0:repeatd_num] + float('-inf')
        out = torch.cat((out[:, :, 0], out[:, :, 1]), dim=1)

        props = F.softmax(out, dim=-1)
        customer_num = left_encoded_node_2.shape[1]

        props = torch.cat((props[:, repeatd_num:repeatd_num+customer_num],
                           props[:, 2*repeatd_num + customer_num :]),dim=1)

        index_small = torch.le(props, 1e-5)
        props_clone = props.clone()
        props_clone[index_small] = props_clone[index_small] + torch.tensor(1e-7, dtype=props_clone[index_small].dtype)  # 防止概率过小
        props = props_clone

        if  use_k_nearest and new_data_len > knearest_num:

            new_props = torch.zeros(batch_size_V, 2 * new_data_len)

            index_1_ = torch.arange(batch_size_V, dtype=torch.long)[:, None].expand(batch_size_V,
                                                                                    sort_index.shape[1]*2)  # shape: [B*(V-1), n]
            index_2_ =torch.cat( ((sort_index).type(torch.long), (left_node.shape[1])+ (sort_index).type(torch.long) ),dim=-1)

            new_props[index_1_, index_2_] = props.reshape(batch_size_V, sort_index.shape[1]*2)

            props = new_props.clone().detach()

        new_props = torch.zeros(batch_size_V, 2 * (problem_size))
        index_1_ = torch.arange(batch_size_V, dtype=torch.long)[:,None].repeat(1,selected_node_list_.shape[1]*2)

        index_2_ =torch.cat( ((selected_node_list_).type(torch.long), (problem_size)+ (selected_node_list_).type(torch.long) ),dim=-1) # shape: [B*V, n]

        new_props[index_1_, index_2_,] = -2
        index = torch.gt(new_props, -1).view(batch_size_V, -1)
        new_props[index] = props.ravel()

        return new_props


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
        out3 = out1 + out2

        q_2 = reshape_by_heads(self.Wq_2(input2), head_num=head_num)
        k_2 = reshape_by_heads(self.Wk_2(out3), head_num=head_num)
        v_2 = reshape_by_heads(self.Wv_2(out3), head_num=head_num)

        out_concat_2 = multi_head_attention(q_2, k_2, v_2)
        multi_head_out_2 = self.multi_head_combine_2(out_concat_2)

        out1_2 = input2 + multi_head_out_2
        out2_2 = self.feedForward_2(out1_2)

        out3_2 = out1_2 + out2_2

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

    input_s = k.size(2)

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
