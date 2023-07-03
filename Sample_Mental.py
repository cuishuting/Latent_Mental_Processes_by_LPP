import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
# TODO: get action predicates' time's embedding, corresponding to "f(D) = h" in the draft

mental_predicate_set = [0]
action_predicate_set = [1, 2]


def get_action_pre_time_embedding(data, embedding_dim, time_horizon, time_interval_len, action_predicate_set):
    # to get one sample_id's certain action predicate's recording time list, call: data[sample_id][p_id]
    time_embedding = {}
    for sample_id in data.keys():
        time_embedding[sample_id] = {}
        for action_id in action_predicate_set:
            time_list = data[sample_id][action_id]["time"]
            time_embedding[sample_id][action_id] = {}
            for time_id, time in enumerate(time_list):
                time_embedding[sample_id][action_id][time_id] = []
                for i in range(embedding_dim):
                    if i % 2 == 0:
                        time_embedding[sample_id][action_id][time_id].append(np.sin(time / (1e4 ** (2*i/embedding_dim))))
                    else:
                        time_embedding[sample_id][action_id][time_id].append(np.cos(time / (1e4 ** (2*i/embedding_dim))))

    # To get sample i's action j's k-th (from 0-th) time stamp embedding: time_embedding[i][j][k], which is a list form embedding

    num_time_intervals = int(time_horizon / time_interval_len)
    cur_interval_neighbor_time_idx = 1  # idx 0 is always time: 0
    action_embeddings = torch.zeros((len(data), num_time_intervals, len(action_predicate_set) * embedding_dim))
    for sample_id in data.keys():
        for a_id, action_id in enumerate(action_predicate_set):
            cur_action_time_list = data[sample_id][action_id]["time"]
            for i in range(num_time_intervals):
                cur_left_time = i * time_interval_len
                cur_right_time = (i+1) * time_interval_len
                cur_mid_time = cur_left_time + time_interval_len / 2
                # choose embedding of the action recording time nearest to cur_mid_time as cur interval's input data
                if cur_right_time <= cur_action_time_list[1]:
                    action_embeddings[sample_id][i][a_id*embedding_dim:(a_id+1)*embedding_dim] = torch.tensor(time_embedding[sample_id][action_id][1])
                elif cur_left_time >= cur_action_time_list[-1]:
                    action_embeddings[sample_id][i][a_id*embedding_dim:(a_id+1)*embedding_dim] = torch.tensor(time_embedding[sample_id][action_id][len(cur_action_time_list)-1])
                else:
                    if (cur_interval_neighbor_time_idx+1 < len(cur_action_time_list)) and ((cur_mid_time - cur_action_time_list[cur_interval_neighbor_time_idx]) < (cur_action_time_list[cur_interval_neighbor_time_idx+1] - cur_mid_time)):
                        action_embeddings[sample_id][i][a_id*embedding_dim:(a_id+1)*embedding_dim] = torch.tensor(time_embedding[sample_id][action_id][cur_interval_neighbor_time_idx])
                    else:
                        cur_interval_neighbor_time_idx += 1
                        if cur_interval_neighbor_time_idx < len(cur_action_time_list):
                            action_embeddings[sample_id][i][a_id*embedding_dim:(a_id+1)*embedding_dim] = torch.tensor(time_embedding[sample_id][action_id][cur_interval_neighbor_time_idx])
                        else:
                            continue
    # todo: action_embedding: [num_samples, num_time_intervals, num_action_predicates * time_embedding_dim]
    return action_embeddings


# TODO : data has the form "data[sample_ID][predicate_idx]['time']", we now only observed action predicates' occur time

'''
class HazardRate_LSTM is used to get each mental predicate i's hazard rate h_l^i in each time interval Vl: [t_(l-1), t_l)
'''


class HazardRate_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob):
        self.input_size = input_size  # input_size: "num_action_predicates * time_embedding_dim" in each time interval
        self.hidden_size = hidden_size # hidden_size: 1, represents hazard rate in each time interval
        self.num_layers = num_layers
        self.dropout_rate = dropout_prob
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            batch_first=True, dropout=self.dropout_rate)

    def forward(self, data, ini_hidden_state):
        # data has shape: [batch_size, num_time_intervals, num_action_predicates * time_embedding_dim]
        # ini_hidden_state: represent Z_(l-1) in the draft
        batch_size = data.shape[0]
        ini_cell_state = torch.zeros((self.num_layers, batch_size, self.hidden_size))
        output, (h_n, c_n) = self.lstm(data, (ini_hidden_state, ini_cell_state))  # todo ???: initial hidden state ([Z_(l-1)]) and initial cell state is zero, correct?
        # todo: output's shape [batch_size, num_time_intervals, 1], because the hidden_size represents hazard rate h_l^i
        #  for each mental predicate at each time interval
        return output


def get_mentals_cat_distribution_cur_interval(cur_interval, all_mentals_hazard_rate_list, num_mental_types):
    # param "all_mentals_hazard_rate_list" has shape: [num_mental_types, batch_size, num_time_intervals, 1]
    batch_size = all_mentals_hazard_rate_list[0].shape[0]
    mentals_cat_prob = torch.zeros((num_mental_types, batch_size, 1))
    for i in range(num_mental_types):
        h_i_cur_interval = all_mentals_hazard_rate_list[i, :, cur_interval, :]
        p_i_cur_interval = h_i_cur_interval # [batch_size, 1]
        for j in range(cur_interval):
            p_i_cur_interval = torch.mul(p_i_cur_interval, 1 - all_mentals_hazard_rate_list[i, :, j, :])  # [batch_size, 1]
        mentals_cat_prob[i] = p_i_cur_interval
    return mentals_cat_prob


def Gumbel_Max_Trick(mentals_unorm_prob_cur_interval): # param shape: [num_mental_types, batch_size, 1]
    batch_size = mentals_unorm_prob_cur_interval.shape[1]
    num_mental_types = mentals_unorm_prob_cur_interval.shape[0]
    std_gumbel_dist = torch.distributions.gumbel.Gumbel(0, 1)
    gumbel_noise = std_gumbel_dist.sample(sample_shape=(num_mental_types, batch_size, 1))
    max_types_in_batch = torch.max(torch.log(mentals_unorm_prob_cur_interval) + gumbel_noise, dim=0).indices  # tensor with shape [batch_size, 1]
    mental_type = [max_types_in_batch[i][0].item() for i in range(batch_size)]
    # list with length: batch_size, each element is the sampled mental type in cur_interval in cur_batch
    return mental_type


def Sample(mental_set, action_set, action_time_emb_dim, lstm_layers_num, lstm_dropout_prob, data_one_batch, time_horizon, time_interval_len):
    """
    param: data_one_batch: [batch_size, num_time_intervals, num_action_predicates * time_embedding_dim]
    """
    num_mental_types = 1 + len(mental_set) # "+ 1" represents the type "no mental predicate occur" at cur time interval
    lstm_input_size = len(action_set) * action_time_emb_dim
    batch_size = data_one_batch.shape[0]
    num_time_intervals = int(time_horizon / time_interval_len)
    imputed_mental_states = torch.zeros((batch_size, num_time_intervals, 1))
    # imputed_mental_states[b_id, interval_id, 0]: mental predicate's type: 0 means no mental occur in cur interval,
    # others mean corresponding mental occur in cur interval

    for l in range(num_time_intervals):
        all_mentals_hazard_rate_list = torch.zeros((num_mental_types, batch_size, num_time_intervals, 1))

        for i in range(num_mental_types):  # each mental type has one lstm
            cur_lstm = HazardRate_LSTM(lstm_input_size, 1, lstm_layers_num, lstm_dropout_prob)  # 1 is hidden_size, represent the mental type in former interval Z_(l-1)
            if l == 0:
                ini_hidden_state = torch.zeros((lstm_layers_num, batch_size, 1))
            else:
                ini_hidden_state = torch.cat([imputed_mental_states[:, l-1, 0]]*lstm_layers_num).reshape(lstm_layers_num, batch_size, 1)

            cur_h_list, (h_n, c_n) = cur_lstm.forward(data_one_batch, ini_hidden_state)   # "cur_h_list": [batch_size, num_time_intervals, 1]
            all_mentals_hazard_rate_list[i][:] = cur_h_list

        mentals_prob_cur_interval = get_mentals_cat_distribution_cur_interval(l, all_mentals_hazard_rate_list, num_mental_types)
        # shape: [num_mental_types, batch_size, 1], represents the prob that cur mental occurs at cur_interval
        sampled_mental_type = Gumbel_Max_Trick(mentals_prob_cur_interval)
        # shape: list with length batch_size, each element is the sampled mental type in cur_interval in cur_batch

        for b_id in range(batch_size):
            imputed_mental_states[b_id, l, 0] = sampled_mental_type[b_id]













data = np.load("./data_new.npy", allow_pickle=True).item()
action_embedding = get_action_pre_time_embedding(data, 3, 10, 0.001, action_predicate_set)
print(action_embedding)