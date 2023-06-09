import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
# TODO: get action predicates' time's embedding, corresponding to "f(D) = h" in the draft

mental_predicate_set = [0]
action_predicate_set = [1, 2]
data = np.load("./data_new.npy", allow_pickle=True).item()
action_time_emb_dim = 3
time_horizon = 10
time_interval_len = 0.1
lstm_layers_num = 2
lstm_dropout_prob = 0.2
batch_size = 5
num_batches = int(len(data) / batch_size)
p_l_i = []


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
        super(HazardRate_LSTM, self).__init__()
        self.input_size = input_size  # input_size: "num_action_predicates * time_embedding_dim" in each time interval
        self.hidden_size = hidden_size  # hidden_size: num_mentals_type, represents each mental type's hazard rate in each time interval
        self.num_layers = num_layers
        self.dropout_rate = dropout_prob
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            batch_first=True, dropout=self.dropout_rate)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data, ini_hidden_state):
        # data has shape: [batch_size, num_time_intervals, num_action_predicates * time_embedding_dim]
        # ini_hidden_state: represent Z_(l-1) in the draft
        batch_size = data.shape[0]
        ini_cell_state = torch.zeros((self.num_layers, batch_size, self.hidden_size))
        output, (h_n, c_n) = self.lstm(data, (ini_hidden_state, ini_cell_state))  # todo ???: initial hidden state ([Z_(l-1)]) and initial cell state is zero, correct?
        # todo: output's shape [batch_size, num_time_intervals, num_mentals' type], because the hidden_size represents hazard rate h_l^i
        #  for each mental predicate at each time interval
        output = self.sigmoid(output)
        return output


def get_mentals_cat_distribution_cur_interval(cur_interval, all_mentals_hazard_rate_list, num_mental_types, all_mentals_last_occur_interval):
    # param "all_mentals_hazard_rate_list" has shape: [batch_size, num_intervals, num_mental_types]
    # param "all_mentals_last_occur_interval" has shape: [batch_size, num_mental_types],
    # each element in param "all_mentals_last_occur_time" records the last occur interval for each mental predicate in each batch
    batch_size = all_mentals_hazard_rate_list.shape[0]
    mentals_cat_prob = torch.zeros((batch_size, num_mental_types))
    for i in range(num_mental_types):
        h_i_cur_interval = all_mentals_hazard_rate_list[:, cur_interval, i]  # shape: [batch_size]
        p_i_cur_interval = h_i_cur_interval # [batch_size]
        for b in range(batch_size):
            cur_last_occur_interval = all_mentals_last_occur_interval[b][i]
            for j in range(int(cur_last_occur_interval), cur_interval): # todo: here, range is not [0, cur_interval) but [last_occur_interval, cur_interval)
                p_i_cur_interval = torch.mul(p_i_cur_interval, 1 - all_mentals_hazard_rate_list[:, j, i])  # [batch_size]
        mentals_cat_prob[:, i] = p_i_cur_interval
    return mentals_cat_prob


def Gumbel_Max_Trick(mentals_unorm_prob_cur_interval):
    # param "mentals_unorm_prob_cur_interval" shape: [num_mentals_occur_batches, num_mental_types]
    # param "batch_idx_mental_occur": one-dim vector represents batch indices during which specific mental occurs
    occur_batch_size = mentals_unorm_prob_cur_interval.shape[0]
    num_mental_types = mentals_unorm_prob_cur_interval.shape[1]
    std_gumbel_dist = torch.distributions.gumbel.Gumbel(0, 1)
    gumbel_noise = std_gumbel_dist.sample(sample_shape=(occur_batch_size, num_mental_types))
    max_types_in_batch = torch.max(torch.log(mentals_unorm_prob_cur_interval) + gumbel_noise, dim=1).indices  # one-dim tensor with length occur_batch_size
    # list with length: occur_batch_size, each element is the sampled mental type in cur_interval in cur_batch
    return max_types_in_batch


def Sample(mental_set, action_set, action_time_emb_dim, lstm_layers_num, lstm_dropout_prob, data_one_batch, time_horizon, time_interval_len):
    """
    param: data_one_batch: [batch_size, num_time_intervals, num_action_predicates * time_embedding_dim]
    """
    num_mental_types = len(mental_set)
    lstm_input_size = len(action_set) * action_time_emb_dim
    batch_size = data_one_batch.shape[0]
    num_time_intervals = int(time_horizon / time_interval_len)
    imputed_mental_states = torch.zeros((batch_size, num_time_intervals, num_mental_types))
    # imputed_mental_states[b_id, interval_id, :]: zeros/one-hot vector with length num_mental_types,
    # represents mental occurring state in cur interval
    # todo: pl_i_sum_in_each_interval is used to see the values of pl_i_sum in each interval
    # pl_i_sum_in_each_interval = torch.zeros((batch_size, num_time_intervals))
    all_mentals_last_occur_interval = np.zeros((batch_size, num_mental_types))
    for l in range(num_time_intervals):
        cur_lstm = HazardRate_LSTM(lstm_input_size, num_mental_types, lstm_layers_num, lstm_dropout_prob)
        if l == 0:
            ini_hidden_state = torch.zeros((lstm_layers_num, batch_size, num_mental_types))
        else:
            ini_hidden_state = torch.cat([imputed_mental_states[:, l - 1, :]] * lstm_layers_num).reshape(lstm_layers_num, batch_size, num_mental_types)
        all_mentals_hazard_rate_list = cur_lstm.forward(data_one_batch, ini_hidden_state)
        # shape: [batch_size, num_intervals, num_mental_types]
        mentals_prob_cur_interval = get_mentals_cat_distribution_cur_interval(l, all_mentals_hazard_rate_list, num_mental_types, all_mentals_last_occur_interval)
        # shape: [batch_size, num_mental_types], represents the prob that cur mental occurs at cur_interval
        # todo: add bernoulli distribution before gumble max trick to see whether a mental occurs in cur_interval or not
        all_mentals_prob_sum_cur_interval = torch.sum(mentals_prob_cur_interval, dim=1)  # shape: [batch_size]
        # pl_i_sum_in_each_interval[:, l] = all_mentals_prob_sum_cur_interval
        mentals_occur_or_not_matrix = torch.bernoulli(all_mentals_prob_sum_cur_interval)  # shape: [batch_size], 0: no mental occur, 1: some mental occur
        if torch.max(mentals_occur_or_not_matrix).item() == 0: # no event will happen in this interval
            continue
        else:
            # first check out which batch occur a mental
            batch_idx_mental_occur = torch.nonzero(mentals_occur_or_not_matrix).view(-1)
            # return the list of batch idx during which some mental occurs
            selected_batch_mentals_prob = mentals_prob_cur_interval[batch_idx_mental_occur, :]
            # shape: [num_mentals_occur_batches, num_mental_types]
            sampled_mental_type = Gumbel_Max_Trick(selected_batch_mentals_prob)
            # list with length: occur_batch_size, each element is the sampled mental type in cur_interval in cur_batch

            for b_id in range(len(batch_idx_mental_occur)):
                one_hot_occur_mental = torch.zeros(num_mental_types)
                one_hot_occur_mental[sampled_mental_type[b_id]] = 1
                imputed_mental_states[batch_idx_mental_occur[b_id], l, :] = one_hot_occur_mental
                all_mentals_last_occur_interval[batch_idx_mental_occur[b_id], sampled_mental_type[b_id]] = l

    # pl_i_sum_in_each_interval = torch.mean(pl_i_sum_in_each_interval, dim=0)
    # print(pl_i_sum_in_each_interval.shape)
    # print(pl_i_sum_in_each_interval)
    return imputed_mental_states  # [batch_size, num_time_intervals, 1]




action_embedding = get_action_pre_time_embedding(data, action_time_emb_dim, time_horizon, time_interval_len, action_predicate_set)
# print(action_embedding.shape)  # [10, 100, 6]

for b in range(num_batches):
    data_one_batch = action_embedding[b*batch_size : (b+1)*batch_size, :, :]
    imputed_mental_states = Sample(mental_predicate_set, action_predicate_set, action_time_emb_dim, lstm_layers_num,
                                   lstm_dropout_prob, data_one_batch, time_horizon, time_interval_len)
# data_one_batch: [batch_size, num_time_intervals, num_action_predicates * time_embedding_dim]



