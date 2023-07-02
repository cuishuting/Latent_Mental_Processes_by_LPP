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
        self.input_size = input_size  # input_size: action time embedding's size in each time interval
        self.hidden_size = hidden_size # hidden_size: 1, represents hazard rate in each time interval
        self.num_layers = num_layers
        self.dropout_rate = dropout_prob
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            batch_first=True, dropout=self.dropout_rate)

    def forward(self, data):
        # data has shape: [batch_size, num_time_intervals, num_action_predicates * time_embedding_dim]
        output, (h_n, c_n) = self.lstm(data)
        # todo: output's shape [batch_size, num_time_intervals, 1], because the hidden_size represents hazard rate h_l^i
        #  for each mental predicate at each time interval
        return output


def Sample():


data = np.load("./data_new.npy", allow_pickle=True).item()
action_embedding = get_action_pre_time_embedding(data, 3, 10, 0.001, action_predicate_set)
print(action_embedding)