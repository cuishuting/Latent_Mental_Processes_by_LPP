import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
# TODO: get action predicates' time's embedding, corresponding to "f(D) = h" in the draft

mental_predicate_set = [0]
action_predicate_set = [1, 2]
def get_action_pre_time_embedding(data, embedding_dim):
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
    # returned "time_embedding" includes all samples' all action predicates' occurred time's embedding with size
    # "embedding_dim" for each time stamp
    return time_embedding


# TODO : data has the form "data[sample_ID][predicate_idx]['time']", we now only observed action predicates' occur time
class HazardRate_LSTM(nn.Module):
    def __init__(self):

        # self.action_embedding
        pass

    def forward(self, data):
        pass

data = np.load("./data_new.npy", allow_pickle=True).item()
action_time_embedding = get_action_pre_time_embedding(data, 3)
print(action_time_embedding[0])