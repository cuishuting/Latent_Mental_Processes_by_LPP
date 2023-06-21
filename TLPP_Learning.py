import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import *
from TLPP_Generation import Logic_Model_Generator
import matplotlib.pyplot as plt
import os
##################################################################
np.random.seed(100)


class Logic_Model(nn.Module):
    # for example, consider three rules:
    # A and B and Equal(A,B), and Before(A, D), then D;
    # C and Before(C, Not D), then  Not D
    # D Then  E, and Equal(D, E)
    # note that define the temporal predicates as compact as possible

    def __init__(self, time_tolerance, decay_rate, initial_params):

        ### the following parameters are used to manually define the logic rules
        self.num_predicate = 3                  # num_predicate is same as num_node
        self.num_formula = 3                 # num of prespecified logic rules
        self.BEFORE = 'BEFORE'
        self.EQUAL = 'EQUAL'
        self.AFTER = 'AFTER'
        self.Time_tolerance = time_tolerance
        self.mental_predicate_set = [0]
        self.action_predicate_set = [1, 2]
        self.head_predicate_set = [0, 1, 2]     # the index set of all head predicates
        self.decay_rate = decay_rate                           # decay kernel
        self.integral_resolution = 0.03

        ### the following parameters are used to generate synthetic data
        ### for the learning part, the following is used to claim variables
        ### self.model_parameter = {0:{},1:{},...,6:{}}
        self.model_parameter = {}
        
        '''
        mental
        '''

        head_predicate_idx = 0
        self.model_parameter[head_predicate_idx] = {}
        self.model_parameter[head_predicate_idx]['base'] = torch.tensor([initial_params[0]], dtype=torch.float64, requires_grad=True)

        formula_idx = 0
        self.model_parameter[head_predicate_idx][formula_idx] = {}
        self.model_parameter[head_predicate_idx][formula_idx]['weight'] = torch.tensor([initial_params[1]], dtype=torch.float64, requires_grad=True)

        '''
        action
        '''
        head_predicate_idx = 1
        self.model_parameter[head_predicate_idx] = {}
        self.model_parameter[head_predicate_idx]['base'] = torch.tensor([initial_params[2]], dtype=torch.float64, requires_grad=True)

        formula_idx = 0
        self.model_parameter[head_predicate_idx][formula_idx] = {}
        self.model_parameter[head_predicate_idx][formula_idx]['weight'] = torch.tensor([initial_params[3]], dtype=torch.float64, requires_grad=True)

        head_predicate_idx = 2
        self.model_parameter[head_predicate_idx] = {}
        self.model_parameter[head_predicate_idx]['base'] = torch.tensor([initial_params[4]], dtype=torch.float64, requires_grad=True)

        formula_idx = 0
        self.model_parameter[head_predicate_idx][formula_idx] = {}
        self.model_parameter[head_predicate_idx][formula_idx]['weight'] = torch.tensor([initial_params[5]], dtype=torch.float64, requires_grad=True)

        #NOTE: set the content of logic rules
        self.logic_template = logic_model_generator.logic_rule()

    def intensity(self, cur_time, head_predicate_idx, history):
        feature_formula = []
        weight_formula = []
        effect_formula = []
        # print("cur head: ", head_predicate_idx)
        for formula_idx in list(self.logic_template[head_predicate_idx].keys()):
            weight_formula.append(self.model_parameter[head_predicate_idx][formula_idx]['weight'])
            cur_feature = logic_model_generator.get_feature(cur_time, head_predicate_idx, history, self.logic_template[head_predicate_idx][formula_idx])
            # print("feature: ", cur_feature)
            feature_formula.append(torch.tensor([cur_feature], dtype=torch.float64))
            cur_effect = logic_model_generator.get_formula_effect(self.logic_template[head_predicate_idx][formula_idx])
            effect_formula.append(torch.tensor([cur_effect], dtype=torch.float64))

        intensity = torch.cat(weight_formula, dim=0) * torch.cat(feature_formula, dim=0) * torch.cat(effect_formula, dim=0)
        intensity = self.model_parameter[head_predicate_idx]['base'] + torch.sum(intensity)
        if intensity.item() >= 0:
            if intensity.item() >= self.model_parameter[head_predicate_idx]['base'].item():
                final_intensity = intensity
            else:
                final_intensity = self.model_parameter[head_predicate_idx]['base']
                print("former intensity:")
                print(intensity)
                print("final intensity:")
                print(final_intensity)
            # final_intensity = torch.max(torch.tensor([intensity.item(), self.model_parameter[head_predicate_idx]['base']]))
            return final_intensity
        else:
            return torch.exp(intensity)


    def log_likelihood(self, dataset, sample_ID_batch, T_max):
        '''
        This function calculates the log-likehood given the dataset
        log-likelihood = \sum log(intensity(transition_time)) + int_0^T intensity dt

        Parameters:
            dataset: 
            sample_ID_batch: list
            T_max:
        '''
        log_likelihood = torch.tensor([0], dtype=torch.float64)
        # iterate over samples
        for sample_ID in sample_ID_batch:
            # iterate over head predicates; each predicate corresponds to one intensity
            data_sample = dataset[sample_ID]
            for head_predicate_idx in self.head_predicate_set:
                #NOTE: compute the summation of log intensities at the transition times
                intensity_log_sum = self.intensity_log_sum(head_predicate_idx, data_sample)
                #NOTE: compute the integration of intensity function over the time horizon
                intensity_integral = self.intensity_integral(head_predicate_idx, data_sample, T_max)
                log_likelihood += (intensity_log_sum - intensity_integral)
        return log_likelihood

    def intensity_log_sum(self, head_predicate_idx, data_sample):
        intensity_transition = []
        for t in data_sample[head_predicate_idx]['time'][1:]:
            #NOTE: compute the intensity at transition times
            cur_intensity = self.intensity(t, head_predicate_idx, data_sample)
            intensity_transition.append(cur_intensity)

        if len(intensity_transition) == 0: # only survival term, no event happens
            log_sum = torch.tensor([0], dtype=torch.float64)
        else:
            log_sum = torch.sum(torch.log(torch.cat(intensity_transition, dim=0)))
        return log_sum

    def intensity_integral(self, head_predicate_idx, data_sample, T_max):
        start_time = 0
        end_time = T_max
        intensity_grid = []
        for t in np.arange(start_time, end_time, self.integral_resolution):
            #NOTE: evaluate the intensity values at the chosen time points
            cur_intensity = self.intensity(t, head_predicate_idx, data_sample)
            intensity_grid.append(cur_intensity)
        #NOTE: approximately calculate the integral
        integral = torch.sum(torch.cat(intensity_grid, dim=0) * self.integral_resolution)
        return integral

    ### the following functions are for optimization
    def optimize_log_likelihood(self, dataset, sample_ID_batch, T_max, optimizer):
        optimizer.zero_grad()  # set gradient zero at the start of a new mini-batch
        loss = torch.tensor(-1) * self.log_likelihood(dataset, sample_ID_batch, T_max)
        batch_size = torch.tensor(len(sample_ID_batch))
        loss = torch.div(loss, batch_size)
        loss.backward()
        optimizer.step()
        return loss


if __name__ == '__main__':
    tuning_params = [{"time_tolerance": 0.1,
                      "decay_rate": 0.8,
                      "initial_params": [0.9, 3.0, 0.6, 2.2, 0.3, 2.5]},
                     {"time_tolerance": 0.2,
                      "decay_rate": 0.9,
                      "initial_params": [0.35, 1.2, 0.25, 0.4, 0.18, 1.5]},
                     {"time_tolerance": 0.1,
                      "decay_rate": 0.9,
                      "initial_params": [0.9, 3.0, 0.6, 2.2, 0.3, 2.5]},
                     {"time_tolerance": 0.1,
                      "decay_rate": 0.8,
                      "initial_params": [0.35, 1.2, 0.25, 0.4, 0.18, 1.5]}
                     ]
    for i in range(len(tuning_params)):
        print("########### tuning epoch: ", i+1)
        logic_model_generator = Logic_Model_Generator(tuning_params[i]["time_tolerance"], tuning_params[i]["decay_rate"])
        num_samples = 2000
        time_horizon = 10
        batch_size = 20
        num_batch = num_samples // batch_size
        num_iter = 500
        lr = 1e-3

        data = logic_model_generator.generate_data(num_samples, time_horizon)
        logic_model = Logic_Model(tuning_params[i]["time_tolerance"], tuning_params[i]["decay_rate"], tuning_params[i]["initial_params"])
        losses = []                     #NOTE: store the loss
        params = []
        model_parameters = [logic_model.model_parameter[0]['base'],
                            logic_model.model_parameter[0][0]['weight'],
                            logic_model.model_parameter[1]['base'],
                            logic_model.model_parameter[1][0]['weight'],
                            logic_model.model_parameter[2]['base'],
                            logic_model.model_parameter[2][0]['weight']]

        optimizer = optim.SGD([{'params': [model_parameters[0],
                                           model_parameters[2],
                                           model_parameters[4]],
                                'lr': lr * 1e-1},
                               {'params': [model_parameters[1],
                                           model_parameters[3],
                                           model_parameters[5]]}],
                              lr=lr, weight_decay=1e-3)
        #optimizer = optim.SGD(params=model_parameters, lr=lr)
        print('start training!')
        for iter in range(num_iter):
            loss = 0
            tmp = []
            for batch_idx in np.arange(0, num_batch, 1):
                indices = np.arange(batch_idx*batch_size, (batch_idx+1)*batch_size, 1)
                loss = logic_model.optimize_log_likelihood(data, indices, time_horizon, optimizer)
                tmp.append(loss.item())
            #print(tmp)
            loss = np.array(tmp).mean()
            losses.append(loss)
            print('[INFO] loss is >>> ', loss)
            print(logic_model.model_parameter)
            a = [item.clone().detach().item() for item in model_parameters]
            params.append(a)

        losses = np.array(losses)
        params = np.array(params)
        print(params)
        # np.save('intensity.npy',intensity)
        # np.save('data.npy',data)
        # np.save('losses.npy',losses)
        # np.save('params.npy',params)

        # ground_truth_param = logic_model_generator.model_parameter
        # fig, ax = plt.subplots()
        # plt.title("model_parameter[0]['base']")
        # ax.plot(np.arange(num_iter), params[:, 0], color='blue')
        # ax.plot(np.arange(num_iter), [ground_truth_param[0]["base"]]*num_iter, color='blue')
        # fig.show()
        #
        #
        #
        # fig, ax = plt.subplots()
        # plt.title("model_parameter[0][0]['weight']")
        # ax.plot(np.arange(num_iter), params[:, 1], color='red')
        # ax.plot(np.arange(num_iter), [ground_truth_param[0][0]["weight"]]*num_iter, color='red')
        # fig.show()
        #
        #
        # fig, ax = plt.subplots()
        # plt.title("model_parameter[1]['base']")
        # ax.plot(np.arange(num_iter), params[:, 2], color='green')
        # ax.plot(np.arange(num_iter), [ground_truth_param[1]["base"]]*num_iter, color='green')
        # fig.show()
        #
        #
        # fig, ax = plt.subplots()
        # plt.title("model_parameter[1][0]['weight']")
        # ax.plot(np.arange(num_iter), params[:, 3], color='black')
        # ax.plot(np.arange(num_iter), [ground_truth_param[1][0]["weight"]]*num_iter, color='black')
        # fig.show()
        #
        # fig, ax = plt.subplots()
        # plt.title("model_parameter[2]['base']")
        # ax.plot(np.arange(num_iter), params[:, 4], color='purple')
        # ax.plot(np.arange(num_iter), [ground_truth_param[2]["base"]]*num_iter, color='purple')
        # fig.show()
        #
        # fig, ax = plt.subplots()
        # plt.title("model_parameter[2][0]['weight']")
        # ax.plot(np.arange(num_iter), params[:, 5], color='pink')
        # ax.plot(np.arange(num_iter), [ground_truth_param[2][0]["weight"]]*num_iter, color='pink')
        # fig.show()
        #
        # plt.title("loss")
        # plt.plot(np.arange(num_iter), losses)
        # plt.show()


