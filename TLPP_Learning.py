import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import *
from TLPP_Generation import Logic_Model_Generator
##################################################################
np.random.seed(100)
logic_model_generator = Logic_Model_Generator()


class Logic_Model(nn.Module):
    # for example, consider three rules:
    # A and B and Equal(A,B), and Before(A, D), then D;
    # C and Before(C, Not D), then  Not D
    # D Then  E, and Equal(D, E)
    # note that define the temporal predicates as compact as possible

    def __init__(self):

        ### the following parameters are used to manually define the logic rules
        self.num_predicate = 3                  # num_predicate is same as num_node
        self.num_formula = 3                    # num of prespecified logic rules
        self.BEFORE = 'BEFORE'
        self.EQUAL = 'EQUAL'
        self.AFTER = 'AFTER'
        self.Time_tolerance = 0.32               
        self.body_predicate_set = []                        # the index set of all body predicates
        self.mental_predicate_set = [0]
        self.action_predicate_set = [1, 2]
        self.head_predicate_set = [0, 1, 2]     # the index set of all head predicates
        self.decay_rate = 0.8                                 # decay kernel
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
        self.model_parameter[head_predicate_idx]['base'] = torch.tensor([-0.43], dtype=torch.float64, requires_grad=True)

        formula_idx = 0
        self.model_parameter[head_predicate_idx][formula_idx] = {}
        self.model_parameter[head_predicate_idx][formula_idx]['weight'] = torch.tensor([0.83], dtype=torch.float64, requires_grad=True)

        '''
        action
        '''
        head_predicate_idx = 1
        self.model_parameter[head_predicate_idx] = {}
        self.model_parameter[head_predicate_idx]['base'] = torch.tensor([-0.33], dtype=torch.float64, requires_grad=True)

        formula_idx = 0
        self.model_parameter[head_predicate_idx][formula_idx] = {}
        self.model_parameter[head_predicate_idx][formula_idx]['weight'] = torch.tensor([0.23], dtype=torch.float64, requires_grad=True)

        head_predicate_idx = 2
        self.model_parameter[head_predicate_idx] = {}
        self.model_parameter[head_predicate_idx]['base'] = torch.tensor([-0.33], dtype=torch.float64, requires_grad=True)

        formula_idx = 0
        self.model_parameter[head_predicate_idx][formula_idx] = {}
        self.model_parameter[head_predicate_idx][formula_idx]['weight'] = torch.tensor([0.97], dtype=torch.float64, requires_grad=True)

        #NOTE: set the content of logic rules
        self.logic_template = logic_model_generator.logic_rule()

    def intensity(self, cur_time, head_predicate_idx, history):
        feature_formula = []
        weight_formula = []
        effect_formula = []

        for formula_idx in list(self.logic_template[head_predicate_idx].keys()):
            weight_formula.append(self.model_parameter[head_predicate_idx][formula_idx]['weight'])

            cur_feature = logic_model_generator.get_feature(cur_time, head_predicate_idx, history, self.logic_template[head_predicate_idx][formula_idx])
            feature_formula.append(torch.tensor([cur_feature], dtype=torch.float64))

            cur_effect = logic_model_generator.get_formula_effect(self.logic_template[head_predicate_idx][formula_idx])
            effect_formula.append(torch.tensor([cur_effect], dtype=torch.float64))

        intensity = torch.cat(weight_formula, dim=0) * torch.cat(feature_formula, dim=0) * torch.cat(effect_formula, dim=0)
        intensity = self.model_parameter[head_predicate_idx]['base'] + torch.sum(intensity)
        intensity = torch.exp(intensity)

        return intensity

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
        loss = -self.log_likelihood(dataset, sample_ID_batch, T_max)
        loss.backward()
        optimizer.step()
        return loss


if __name__ == '__main__':
    #NOTE: some parameters
    num_samples = 10
    time_horizon = 20
    batch_size = 10
    num_batch = num_samples // batch_size
    num_iter = 10
    lr = 1e-3

    data = logic_model_generator.generate_data(num_samples, time_horizon)

    logic_model = Logic_Model()
    losses = []                     #NOTE: store the loss
    params = []
    model_parameters = [logic_model.model_parameter[0]['base'],
                        logic_model.model_parameter[0][0]['weight'],
                        logic_model.model_parameter[1]['base'],
                        logic_model.model_parameter[1][0]['weight'],
                        logic_model.model_parameter[2]['base'],
                        logic_model.model_parameter[2][0]['weight'],
                        ]

    optimizer = optim.Adam(params=model_parameters, lr=lr)
    #optimizer = optim.SGD(params=model_parameters, lr=lr)
    print('start training!')
    print(logic_model.log_likelihood(data, np.arange(0, len(data), 1), T_max=time_horizon))
    
    for iter in tqdm(range(num_iter)):
        loss = 0
        tmp = []
        for batch_idx in tqdm(np.arange(0, num_batch, 1)):
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
    # np.save('intensity.npy',intensity)
    # np.save('data.npy',data)
    # np.save('losses.npy',losses)
    # np.save('params.npy',params)
