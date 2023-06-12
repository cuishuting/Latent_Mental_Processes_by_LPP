import numpy as np
import itertools

np.random.seed(1)


class Logic_Model_Generator:

    def __init__(self):

        ### the following parameters are used to manually define the logic rules
        self.num_predicate = 3
        self.num_formula = 3  # num of prespecified logic rules
        self.BEFORE = 'BEFORE'
        self.EQUAL = 'EQUAL'
        self.AFTER = 'AFTER'
        self.Time_tolerance = 0.32
        self.mental_predicate_set = [0]
        self.action_predicate_set = [1, 2]
        self.head_predicate_set = [0, 1, 2]  # the index set of all head predicates
        self.decay_rate = 0.8  # decay kernel

        ### the following parameters are used to generate synthetic data
        ### for the learning part, the following is used to claim variables
        ### self.model_parameter = {0:{},1:{},...,6:{}}
        self.model_parameter = {}

        '''
        mental
        '''

        head_predicate_idx = 0
        self.model_parameter[head_predicate_idx] = {}
        self.model_parameter[head_predicate_idx]['base'] = -0.4

        formula_idx = 0
        self.model_parameter[head_predicate_idx][formula_idx] = {}
        self.model_parameter[head_predicate_idx][formula_idx]['weight'] = 0.8

        '''
        action
        '''
        head_predicate_idx = 1
        self.model_parameter[head_predicate_idx] = {}
        self.model_parameter[head_predicate_idx]['base'] = -0.3

        formula_idx = 0
        self.model_parameter[head_predicate_idx][formula_idx] = {}
        self.model_parameter[head_predicate_idx][formula_idx]['weight'] = 0.2

        head_predicate_idx = 2
        self.model_parameter[head_predicate_idx] = {}
        self.model_parameter[head_predicate_idx]['base'] = -0.3

        formula_idx = 0
        self.model_parameter[head_predicate_idx][formula_idx] = {}
        self.model_parameter[head_predicate_idx][formula_idx]['weight'] = 1.0

        # NOTE: set the content of logic rules
        self.logic_template = self.logic_rule()

    def logic_rule(self):
        '''
        This function encodes the content of logic rules
        logic_template = {0:{},1:{},...,6:{}}
        '''

        '''
        Only head predicates may have negative values, because we only record each body predicate's boosted time 
        (states of body predicates are always be 1) 
        '''
        logic_template = {}

        '''
        Mental predicate: [0]
        '''

        head_predicate_idx = 0
        logic_template[head_predicate_idx] = {}

        # NOTE: rule content: 1 and before(1, 0) \to \neg 0
        formula_idx = 0
        logic_template[head_predicate_idx][formula_idx] = {}
        logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [1]
        logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1]  # use 1 to indicate True; use 0 to indicate False
        logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [0]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[1, 0]]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [self.BEFORE]

        '''
        Action predicates: [1, 2]
        '''

        head_predicate_idx = 1
        logic_template[head_predicate_idx] = {}

        # NOTE: rule content: 0 and 1 and before(0,1) to 1
        formula_idx = 0
        logic_template[head_predicate_idx][formula_idx] = {}
        logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [0, 1]
        logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1, 1]
        logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[0, 1]]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [self.BEFORE]

        head_predicate_idx = 2
        logic_template[head_predicate_idx] = {}

        # NOTE: rule content: 0 and 1 and before(0,1) to 2
        formula_idx = 0
        logic_template[head_predicate_idx][formula_idx] = {}
        logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [0, 1]
        logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1, 1]
        logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[0, 1]]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [self.BEFORE]

        return logic_template

    def intensity(self, cur_time, head_predicate_idx, history):
        feature_formula = []
        weight_formula = []
        effect_formula = []

        for formula_idx in list(self.logic_template[head_predicate_idx].keys()):
            # range all the formula for the chosen head_predicate
            weight_formula.append(self.model_parameter[head_predicate_idx][formula_idx]['weight'])
            feature_formula.append(self.get_feature(cur_time=cur_time, head_predicate_idx=head_predicate_idx,
                                                    history=history,
                                                    template=self.logic_template[head_predicate_idx][formula_idx]))
            effect_formula.append(self.get_formula_effect(template=self.logic_template[head_predicate_idx][formula_idx]))

        intensity = np.array(weight_formula) * np.array(feature_formula) * np.array(effect_formula)
        intensity = self.model_parameter[head_predicate_idx]['base'] + np.sum(intensity)
        # if intensity >= 0: intensity = np.max([ intensity, self.model_parameter[head_predicate_idx]['base'] ])
        # else: intensity = np.exp( self.model_parameter[head_predicate_idx]['base'] + intensity )
        intensity = np.exp(intensity)
        return intensity

    def get_feature(self, cur_time, head_predicate_idx, history, template):
        occur_time_dic = {}
        feature = 0
        for idx, body_predicate_idx in enumerate(template['body_predicate_idx']):
            occur_time = np.array(history[body_predicate_idx]['time'][1:])
            mask = (occur_time <= cur_time)  # find corresponding history
            # mask: filter all the history time points that satisfies the conditions which will boost the head predicate
            occur_time_dic[body_predicate_idx] = occur_time[mask]
        occur_time_dic[head_predicate_idx] = [cur_time]
        ### get weights
        # compute features whenever any item of the transition_item_dic is nonempty
        history_transition_len = [len(i) for i in occur_time_dic.values()]
        if min(history_transition_len) > 0:
            # need to compute feature using logic rules
            time_combination = np.array(list(itertools.product(*occur_time_dic.values())))
            # get all possible time combinations
            time_combination_dic = {}
            for i, idx in enumerate(list(occur_time_dic.keys())):
                time_combination_dic[idx] = time_combination[:, i]
            temporal_kernel = np.ones(len(time_combination))
            for idx, temporal_relation_idx in enumerate(template['temporal_relation_idx']):
                time_difference = time_combination_dic[temporal_relation_idx[0]] - time_combination_dic[temporal_relation_idx[1]]
                if template['temporal_relation_type'][idx] == 'BEFORE':
                    temporal_kernel *= (time_difference < -self.Time_tolerance) * np.exp(
                        -self.decay_rate * (cur_time - time_combination_dic[temporal_relation_idx[0]]))
                if template['temporal_relation_type'][idx] == 'EQUAL':
                    temporal_kernel *= (abs(time_difference) <= self.Time_tolerance) * np.exp(
                        -self.decay_rate * (cur_time - time_combination_dic[temporal_relation_idx[0]]))
                if template['temporal_relation_type'][idx] == 'AFTER':
                    temporal_kernel *= (time_difference > self.Time_tolerance) * np.exp(
                        -self.decay_rate * (cur_time - time_combination_dic[temporal_relation_idx[1]]))
            feature = np.sum(temporal_kernel)

        return feature

    '''
    get_formula_effect(): the body condition will boost the head to be 1 or 0 (positive or neg effect to head predicate) 
    (self.model_parameter[head_predicate_idx][formula_idx]['weight'] represents the effect's degree)
    '''

    def get_formula_effect(self, template):
        if template['head_predicate_sign'][0] == 1:
            formula_effect = 1
        else:
            formula_effect = -1
        return formula_effect

    def generate_data(self, num_sample, time_horizon):
        data = {}

        # NOTE: data = {0:{},1:{},....,num_sample:{}}
        for sample_ID in np.arange(0, num_sample, 1):
            data[sample_ID] = {}  # each data[sample_ID] stores one realization of the point process
            # initialize data
            for predicate_idx in np.arange(0, self.num_predicate, 1):
                data[sample_ID][predicate_idx] = {}
                data[sample_ID][predicate_idx]['time'] = [0]
            t = 0  # cur_time
            sep = 0.03
            while t < time_horizon:
                grid = np.arange(t, time_horizon, sep)
                intensity_potential = []
                for time in grid:
                    intensity_potential.append(
                        [self.intensity(time, head_predicate_idx, data[sample_ID]) for head_predicate_idx in
                         self.head_predicate_set])
                intensity_potential = [sum(item) for item in intensity_potential]
                intensity_max = np.max(np.array(intensity_potential))
                time_to_event = np.random.exponential(scale=1.0 / intensity_max)  # sample the interarrival time
                t = t + time_to_event
                if t > time_horizon:
                    break
                # accept reject
                intensity_potential = []
                for head_predicate_idx in self.head_predicate_set:
                    intensity_potential.append(self.intensity(t, head_predicate_idx, data[sample_ID]))
                # accept
                if (np.random.uniform() < np.sum(np.array(intensity_potential)) / intensity_max):
                    tmp = np.random.multinomial(1, pvals=np.array(intensity_potential) / np.sum(np.array(intensity_potential)))
                    idx = np.argmax(tmp)
                    data[sample_ID][idx]['time'].append(t)

        return data

    # def compute_intensity(self, data, time_horizon, step=0.03):
    #     intensity = {}
    #     time_grid = np.arange(0, time_horizon, step)
    #     for sample_ID in data:
    #         intensity[sample_ID] = {}
    #         for predicate_idx in self.head_predicate_set:
    #             intensity[sample_ID][predicate_idx] = []
    #     for sample_ID in data:
    #         for t in time_grid:
    #             for predicate_idx in self.head_predicate_set:
    #                 intensity[sample_ID][predicate_idx].append(
    #                     self.intensity(t, predicate_idx, history=data[sample_ID]))
    #     return intensity

if __name__ == "__main__":
    logic_model_generator = Logic_Model_Generator()
    data = logic_model_generator.generate_data(num_sample=10, time_horizon=10)
    np.save('./data.npy', data)