import operator
import numpy as np
import tensorflow as tf

from common.utils import BaseAgent, softmax

class RuleAgent(BaseAgent):
    def __init__(self, env, name, scenario):
        super().__init__(env, name)
        self.scenario = scenario
        support = set(['simple_speaker_listener'])
        assert(self.scenario in support)

    def act(self, obs_n):
        if self.scenario == 'simple_speaker_listener': 
            act_n = [None, None]
            act_n[0] = obs_n[0]
            label = np.argmax(obs_n[1][-3:])+1
            act_n[1] = [0, 0, 0, 0, 0]
            target = obs_n[1][label*2:label*2+1+1]
            normalize_target = softmax(np.abs(target))
            if target[0]>=0:
                act_n[1][1] = normalize_target[0] # target[0]
            else:
                act_n[1][2] = normalize_target[0] # -target[0]
            if target[1]>=0:
                act_n[1][3] = normalize_target[1] # target[1]
            else:
                act_n[1][4] = normalize_target[1] # -target[1]

        return act_n
