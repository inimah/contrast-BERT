# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''

Generic sentence evaluation scripts wrapper

'''
from __future__ import absolute_import, division, unicode_literals

import os
import sys

from senteval import utils
from senteval.ensembleEval import EnsembleClassifierTrain
from senteval.probing import *



class SE(object):
    def __init__(self, args, params, batcher, prepare=None):
        # parameters
        self.args = args
        params = utils.dotdict(params)
        params.usepytorch = True if 'usepytorch' not in params else params.usepytorch
        params.seed = 1111 if 'seed' not in params else params.seed

        params.batch_size = 128 if 'batch_size' not in params else params.batch_size
        params.nhid = 0 if 'nhid' not in params else params.nhid
        params.kfold = 5 if 'kfold' not in params else params.kfold

        if 'classifier' not in params or not params['classifier']:
            params.classifier = {'nhid': 0}

        assert 'nhid' in params.classifier, 'Set number of hidden units in classifier config!!'

        self.params = params

        # batcher and prepare
        self.batcher = batcher
        self.prepare = prepare if prepare else lambda x, y: None

        self.list_tasks = ['EnsembleClassifier']

    def eval(self, name):
        # evaluate on evaluation [name], either takes string or list of strings
        if (isinstance(name, list)):
            self.results = {x: self.eval(x) for x in name}
            return self.results

        tpath = self.params.task_path
        assert name in self.list_tasks, str(name) + ' not in ' + str(self.list_tasks)
        
        print("name:", name)
        sys.stdout.flush()

        # Original SentEval tasks
        if name == 'EnsembleClassifier':
            self.evaluation = EnsembleClassifierTrain("nlp-brin-id/id-hoax-report-merge-v2", "title-fact", seed=self.params.seed)
        

        self.params.current_task = name
        
        # for eval only
        #self.evaluation.do_prepare(self.params, self.prepare)
        #self.results = self.evaluation.eval_run(self.params, self.batcher)
        
        # for eval + MLP training
        self.evaluation.do_preparetrain(self.params, self.prepare)
        self.results = self.evaluation.evaltrain_run(self.params, self.batcher)

        return self.results
