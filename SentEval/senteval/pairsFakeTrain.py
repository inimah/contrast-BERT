# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
Binary classifier and corresponding datasets : MR, CR, SUBJ, MPQA, FakeCLS (Fake news classification)
'''
from __future__ import absolute_import, division, unicode_literals

import io
import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import logging

from senteval.tools.validation import InnerKFoldClassifier, EvalClassifier, EvalTrainClassifier

import transformers
from datasets import load_dataset
from sklearn.utils import resample,shuffle

'''
This class module is for validation task: Fake News classification
it requires training MLP model
'''


class BinaryClassifierEval(object):
    
    def __init__(self, train_text, train_label, test_text, test_label, do_eval, seed=1111):
        self.seed = seed
        self.do_eval = do_eval
        self.train_samples, self.train_labels = train_text, train_label
        self.test_samples, self.test_labels = test_text, test_label
        self.n_train_samples = len(self.train_samples)
        self.n_test_samples = len(self.test_samples)
        
    def do_prepare(self, params, prepare):
        # prepare is given the whole text
        return prepare(params, self.samples)
        # prepare puts everything it outputs in "params" : params.word2id etc
        # Those output will be further used by "batcher".
        
    def do_preparetrain(self, params, prepare):
        # prepare is given the whole text
        return prepare(params, self.n_train_samples)
        # prepare puts everything it outputs in "params" : params.word2id etc
        # Those output will be further used by "batcher".

    
    def loadFile(self, fpath):
        with io.open(fpath, 'r', encoding='latin-1') as f:
            return [line.split() for line in f.read().splitlines()]
        
        
    def loadFakeNews(self, task_path, split, attribute):
        
        print("attribute: ", attribute)
        sys.stdout.flush()
        
        
        data_files = {
                            "train_hoax_fct": "train/train_hoax_fct.csv", \
                            "train_nonhoax_fct": "train/train_nonhoax_fct.csv", \
                            "test_hoax_fct": "test/test_hoax_fct.csv", \
                            "test_nonhoax_fct": "test/test_nonhoax_fct.csv", \
                            }
        
        # are we using Title, Content, or Fact?
        if attribute == "title-fact":
            
            print("split:", split)
            sys.stdout.flush()
            
            if split=="train":
                
                hoax_fct = load_dataset(task_path, data_files=data_files, \
                                              split="train_hoax_fct", trust_remote_code=True, token="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")
                nonhoax_fct = load_dataset(task_path, data_files=data_files, \
                                                 split="train_nonhoax_fct", trust_remote_code=True, \
                                                 token="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")
            else:
                # test set
                
                hoax_fct = load_dataset(task_path, data_files=data_files, \
                                              split="test_hoax_fct", trust_remote_code=True, token="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")
                nonhoax_fct = load_dataset(task_path, data_files=data_files, \
                                                 split="test_nonhoax_fct", trust_remote_code=True, \
                                                 token="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")
                
            hoax_fct = hoax_fct.to_pandas()
            nonhoax_fct = nonhoax_fct.to_pandas()
            
            # upsampling nonhoax into nsamples of hoax
            nonhoax_fct_upsampled = resample(nonhoax_fct, random_state=42, n_samples=len(hoax_fct), replace=True)
            
            print("len hoax_fct:", len(hoax_fct))
            sys.stdout.flush()
            
            print("len nonhoax_fct:", len(nonhoax_fct))
            sys.stdout.flush()
            
            print("len nonhoax_fct_upsampled:", len(nonhoax_fct_upsampled))
            sys.stdout.flush()
            
            new_contents = []
            new_labels = []
            for ttl, fct in hoax_fct.values:
                text =  "Klaim: " + str(ttl) + ".\nFakta: " + str(fct) +"."
                new_contents.append(text)
                new_labels.append(1)
            for ttl, fct in nonhoax_fct_upsampled.values:
                text = "Klaim: " + str(ttl) + ".\nFakta: " + str(fct) +"."
                new_contents.append(text)
                new_labels.append(0)
                
            return (new_contents, new_labels)

    # train MLP during validation task
    def run(self, params, batcher):
        
            
        # inference only run in self.model.eval() mode
        train_enc_input = []
        test_enc_input = []
        # Sort to reduce padding
        train_sorted_corpus = sorted(zip(self.train_samples, self.train_labels),
                               key=lambda z: (len(z[0]), z[1]))
        test_sorted_corpus = sorted(zip(self.test_samples, self.test_labels),
                               key=lambda z: (len(z[0]), z[1]))
        train_sorted_samples = [x for (x, y) in train_sorted_corpus]
        train_sorted_labels = [y for (x, y) in train_sorted_corpus]
        
        test_sorted_samples = [x for (x, y) in test_sorted_corpus]
        test_sorted_labels = [y for (x, y) in test_sorted_corpus]
        
        logging.info('Generating sentence embeddings for training subset')
        for ii in range(0, self.n_train_samples, params.batch_size):
            batch = train_sorted_samples[ii:ii + params.batch_size]
            
            embeddings = batcher(params, batch)
            train_enc_input.append(embeddings)
        train_enc_input = np.vstack(train_enc_input)
        logging.info('Generated sentence embeddings')
        
        logging.info('Generating sentence embeddings for test subset')
        for ii in range(0, self.n_test_samples, params.batch_size):
            batch = test_sorted_samples[ii:ii + params.batch_size]
            
            embeddings = batcher(params, batch)
            test_enc_input.append(embeddings)
        test_enc_input = np.vstack(test_enc_input)
        logging.info('Generated sentence embeddings')

        config = {'nclasses': 2, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'classifier': params.classifier,
                  'nhid': params.nhid, 'kfold': params.kfold}
        
        # define classifier for the resulting embeddings
        clf = EvalTrainClassifier(self.do_eval, train_enc_input, np.array(train_sorted_labels), test_enc_input, np.array(test_sorted_labels), config)
        # predict based on prediction subset
        scores = clf.run()
        
        return scores
        
        #return {'f1': scores}
    
    def evaltrain_run(self, params, batcher):
        
            
        # inference only run in self.model.eval() mode
        train_enc_input = []
        test_enc_input = []
        # Sort to reduce padding
        train_sorted_corpus = sorted(zip(self.train_samples, self.train_labels),
                               key=lambda z: (len(z[0]), z[1]))
        test_sorted_corpus = sorted(zip(self.test_samples, self.test_labels),
                               key=lambda z: (len(z[0]), z[1]))
        train_sorted_samples = [x for (x, y) in train_sorted_corpus]
        train_sorted_labels = [y for (x, y) in train_sorted_corpus]
        
        test_sorted_samples = [x for (x, y) in test_sorted_corpus]
        test_sorted_labels = [y for (x, y) in test_sorted_corpus]
        
        logging.info('Generating sentence embeddings for training subset')
        for ii in range(0, self.n_train_samples, params.batch_size):
            batch = train_sorted_samples[ii:ii + params.batch_size]
            
            embeddings = batcher(params, batch)
            train_enc_input.append(embeddings)
        train_enc_input = np.vstack(train_enc_input)
        logging.info('Generated sentence embeddings')
        
        logging.info('Generating sentence embeddings for test subset')
        for ii in range(0, self.n_test_samples, params.batch_size):
            batch = test_sorted_samples[ii:ii + params.batch_size]
            
            embeddings = batcher(params, batch)
            test_enc_input.append(embeddings)
        test_enc_input = np.vstack(test_enc_input)
        logging.info('Generated sentence embeddings')

        config = {'nclasses': 2, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'classifier': params.classifier,
                  'nhid': params.nhid, 'kfold': params.kfold}
        
        # define classifier for the resulting embeddings
        clf = EvalTrainClassifier(train_enc_input, np.array(train_sorted_labels), test_enc_input, np.array(test_sorted_labels), config)
        # predict based on prediction subset
        scores = clf.run()
        
        return scores


class PairsFakeCLSTrain(BinaryClassifierEval):
    def __init__(self, task_path, attribute, do_eval, seed=1111):
        logging.debug('***** FakeClassification Evaluation *****\n\n')
        #train = load_dataset(task_path, split='train', trust_remote_code=True, token="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")
        #test = load_dataset(task_path, split='test', trust_remote_code=True, token="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")
        train_samples, train_labels = self.loadFakeNews(task_path, "train", attribute) 
        test_samples, test_label = self.loadFakeNews(task_path, "test", attribute) 
        super(self.__class__, self).__init__(train_samples, train_labels, test_samples, test_label, do_eval, seed)
