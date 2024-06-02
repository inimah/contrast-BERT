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
import pandas as pd
import logging
import random
import torch

from senteval.tools.validation import EnsembleClassifier

import transformers
from datasets import load_dataset


class BinaryClassifierEval(object):
    '''
    def __init__(self, text, label, seed=1111):
        self.seed = seed
        self.samples, self.labels = text, label
        self.n_samples = len(self.samples)
    '''
    
    def __init__(self, train_text, train_label, test_text, test_label, seed=1111):
        self.seed = seed
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
        
        
    def loadFakeNews2(self, dataset, attribute, split):
        
        print("attribute: ", attribute)
        sys.stdout.flush()
        
        df = dataset.to_pandas()
        print("df: ", df)
        sys.stdout.flush()
        
        # are we using Title, Content, or Fact?
        if attribute == "title-fact":
            #elif attribute == 'fact':
            # divide data into 10 sub-batch ensemble , each batch contains 500 HOAX - 500 NON-HOAX
            if split == "train":
                new_contents = {}
                new_labels = {}
                klinikjatim = df[df['Datasource'] == 'Klinikhoax Jatim']
                mafindo = df[df['Datasource'] == 'Mafindo']
                opendata = df[df['Datasource'] == 'Opendata Jabar']
                saberhoax = df[df['Datasource'] == 'Saberhoax Jabar']
                hoax_klinikjatim = klinikjatim[klinikjatim['Label_id']==1]
                nonhoax_klinikjatim = klinikjatim[klinikjatim['Label_id']==0]
                hoax_Mafindo = mafindo[mafindo['Label_id']==1]
                nonhoax_Mafindo = mafindo[mafindo['Label_id']==0]
                hoax_opendata = opendata[opendata['Label_id']==1]
                nonhoax_opendata = opendata[opendata['Label_id']==0]
                hoax_saberhoax = saberhoax[saberhoax['Label_id']==1]
                nonhoax_saberhoax = saberhoax[saberhoax['Label_id']==0]
                # klinikjatim HOAX: 0 - klinikjatim NONHOAX: 492 # sampling 203 to be in total 500
                # Mafindo HOAX: 6898 - Mafindo NONHOAX: 85
                # opendata HOAX: 299 - opendata NONHOAX: 55
                # saberhoax HOAX: 4366 - saberhoax NONHOAX: 157
                for j in range(10):
                    new_contents[j] = []
                    new_labels[j] = []
                    perm_klinik = np.random.permutation(len(nonhoax_klinikjatim))
                    # sample 203 from 492
                    rand_ids_klinik = np.random.choice(len(perm_klinik), 203, replace=False)
                    sampled_klinik = nonhoax_klinikjatim.iloc[perm_klinik[rand_ids_klinik]]                                               
                    # for hoax class: sample mafindo 300, saber 150, opendata 50
                    perm_mafindo = np.random.permutation(len(hoax_Mafindo))
                    rand_ids_mafindo = np.random.choice(len(perm_mafindo), 300, replace=False)
                    sampled_hoax_Mafindo = hoax_Mafindo.iloc[perm_mafindo[rand_ids_mafindo]]
                    # opendata                                      
                    perm_opendata = np.random.permutation(len(hoax_opendata))
                    rand_ids_opendata = np.random.choice(len(perm_opendata), 50, replace=False)
                    sampled_hoax_opendata = hoax_opendata.iloc[perm_opendata[rand_ids_opendata]]
                    # saberhoax
                    perm_saber = np.random.permutation(len(hoax_saberhoax))
                    rand_ids_saber = np.random.choice(len(perm_saber), 150, replace=False)
                    sampled_hoax_saberhoax = hoax_saberhoax.iloc[perm_saber[rand_ids_saber]]                                               
                    non_hoax_df = pd.concat([sampled_klinik, nonhoax_Mafindo, nonhoax_opendata, nonhoax_saberhoax ])
                    hoax_df = pd.concat([sampled_hoax_Mafindo, sampled_hoax_opendata, sampled_hoax_saberhoax])
                    all_df = pd.concat([non_hoax_df, hoax_df])
                    for ttl, fct, lbl in zip(all_df.Title.values, all_df.Fact.values, all_df.Label_id.values):
                        text = "Klaim: " + str(ttl) + ".\nFakta: " + str(fct)
                        new_contents[j].append(text)
                        new_labels[j].append(lbl)
            else:
                new_contents = []
                new_labels = []
                for ttl, fct, lbl in zip(df.Title.values, df.Fact.values, df.Label_id.values):
                        text = "Klaim: " +  str(ttl) + ".\nFakta: " + str(fct)
                        new_contents.append(text)
                        new_labels.append(lbl)                                                     
            return (new_contents, new_labels)
            
            
    
    def evaltrain_run(self, params, batcher):
        
            
        # inference only run in self.model.eval() mode
        train_enc_input = {}
        train_sorted_samples = {}
        train_sorted_labels = {}
        # Sort to reduce padding
        # transform into embedding
        for m in range(10):
            train_enc_input[m] = []
            train_sorted_samples[m] = []
            train_sorted_labels[m] = []
        
            trainlabel_smp = self.train_labels[m]
            train_smp = self.train_samples[m]
            
            train_sorted_corpus = sorted(zip(train_smp, trainlabel_smp),
                                   key=lambda z: (len(z[0]), z[1]))
            
            train_sorted_samples = [x for (x, y) in train_sorted_corpus]
            train_sorted_labels[m] = [y for (x, y) in train_sorted_corpus]

            logging.info('Generating sentence embeddings for training subset')
            for ii in range(0, len(train_smp), params.batch_size):
                batch = train_sorted_samples[ii:ii + params.batch_size]
            
                embeddings = batcher(params, batch)
                train_enc_input[m].append(embeddings)
            
            train_enc_input[m] = np.vstack(train_enc_input[m])
            logging.info('Generated sentence embeddings')
        
        test_sorted_corpus = sorted(zip(self.test_samples, self.test_labels),
                                   key=lambda z: (len(z[0]), z[1]))
        test_sorted_samples = [x for (x, y) in test_sorted_corpus]
        test_sorted_labels = [y for (x, y) in test_sorted_corpus]
            
        test_enc_input = []
        
        logging.info('Generating sentence embeddings for test subset')
        for ii in range(0, len(self.test_samples), params.batch_size):
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
        clf = EnsembleClassifier(train_enc_input, train_sorted_labels, test_enc_input, test_sorted_labels, config)
        # predict based on prediction subset
        scores = clf.run()
        
        return scores


class EnsembleClassifierTrain(BinaryClassifierEval):
    def __init__(self, task_path, attribute, seed=1111):
        
        print("task_path: ", task_path)
        sys.stdout.flush()
        
        def set_all_seeds(seed=1111):
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
        
        set_all_seeds(seed=1111)
        
        logging.debug('***** FakeClassification Evaluation with trained MLP *****\n\n')
        train = load_dataset(task_path, split='train', trust_remote_code=True, token="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")
        test = load_dataset(task_path, split='test', trust_remote_code=True, token="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")
        train_samples, train_labels = self.loadFakeNews2(train, attribute, split="train") 
        test_samples, test_label = self.loadFakeNews2(test, attribute, split="test") 
        super(self.__class__, self).__init__(train_samples, train_labels, test_samples, test_label, seed)
