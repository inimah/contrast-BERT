# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Validation and classification
(train)            :  inner-kfold classifier
(train, test)      :  kfold classifier
(train, dev, test) :  split classifier

"""
from __future__ import absolute_import, division, unicode_literals

import os
import sys

import logging
import numpy as np
from senteval.tools.classifier import MLP

import torch
import transformers
import evaluate
from evaluate import evaluator

import sklearn
assert(sklearn.__version__ >= "0.18.0"), \
    "need to update sklearn to version >= 0.18.0"
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import brier_score_loss

from sklearn.calibration import calibration_curve
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)


def get_classif_name(classifier_config, usepytorch):
    if not usepytorch:
        modelname = 'sklearn-LogReg'
    else:
        nhid = classifier_config['nhid']
        optim = 'adam' if 'optim' not in classifier_config else classifier_config['optim']
        bs = 64 if 'batch_size' not in classifier_config else classifier_config['batch_size']
        modelname = 'pytorch-MLP-nhid%s-%s-bs%s' % (nhid, optim, bs)
    return modelname

# Pytorch version
class InnerKFoldClassifier(object):
    """
    (train) split classifier : InnerKfold.
    """
    def __init__(self, X, y, config):
        self.X = X
        self.y = y
        self.featdim = X.shape[1]
        self.nclasses = config['nclasses']
        self.seed = config['seed']
        self.devresults = []
        self.testresults = []
        self.usepytorch = config['usepytorch']
        self.classifier_config = config['classifier']
        self.modelname = get_classif_name(self.classifier_config, self.usepytorch)

        self.k = 5 if 'kfold' not in config else config['kfold']

    def run(self):
        logging.info('Training {0} with (inner) {1}-fold cross-validation'
                     .format(self.modelname, self.k))

        regs = [10**t for t in range(-5, -1)] if self.usepytorch else \
               [2**t for t in range(-2, 4, 1)]
        skf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=1111)
        innerskf = StratifiedKFold(n_splits=self.k, shuffle=True,
                                   random_state=1111)
        count = 0
        for train_idx, test_idx in skf.split(self.X, self.y):
            count += 1
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            scores = []
            for reg in regs:
                regscores = []
                for inner_train_idx, inner_test_idx in innerskf.split(X_train, y_train):
                    X_in_train, X_in_test = X_train[inner_train_idx], X_train[inner_test_idx]
                    y_in_train, y_in_test = y_train[inner_train_idx], y_train[inner_test_idx]
                    if self.usepytorch:
                        clf = MLP(self.classifier_config, inputdim=self.featdim,
                                  nclasses=self.nclasses, l2reg=reg,
                                  seed=self.seed)
                        clf.fit(X_in_train, y_in_train,
                                validation_data=(X_in_test, y_in_test))
                    else:
                        clf = LogisticRegression(C=reg, random_state=self.seed)
                        clf.fit(X_in_train, y_in_train)
                    regscores.append(clf.score(X_in_test, y_in_test))
                scores.append(round(100*np.mean(regscores), 2))
            optreg = regs[np.argmax(scores)]
            logging.info('Best param found at split {0}: l2reg = {1} \
                with score {2}'.format(count, optreg, np.max(scores)))
            self.devresults.append(np.max(scores))

            if self.usepytorch:
                clf = MLP(self.classifier_config, inputdim=self.featdim,
                          nclasses=self.nclasses, l2reg=optreg,
                          seed=self.seed)

                clf.fit(X_train, y_train, validation_split=0.05)
            else:
                clf = LogisticRegression(C=optreg, random_state=self.seed)
                clf.fit(X_train, y_train)

            self.testresults.append(round(100*clf.score(X_test, y_test), 2))

        devaccuracy = round(np.mean(self.devresults), 2)
        testaccuracy = round(np.mean(self.testresults), 2)
        return devaccuracy, testaccuracy


class KFoldClassifier(object):
    """
    (train, test) split classifier : cross-validation on train.
    """
    def __init__(self, train, test, config):
        self.train = train
        self.test = test
        self.featdim = self.train['X'].shape[1]
        self.nclasses = config['nclasses']
        self.seed = config['seed']
        self.usepytorch = config['usepytorch']
        self.classifier_config = config['classifier']
        self.modelname = get_classif_name(self.classifier_config, self.usepytorch)

        self.k = 5 if 'kfold' not in config else config['kfold']

    def run(self):
        # cross-validation
        logging.info('Training {0} with {1}-fold cross-validation'
                     .format(self.modelname, self.k))
        regs = [10**t for t in range(-5, -1)] if self.usepytorch else \
               [2**t for t in range(-1, 6, 1)]
        skf = StratifiedKFold(n_splits=self.k, shuffle=True,
                              random_state=self.seed)
        scores = []

        for reg in regs:
            scanscores = []
            for train_idx, test_idx in skf.split(self.train['X'],
                                                 self.train['y']):
                # Split data
                X_train, y_train = self.train['X'][train_idx], self.train['y'][train_idx]

                X_test, y_test = self.train['X'][test_idx], self.train['y'][test_idx]

                # Train classifier
                if self.usepytorch:
                    clf = MLP(self.classifier_config, inputdim=self.featdim,
                              nclasses=self.nclasses, l2reg=reg,
                              seed=self.seed)
                    clf.fit(X_train, y_train, validation_data=(X_test, y_test))
                else:
                    clf = LogisticRegression(C=reg, random_state=self.seed)
                    clf.fit(X_train, y_train)
                score = clf.score(X_test, y_test)
                scanscores.append(score)
            # Append mean score
            scores.append(round(100*np.mean(scanscores), 2))

        # evaluation
        logging.info([('reg:' + str(regs[idx]), scores[idx])
                      for idx in range(len(scores))])
        optreg = regs[np.argmax(scores)]
        devaccuracy = np.max(scores)
        logging.info('Cross-validation : best param found is reg = {0} \
            with score {1}'.format(optreg, devaccuracy))

        logging.info('Evaluating...')
        if self.usepytorch:
            clf = MLP(self.classifier_config, inputdim=self.featdim,
                      nclasses=self.nclasses, l2reg=optreg,
                      seed=self.seed)
            clf.fit(self.train['X'], self.train['y'], validation_split=0.05)
        else:
            clf = LogisticRegression(C=optreg, random_state=self.seed)
            clf.fit(self.train['X'], self.train['y'])
        yhat = clf.predict(self.test['X'])

        testaccuracy = clf.score(self.test['X'], self.test['y'])
        testaccuracy = round(100*testaccuracy, 2)

        return devaccuracy, testaccuracy, yhat


class SplitClassifier(object):
    """
    (train, valid, test) split classifier.
    """
    def __init__(self, X, y, config):
        self.X = X
        self.y = y
        
        
            
        self.nclasses = config['nclasses']
        self.featdim = self.X['train'].shape[1]
        self.seed = config['seed']
        self.usepytorch = config['usepytorch']
        self.classifier_config = config['classifier']
        self.cudaEfficient = False if 'cudaEfficient' not in config else \
            config['cudaEfficient']
        self.modelname = get_classif_name(self.classifier_config, self.usepytorch)
        self.noreg = False if 'noreg' not in config else config['noreg']
        self.config = config

    def run(self):
        logging.info('Training {0} with standard validation..'
                     .format(self.modelname))
        regs = [10**t for t in range(-5, -1)] if self.usepytorch else \
               [2**t for t in range(-2, 4, 1)]
        if self.noreg:
            regs = [1e-9 if self.usepytorch else 1e9]
        scores = []
        for reg in regs:
            if self.usepytorch:
                clf = MLP(self.classifier_config, inputdim=self.featdim,
                          nclasses=self.nclasses, l2reg=reg,
                          seed=self.seed, cudaEfficient=self.cudaEfficient)

                # TODO: Find a hack for reducing nb epoches in SNLI
                clf.fit(self.X['train'], self.y['train'],
                        validation_data=(self.X['valid'], self.y['valid']))
            else:
                clf = LogisticRegression(C=reg, random_state=self.seed)
                clf.fit(self.X['train'], self.y['train'])
            scores.append(round(100*clf.score(self.X['valid'],
                                self.y['valid']), 2))
        logging.info([('reg:'+str(regs[idx]), scores[idx])
                      for idx in range(len(scores))])
        optreg = regs[np.argmax(scores)]
        devaccuracy = np.max(scores)
        logging.info('Validation : best param found is reg = {0} with score \
            {1}'.format(optreg, devaccuracy))
        clf = LogisticRegression(C=optreg, random_state=self.seed)
        logging.info('Evaluating...')
        if self.usepytorch:
            clf = MLP(self.classifier_config, inputdim=self.featdim,
                      nclasses=self.nclasses, l2reg=optreg,
                      seed=self.seed, cudaEfficient=self.cudaEfficient)

            # TODO: Find a hack for reducing nb epoches in SNLI
            clf.fit(self.X['train'], self.y['train'],
                    validation_data=(self.X['valid'], self.y['valid']))
        else:
            clf = LogisticRegression(C=optreg, random_state=self.seed)
            clf.fit(self.X['train'], self.y['train'])

        testaccuracy = clf.score(self.X['test'], self.y['test'])
        testaccuracy = round(100*testaccuracy, 2)
        return devaccuracy, testaccuracy
    

class EvalClassifier(object):
    """
    (train, valid, test) split classifier.
    """
    def __init__(self, do_eval, X, y, config):
        self.X = X
        self.y = y
        self.do_eval = do_eval
        print("shape self.X:", self.X.shape)
        sys.stdout.flush()
        print("shape self.y:", self.y.shape)
        sys.stdout.flush()
        
        self.batch_size = 128
        
        self.nclasses = config['nclasses']
        self.featdim = self.X.shape[1]
        self.seed = config['seed']
        self.usepytorch = config['usepytorch']
        self.classifier_config = config['classifier']
        self.cudaEfficient = False if 'cudaEfficient' not in config else \
            config['cudaEfficient']
        self.modelname = get_classif_name(self.classifier_config, self.usepytorch)
        self.noreg = False if 'noreg' not in config else config['noreg']
        self.config = config
        # initial MLP randomly (without training them)
        self.clf = MLP(self.classifier_config, inputdim=self.featdim,
                          nclasses=2,
                          seed=self.seed, cudaEfficient=self.cudaEfficient)
    
    # for validation task (do_eval == True)
    def custom_score_valid(self, devX, devy):
        #metric_names = ["accuracy", "f1", "precision", "recall", "confusion_matrix"]
        
        # use precision as validation metric
        metric = evaluate.load("precision")
        
            
        if not isinstance(devX, torch.cuda.FloatTensor) or self.cudaEfficient:
            devX = torch.FloatTensor(devX).cuda()
            devy = torch.LongTensor(devy).cuda()
        with torch.no_grad():
            
            all_pred = np.array([])
            all_ytrue = np.array([])
            for i in range(0, len(devX), self.batch_size):
                Xbatch = devX[i:i + self.batch_size]
                ybatch = devy[i:i + self.batch_size]
                if self.cudaEfficient:
                    Xbatch = Xbatch.cuda()
                    ybatch = ybatch.cuda()
                pred = self.clf.predict(Xbatch)
                #pred = output.data.max(1)[1]
                all_pred = np.append(all_pred,
                                 pred)
                all_ytrue = np.append(all_ytrue,
                                 ybatch.cpu().numpy())
            all_pred = np.vstack(all_pred)
            all_ytrue = np.vstack(all_ytrue)
                
            scores = metric.compute(predictions=all_pred, references=all_ytrue)
        
            print(scores)
            sys.stdout.flush()
        
        return scores
    # for inference testing 
    def custom_score_test(self, devX, devy):
        metric_names = ["accuracy", "f1", "precision", "recall", "confusion_matrix"]
        
        all_metric = []
        for nm in metric_names:
            all_metric.append(evaluate.load(str(nm)))
            
        if not isinstance(devX, torch.cuda.FloatTensor) or self.cudaEfficient:
            devX = torch.FloatTensor(devX).cuda()
            devy = torch.LongTensor(devy).cuda()
        with torch.no_grad():
            scores = {}
            all_pred = np.array([])
            all_ytrue = np.array([])
            for i in range(0, len(devX), self.batch_size):
                Xbatch = devX[i:i + self.batch_size]
                ybatch = devy[i:i + self.batch_size]
                if self.cudaEfficient:
                    Xbatch = Xbatch.cuda()
                    ybatch = ybatch.cuda()
                pred = self.clf.predict(Xbatch)
                #pred = output.data.max(1)[1]
                all_pred = np.append(all_pred,
                                 pred)
                all_ytrue = np.append(all_ytrue,
                                 ybatch.cpu().numpy())
            all_pred = np.vstack(all_pred)
            all_ytrue = np.vstack(all_ytrue)
                
            for nm, met in zip(metric_names, all_metric):
                scores[str(nm)] = met.compute(predictions=all_pred, references=all_ytrue)
        
            print(scores)
            sys.stdout.flush()
        
        return scores
    
    def run(self):
        logging.info('Training {0} with standard validation..'
                     .format(self.modelname))
        
        if self.do_eval:
            scores = self.clf.custom_score_valid(self.X, self.y)
        else:
            scores = self.clf.custom_score_test(self.X, self.y)
        return scores
    
class EvalTrainClassifier(object):
    """
    (train, valid, test) split classifier.
    """
    def __init__(self, X_train, y_train, X_test, y_test, config):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.do_eval = False # for inference
        #self.do_eval = True # for training stage
        
        print("shape self.X_train:", self.X_train.shape)
        sys.stdout.flush()
        print("shape self.y_train:", self.y_train.shape)
        sys.stdout.flush()
        
        print("shape self.X_test:", self.X_test.shape)
        sys.stdout.flush()
        print("shape self.y_test:", self.y_test.shape)
        sys.stdout.flush()
        
        #self.batch_size = 128
        self.batch_size = 64
        
        self.nclasses = 2
        self.featdim = self.X_train.shape[1]
        self.seed = config['seed']
        self.usepytorch = config['usepytorch']
        self.classifier_config = config['classifier']
        self.cudaEfficient = False if 'cudaEfficient' not in config else \
            config['cudaEfficient']
        self.modelname = get_classif_name(self.classifier_config, self.usepytorch)
        self.noreg = False if 'noreg' not in config else config['noreg']
        self.config = config
        self.clf = MLP(self.classifier_config, inputdim=self.featdim,
                          nclasses=2,
                          seed=self.seed, cudaEfficient=self.cudaEfficient)
        self.k = 5 if 'kfold' not in config else config['kfold']
    
    
    
    def run(self):
        
        # inference function
        def custom_score_test(devX, devy, model):
            metric_names = ["accuracy", "f1", "precision", "recall", "confusion_matrix"]

            all_metric = []
            for nm in metric_names:
                all_metric.append(evaluate.load(str(nm)))

            if not isinstance(devX, torch.cuda.FloatTensor) or self.cudaEfficient:
                devX = torch.FloatTensor(devX).cuda()
                devy = torch.LongTensor(devy).cuda()
            with torch.no_grad():
                scores = {}
                all_pred = np.array([])
                all_ytrue = np.array([])
                all_pred_probs = np.array([])
                for i in range(0, len(devX), self.batch_size):
                    Xbatch = devX[i:i + self.batch_size]
                    ybatch = devy[i:i + self.batch_size]
                    if self.cudaEfficient:
                        Xbatch = Xbatch.cuda()
                        ybatch = ybatch.cuda()
                    pred = model.predict(Xbatch)
                    #pred_probs = model.predict_proba(Xbatch).data.cpu().numpy()
                    pred_probs = model.predict_proba(Xbatch)
                    #pred = output.data.max(1)[1]
                    all_pred = np.append(all_pred,
                                     pred)
                    all_ytrue = np.append(all_ytrue,
                                     ybatch.cpu().numpy())
                    all_pred_probs = np.append(all_pred_probs, pred_probs)

                all_pred = np.vstack(all_pred)
                all_ytrue = np.vstack(all_ytrue)
                all_pred_probs = np.array(all_pred_probs)
                all_pred_probs = all_pred_probs.reshape((-1,2))

                for nm, met in zip(metric_names, all_metric):
                    tmp = met.compute(predictions=all_pred, references=all_ytrue)[str(nm)]
                    if str(nm) in ["accuracy", "f1", "precision", "recall"]:
                        scores[str(nm)] = [tmp]
                    else:
                        scores[str(nm)]= tmp.tolist()
                    
                # add sklearn classification report to the dictionary of the evaluation scores
                #target_names = ['NON-HOAX', 'HOAX']
                scores['classification_report'] = {"all_ytrue": all_ytrue.tolist(), "all_pred": all_pred.tolist(),  "all_pred_probs": all_pred_probs.tolist() }
                
                # add brier score based on confidence level 
                brier = evaluate.load("brier_score")
                # predictions here = proba true class as predicted
                # references = the expectation of true probability
                # binning the prediction
                prob_true, prob_pred = calibration_curve(all_ytrue, all_pred_probs[:, 1], n_bins=10)
                #scores["brier_score"] = brier_score_loss(all_ytrue, prob_pred)
                scores["calibration_curve"] = {"prob_true": prob_true.tolist(), "prob_pred": prob_pred.tolist()}

                print(scores)
                sys.stdout.flush()

            return scores
        
        # for validation task (do_eval == True)
        def custom_score_valid(devX, devy, model):
            #metric_names = ["accuracy", "f1", "precision", "recall", "confusion_matrix"]

            # use precision as validation metric
            metric = evaluate.load("f1")


            if not isinstance(devX, torch.cuda.FloatTensor) or self.cudaEfficient:
                devX = torch.FloatTensor(devX).cuda()
                devy = torch.LongTensor(devy).cuda()
            with torch.no_grad():

                all_pred = np.array([])
                all_ytrue = np.array([])
                for i in range(0, len(devX), self.batch_size):
                    Xbatch = devX[i:i + self.batch_size]
                    ybatch = devy[i:i + self.batch_size]
                    if self.cudaEfficient:
                        Xbatch = Xbatch.cuda()
                        ybatch = ybatch.cuda()
                    pred = model.predict(Xbatch)
                    #pred = output.data.max(1)[1]
                    all_pred = np.append(all_pred,
                                     pred)
                    all_ytrue = np.append(all_ytrue,
                                     ybatch.cpu().numpy())
                all_pred = np.vstack(all_pred)
                all_ytrue = np.vstack(all_ytrue)

                scores = metric.compute(predictions=all_pred, references=all_ytrue)

                print(scores)
                sys.stdout.flush()

            return scores
        
        logging.info('Training {0} MLP with standard validation given sentence embedding from BERT..'
                     .format(self.modelname))
        
        regs = [10**t for t in range(-5, -1)] if self.usepytorch else \
               [2**t for t in range(-2, 4, 1)]
        skf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=1111)
        innerskf = StratifiedKFold(n_splits=self.k, shuffle=True,
                                   random_state=1111)
        count = 0
        for train_idx, dev_idx in skf.split(self.X_train, self.y_train):
            count += 1
            X_train, X_dev = self.X_train[train_idx], self.X_train[dev_idx]
            y_train, y_dev = self.y_train[train_idx], self.y_train[dev_idx]

            self.clf.fit(X_train, y_train, validation_split=0.05)
            
        if self.do_eval:
            scores = custom_score_valid(self.X_test, self.y_test, self.clf)
        else:
            # get calibration plot from model prediction
            #plot_calibration(self.X_test, self.y_test, self.clf)
            scores = custom_score_test(self.X_test, self.y_test, self.clf)
            
        return scores

    

class EnsembleClassifier(object):
    """
    (train, valid, test) split classifier.
    """
    def __init__(self, X_train, y_train, X_test, y_test, config):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.do_eval = False
        
        
        self.batch_size = 128
        
        self.nclasses = 2
        self.featdim = self.X_train[0].shape[1]
        self.seed = config['seed']
        self.usepytorch = config['usepytorch']
        self.classifier_config = config['classifier']
        self.cudaEfficient = False if 'cudaEfficient' not in config else \
            config['cudaEfficient']
        self.modelname = get_classif_name(self.classifier_config, self.usepytorch)
        self.noreg = False if 'noreg' not in config else config['noreg']
        self.config = config
        self.clf = {}
        self.clf[0] = MLP(self.classifier_config, inputdim=self.featdim,
                          nclasses=2,
                          seed=self.seed, cudaEfficient=self.cudaEfficient)
        self.clf[1] = MLP(self.classifier_config, inputdim=self.featdim,
                          nclasses=2,
                          seed=self.seed, cudaEfficient=self.cudaEfficient)
        self.clf[2] = MLP(self.classifier_config, inputdim=self.featdim,
                          nclasses=2,
                          seed=self.seed, cudaEfficient=self.cudaEfficient)
        self.clf[3] = MLP(self.classifier_config, inputdim=self.featdim,
                          nclasses=2,
                          seed=self.seed, cudaEfficient=self.cudaEfficient)
        self.clf[4] = MLP(self.classifier_config, inputdim=self.featdim,
                          nclasses=2,
                          seed=self.seed, cudaEfficient=self.cudaEfficient)
        self.clf[5] = MLP(self.classifier_config, inputdim=self.featdim,
                          nclasses=2,
                          seed=self.seed, cudaEfficient=self.cudaEfficient)
        self.clf[6] = MLP(self.classifier_config, inputdim=self.featdim,
                          nclasses=2,
                          seed=self.seed, cudaEfficient=self.cudaEfficient)
        self.clf[7] = MLP(self.classifier_config, inputdim=self.featdim,
                          nclasses=2,
                          seed=self.seed, cudaEfficient=self.cudaEfficient)
        self.clf[8] = MLP(self.classifier_config, inputdim=self.featdim,
                          nclasses=2,
                          seed=self.seed, cudaEfficient=self.cudaEfficient)
        self.clf[9] = MLP(self.classifier_config, inputdim=self.featdim,
                          nclasses=2,
                          seed=self.seed, cudaEfficient=self.cudaEfficient)
        self.k = 5 if 'kfold' not in config else config['kfold']
        
        # results to save: scores for all metrics, prediction and true probability for plotting confidence calibration plot 
        
    
    
    
    def run(self):
        
        # calibration plt function
        def plot_calibration(devX, devy, model):
            
            print("shape devX:", devX.shape)
            sys.stdout.flush()
            
            print("shape devy: ", devy.shape)
            sys.stdout.flush()
            
            if not isinstance(devX, torch.cuda.FloatTensor) or self.cudaEfficient:
                devX = torch.FloatTensor(devX).cuda()
                devy = torch.LongTensor(devy).cuda()
            
            with torch.no_grad():
                all_pred = np.array([])
                all_ytrue = np.array([])
                for i in range(0, len(devX), self.batch_size):
                    Xbatch = devX[i:i + self.batch_size]
                    ybatch = devy[i:i + self.batch_size]
                    if self.cudaEfficient:
                        Xbatch = Xbatch.cuda()
                        ybatch = ybatch.cuda()
                    
                    probs_all = []
                    for m in range(10):
                        probs = model[m].predict_proba(Xbatch).data.cpu().numpy()
                        probs = probs.reshape((-1, 2))
                        probs_all.append(probs)
                    
                    probs_all = np.array(probs_all)
                    print("probs_all.shape: ", probs_all.shape)
                    sys.stdout.flush()
                        
                    pred = np.mean(probs_all, axis=0)
                    all_pred = np.append(all_pred, pred)
                    all_ytrue = np.append(all_ytrue, ybatch.cpu().numpy())

                all_pred = np.vstack(all_pred)
                all_ytrue = np.vstack(all_ytrue)
                        
            all_pred = all_pred.reshape((-1, 2))
                
            print("shape all_pred:", all_pred.shape)
            sys.stdout.flush()
            
            print("shape all_ytrue: ", all_ytrue.shape)
            sys.stdout.flush()
            
            print("all_pred[:, 1]:", all_pred[:, 1])
            sys.stdout.flush()
            
            print("all_ytrue:", all_ytrue)
            sys.stdout.flush()
            
            # binning the prediction
            prob_true, prob_pred = calibration_curve(all_ytrue, all_pred[:, 1], n_bins=10)
            
            print("prob_true:", prob_true)
            sys.stdout.flush()
            
            print("prob_pred:", prob_pred)
            sys.stdout.flush()
            
            # plot calibration based on model prediction
            plt.plot(prob_pred,
                     prob_true, 
                     marker='o', 
                     linewidth=1, 
                     #label='indobert-base-p2')
                     #label='indobert-base-p2-ensemble')
                     #label='simcse-indobert-title-FakeCLSTrain')
                     #label='simcse-indobert-title-ensemble')
                     #label='simcse-indobert-content-ensemble')
                     #label='simcse-indobert-content-FakeCLSTrain')
                     #label='simcse-indobert-fact-ensemble')
                     #label='simcse-indobert-fact-FakeCLSTrain')
                     #label='simcse-indobert-triplets-ensemble')
                     label='LazarusNLP-simcse-indobert-ensemble')

            #Plot the Perfectly Calibrated by Adding the 45-degree line to the plot
            plt.plot([0, 1], 
                     [0, 1], 
                     linestyle='--', 
                     label='Perfectly Calibrated')


            # Set the title and axis labels for the plot
            plt.title('Probability Calibration Curve')
            plt.xlabel('Predicted Probability')
            plt.ylabel('True Probability')

            # Add a legend to the plot
            plt.legend(loc='best')

            # Show the plot
            print("saving plots")
            sys.stdout.flush()
            #plt.savefig("./plots/mlp_nocalib.png")
            #plt.savefig("./plots/indobert-base-p2-ensemble.png")
            #plt.savefig("./plots/simcse-indobert-title-FakeCLSTrain.png")
            #plt.savefig("./plots/simcse-indobert-title-ensemble.png")
            #plt.savefig("./plots/simcse-indobert-content-ensemble.png")
            #plt.savefig("./plots/simcse-indobert-content-FakeCLSTrain.png")
            #plt.savefig("./plots/simcse-indobert-fact-ensemble.png")
            #plt.savefig("./plots/simcse-indobert-fact-FakeCLSTrain.png")
            #plt.savefig("./plots/simcse-indobert-triplets-ensemble.png")
            plt.savefig("./plots/LazarusNLP-simcse-indobert-ensemble.png")

        
        # inference function
        def custom_score_test(devX, devy, model):
            metric_names = ["accuracy", "f1", "precision", "recall", "confusion_matrix"]

            all_metric = []
            for nm in metric_names:
                all_metric.append(evaluate.load(str(nm)))

            if not isinstance(devX, torch.cuda.FloatTensor) or self.cudaEfficient:
                devX = torch.FloatTensor(devX).cuda()
                devy = torch.LongTensor(devy).cuda()
            with torch.no_grad():
                scores = {}
                all_pred = np.array([])
                all_ytrue = np.array([])
                all_pred_probs = np.array([])
                for i in range(0, len(devX), self.batch_size):
                    Xbatch = devX[i:i + self.batch_size]
                    ybatch = devy[i:i + self.batch_size]
                    if self.cudaEfficient:
                        Xbatch = Xbatch.cuda()
                        ybatch = ybatch.cuda()
                        
                    #print("Xbatch:", Xbatch)
                    #sys.stdout.flush()
    
                    pred = model.predict(Xbatch)
                    #pred_probs = model.predict_proba(Xbatch).data.cpu().numpy()
                    pred_probs = model.predict_proba(Xbatch)
                    #pred = output.data.max(1)[1]
                    all_pred = np.append(all_pred,
                                     pred)
                    all_ytrue = np.append(all_ytrue,
                                     ybatch.cpu().numpy())
                    all_pred_probs = np.append(all_pred_probs, pred_probs)

                all_pred = np.vstack(all_pred)
                all_ytrue = np.vstack(all_ytrue)
                all_pred_probs= np.array(all_pred_probs)
                all_pred_probs = all_pred_probs.reshape((-1,2))
                
                print("all_pred_probs.shape: ", all_pred_probs.shape)
                sys.stdout.flush()
                print("all_ytrue.shape: ", all_ytrue.shape)
                sys.stdout.flush()

                for nm, met in zip(metric_names, all_metric):
                    tmp = met.compute(predictions=all_pred, references=all_ytrue)[str(nm)]
                    if str(nm) in ["accuracy", "f1", "precision", "recall"]:
                        scores[str(nm)] = [tmp]
                    else:
                        scores[str(nm)]= tmp.tolist()
                    
                # add sklearn classification report to the dictionary of the evaluation scores
                #target_names = ['NON-HOAX', 'HOAX']
                scores['classification_report'] = {"all_ytrue": all_ytrue.tolist(), "all_pred": all_pred.tolist(),  "all_pred_probs": all_pred_probs.tolist() }
                
                # add brier score based on confidence level 
                brier = evaluate.load("brier_score")
                # predictions here = proba true class as predicted
                # references = the expectation of true probability
                # binning the prediction
                prob_true, prob_pred = calibration_curve(all_ytrue, all_pred_probs[:, 1], n_bins=10)
                #scores["brier_score"] = brier_score_loss(all_ytrue, prob_pred)
                scores["calibration_curve"] = {"prob_true": prob_true.tolist(), "prob_pred": prob_pred.tolist()}

                print(scores)
                sys.stdout.flush()

            return scores
        
        # averaging the probability of 10 ensemble models
        def avg_ensemble_score_test(devX, devy, model):
            metric_names = ["accuracy", "f1", "precision", "recall", "confusion_matrix"]

            all_metric = []
            for nm in metric_names:
                all_metric.append(evaluate.load(str(nm)))

            if not isinstance(devX, torch.cuda.FloatTensor) or self.cudaEfficient:
                devX = torch.FloatTensor(devX).cuda()
                devy = torch.LongTensor(devy).cuda()
            with torch.no_grad():
                scores = {}
                all_pred = np.array([])
                all_ytrue = np.array([])
                all_pred_probs = np.array([])
                for i in range(0, len(devX), self.batch_size):
                    Xbatch = devX[i:i + self.batch_size]
                    ybatch = devy[i:i + self.batch_size]
                    if self.cudaEfficient:
                        Xbatch = Xbatch.cuda()
                        ybatch = ybatch.cuda()
                    probs_all = []
                    for m in range(10):
                        #probs = model[m].predict_proba(Xbatch).data.cpu().numpy()
                        probs = model[m].predict_proba(Xbatch)
                        probs = np.array(probs)
                        probs = probs.reshape((-1, 2))
                        probs_all.append(probs)
                    
                    probs_all = np.array(probs_all)
                    print("probs_all.shape: ", probs_all.shape)
                    sys.stdout.flush()
                        
                    probs_avg = np.mean(probs_all, axis=0)
                    pred = np.argmax(probs_avg, axis=1)
                    
                    all_pred_probs = np.append(all_pred_probs, probs_avg)
                    
                    #pred = output.data.max(1)[1]
                    all_pred = np.append(all_pred,
                                     pred)
                    all_ytrue = np.append(all_ytrue,
                                     ybatch.cpu().numpy())

                all_pred = np.vstack(all_pred)
                all_ytrue = np.vstack(all_ytrue)
                all_pred_probs= np.array(all_pred_probs)
                all_pred_probs = all_pred_probs.reshape((-1,2))
                
                print("all_pred_probs.shape: ", all_pred_probs.shape)
                sys.stdout.flush()

                for nm, met in zip(metric_names, all_metric):
                    tmp = met.compute(predictions=all_pred, references=all_ytrue)[str(nm)]
                    if str(nm) in ["accuracy", "f1", "precision", "recall"]:
                        scores[str(nm)] = [tmp]
                    else:
                        scores[str(nm)] = tmp.tolist()
                    
                # add sklearn classification report to the dictionary of the evaluation scores
                target_names = ['NON-HOAX', 'HOAX']
                #scores['classification_report'] = classification_report(all_ytrue, all_pred, target_names=target_names)
                scores['classification_report'] = {"all_ytrue": all_ytrue.tolist(), "all_pred": all_pred.tolist(),  "all_pred_probs": all_pred_probs.tolist() }
                
                # add brier score based on confidence level 
                brier = evaluate.load("brier_score")
                # predictions here = proba true class as predicted
                # references = the expectation of true probability
                # binning the prediction
                prob_true, prob_pred = calibration_curve(all_ytrue, all_pred_probs[:, 1], n_bins=10)
                #scores["brier_score"] = brier_score_loss(all_ytrue, prob_pred)
                #scores["brier_score"] = brier.compute(predictions=prob_pred, references=all_ytrue)
                scores["calibration_curve"] = {"prob_true": prob_true.tolist(), "prob_pred": prob_pred.tolist()}

                print(scores)
                sys.stdout.flush()

            return scores
        
        # based on probability threshold and max probability
        # here from all ensemble models 1-10, use the first model as anchor and the remaining as potential new anchor
        # if p[1] > 0.7 use p[1]
        # else use max(p)
        # instance-level best model
        
        def best_ensemble_score_test(devX, devy, model):
            
            CONFIDENCE_THRESHOLD = 0.7
            
            metric_names = ["accuracy", "f1", "precision", "recall", "confusion_matrix"]

            all_metric = []
            for nm in metric_names:
                all_metric.append(evaluate.load(str(nm)))

            if not isinstance(devX, torch.cuda.FloatTensor) or self.cudaEfficient:
                devX = torch.FloatTensor(devX).cuda()
                devy = torch.LongTensor(devy).cuda()
            with torch.no_grad():
                scores = {}
                all_pred = np.array([])
                all_ytrue = np.array([])
                all_pred_probs = np.array([])
                for i in range(0, len(devX), self.batch_size):
                    Xbatch = devX[i:i + self.batch_size]
                    ybatch = devy[i:i + self.batch_size]
                    if self.cudaEfficient:
                        Xbatch = Xbatch.cuda()
                        ybatch = ybatch.cuda()
                    probs_all = []
                    for m in range(10):
                        probs = model[m].predict_proba(Xbatch).data.cpu().numpy()
                        probs = probs.reshape((-1, 2))
                        probs_all.append(probs)
                    
                    probs_all = np.array(probs_all)
                    print("probs_all.shape: ", probs_all.shape)
                    sys.stdout.flush()
                    
                    ensemble_probs = []
                    
                    # get anchor/best model
                    probs_best = probs_all[0,:,:].reshape((-1,2))
                    probs_best_cls0 = probs_all[0,:,0].reshape((-1,))
                    probs_best_cls0_ = probs_best_cls0 > CONFIDENCE_THRESHOLD
                    probs_best_cls1 = probs_all[0,:,1].reshape((-1,))
                    probs_best_cls1_ = probs_best_cls1 > CONFIDENCE_THRESHOLD
                    probs_model = {}
                    probs_model_cls0 = {}
                    probs_model_cls1 = {}
                    for j in range(1,10):
                        probs_model[j] = probs_all[j,:,:].reshape((-1,2))
                        probs_model_cls0[j] = probs_all[j,:,0].reshape((-1,))
                        probs_model_cls1[j] = probs_all[j,:,1].reshape((-1,))
                        
                    
                        
                    # inspect per instance
                    for j, (p0, p1) in enumerate(zip(probs_best_cls0_, probs_best_cls1_)):
                        if p0 or p1:
                            ensemble_probs.append((probs_best_cls0[j], probs_best_cls1[j]))
                        else:
                            avg_cls0 = np.mean(probs_model_cls0[1][j] + probs_model_cls0[2][j] + probs_model_cls0[3][j] + \
                                               probs_model_cls0[4][j] + probs_model_cls0[5][j] + probs_model_cls0[6][j] + \
                                               probs_model_cls0[7][j] + probs_model_cls0[8][j] + probs_model_cls0[9][j])
                            avg_cls1 = np.mean(probs_model_cls1[1][j] + probs_model_cls1[2][j] + probs_model_cls1[3][j] + \
                                               probs_model_cls1[4][j] + probs_model_cls1[5][j] + probs_model_cls1[6][j] + \
                                               probs_model_cls1[7][j] + probs_model_cls1[8][j] + probs_model_cls1[9][j])
                            #max_cls0 = np.maximum.reduce([probs_model_cls0[1][j] + probs_model_cls0[2][j] + probs_model_cls0[3][j] + \
                            #                   probs_model_cls0[4][j] + probs_model_cls0[5][j] + probs_model_cls0[6][j] + \
                            #                   probs_model_cls0[7][j] + probs_model_cls0[8][j] + probs_model_cls0[9][j]])
                            #max_cls1 = np.maximum.reduce([probs_model_cls1[1][j] + probs_model_cls1[2][j] + probs_model_cls1[3][j] + \
                            #                   probs_model_cls1[4][j] + probs_model_cls1[5][j] + probs_model_cls1[6][j] + \
                            #                   probs_model_cls1[7][j] + probs_model_cls1[8][j] + probs_model_cls1[9][j]])
                            ensemble_probs.append((avg_cls0, avg_cls1))
                            
                    ensemble_probs = np.array(ensemble_probs)
                    ensemble_probs = ensemble_probs.reshape((-1,2))
                        
                    pred = np.argmax(ensemble_probs, axis=1)
                    
                    all_pred_probs = np.append(all_pred_probs, ensemble_probs)
                    
                    #pred = output.data.max(1)[1]
                    all_pred = np.append(all_pred,
                                     pred)
                    all_ytrue = np.append(all_ytrue,
                                     ybatch.cpu().numpy())

                all_pred = np.vstack(all_pred)
                all_ytrue = np.vstack(all_ytrue)
                
                all_pred_probs= np.array(all_pred_probs)
                all_pred_probs = all_pred_probs.reshape((-1,2))
                
                print("all_pred_probs.shape: ", all_pred_probs.shape)
                sys.stdout.flush()
                
                print("all_pred_probs.shape best ensemble: ", all_pred_probs)
                sys.stdout.flush()

                for nm, met in zip(metric_names, all_metric):
                    scores[str(nm)] = met.compute(predictions=all_pred, references=all_ytrue)
                    
                # add sklearn classification report to the dictionary of the evaluation scores
                target_names = ['NON-HOAX', 'HOAX']
                scores['classification_report'] = classification_report(all_ytrue, all_pred, target_names=target_names)
                
                # add brier score based on confidence level 
                brier = evaluate.load("brier_score")
                # predictions here = proba true class as predicted
                # references = the expectation of true probability
                # binning the prediction
                prob_true, prob_pred = calibration_curve(all_ytrue, all_pred_probs[:, 1], n_bins=10)
                #scores["brier_score"] = brier.compute(predictions=prob_pred, references=all_ytrue)
                #scores["brier_score"] = brier_score_loss(all_ytrue, prob_pred)
                scores["calibration_curve"] = {"prob_true": prob_true, "prob_pred": prob_pred}

                print(scores)
                sys.stdout.flush()

            return scores
        
        # for validation task (do_eval == True)
        def custom_score_valid(devX, devy, model):
            #metric_names = ["accuracy", "f1", "precision", "recall", "confusion_matrix"]

            # use precision as validation metric
            metric = evaluate.load("precision")


            if not isinstance(devX, torch.cuda.FloatTensor) or self.cudaEfficient:
                devX = torch.FloatTensor(devX).cuda()
                devy = torch.LongTensor(devy).cuda()
            with torch.no_grad():

                all_pred = np.array([])
                all_ytrue = np.array([])
                for i in range(0, len(devX), self.batch_size):
                    Xbatch = devX[i:i + self.batch_size]
                    ybatch = devy[i:i + self.batch_size]
                    if self.cudaEfficient:
                        Xbatch = Xbatch.cuda()
                        ybatch = ybatch.cuda()
                    pred = model.predict(Xbatch)
                    #pred = output.data.max(1)[1]
                    all_pred = np.append(all_pred,
                                     pred)
                    all_ytrue = np.append(all_ytrue,
                                     ybatch.cpu().numpy())
                all_pred = np.vstack(all_pred)
                all_ytrue = np.vstack(all_ytrue)

                scores = metric.compute(predictions=all_pred, references=all_ytrue)

                print(scores)
                sys.stdout.flush()

            return scores
        
        logging.info('Training {0} MLP with standard validation given sentence embedding from BERT..'
                     .format(self.modelname))
        
        
        regs = [10**t for t in range(-5, -1)] if self.usepytorch else \
               [2**t for t in range(-2, 4, 1)]
        skf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=1111)
        innerskf = StratifiedKFold(n_splits=self.k, shuffle=True,
                                   random_state=1111)
        count = 0
        for j in range(10):
            xtrain = self.X_train[j]
            ytrain = self.y_train[j]
            self.clf[j].fit(np.array(xtrain), np.array(ytrain), validation_split=0.05)
        
        scores ={}
        if self.do_eval:
            scores = custom_score_valid(self.X_test, self.y_test, self.clf)
        else:
            # get calibration plot from model prediction
            #plot_calibration(np.array(self.X_test), np.array(self.y_test), self.clf)
            # individual model
            for j in range(10):
                scores[j] = custom_score_test(np.array(self.X_test), np.array(self.y_test), self.clf[j])
            # score average
            scores[10] = avg_ensemble_score_test(np.array(self.X_test), np.array(self.y_test), self.clf)
            # score anchor and average
            #scores[11] = best_ensemble_score_test(np.array(self.X_test), np.array(self.y_test), self.clf)
            
            
        return scores
    
class LogitClassifier(object):
    """
    (train, valid, test) split classifier.
    """
    def __init__(self, X_train, y_train, X_test, y_test, config):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        print("shape self.X_train:", self.X_train.shape)
        sys.stdout.flush()
        print("shape self.y_train:", self.y_train.shape)
        sys.stdout.flush()
        
        print("shape self.X_test:", self.X_test.shape)
        sys.stdout.flush()
        print("shape self.y_test:", self.y_test.shape)
        sys.stdout.flush()
        
        self.batch_size = 128
        
        self.nclasses = 2
        self.featdim = self.X_train.shape[1]
        self.seed = config['seed']
        self.usepytorch = config['usepytorch']
        self.classifier_config = config['classifier']
        self.cudaEfficient = False if 'cudaEfficient' not in config else \
            config['cudaEfficient']
        self.modelname = get_classif_name(self.classifier_config, self.usepytorch)
        self.noreg = False if 'noreg' not in config else config['noreg']
        self.config = config
        self.clf = MLP(self.classifier_config, inputdim=self.featdim,
                          nclasses=2,
                          seed=self.seed, cudaEfficient=self.cudaEfficient)
        self.k = 5 if 'kfold' not in config else config['kfold']
    
    
    
    def run(self):
        
       
        logging.info('Training {0} MLP with standard validation given sentence embedding from BERT..'
                     .format(self.modelname))
        
        
        regs = [10**t for t in range(-5, -1)] if self.usepytorch else \
               [2**t for t in range(-2, 4, 1)]
        skf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=1111)
        innerskf = StratifiedKFold(n_splits=self.k, shuffle=True,
                                   random_state=1111)
        count = 0
        for train_idx, dev_idx in skf.split(self.X_train, self.y_train):
            count += 1
            X_train, X_dev = self.X_train[train_idx], self.X_train[dev_idx]
            y_train, y_dev = self.y_train[train_idx], self.y_train[dev_idx]

            self.clf.fit(X_train, y_train, validation_split=0.05)
            
        # return logit
        logit = self.clf.logitscore(self.X_test)
            
        return logit