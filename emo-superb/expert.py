import os
import math
import torch
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, DistributedSampler
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from .model import *
from ..model import *
from .dataset import collate_fn, DataManager, WavExtractor, WavSet, collate_fn_padd, get_norm_stat_for_wav
import json
from sklearn.metrics import classification_report
import numpy as np
import warnings
import pickle as pk

warnings.filterwarnings("ignore")


def class_balanced_softmax_cross_entropy_with_softtarget(input,target,weights,reduction='mean'):
    """
    :param input: (batch, *)
    :param target: (batch, *) same shape as input, each item must be a valid distribution: target[i, :].sum() == 1.
    :weights from the training set: [0.2139, 0.3445, 0.4665, 3.3741, 0.3337, 2.2446, 0.2398, 1.3928, 0.3900]
    :original batch loss: [3.0727, 2.7817, 1.7426, 3.5614, 2.9689, 4.5142, 3.8597, 2.8744]
    :weighted batch loss: [1.2815, 0.7407, 0.5930, 1.4853, 3.0804, 1.5423, 1.9164, 1.1688]
    """

    weights = weights.unsqueeze(0)
    weights = weights.repeat(target.shape[0],1) * target
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1,target.shape[1]) # Dim = (batch,K_emotions)

    logprobs = torch.nn.functional.log_softmax(input.view(input.shape[0], -1), dim=1)
    batchloss = - torch.sum(weights*target.view(target.shape[0], -1) * logprobs, dim=1)

    if reduction == 'none':
        return batchloss
    elif reduction == 'mean':
        return torch.mean(batchloss)
    elif reduction == 'sum':
        return torch.sum(batchloss)
    else:
        raise NotImplementedError('Unsupported reduction mode.')

def softmax_cross_entropy_with_softtarget(input, target, reduction='mean'):
    """
    soft-version cross entropy 
    :param input: (batch, *)
    :param target: (batch, *) same shape as input, each item must be a valid distribution: target[i, :].sum() == 1.
    """
    logprobs = torch.nn.functional.log_softmax(input.view(input.shape[0], -1), dim=1)
    batchloss = - torch.sum(target.view(target.shape[0], -1) * logprobs, dim=1)
    if reduction == 'none':
        return batchloss
    elif reduction == 'mean':
        return torch.mean(batchloss)
    elif reduction == 'sum':
        return torch.sum(batchloss)
    else:
        raise NotImplementedError('Unsupported reduction mode.')
    

def SCE_category(pred, lab, weights):
    lsm = F.log_softmax(pred, -1)
    #loss = -weights*(lab * lsm)
    loss = -(lab * lsm)
    loss = loss.sum(-1)
    return loss.mean()

def load_pk_file(input_path):
    with open(input_path, 'rb') as fp:
        dataset = pk.load(fp) 
    return dataset  

class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']

        self.fold = self.datarc.get('test_fold') or kwargs.get("downstream_variant")
        print(f"[Expert] - using the testing fold: \"{self.fold}\". Ps. Use -o config.downstream_expert.datarc.test_fold=fold2 to change test_fold in config.")

        self.audio_path = self.datarc['root'] + self.datarc['corpus'] + "/Audios"
        self.labels_path = self.datarc['root'] + self.datarc['corpus'] + '/' + self.datarc['p_or_s'] + "/labels_consensus_" + self.datarc['test_fold'].replace("fold","") + ".csv"
        self.config_path = self.datarc['root'] + self.datarc['corpus'] + '/' + self.datarc['p_or_s'] + "/config.json"

        dam = DataManager(self.config_path) #(self.datarc['config'])
        snum=10000000000000000

        train_wav_path = dam.get_wav_path(split_type="train",wav_loc=self.audio_path , lbl_loc=self.labels_path)[:snum]
        dev_wav_path = dam.get_wav_path(split_type="dev",wav_loc=self.audio_path,lbl_loc=self.labels_path)[:snum]
        test_wav_path = dam.get_wav_path(split_type="test",wav_loc=self.audio_path, lbl_loc=self.labels_path)

        # Class balanced weights
        train_utts = dam.get_utt_list("train",lbl_loc=self.labels_path)[:snum]
        train_labs = dam.get_msp_labels(train_utts, lab_type='categorical',lbl_loc=self.labels_path)
        self.k_thresold = 1/train_labs.shape[1]
        train_labs_PT = torch.Tensor(np.asarray(train_labs))
        train_labs_binary_PT = torch.where(train_labs_PT>self.k_thresold,1.0,0.0)

        samples_per_cls = torch.sum(train_labs_binary_PT,dim=0)
        beta = (train_labs.shape[0]-1)/train_labs.shape[0]
        no_of_classes = train_labs.shape[1]
        effective_num = 1.0 - torch.pow(beta, samples_per_cls)
        weights = (1.0 - beta) / effective_num
        self.class_balanced_weights = weights / torch.sum(weights) * no_of_classes
        print("samples_per_cls",samples_per_cls)
        print("self.class_balanced_weights",self.class_balanced_weights)

        # Save/Load Wavs Numpy Files
        train_wavs_np_path = self.datarc['root'] + self.datarc['corpus'] + '/' + self.datarc['p_or_s'] + "/Train_wavs_numpy_" + self.datarc['test_fold'] + ".pkl"

        if not os.path.exists(train_wavs_np_path):
            print("Saving Wavs Numpy files ",train_wavs_np_path)
            # Train
            train_wavs = WavExtractor(train_wav_path).extract()
            wav_mean, wav_std = get_norm_stat_for_wav(train_wavs)
            
            stats = {"wav_mean": wav_mean, "wav_std": wav_std}
            with open(train_wavs_np_path, 'wb') as f:
                pk.dump(stats, f)
        else:
            stats = load_pk_file(train_wavs_np_path)
            wav_mean = stats["wav_mean"]
            wav_std = stats["wav_std"]

        self.train_dataset = WavSet(train_wav_path, train_labs, train_utts, 
            print_dur=True, lab_type='categorical',
            label_config = dam.get_label_config(label_type='categorical'),
            wav_mean = wav_mean, wav_std = wav_std
        )

        # Dev
        dev_utts = dam.get_utt_list("dev",lbl_loc=self.labels_path)[:snum]
        dev_labs = dam.get_msp_labels(dev_utts, lab_type='categorical',lbl_loc=self.labels_path)

        self.dev_dataset  = WavSet(dev_wav_path, dev_labs, dev_utts, 
            print_dur=True, lab_type='categorical',
            wav_mean = self.train_dataset.wav_mean, wav_std = self.train_dataset.wav_std,
            label_config = dam.get_label_config(label_type='categorical')
        )

        # Test
        test_utts = dam.get_utt_list("test",lbl_loc=self.labels_path)
        test_labs = dam.get_msp_labels(test_utts, lab_type='categorical',lbl_loc=self.labels_path)
        self.test_dataset  = WavSet(test_wav_path, test_labs, test_utts, 
            print_dur=True, lab_type='categorical',
            wav_mean = self.train_dataset.wav_mean, wav_std = self.train_dataset.wav_std, 
            label_config = dam.get_label_config(label_type='categorical')
        )


        with open(self.config_path, 'r') as f:
            self.config = json.load(f)

        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc.get(self.modelrc['select'], {})
        self.projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
        self.model = model_cls(
            input_dim = self.modelrc['projector_dim'],
            output_dim = len(self.config['categorical']["emo_type"]),#dataset.class_num,
            **model_conf,
        )
        #self.objective = nn.CrossEntropyLoss()
        #self.objective = SCE_category
        self.objective = class_balanced_softmax_cross_entropy_with_softtarget 
        #self.objective = softmax_cross_entropy_with_softtarget
        self.expdir = expdir

        self.register_buffer('best_score', torch.ones(1)*99999)
        #self.register_buffer('best_score', torch.Tensor(np.array[99]))


    def get_downstream_name(self):
        return self.fold.replace('fold', 'emotion')


    def _get_train_dataloader(self, dataset):
        sampler = DistributedSampler(dataset) if is_initialized() else None
        return DataLoader(
            dataset, batch_size=self.datarc['train_batch_size'],
            shuffle=(sampler is None), sampler=sampler,
            num_workers=self.datarc['num_workers'],
            collate_fn= collate_fn_padd #collate_fn
        )
        #DataLoader(train_set, batch_size=args.batch_size, collate_fn=sg_utils.collate_fn_padd, shuffle=True)

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=self.datarc['eval_batch_size'],
            shuffle=False, num_workers=self.datarc['num_workers'],
            collate_fn=collate_fn_padd #collate_fn
        )

    def get_train_dataloader(self):
        return self._get_train_dataloader(self.train_dataset)

    def get_dev_dataloader(self):
        return self._get_eval_dataloader(self.dev_dataset)

    def get_test_dataloader(self):
        return self._get_eval_dataloader(self.test_dataset)

    # Interface
    def get_dataloader(self, mode):
        return eval(f'self.get_{mode}_dataloader')()

    # Interface
    def forward(self, mode, features, labels, filenames, records, **kwargs):
        
        device = features[0].device
        features_len = torch.IntTensor([len(feat) for feat in features]).to(device=device)

        features = pad_sequence(features, batch_first=True)
        features = self.projector(features)
        predicted, _ = self.model(features, features_len)

        labels = labels.to(features.device)
        loss = self.objective(predicted, labels,self.class_balanced_weights.to(features.device),reduction='mean') #(input,target,weights,reduction='mean')

        prediction_distribution = torch.nn.functional.softmax(predicted,dim=1)
        preditions_binary = torch.where(prediction_distribution> self.k_thresold, 1.0, 0.0)
        labels_binary = torch.where(labels > self.k_thresold, 1.0, 0.0)

        all_emotions = self.config['categorical']["emo_type"]
        reprot_dict = classification_report(labels_binary.cpu().numpy(force=True), preditions_binary.cpu().numpy(force=True),target_names=all_emotions,output_dict=True)
        records['acc'] += [reprot_dict['macro avg']['f1-score']]
        
        records['loss'].append(loss.item())

        records["filename"] += filenames

        predict=[]
        truth=[]
        all_emotions_np = np.array(all_emotions)
        for idx in range(len(labels_binary)):
            turth_emo = ";".join(list(all_emotions_np[np.where(labels_binary[idx].cpu().numpy(force=True)==1.0)[0]]))
            predict_emo = ";".join(list(all_emotions_np[np.where(preditions_binary[idx].cpu().numpy(force=True)==1.0)[0]]))

            truth.append(turth_emo)
            predict.append(predict_emo)

        records["predict"] +=  predict
        records["truth"] += truth

        return loss

    # interface
    def log_records(self, mode, records, logger, global_step, **kwargs):
        save_names = []
        for key in ["acc", "loss"]:
            values = records[key]
            average = torch.FloatTensor(values).mean().item()
            logger.add_scalar(
                f'emotion-{self.fold}/{mode}-{key}',
                average,
                global_step=global_step
            )
            
            with open(Path(self.expdir) / "log.log", 'a') as f:
                if key == 'loss':
                    print(f"{mode} {key}: {average}")
                    f.write(f'{mode} {key} at step {global_step}: {average}\n')
                    #if mode == 'dev' and average > self.best_score:
                    if mode == 'dev' and average < self.best_score:
                        self.best_score = torch.ones(1) * average
                        f.write(f'New best on {mode} {key} at step {global_step}: {average}\n')
                        save_names.append(f'{mode}-best.ckpt')
                elif key == 'acc':
                    print(f"{mode} {key}: {average}")
                    f.write(f'{mode} {key} at step {global_step}: {average}\n')


        if mode in ["dev", "test"]:
            with open(Path(self.expdir) / f"{mode}_{self.fold}_predict.txt", "w") as file:
                line = [f"{f} {e}\n" for f, e in zip(records["filename"], records["predict"])]
                file.writelines(line)

            with open(Path(self.expdir) / f"{mode}_{self.fold}_truth.txt", "w") as file:
                line = [f"{f} {e}\n" for f, e in zip(records["filename"], records["truth"])]
                file.writelines(line)

        return save_names
