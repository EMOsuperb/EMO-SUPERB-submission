# -*- coding: utf-8 -*- #
"""
    FileName     [ dataset.py ]
    Synopsis     [ the emotion classifier dataset ]
    Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""

import json
from pathlib import Path
from os.path import join as path_join

import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import Resample

import pandas as pd
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
import glob
import os
import librosa
import csv
import torch
import torch.utils as torch_utils
import torch.nn as nn
import pickle as pk

SAMPLE_RATE = 16000


def get_norm_stat_for_wav(wav_list, verbose=False):
    count = 0
    wav_sum = 0
    wav_sqsum = 0
    
    for cur_wav in tqdm(wav_list):
        wav_sum += np.sum(cur_wav)
        wav_sqsum += np.sum(cur_wav**2)
        count += len(cur_wav)
    
    wav_mean = wav_sum / count
    wav_var = (wav_sqsum / count) - (wav_mean**2)
    wav_std = np.sqrt(wav_var)

    return wav_mean, wav_std

class WavSet(torch_utils.data.Dataset):
    def __init__(self, *args, **kwargs):
        super(WavSet, self).__init__()
        self.wav_list = kwargs.get("wav_list", args[0]) # (N, D, T)
        self.lab_list = kwargs.get("lab_list", args[1])
        self.utt_list = kwargs.get("utt_list", args[2])
        self.print_dur = kwargs.get("print_dur", False)
        self.lab_type = kwargs.get("lab_type", False)

        self.wav_mean = kwargs.get("wav_mean", 0)
        self.wav_std = kwargs.get("wav_std", 1)

        self.label_config = kwargs.get("label_config", None)

        ## Assertion
        if self.lab_type == "categorical":
            assert len(self.label_config.get("emo_type", [])) != 0, "Wrong emo_type in config file"

        # check max duration
        self.max_dur = 10*16000
        if self.wav_mean is None or self.wav_std is None:
            self.wav_mean, self.wav_std = get_norm_stat_for_wav(self.wav_list)

    def save_norm_stat(self, norm_stat_file):
        with open(norm_stat_file, 'wb') as f:
            pk.dump((self.wav_mean, self.wav_std), f)
            
    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        cur_wav = extract_wav(self.wav_list[idx])[:self.max_dur]
        cur_dur = len(cur_wav)
        cur_wav = (cur_wav - self.wav_mean) / (self.wav_std+0.000001)
        cur_utt = self.utt_list[idx]

        if self.lab_type == "categorical":
            cur_lab = self.lab_list[idx]

        result = (cur_wav, cur_lab, cur_utt)
        if self.print_dur:
            result = (cur_wav, cur_lab, cur_utt, cur_dur)
        return result

def collate_fn_padd(batch):
    '''
    Padds batch of variable length
    '''
    total_wav = []
    total_lab = []
    total_dur = []
    total_utt = []
    for wav, lab, utt, dur in batch:
        total_wav.append(torch.Tensor(wav))
        total_lab.append(lab)
        total_dur.append(dur)
        total_utt.append(utt)
    #total_wav = nn.utils.rnn.pad_sequence(total_wav, batch_first=True)

    total_lab = torch.Tensor(np.asarray(total_lab))
    
    #total_utt = torch.Tensor(total_utt)
    max_dur = np.max(total_dur)
    #attention_mask = torch.zeros(total_wav.shape[0], max_dur)
    #for dur in total_dur:
    #    attention_mask[:,:dur] = 1
    ## compute mask
    #return total_wav, total_lab, total_utt, attention_mask
    return total_wav, total_lab, total_utt

def extract_wav(wav_path):
    raw_wav, _ = librosa.load(wav_path, sr=16000)
    return raw_wav

class WavExtractor:
    def __init__(self, *args, **kwargs):
        self.wav_path_list = kwargs.get("wav_paths", args[0])
        # self.sr = kwargs.get("sampling_rate", 16000)
        self.nj = kwargs.get("nj", 24)
    def extract(self):
        print("Extracting wav files")
        with Pool(self.nj) as p:
            wav_list = list(tqdm(p.imap(extract_wav, self.wav_path_list), total=len(self.wav_path_list)))
            
        return wav_list

def load_env(env_path):
    with open(env_path, 'r') as f:
        env_dict = json.load(f)
    return env_dict

class DataManager:
    def __load_env__(self, env_path):
        with open(env_path, 'r') as f:
            env_dict = json.load(f)
        return env_dict

    def __init__(self, env_path):
        self.env_dict=self.__load_env__(env_path)
        self.msp_label_dict = None

    def get_wav_path(self, split_type=None, wav_loc=None, lbl_loc=None , *args, **kwargs):
        wav_root=wav_loc
        if split_type == None:
            wav_list = glob.glob(os.path.join(wav_root, "*.wav"))
        else:
            utt_list = self.get_utt_list(split_type,  lbl_loc)
            wav_list = [os.path.join(wav_root, utt_id) for utt_id in utt_list]
        
        wav_list.sort()
        return wav_list

    def get_utt_list(self, split_type, lbl_loc):
        label_path = lbl_loc
        utt_list=[]
        sid = self.env_dict["data_split_type"][split_type]
        with open(label_path, 'r') as f:
            f.readline()
            csv_reader = csv.reader(f)
            for row in csv_reader:
                utt_id = row[0]
                stype = row[-1]
                if stype == sid:
                    utt_list.append(utt_id)
        utt_list.sort()
        return utt_list

    def __load_msp_cat_label_dict__(self,lbl_loc):
        label_path = lbl_loc
        self.msp_label_dict=dict()
        emo_class_list =  self.get_categorical_emo_class()
        print(emo_class_list)
        with open(label_path, 'r') as f:
            header = f.readline().split(",")
            emo_idx_list = []
            for emo_class in emo_class_list:
                emo_idx_list.append(header.index(emo_class))

            csv_reader = csv.reader(f)
            for row in csv_reader:
                utt_id = row[0]
                self.msp_label_dict[utt_id]=dict()
                cur_emo_lab = []
                for emo_idx in emo_idx_list:
                    cur_emo_lab.append(float(row[emo_idx]))
                self.msp_label_dict[utt_id]=cur_emo_lab

    def get_msp_labels(self, utt_list, lab_type=None,lbl_loc=None):
        if lab_type == "categorical":
            self.__load_msp_cat_label_dict__(lbl_loc)
        return np.array([self.msp_label_dict[utt_id] for utt_id in utt_list])

    def get_categorical_emo_class(self):
        return self.env_dict["categorical"]["emo_type"]
    def get_categorical_emo_num(self):
        cat_list = self.get_categorical_emo_class()
        return len(cat_list)

    def get_label_config(self, label_type):
        assert label_type in ["categorical", "dimensional"]
        return self.env_dict[label_type]

class IEMOCAPDataset(Dataset):
    def __init__(self, data_dir, meta_path, pre_load=True):
        self.data_dir = data_dir
        self.pre_load = pre_load
        with open(meta_path, 'r') as f:
            self.data = json.load(f)
        self.class_dict = self.data['labels']
        self.idx2emotion = {value: key for key, value in self.class_dict.items()}
        self.class_num = len(self.class_dict)
        self.meta_data = self.data['meta_data']
        _, origin_sr = torchaudio.load(
            path_join(self.data_dir, self.meta_data[0]['path']))
        self.resampler = Resample(origin_sr, SAMPLE_RATE)
        if self.pre_load:
            self.wavs = self._load_all()

    def _load_wav(self, path):
        wav, _ = torchaudio.load(path_join(self.data_dir, path))
        wav = self.resampler(wav).squeeze(0)
        return wav

    def _load_all(self):
        wavforms = []
        for info in self.meta_data:
            wav = self._load_wav(info['path'])
            wavforms.append(wav)
        return wavforms

    def __getitem__(self, idx):
        label = self.meta_data[idx]['label']
        label = self.class_dict[label]
        if self.pre_load:
            wav = self.wavs[idx]
        else:
            wav = self._load_wav(self.meta_data[idx]['path'])
        #print("Default IEMOCAP",(wav.numpy(), label, Path(self.meta_data[idx]['path']).stem))
        return wav.numpy(), label, Path(self.meta_data[idx]['path']).stem

    def __len__(self):
        return len(self.meta_data)

class IEMOCAPDataset_Dev(Dataset):
    def __init__(self, data_dir, meta_path, pre_load=True):
        # data_dir =  ./data/IEMOCAP/Audios
        # meata_path = ./data/IEMOCAP/labels_consensus_1.csv
        self.data_dir = data_dir
        self.pre_load = pre_load
        

        #with open(meta_path, 'r') as f:
        #    self.data = json.load(f)
        self.data = pd.read_csv(meta_path,index_col=0)

        # './data/IEMOCAP/config.json'
        config_path = data_dir.replace("Audios","config.json")
        with open(config_path, 'r') as f:
            self.config = json.load(f)        
        
        #self.class_dict = self.data['labels']
        #self.idx2emotion = {value: key for key, value in self.class_dict.items()}
        #self.class_num = len(self.class_dict)
        self.class_num = len([self.config["categorical"]["emo_type"]])
        

        #self.meta_data = self.data['meta_data'] #use self.data to replace
        

        # _, origin_sr = torchaudio.load(
        #     path_join(self.data_dir, self.meta_data[0]['path']))


        _, origin_sr = torchaudio.load(
            path_join(self.data_dir, self.meta_data[0]['path']))

        self.resampler = Resample(origin_sr, SAMPLE_RATE)
        if self.pre_load:
            self.wavs = self._load_all()

    def _load_wav(self, path):
        wav, _ = torchaudio.load(path_join(self.data_dir, path))
        wav = self.resampler(wav).squeeze(0)
        return wav

    def _load_all(self):
        wavforms = []
        for info in self.meta_data:
            wav = self._load_wav(info['path'])
            wavforms.append(wav)
        return wavforms

    def __getitem__(self, idx):
        label = self.meta_data[idx]['label']
        label = self.class_dict[label]
        if self.pre_load:
            wav = self.wavs[idx]
        else:
            wav = self._load_wav(self.meta_data[idx]['path'])
        return wav.numpy(), label, Path(self.meta_data[idx]['path']).stem

    def __len__(self):
        return len(self.meta_data)

def collate_fn(samples):
    return zip(*samples)
