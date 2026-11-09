#! for fine_tune_baseline_v1_train_fully.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from config import cfg
import imagebind
# from imagebind import data

import os
import os.path as osp
import numpy as np
import pandas as pd
import csv
import json
import pdb



# meta_csv_path = cfg.META_CSV_PATH
# meta_df = pd.read_csv(meta_csv_path)
# train_df = meta_df[meta_df['split'] == 'train']
# train_close_categories = list(set(train_df['cls_name'].tolist())) #! 46
# train_close_categories.sort()
# csv_file = "/zhoujinxing/AVEL/OV_AVEL/ovave_train_close_categories.csv"
# with open(csv_file, 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows([[item] for item in train_close_categories])
# pdb.set_trace()


# meta_csv_path = cfg.META_CSV_PATH
# meta_df = pd.read_csv(meta_csv_path)
# train_df = meta_df
# train_close_categories = list(set(train_df['cls_name'].tolist())) #! 67
# train_close_categories.sort()
# csv_file = "/zhoujinxing/AVEL/OV_AVEL/ovave_total_close_open_categories.csv"
# with open(csv_file, 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows([[item] for item in train_close_categories])
# pdb.set_trace()



# category_csv_path = cfg.TOTAL_CLOSE_OPEN_CATEGORY_CSV_PATH
# categories_df = pd.read_csv(category_csv_path, header=None)
# category_list = categories_df[0].tolist()
# category_dict = {cls_name: index for index, cls_name in enumerate(category_list)}



class OVAVE_Dataset(Dataset):
    def __init__(self, split, test_data_type='total', device=None, debug=False):
    # def __init__(self, split, test_data_type='total', device=None, debug=True):
        meta_csv_path = cfg.META_CSV_PATH
        meta_df = pd.read_csv(meta_csv_path)
        # ave_dataset_meta_combined.csv carries a blank record ("train,,,") between the
        # train and test blocks; it has no vid_name, so it cannot be loaded.
        meta_df = meta_df.dropna(subset=['cls_name', 'cls_type', 'vid_name'])
        anno_json_path = cfg.ANNO_JSON_PATH
        with open(anno_json_path, 'r') as fr:
            self.anno_data = json.load(fr)
        self.data_base_dir = cfg.DATA_BASE_DIR
        self.split = split
        self.test_data_type = test_data_type
        self.split_meta_df = meta_df[meta_df['split'] == self.split]
        if self.split == 'test':
            if self.test_data_type != 'total':
                self.split_meta_df = self.split_meta_df[ self.split_meta_df['cls_type'] == test_data_type ]

        self.debug = debug
        if self.debug:
            self.split_meta_df = self.split_meta_df.tail(32)
            print("==>>> debug mode is running")
        # self.device = device
        # if self.device is None:
        #     # self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        #     self.device = "cpu"

        if self.split == 'train':
            self.text_list = pd.read_csv(cfg.TRAIN_CLOSE_CATEGORY_CSV_PATH, header=None)[0].tolist()
        else:
            self.text_list = pd.read_csv(cfg.TOTAL_CLOSE_OPEN_CATEGORY_CSV_PATH, header=None)[0].tolist()
        self.text_inputs   = imagebind.data.load_and_transform_text(self.text_list) #! [bs, 46, 77] remove to main.py later 
 

    def __len__(self):
        return len(self.split_meta_df)


    def _gene_event_labels_train(self, category, vid_name):
        avc_label = json.loads(self.anno_data[vid_name]['label'])
        avc_label = torch.FloatTensor(avc_label)
        if torch.sum(avc_label) == 0:
            category_label = torch.LongTensor([cfg.TRAIN_BG_CLASS_ID]) #! background video
            full_label = torch.ones_like(avc_label) * cfg.TRAIN_BG_CLASS_ID # [10]
        else:
            train_close_categories = pd.read_csv(cfg.TRAIN_CLOSE_CATEGORY_CSV_PATH, header=None)[0].tolist()
            # label_mat = torch.zeros([10, len(train_close_categories)]) # [10, 46]
            label_idx = train_close_categories.index(category)
            # label_mat[:, label_idx] = avc_label
            # return label_mat.to(self.device)
            # category_label = torch.zeros([len(train_close_categories)])
            # category_label[label_idx] = 1
            category_label = torch.LongTensor([label_idx])
            full_label = torch.ones_like(avc_label) * cfg.TRAIN_BG_CLASS_ID # [10]
            ave_pos_mask = (avc_label == 1)
            full_label.masked_fill_(ave_pos_mask, label_idx)
        return avc_label, category_label, full_label.long()

    def _gene_event_labels_test(self, category, vid_name):
        avc_label = json.loads(self.anno_data[vid_name]['label'])
        avc_label = torch.FloatTensor(avc_label)
        if torch.sum(avc_label) == 0:
            category_label = torch.LongTensor([cfg.TOTAL_BG_CLASS_ID]) #! background video
            full_label = torch.ones_like(avc_label) * cfg.TOTAL_BG_CLASS_ID # [10]
        else:
            test_close_open_categories = pd.read_csv(cfg.TOTAL_CLOSE_OPEN_CATEGORY_CSV_PATH, header=None)[0].tolist()
            # label_mat = torch.zeros([10, len(test_close_open_categories)]) # [10, 67]
            label_idx = test_close_open_categories.index(category)
            # label_mat[:, label_idx] = avc_label
            # return label_mat.to(self.device)
            # category_label = torch.zeros([len(test_close_open_categories)])
            # category_label[label_idx] = 1
            category_label = torch.LongTensor([label_idx])
            full_label = torch.ones_like(avc_label) * cfg.TOTAL_BG_CLASS_ID # [10]
            ave_pos_mask = (avc_label == 1)
            full_label.masked_fill_(ave_pos_mask, label_idx)
        return avc_label, category_label, full_label.long()


    def __getitem__(self, idx):
        _, category, cls_type, vid_name = self.split_meta_df.iloc[idx, :]
        audio_paths = [osp.join(self.data_base_dir, self.split, 'audio', category, vid_name+'.wav')]
        visual_frames_dir = osp.join(self.data_base_dir, self.split, 'video', category, vid_name)
        visual_frames_paths = []
        assert len(os.listdir(visual_frames_dir)) == 10
        img_names = os.listdir(visual_frames_dir)
        img_names.sort()
        for img_name in img_names:
            visual_frames_paths.append(osp.join(visual_frames_dir, img_name))
        
        if cfg.COMPLEX_REPR.ENABLED:
            #! Complex representation mode:
            #! Vision: HSV → Complex → hsv_to_rgb(real) and hsv_to_rgb(imag) (always)
            visual_inputs, visual_imag_inputs = imagebind.data.load_and_transform_vision_data_complex(
                visual_frames_paths)  # both [10, 3, 224, 224]

            if cfg.COMPLEX_REPR.AUDIO_MODE == 'fft_real_imag':
                #! Audio: |FFT.real| → mel & |FFT.imag| → mel (original COMPLEX_REPR)
                audio_inputs, audio_imag_inputs = imagebind.data.load_and_transform_audio_data_complex(
                    audio_paths)  # both [1, 10, 1, 128, 204]
                phase_inputs = None
            else:
                #! Audio: standard magnitude mel + phase IF/GD (magnitude_phase mode)
                audio_inputs, phase_inputs = imagebind.data.load_and_transform_audio_data(
                    audio_paths, return_phase=True)  # audio: [1, 10, 1, 128, 204], phase: [1, 10, 1, 257, 204]
                audio_imag_inputs = None
        elif cfg.COMPLEX_TEMPORAL.ENABLED:
            #! Complex Temporal mode:
            #! Vision: HSV → Complex → hsv_to_rgb(real) and hsv_to_rgb(imag) (same as COMPLEX_REPR)
            visual_inputs, visual_imag_inputs = imagebind.data.load_and_transform_vision_data_complex(
                visual_frames_paths)  # both [10, 3, 224, 224]
            #! Audio: standard magnitude mel + phase IF/GD
            audio_inputs, phase_inputs = imagebind.data.load_and_transform_audio_data(
                audio_paths, return_phase=True)  # audio: [1, 10, 1, 128, 204], phase: [1, 10, 1, 257, 204]
        else:
            if cfg.PHASE.ENABLED and cfg.PHASE.MODE == 'hilbert':
                audio_inputs, phase_inputs = imagebind.data.load_and_transform_audio_data(
                    audio_paths, return_hilbert=True)  # audio: [1, 10, 1, 128, 204], hilbert: [1, 10, 1, 128, 204]
            elif cfg.PHASE.ENABLED:
                audio_inputs, phase_inputs = imagebind.data.load_and_transform_audio_data(
                    audio_paths, return_phase=True)  # audio: [1, 10, 1, 128, 204], phase: [1, 10, 1, 257, 204]
            else:
                audio_inputs = imagebind.data.load_and_transform_audio_data(audio_paths) # [bs, 1, 10, 1, 128, 204]
                phase_inputs = None
            # Visual-imag stream: dispatched by cfg.IHSV.MODE
            #   'ihsv'       = HSV→complex→hsv_to_rgb imag part (original FCCNs-style)
            #   'phase_ifft' = 2D-FFT phase-only IFFT + CLAHE (3ch broadcast)
            #   'luma'       = BT.709 luma replicated to 3 channels
            if cfg.IHSV.ENABLED:
                ihsv_mode = getattr(cfg.IHSV, 'MODE', 'ihsv')
                if ihsv_mode == 'phase_ifft':
                    visual_inputs, visual_imag_inputs = imagebind.data.load_and_transform_vision_data_phase_ifft(
                        visual_frames_paths)  # both [10, 3, 224, 224], ImageBind-normalized
                elif ihsv_mode == 'luma':
                    visual_inputs, visual_imag_inputs = imagebind.data.load_and_transform_vision_data_luma(
                        visual_frames_paths)  # both [10, 3, 224, 224], ImageBind-normalized
                else:
                    visual_inputs, visual_imag_inputs = imagebind.data.load_and_transform_vision_data_complex(
                        visual_frames_paths)  # both [10, 3, 224, 224], ImageBind-normalized
            else:
                visual_inputs = imagebind.data.load_and_transform_vision_data(visual_frames_paths) # [bs, 10, 3, 224, 224]
                visual_imag_inputs = None

        # Depth inputs (from Depth-Anything-V2 pre-generated depth maps)
        depth_inputs = None
        if cfg.DEPTH.ENABLED:
            depth_frames_dir = osp.join(self.data_base_dir, self.split, 'depth', category, vid_name)
            depth_frame_names = sorted(os.listdir(depth_frames_dir))
            assert len(depth_frame_names) == 10, \
                f"Expected 10 depth frames in {depth_frames_dir}, got {len(depth_frame_names)}"
            depth_frames_paths = [osp.join(depth_frames_dir, fn) for fn in depth_frame_names]
            depth_inputs = imagebind.data.load_and_transform_depth_data(depth_frames_paths)  # [10, 1, 224, 224]

        if self.split == 'train':
            avc_label, category_label, full_label = self._gene_event_labels_train(category, vid_name) # [bs, 10, 46]
        else:
            avc_label, category_label, full_label = self._gene_event_labels_test(category, vid_name) # [bs, 10, 67]

        # Unified return format: always return all fields, use None for disabled ones
        if cfg.COMPLEX_REPR.ENABLED:
            if cfg.COMPLEX_REPR.AUDIO_MODE == 'fft_real_imag':
                # audio_imag_inputs replaces phase_inputs (already mel-shaped)
                return audio_inputs, visual_inputs, visual_imag_inputs, self.text_inputs, full_label, avc_label, category_label, vid_name, audio_imag_inputs
            else:
                # magnitude_phase: standard audio + phase IF/GD (same format as non-COMPLEX_REPR with phase)
                return audio_inputs, visual_inputs, visual_imag_inputs, self.text_inputs, full_label, avc_label, category_label, vid_name, phase_inputs
        elif cfg.COMPLEX_TEMPORAL.ENABLED:
            # visual_inputs=real_iHSV, visual_imag_inputs=imag_iHSV, phase_inputs=phase IF/GD
            return audio_inputs, visual_inputs, visual_imag_inputs, self.text_inputs, full_label, avc_label, category_label, vid_name, phase_inputs
        elif cfg.DEPTH.ENABLED:
            if phase_inputs is not None:
                return audio_inputs, visual_inputs, depth_inputs, self.text_inputs, full_label, avc_label, category_label, vid_name, phase_inputs
            return audio_inputs, visual_inputs, depth_inputs, self.text_inputs, full_label, avc_label, category_label, vid_name
        elif cfg.IHSV.ENABLED:
            if phase_inputs is not None:
                return audio_inputs, visual_inputs, visual_imag_inputs, self.text_inputs, full_label, avc_label, category_label, vid_name, phase_inputs
            return audio_inputs, visual_inputs, visual_imag_inputs, self.text_inputs, full_label, avc_label, category_label, vid_name
        if phase_inputs is not None:
            return audio_inputs, visual_inputs, self.text_inputs, full_label, avc_label, category_label, vid_name, phase_inputs
        return audio_inputs, visual_inputs, self.text_inputs, full_label, avc_label, category_label, vid_name
        # return audio_inputs, visual_inputs, avc_label, category_label, vid_name



if __name__ == "__main__":
    from imagebind.models import imagebind_model
    from imagebind.models.imagebind_model import ModalityType
    from einops import rearrange, repeat, reduce

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Instantiate model
    # model = imagebind_model.imagebind_huge(pretrained=True)
    # model.eval()
    # model.to(device)


    # split = 'train'
    split = 'test'
    BS = 32
    split_dataset = OVAVE_Dataset(split, device=None, debug=False)
    split_dataloader = torch.utils.data.DataLoader(
        split_dataset,
        batch_size=BS,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    ) 
    count = 0
    for n_iter, batch_data in enumerate(split_dataloader):
        audio, visual, text, full_label, avc_label, category_label, vid_name = batch_data
        # text = split_dataset.text_inputs
        avc_label = avc_label.cuda()
        category_label = category_label.cuda()
        full_label = full_label.cuda()
        assert len(audio[0]) == len(vid_name)
        # pdb.set_trace()
        for i in range(len(audio[0])):
            count += 1
            print(count, vid_name[i])
            # if vid_name[i] == 'VP2WctM2tzU':
            audio_input = imagebind.data.load_and_transform_audio_data([audio[0][i]])  
    
        # audio = audio.squeeze(1) # [bs, 10, 1, 128, 204]]
        # print(vid_name)
        # pdb.set_trace()
        # text = rearrange(text, 'b k d -> (b k) d')
        # single_text = text[0]
        # inputs = {
        #     ModalityType.TEXT: single_text.cuda(), # [46/67, 77]
        #     ModalityType.VISION: visual.cuda(), # [bs, 10, 3, 224, 224]
        #     ModalityType.AUDIO: audio.cuda(), # [bs, 10, 1, 128, 204]
        # }
        # with torch.no_grad():
        #     embeddings = model(inputs)
            # audio: [bs, 10, 1024], vision: [bs, 10, 1024], text: [46 or 67, 1024]
    pdb.set_trace()