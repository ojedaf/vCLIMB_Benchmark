# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import torch.utils.data as data
from torch.utils.data import DataLoader

from PIL import Image
import os
import numpy as np
from numpy.random import randint
import random


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]
    
    @property
    def frames(self):
        return self._data[1]

    @property
    def num_frames(self):
        return self._data[2]

    @property
    def label(self):
        return self._data[3]
    

class CILSetTask:
    def __init__(self, set_tasks, path_frames, memory_size, batch_size, shuffle, num_workers, 
                 drop_last=False, pin_memory=False, num_segments=3, new_length=1, modality='RGB', 
                 transform=None, random_shift=True, test_mode=False, 
                 remove_missing=False, dense_sample=False, twice_sample=False, train_enable = True):
        
        self.memory = {}
        self.num_tasks = len(set_tasks)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.current_task = 0
        self.current_task_dataset = None
        self.memory_size = memory_size
        self.set_tasks = set_tasks
        self.path_frames = path_frames
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.remove_missing = remove_missing
        self.dense_sample = dense_sample
        self.twice_sample = twice_sample
        self.train_enable = train_enable
    
    def __iter__(self):
        self.memory = {}
        self.current_task_dataset = None
        self.current_task = 0
        return self
    
    def __next__(self):
        data = self.set_tasks[self.current_task]
        if self.train_enable:
            comp_data = {**self.memory, **data}
        else:
            comp_data = data
        current_task_dataset = TSNDataSet(self.path_frames, comp_data, None, self.num_segments, self.new_length, 
                                          self.modality, self.transform, self.random_shift, self.test_mode, 
                                          self.remove_missing, self.dense_sample, self.twice_sample)
        self.current_task_dataloader = DataLoader(current_task_dataset, batch_size = self.batch_size, shuffle = self.shuffle, 
                                     num_workers = self.num_workers, pin_memory = self.pin_memory, drop_last = self.drop_last)
        if self.train_enable:
            self.rehearsal_randomMethod(data)
            
        self.current_task += 1
        if self.current_task < len(self.set_tasks):
            return self.current_task_dataloader, len(self.set_tasks[self.current_task].keys())
        else:
            return self.current_task_dataloader, None
    
    def get_valSet_by_taskNum(self, num_task):
        eval_data = {}
        total_data = []
        list_val_loaders = []
        list_num_classes = []
        for k in range(num_task):
            data = self.set_tasks[k]
            eval_data = {**eval_data, **data}
            total_data.append(data)
            list_num_classes.append(len(data.keys()))
        classes = eval_data.keys()
        for i, data_i in enumerate(total_data):
            val_task_dataset = TSNDataSet(self.path_frames, data_i, classes, self.num_segments, self.new_length, 
                                              self.modality, self.transform, self.random_shift, self.test_mode, 
                                              self.remove_missing, self.dense_sample, self.twice_sample)
            val_task_dataloader = DataLoader(val_task_dataset, batch_size = self.batch_size, shuffle = self.shuffle, 
                                                      num_workers = self.num_workers, pin_memory = self.pin_memory, 
                                                      drop_last = self.drop_last)
            list_val_loaders.append((val_task_dataloader, list_num_classes[i]))
        return list_val_loaders
        
    
    def rehearsal_randomMethod(self, current_task):
        saved_classes = self.memory.keys()
        current_classes = current_task.keys()
        num_classes = len(saved_classes) + len(current_classes)
        elem_to_save = {**self.memory, **current_task}
        if self.memory_size != 'ALL':
            num_instances_per_class = self.memory_size // num_classes
            for class_n, elems in elem_to_save.items():
                random.shuffle(elems)
                elem_to_save[class_n] = elems[:num_instances_per_class]
        self.memory = elem_to_save

class TSNDataSet(data.Dataset):
    def __init__(self, path_frames, data, classes=None, 
                 num_segments=3, new_length=1, modality='RGB', 
                 transform=None, random_shift=True, test_mode=False,
                 remove_missing=False, dense_sample=False, twice_sample=False):

        self.path_frames = path_frames
        self.data = data
        self.classes = classes if classes != None else data.keys()
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.remove_missing = remove_missing
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.twice_sample = twice_sample  # twice sample for more validation
        if self.dense_sample:
            print('=> Using dense sample for the dataset...')
        if self.twice_sample:
            print('=> Using twice sample for the dataset...')

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, name_frame):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            try:
                return [Image.open(os.path.join(directory, name_frame)).convert('RGB')]
            except Exception:
                print('error loading image:', os.path.join(directory, name_frame))
                return [Image.open(os.path.join(directory, name_frame)).convert('RGB')]
        elif self.modality == 'Flow':
            if self.image_tmpl == 'flow_{}_{:05d}.jpg':  # ucf
                x_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('x', idx))).convert(
                    'L')
                y_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('y', idx))).convert(
                    'L')
            elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':  # something v1 flow
                x_img = Image.open(os.path.join(self.root_path, '{:06d}'.format(int(directory)), self.image_tmpl.
                                                format(int(directory), 'x', idx))).convert('L')
                y_img = Image.open(os.path.join(self.root_path, '{:06d}'.format(int(directory)), self.image_tmpl.
                                                format(int(directory), 'y', idx))).convert('L')
            else:
                try:
                    # idx_skip = 1 + (idx-1)*5
                    flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert(
                        'RGB')
                except Exception:
                    print('error loading flow file:',
                          os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                    flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')
                # the input flow file is RGB image with (flow_x, flow_y, blank) for each channel
                flow_x, flow_y, _ = flow.split()
                x_img = flow_x.convert('L')
                y_img = flow_y.convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        
        class2label = {name:i for i, name in enumerate(self.classes)}
        self.video_list = []
        for class_name, videos in self.data.items():
            for video_name in videos: 
                path_video = os.path.join(self.path_frames, class_name, video_name)
                frames = os.listdir(path_video)
                frames.sort(key = lambda x: int(x.split('.')[0].replace('frame','')))
                num_frames = len(frames)
                if num_frames >= self.num_segments:
                    item = [path_video, frames, num_frames, class2label[class_name]]
                    self.video_list.append(VideoRecord(item))

        print('video number:%d' % (len(self.video_list)))

    def _sample_indices(self, record):
        """
    
        :param record: VideoRecord
        :return: list
        """
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets)
        else:  # normal sample
            average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                                  size=self.num_segments)
            elif record.num_frames > self.num_segments:
                offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets

    def _get_val_indices(self, record):
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets)
        else:
            if record.num_frames > self.num_segments + self.new_length - 1:
                tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets

    def _get_test_indices(self, record):
        if self.dense_sample:
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets)
        elif self.twice_sample:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)] +
                               [int(tick * x) for x in range(self.num_segments)])

            return offsets
        else:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            return offsets

    def __getitem__(self, index):
        record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
        return self.get(record, segment_indices)

    def get(self, record, indices):

        images = list()
        list_frames = record.frames
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
#                 print('frames: ',list_frames[p])
                seg_imgs = self._load_image(record.path, list_frames[p])
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)
