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
import copy
import torch

class VideoRecord(object):
    def __init__(self, row):
        self._data = row
    
    @property
    def id(self):
        return self._data[0]
        
    @property
    def path(self):
        return self._data[1]
    
    @property
    def frames(self):
        return self._data[2]

    @property
    def num_frames(self):
        return self._data[3]

    @property
    def label(self):
        return self._data[4]
    
    @property
    def is_memory(self):
        return self._data[5]
    
    @property
    def global_id(self):
        return self._data[6]
    
    def set_frames(self, frames):
        self._data[2] = frames
        
    def set_num_frames(self, num_frames):
        self._data[3] = num_frames
        
    def set_is_memory(self, is_memory):
        self._data[5] = is_memory
    

class CILSetTask:
    def __init__(self, set_tasks, perc, path_frames, memory_size, batch_size, shuffle, num_workers, 
                 num_frame_to_save = 'ALL', is_activityNet = False, per_noise = 0, co_threshold = 0.3,
                 drop_last=False, pin_memory=False, num_segments=3, new_length=1, modality='RGB', 
                 transform=None, random_shift=True, test_mode=False, remove_missing=False, 
                 dense_sample=False, twice_sample=False, train_enable = True):
        
        self.is_activityNet = is_activityNet
        self.per_noise = per_noise
        self.co_threshold = co_threshold
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
        self.num_frame_to_save = num_frame_to_save
        self.perc = perc
        
        if self.is_activityNet:
            self.int_tasks()
        
    
    def get_videos(self):
        list_videos = {}
        for task in self.set_tasks:
            for class_n, videos in task.items():
                for video in videos: 
                    filename = video['filename']
                    video['label'] = class_n
                    if filename in list_videos:
                        list_videos[filename].append(video)
                    else:
                        list_videos[filename] = [video]   
        return list_videos
    
    
    def int_tasks(self):
        print('Noise: {}%'.format(self.per_noise*100))
        if self.per_noise > 0:
            list_videos = self.get_videos()
            num_vid = len(list_videos)
            num_aug_vid =  int(num_vid*self.per_noise)
            print('...Adding noise...')
            list_aug_vid = random.sample(list(range(num_vid)), num_aug_vid)

            new_list_vid = {}
            for i, (vid_name, actions) in enumerate(list_videos.items()):
                cover_actions = []
                list_labels = []
                if i in list_aug_vid:
                    for action in actions:
                        video_duration = float(action['video_duration'])
                        t_end = float(action['t_end'])
                        t_start = float(action['t_start'])
                        per_act = (t_end - t_start)/video_duration
                        cover_actions.append(per_act)
                        if not action['label'] in list_labels:
                            list_labels.append(action['label'])
                    ind = cover_actions.index(max(cover_actions))
                    action = actions[ind]
                    if cover_actions[ind] > self.co_threshold and len(list_labels) == 1:
                        action['t_start'] = str(0)
                        action['t_end'] = action['video_duration']
                        new_list_vid[vid_name] = [action]
                else:
                    new_list_vid[vid_name] = actions

            vid_class = {}
            idx = 0
            for vid_name, actions in new_list_vid.items():
                for action in actions:
                    action['filename'] = vid_name
                    action['id'] = idx
                    if action['label'] in vid_class:
                        vid_class[action['label']].append(action)
                    else:
                        vid_class[action['label']] = [action]
                    idx+=1

            new_Ntasks = []
            for task_i in self.set_tasks:
                task_n = {}
                for class_n, _ in task_i.items():
                    task_n[class_n] = vid_class[class_n]
                new_Ntasks.append(task_n)
            
            print('...Replacing training Data...')
            self.set_tasks = new_Ntasks
        else:
            print('...Adding ids...')
            idx = 0
            new_Ntasks = []
            for task in self.set_tasks:
                task_n = {}
                for class_n, videos in task.items():
                    for video in videos: 
                        video['id'] = idx
                        if class_n in task_n:
                            task_n[class_n].append(video)
                        else:
                            task_n[class_n] = [video]    
                        idx += 1
                new_Ntasks.append(task_n)
            print('...Replacing training Data...')
            self.set_tasks = new_Ntasks
        
    
    def __iter__(self):
        self.memory = {}
        self.current_task_dataset = None
        self.current_task = 0
        return self
    
    def get_dataloader(self, data, batch_size = 1, memory = None, sample_frame = False):
        num_frm = self.num_frame_to_save if sample_frame else 'ALL'
        is_memory = True if sample_frame else False
        data = self.get_frames(data, num_frm, is_memory)
        if memory != None:
            new_mem = self.get_frames(memory, self.num_frame_to_save, is_memory = True)
            data = {**new_mem, **data}
        dataset = TSNDataSet(self.path_frames, data, self.num_frame_to_save, True, None, self.num_segments, self.new_length, 
                             self.modality, self.transform, self.random_shift, self.test_mode, 
                             self.remove_missing, self.dense_sample, self.twice_sample)
        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = self.shuffle, 
                                num_workers = self.num_workers, pin_memory = self.pin_memory, drop_last = self.drop_last)
        
        return dataloader
    
    def set_memory(self, memory):
        self.memory = memory
    
    def sampling(self, data, num_sel_frames, is_memory):
        new_data = {}
        for class_n, videos in data.items():
            new_data[class_n] = []
            for video in videos:
                if self.is_activityNet:
                    video_name = video['filename']
                    idx = video['id']
                    start_f = round(float(video['t_start'])*float(video['fps']))
                    end_f = round(float(video['t_end'])*float(video['fps']))

                    path_video = os.path.join(self.path_frames, video_name)
                    frames = os.listdir(path_video)
                    frames.sort(key = lambda x: int(x.split('.')[0].replace('frame','')))
                    frames = frames[start_f:end_f]
                else:
                    video_name = video
                    idx = None
                    path_video = os.path.join(self.path_frames, class_n, video_name)
                    if not os.path.exists(path_video):
                        path_video = os.path.join(self.path_frames, video_name)
                    frames = os.listdir(path_video)
                    frames.sort(key = lambda x: int(x.split('.')[0].replace('frame','')))   
                
                num_frames = len(frames)
                if num_frames >= self.num_segments:
                    if num_sel_frames == 'ALL':
                        vid = {'path': path_video, 'frames': frames, 'is_memory': is_memory, 'id': idx}
                        new_data[class_n].append(vid)
                    else:
                        tick = (num_frames) / float(num_sel_frames)
                        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(num_sel_frames)])
                        frames = [frames[i] for i in offsets]
                        vid = {'path': path_video, 'frames': frames, 'is_memory': is_memory, 'id':idx}
                        new_data[class_n].append(vid)
        return new_data
        
#     def standard_sampling(self, data, num_sel_frames, is_memory):
#         new_data = {}
#         for class_n, videos in data.items():
#             new_data[class_n] = []
#             for video_name in videos: 
#                 path_video = os.path.join(self.path_frames, class_n, video_name)
#                 if not os.path.exists(path_video):
#                     path_video = os.path.join(self.path_frames, video_name)
#                 frames = os.listdir(path_video)
#                 frames.sort(key = lambda x: int(x.split('.')[0].replace('frame','')))
#                 num_frames = len(frames)
#                 if num_frames >= self.num_segments:
#                     if num_sel_frames == 'ALL':
#                         vid = {'path': path_video, 'frames': frames, 'is_memory': is_memory}
#                         new_data[class_n].append(vid)
#                     else:
#                         tick = (num_frames) / float(num_sel_frames)
#                         offsets = np.array([int(tick / 2.0 + tick * x) for x in range(num_sel_frames)])
#                         frames = [frames[i] for i in offsets]
#                         vid = {'path': path_video, 'frames': frames, 'is_memory': is_memory}
#                         new_data[class_n].append(vid)
#         return new_data
        
    def get_frames(self, data, num_sel_frames = 'ALL', is_memory = False):
        return self.sampling(data, num_sel_frames, is_memory)            
      
    def __next__(self):
        data = self.set_tasks[self.current_task]
        new_data = self.get_frames(data, is_memory = False)
        if self.train_enable:
            new_mem = self.get_frames(self.memory, self.num_frame_to_save, is_memory = True)
            comp_data = {**new_mem, **new_data}
        else:
            comp_data = new_data
            
        if self.current_task == 0:
            train_train_dataset = TSNDataSet(self.path_frames, comp_data, self.num_frame_to_save, False, None, self.num_segments,
                                          self.new_length, self.modality, self.transform, self.random_shift, self.test_mode, 
                                          self.remove_missing, self.dense_sample, self.twice_sample)
            len_train_data = len(train_train_dataset.video_list)
            train_train_dataloader = DataLoader(train_train_dataset, batch_size = self.batch_size, shuffle = self.shuffle, 
                                     num_workers = self.num_workers, pin_memory = self.pin_memory, drop_last = self.drop_last)
            
            self.current_task += 1
            return comp_data.keys(), data, train_train_dataloader, None, len_train_data, None, len(self.set_tasks[self.current_task].keys())
        
        else:
            train_train_data = {}
            train_val_data = {}
            for key, values in comp_data.items():
                total_data_value = len(values)
                len_train_train_data = int(total_data_value*self.perc)
                train_train_data[key] = values[:len_train_train_data]
                train_val_data[key] = values[len_train_train_data:]
                
            train_train_dataset = TSNDataSet(self.path_frames, train_train_data, self.num_frame_to_save, False, None, self.num_segments,
                                          self.new_length, self.modality, self.transform, self.random_shift, self.test_mode, 
                                          self.remove_missing, self.dense_sample, self.twice_sample)
            len_train_data = len(train_train_dataset.video_list)
            
            train_train_dataloader = DataLoader(train_train_dataset, batch_size = self.batch_size, shuffle = self.shuffle, 
                                     num_workers = self.num_workers, pin_memory = self.pin_memory, drop_last = self.drop_last)
            
            
            train_val_dataset = TSNDataSet(self.path_frames, train_val_data, self.num_frame_to_save, False, None, self.num_segments,
                                          self.new_length, self.modality, self.transform, self.random_shift, self.test_mode, 
                                          self.remove_missing, self.dense_sample, self.twice_sample)
            len_val_data = len(train_val_dataset.video_list)
            
            train_val_dataloader = DataLoader(train_val_dataset, batch_size = self.batch_size, shuffle = self.shuffle, 
                                        num_workers = self.num_workers, pin_memory = self.pin_memory, drop_last = self.drop_last)
            
            if self.current_task >=1 and self.current_task < len(self.set_tasks) - 1:
                self.current_task += 1
                return comp_data.keys(), data, train_train_dataloader, train_val_dataloader, len_train_data, len_val_data, len(self.set_tasks[self.current_task].keys())
            else:
                return comp_data.keys(), data, train_train_dataloader, train_val_dataloader, len_train_data, len_val_data, None
            
#         current_task_dataset = TSNDataSet(self.path_frames, comp_data, self.num_frame_to_save, False, None, self.num_segments,
#                                           self.new_length, self.modality, self.transform, self.random_shift, self.test_mode, 
#                                           self.remove_missing, self.dense_sample, self.twice_sample)
#         len_data = len(current_task_dataset.video_list)
#         self.current_task_dataloader = DataLoader(current_task_dataset, batch_size = self.batch_size, shuffle = self.shuffle, 
#                                      num_workers = self.num_workers, pin_memory = self.pin_memory, drop_last = self.drop_last)
            
#         self.current_task += 1
#         if self.current_task < len(self.set_tasks):
#             return comp_data.keys(), data, self.current_task_dataloader, len_data, len(self.set_tasks[self.current_task].keys())
#         else:
#             return comp_data.keys(), data, self.current_task_dataloader, len_data, None

    def get_valSet_by_taskNum(self, num_task):
        eval_data = {}
        total_data = []
        list_val_loaders = []
        list_num_classes = []
        for k in range(num_task):
            data = self.set_tasks[k]
            eval_data = {**eval_data, **data}
            new_data = self.get_frames(data)
            total_data.append(new_data)
            list_num_classes.append(len(data.keys()))
        classes = eval_data.keys()
        for i, data_i in enumerate(total_data):
            val_task_dataset = TSNDataSet(self.path_frames, data_i, self.num_frame_to_save, True, classes, self.num_segments,
                                          self.new_length, self.modality, self.transform, self.random_shift, self.test_mode, 
                                          self.remove_missing, self.dense_sample, self.twice_sample)
            val_task_dataloader = DataLoader(val_task_dataset, batch_size = self.batch_size, shuffle = self.shuffle, 
                                             num_workers = self.num_workers, pin_memory = self.pin_memory, 
                                             drop_last = self.drop_last)
            list_val_loaders.append((val_task_dataloader, list_num_classes[i]))
        return list_val_loaders
        

class TSNDataSet(data.Dataset):
    def __init__(self, path_frames, data, num_fram_self, inference, classes=None, 
                 num_segments=3, new_length=1, modality='RGB', 
                 transform=None, random_shift=True, test_mode=False,
                 remove_missing=False, dense_sample=False, twice_sample=False):

        self.path_frames = path_frames
        self.data = data
        self.classes = classes if classes != None else data.keys()
        self.num_segments = num_segments
        self.num_fram_self = num_fram_self
        self.inference = inference
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
        id_vid = 0
        for class_name, videos in self.data.items():
            for vid in videos: 
                frames = vid['frames']
                path_video = vid['path']
                global_id = vid['id'] if 'id' in vid else None
                num_frames = len(frames)
#                 if num_frames >= self.num_segments:
                item = [id_vid, path_video, frames, num_frames, class2label[class_name], vid['is_memory'], global_id]
                self.video_list.append(VideoRecord(item))
                id_vid+=1

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
#                 print('normal. Len frm: {}, is_memory: {}'.format(record.num_frames, record.is_memory))
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
            elif record.is_memory:
#                 print('Is memory: {} Frames: {}'.format(record.is_memory, record.frames))
                new_num_segments = self.num_segments if record.num_frames - self.new_length + 1 >= self.num_segments else record.num_frames - self.new_length + 1
                average_duration = (record.num_frames - self.new_length + 1) // new_num_segments
                offsets = np.multiply(list(range(new_num_segments)), average_duration) + randint(average_duration, size=new_num_segments)
                num_repeat = self.num_segments / new_num_segments
                offsets = np.repeat(offsets, num_repeat)
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
            if record.num_frames >= self.num_segments + self.new_length - 1:
                tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            elif record.is_memory:
                new_num_segments = self.num_segments if record.num_frames - self.new_length + 1 >= self.num_segments else record.num_frames - self.new_length + 1
                tick = (record.num_frames - self.new_length + 1) / float(new_num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(new_num_segments)])
                num_repeat = self.num_segments / new_num_segments
                offsets = np.repeat(offsets, num_repeat)
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
            id_vid, video_name, process_data, label = self.get(record, segment_indices)
            process_data_aug = torch.zeros_like(process_data)
            if (not self.inference) and (self.num_fram_self == 'ALL' or len(record.frames) > self.num_fram_self) and (self.num_fram_self != 0): 
                new_record = copy.deepcopy(record)
                if self.num_fram_self != 'ALL':
                    frames = new_record.frames
                    num_frames = len(frames)
                    tick = (num_frames) / float(self.num_fram_self)
                    offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_fram_self)])
                    frames = [frames[i] for i in offsets]
    #                         print('num frames trimmed: ',len(frames))
                    new_record.set_frames(frames)
                    new_record.set_num_frames(len(frames))
                    new_record.set_is_memory(True)
                aug_segment_indices = self._sample_indices(new_record)
                _, _, process_data_aug, _ = self.get(record, aug_segment_indices)
            return id_vid, video_name, process_data, process_data_aug, label
        else:
            segment_indices = self._get_test_indices(record)
            id_vid, video_name, process_data, label = self.get(record, segment_indices)
            process_data_aug = torch.zeros_like(process_data)
            return id_vid, video_name, process_data, process_data_aug, label

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
        video_name = record.path.split('/')[-1]
        if record.global_id != None: 
            video_name = '{}_{}'.format(video_name, record.global_id)
        
        return record.id, video_name, process_data, record.label

    def __len__(self):
        return len(self.video_list)
