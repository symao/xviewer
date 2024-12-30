#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2016-2023 maoshuyuan123@gmail.com. All Rights Reserved.

import os
import sys
import cv2
import yaml
import numpy as np
from tqdm import tqdm

''' @brief load data from single file '''
class FileData():
    def __init__(self, file=None, colnames=None):
        self.ts = None
        self.col_dict = {}  # <col_name, col_data>
        if file is not None and os.path.exists(file):
            self.__load_from_file(file, colnames)

    def __str2num(self, s):
        if s[:2] == '0x':
            return int(s, 16)
        elif '.' in s:
            return float(s)
        else:
            return int(s)

    def __colname_prune(self, s):
        ret = s.strip().split(' ')[0].strip()
        for c in '[]()*&^%$+-/':
            ret = ret.replace(c, '_')
        return ret

    def __find_ts_col(self, colnames, data):
        # find ts from colnames
        for i, colname in enumerate(colnames):
            s = colname.lower()
            if s in ['ts', 'timestamp', 't', 'time', 'timestamps']:
                return i
            for tar in ['timestamp', 'time', 'timestamps']:
                if tar in s:
                    return i
        # try use the first col as ts col if it's positive and incremental
        if np.all(data[:, 0] > 0) and np.all((data[1:, 0]-data[:-1, 0]) > 0):
            return 0
        return -1

    def __load_from_file(self, file, colnames=None):
        """ Load data from file, deduce colnames from first comment line if input colnames is None or empty.
            deduce timestamp col from colnames, if no colnames, try to use the first col as timestamp.
        """
        lines = open(file).readlines()[:-1]  # drop last line which may be incompleted
        first_line = lines[0].strip()
        spliter = ',' if ',' in first_line else ' '
        data = np.array([[self.__str2num(i) for i in l.strip().split(spliter)] for l in lines[1:] if '#' not in l])
        data_cols = data.shape[1]

        # find colnames for each data col
        if colnames is None or len(colnames) == 0:
            if first_line.startswith('#'):
                colnames = [self.__colname_prune(x) for x in first_line.replace('#', '').split(spliter)]
            else:
                colnames = ['col%d' % i for i in range(data_cols)]
        if colnames is not None and len(colnames) != data_cols:
            print('[WARN]colnames %s not match data cols(%d), use default colnames' % (str(colnames), data_cols))
            colnames = ['col%d' % i for i in range(data_cols)]

        # find the ts col
        ts_col_idx = self.__find_ts_col(colnames, data)

        # set data to output
        if ts_col_idx >= 0:
            self.ts = data[:, ts_col_idx]
            self.col_dict = dict([(name, data[:, i]) for i, name in enumerate(colnames)])
        else:
            self.ts = np.arange(data.shape[0], dtype=float)
            self.col_dict = dict([(name, data[:, i]) for i, name in enumerate(colnames)])

class ImageLoaderEuroc():
    def __init__(self, img_dir, ts_file, start_idx=0, stop_idx=-1, idx_step=1):
        if not os.path.exists(ts_file):
            print("[ERROR]cannot open file: %s\n" % ts_file)
            self.ok = False
            return
        self.img_dir = img_dir
        self.start_idx = start_idx
        self.idx_step = idx_step
        self.lines = [l for l in open(ts_file).readlines() if not l.strip().startswith('#')]
        self.spliter = ',' if len(self.lines) > 0 and ',' in self.lines[0] else ' '
        frame_cnt = len(self.lines)
        self.stop_idx = frame_cnt if stop_idx <= 0 else min(frame_cnt, stop_idx)
        self.reset()
        self.ok = self.stop_idx >= self.start_idx

    def reset(self):
        self.cur_idx = self.start_idx

    # return ts,img
    def read_next(self, grayscale=False):
        if self.cur_idx >= self.stop_idx:
            return None, None
        ts, fimg = [x.strip() for x in self.lines[self.cur_idx].strip().split(self.spliter)[:2]]
        ts = float(ts)
        img = cv2.imread(os.path.join(self.img_dir, fimg), cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
        self.cur_idx += self.idx_step
        return ts, img

    def cnt(self):
        return (self.stop_idx - self.start_idx) // self.idx_step


class ImageLoaderVideo():
    def __init__(self, video_file, ts_file, ts_col=0, start_idx=0, stop_idx=-1, idx_step=1):
        self.cap = cv2.VideoCapture(video_file)
        if not self.cap.isOpened():
            self.ok = False
            return
        self.start_idx = start_idx
        self.idx_step = idx_step
        self.ts_list = [float(l.strip().split(',')[ts_col])
                        for l in open(ts_file).readlines() if not l.strip().startswith('#')]
        frame_cnt = len(self.ts_list)
        self.stop_idx = frame_cnt if stop_idx <= 0 else min(frame_cnt, stop_idx)
        self.reset()
        self.ok = self.stop_idx >= self.start_idx

    def reset(self):
        self.cur_idx = self.start_idx

    # return ts,img
    def read_next(self, grayscale=False):
        if self.cur_idx >= self.stop_idx:
            return None, None
        # read image
        ret, img = self.cap.read()
        ts = self.ts_list[self.cur_idx]
        self.cur_idx += 1
        if not ret or img is None:
            return None, None
        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # skip stride
        for _ in range(self.idx_step-1):
            self.cap.read()
            self.cur_idx += 1

        return ts, img

    def cnt(self):
        return (self.stop_idx - self.start_idx) // self.idx_step


class ImageLoaderImageRaw():
    def __init__(self, raw_file, ts_file, img_row, img_col, img_channel=1, ts_col=0, start_idx=0, stop_idx=-1, idx_step=1):
        if not os.path.exists(raw_file) or not os.path.exists(ts_file):
            self.ok = False
            return
        self.bin = open(raw_file, 'rb')
        self.ts_list = [float(l.strip().split(',')[ts_col])
                        for l in open(ts_file).readlines() if not l.strip().startswith('#')]
        self.img_row = img_row
        self.img_col = img_col
        self.img_channel = img_channel
        self.img_size = img_row * img_col * img_channel
        frame_cnt = len(self.ts_list)
        img_cnt = os.path.getsize(raw_file) // self.img_size
        frame_cnt = min(frame_cnt, img_cnt)
        self.start_idx = start_idx
        self.idx_step = idx_step
        self.stop_idx = frame_cnt if stop_idx <= 0 else min(frame_cnt, stop_idx)
        self.reset()
        self.ok = self.stop_idx >= self.start_idx

    def reset(self):
        self.cur_idx = self.start_idx

    # return ts,img
    def read_next(self, grayscale=False):
        if self.cur_idx >= self.stop_idx:
            return None, None
        # load data
        data = self.bin.read(self.img_size)
        ts = self.ts_list[self.cur_idx]
        self.cur_idx += 1
        # convert to image
        if data is None or len(data) != self.img_size:
            return None, None
        else:
            data = np.array([int(x) for x in data], dtype='uint8')
            if self.img_channel == 1:
                img = np.reshape(data, (self.img_row, self.img_col))
                if not grayscale:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img = np.reshape(data, (self.img_row, self.img_col, self.img_channel))
                if grayscale:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # skip stride
        for _ in range(self.idx_step-1):
            self.bin.read(self.img_size)
            self.cur_idx += 1
        return ts, img

    def cnt(self):
        return (self.stop_idx - self.start_idx) // self.idx_step


class DataParser():
    def __init__(self):
        self.reset()

    def reset(self):
        self.file_dict = {}  # <file_name, FileData>
        self.img_dict = {}  # <image_name, image_list>

    def load_file(self, file):
        try:
            file_data = FileData(file)
            self.__auto_adjust_ts(file_data)
        except:
            print('[ERROR]failed load input file:%s, only support pure numeric (no string) table file.' % file)
            return
        self.file_dict[os.path.basename(file)] = file_data  # read file data

    def __auto_adjust_ts(self, file_data):
        # auto convert timestamp from nanoseconds to seconds
        # if ts duration beyond 1e7, we assume it's in nanoseconds
        if file_data.ts[-1] - file_data.ts[0] > 1e7:
            file_data.ts /= 1e9

    def __read_img_from_cfg(self, cfg, data_dir):
        type = cfg['type']
        path = cfg['path']
        ts_path = cfg['ts_path']
        ts_col = cfg['ts_col'] if 'ts_col' in cfg else 0
        ts_multiplier = float(cfg['ts_multiplier']) if 'ts_multiplier' in cfg else 1.0
        roi = [float(x) for x in cfg['roi']]
        resize_rate = float(cfg['resize_rate']) if 'resize_rate' in cfg else 1.0
        start_idx = int(cfg['start_idx']) if 'start_idx' in cfg else 0
        stop_idx = int(cfg['stop_idx']) if 'stop_idx' in cfg else -1
        idx_step = int(cfg['idx_step']) if 'idx_step' in cfg else 1
        grayscale = int(cfg['grayscale']) if 'grayscale' in cfg else 1
        max_load_mem = float(cfg['max_load_mem']) if 'grayscale' in cfg else 1e10
        img_loader = None
        if type == 'image':
            img_loader = ImageLoaderEuroc(os.path.join(data_dir, path), ts_file=os.path.join(data_dir, ts_path),
                                          start_idx=start_idx, stop_idx=stop_idx, idx_step=idx_step)
        elif type == 'video':
            img_loader = ImageLoaderVideo(os.path.join(data_dir, path), ts_file=os.path.join(
                data_dir, ts_path), ts_col=ts_col, start_idx=start_idx, stop_idx=stop_idx, idx_step=idx_step)
        elif type == 'image_raw':
            img_col = int(cfg['img_col'])
            img_row = int(cfg['img_row'])
            img_channel = int(cfg['img_channel']) if 'img_channel' in cfg else 1
            img_loader = ImageLoaderImageRaw(os.path.join(data_dir, path),
                                             os.path.join(data_dir, ts_path),
                                             img_row, img_col, img_channel=img_channel, ts_col=ts_col,
                                             start_idx=start_idx, stop_idx=stop_idx, idx_step=idx_step)
        if img_loader is None or not img_loader.ok:
            return []

        print('loading images...')
        img_data = []  # output
        sum_img_mem = 0
        for _ in tqdm(range(img_loader.cnt())):
            ts, img = img_loader.read_next(grayscale)
            if ts is None or img is None:
                break
            # post process
            ts *= ts_multiplier
            if resize_rate != 1.0:
                img = cv2.resize(img, None, fx=resize_rate, fy=resize_rate)
            if len(roi) == 4:
                img_rows, img_cols = img.shape[:2]
                x0, y0, w, h = roi
                x0, y0, w, h = int(img_cols*x0), int(img_rows*y0), int(img_cols*w), int(img_rows*h)
                img = img[y0:h, x0:w]
            img_data.append((ts, img))
            sum_img_mem += np.prod(img.shape)
            if sum_img_mem > max_load_mem * 1e6:
                print("WARN: stop load image since out of memory. If you still want to load all image, try increase 'max_load_mem' or decrease 'resize_rate' in format.yaml")
                break
        return img_data

    def __read_file_from_cfg(self, cfg, data_dir):
        path = cfg['path']
        abs_path = os.path.join(data_dir, path)
        if not os.path.exists(abs_path) or not os.path.isfile(abs_path):
            return None
        colnames = cfg['col_names'] if 'col_names' in cfg else []
        if 'ts_multiplier' in cfg:
            ts_multiplier = float(cfg['ts_multiplier'])
            file_data = FileData(abs_path, colnames)
            if ts_multiplier != 1:
                file_data.ts *= ts_multiplier
        else:
            file_data = FileData(abs_path, colnames)
            self.__auto_adjust_ts(file_data)
        return file_data

    def load_dir(self, cfg_file, data_dir=None):
        print('open data dir by config:', cfg_file)
        if cfg_file is None or not os.path.exists(cfg_file):
            print('[ERROR]cfg file not exist %s' % str(cfg_file))
            return
        cfg_dict = yaml.load(open(cfg_file, encoding='UTF-8'), Loader=yaml.FullLoader)
        if data_dir is None:
            if 'data_dir' in cfg_dict:
                data_dir = cfg_dict['data_dir']
            else:
                print("[ERROR]No data dir found.")
                return
        for name, cfg in cfg_dict.items():
            if not isinstance(cfg, dict) or not 'type' in cfg:
                continue
            if cfg['type'] in ['image', 'video', 'image_raw']:
                img_list = self.__read_img_from_cfg(cfg, data_dir)
                if len(img_list) > 0:
                    self.img_dict[name] = img_list
                else:
                    print("[ERROR]Load image data '%s' failed.\n" % name)
            elif cfg['type'] == 'file':
                file_data = self.__read_file_from_cfg(cfg, data_dir)
                if file_data is not None:
                    self.file_dict[name] = file_data

    def load(self, input, cfg_file=None):
        self.reset()
        if isinstance(input, list):
            for i in input:
                self.load_file(i)
        elif isinstance(input, str):
            if os.path.isfile(input):
                self.load_file(input)
            elif os.path.isdir(input):
                self.load_dir(cfg_file, input)
        else:
            print('[ERROR]DataParser: unsupport input.')

    def get_files(self):
        return sorted(self.file_dict.keys())

    def get_variables(self, filename=None):
        variables = []
        if filename is None:
            for filename, file_data in self.file_dict.items():
                for colname in file_data.col_dict.keys():
                    variables.append((filename, colname))
        elif filename in self.file_dict:
            for colname in self.file_dict[filename].col_dict.keys():
                variables.append((filename, colname))
        return variables

    def get_variable_data(self, filename, colname):
        if filename in self.file_dict and colname in self.file_dict[filename].col_dict:
            return self.file_dict[filename].col_dict[colname]
        return None

    def get_variable_datas(self, variable_list):
        return [self.get_variable_data(filename, colname) for filename, colname in variable_list]


if __name__ == '__main__':
    a = DataParser()
    a.load(sys.argv[1:])
    variable_list = a.get_variables()
    print(variable_list[:2], a.get_variable_datas(variable_list[:2]))
