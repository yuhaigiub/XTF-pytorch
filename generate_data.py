from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pandas as pd
from util import load_pickle

import os
import argparse

def generate_graph_seq2seq_io_data(data: np.ndarray, 
                                   x_offsets, 
                                   y_offsets,
                                   df_index=None, 
                                   add_time_in_day=True,
                                   add_day_in_week=False):
    '''
        data: [n_samples, num_nodes, channels]
        return:
            x: [size, seq_len, num_nodes, in_channels]
            y: [size, seq_len, num_nodes, out_channels]
    '''
    num_samples, num_nodes, _ = data.shape
    data_list = [data]
    
    if add_time_in_day:
        try:
            time_ind = (df_index.values - df_index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
            time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose(2, 1, 0)
            data_list.append(time_in_day) 
            print('added time in day feature')
        except:
            print('fail to add time in day')
    if add_day_in_week:
        try:
            day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
            day_in_week[np.arange(num_samples), :, df_index.dayofweek] = 1
            data_list.append(day_in_week)
            print('added day in week feature')
        except:
            print('fail to add day in week')
    
    data = np.concatenate(data_list, axis=-1)
    
    x, y = [], []
    
    # t is the index of the last observation
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))
    
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    
    return x, y

def main(args, train_ratio=0.7, test_ratio=0.2):
    
    print("Generating training data:")
    print("Reading input file:", args.data_file_path)
    if args.file_type == "h5":
        df = pd.read_hdf(args.data_file_path)
        data = np.expand_dims(df.values, axis=-1)
    elif args.file_type == "npz":
        data = np.load(args.data_file_path)['data']
    elif args.file_type == "pkl":
        data = load_pickle(args.data_file_path)['processed_data']
    else:
        raise Exception("Invalid file type")
    
    # 0 is the latest observed sample
    x_offsets = np.sort(np.arange(-11, 1, 1))
    # predict the next one hour (12 steps)
    y_offsets = np.sort(np.arange(1, 13, 1))
    
    x, y = generate_graph_seq2seq_io_data(data,
                                          x_offsets,
                                          y_offsets,
                                          df_index= df.index if args.file_type == 'h5' else None,
                                          add_time_in_day=args.dataset in ['METR-LA', 'PEMS-BAY'],
                                          add_day_in_week=False)
    print("x shape:", x.shape)
    print("y shape:", y.shape)
    
    # write data into npz file
    
    num_samples = x.shape[0]
    num_test = round(num_samples * test_ratio)
    num_train = round(num_samples * train_ratio)
    num_val = num_samples - num_test - num_train
    
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = x[num_train: num_train + num_val], y[num_train:num_train + num_val]
    x_test, y_test = x[-num_test:], y[-num_test:]
    
    print("Writing output file:", args.output_dir)
    
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x:", _x.shape, "y:", _y.shape)
        np.savez_compressed(os.path.join(args.output_dir, "%s.npz" % cat), 
                            x=_x, 
                            y=_y,
                            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
                            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]))
                            
    

DATASET = ["METR-LA", "PEMS-BAY", "PEMS03", "PEMS04"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='METR-LA', choices=DATASET)
    
    args = parser.parse_args()
    
    class MyArgs():
        def __init__(self):
            self.dataset = args.dataset
            if self.dataset in DATASET[0:2]:
                self.file_type = 'h5'
            elif self.dataset in DATASET[2:]:
                self.file_type = 'npz'
            else:
                self.file_type = 'pkl'
            self.data_file_path = './store/raw/{}/data.{}'.format(self.dataset, self.file_type)
            self.output_dir = './store/{}'.format(args.dataset)
    args = MyArgs()
    
    if args.dataset in DATASET[0:2]:
        train_ratio = 0.7
        test_ratio = 0.2
    else:
        train_ratio = 0.6
        test_ratio = 0.2
    print('ratios (train, test, val):', train_ratio, test_ratio, 1 - (train_ratio + test_ratio))
    main(args, train_ratio, test_ratio)