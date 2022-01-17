import os, time, csv, shutil
import numpy as np
import matplotlib.pyplot as plt

from intervaltree import IntervalTree
from scipy.io import wavfile

extratest = ['2224.wav', '2241.wav', '1772.wav', '2502.wav', '2614.wav',
             '2398.wav', '2516.wav', '2202.wav', '1760.wav', '2307.wav',
             '1916.wav', '1790.wav', '2366.wav', '2302.wav', '2476.wav',
             '2436.wav', '1729.wav', '2540.wav', '2157.wav', '1775.wav',
             '2336.wav', '2403.wav', '2335.wav', '2151.wav', '1730.wav',
             '1791.wav', '2148.wav', '2397.wav', '2292.wav', '2473.wav']

for file in extratest:
    shutil.move('musicnet/train_data/' + file, 'musicnet/test_data/' + file)
    shutil.move('musicnet/train_labels/' + file[:-4] + '.csv', 'musicnet/test_labels/' + file[:-4] + '.csv')

validation = ['2390.wav', '2391.wav', '2341.wav', '2227.wav', '1755.wav', 
              '2244.wav', '2532.wav', '2560.wav', '2611.wav', '2575.wav', 
              '2117.wav', '1829.wav', '2422.wav', '2215.wav', '2178.wav', 
              '2119.wav', '2229.wav', '2632.wav', '2371.wav', '2383.wav', 
              '2491.wav', '2506.wav', '2140.wav', '2195.wav', '2393.wav', 
              '1818.wav', '2228.wav', '2075.wav', '1811.wav', '1788.wav', 
              '1807.wav', '2593.wav', '1735.wav', '2510.wav', '2373.wav', 
              '2627.wav', '2308.wav', '2156.wav', '2112.wav', '1931.wav']

os.mkdir('musicnet/valid_data')
os.mkdir('musicnet/valid_labels')
for file in validation:
    shutil.move('musicnet/train_data/' + file, 'musicnet/valid_data/' + file)
    shutil.move('musicnet/train_labels/' + file[:-4] + '.csv', 'musicnet/valid_labels/' + file[:-4] + '.csv')

d = 4096 # 4,096 sample context
splits = ['train', 'valid', 'test']

os.mkdir('minimusic')
for split in splits:
    t0 = time.time()
    print('Split', split)
    wavs = f'musicnet/{split}_data'
    labels = f'musicnet/{split}_labels'

    X, Y = [], []
    for i,file in enumerate(os.listdir(wavs)):
        print('.', end='')
        if not file.endswith('.wav'): continue
        fs, wav = wavfile.read(os.path.join(wavs, file))
        length = wav.shape[0]

        tree = IntervalTree()
        with open(os.path.join(labels,file[:-4] + '.csv'), 'r') as f:
            reader = csv.DictReader(f, delimiter=',')
            for label in reader:
                start_time = int(label['start_time'])
                end_time = int(label['end_time'])
                note = int(label['note'])
                tree[start_time:end_time] = note

        # take 250 samples from each record
        # cut first/last 10 seconds to avoid silence
        samples = np.linspace(10*fs, length - 10*fs, num=250).astype(np.int32)
        foo = 100
        for j, s in enumerate(samples):
            x = wav[s-d//2:s + d//2]
            foo = min(foo, np.max(np.abs(x)))
            x /= np.max(np.abs(x)) # normalize to [-1,1]
            X.append(x)

            label = [item[2] for item in tree[s]]
            y = np.zeros(128)
            y[label] = 1
            Y.append(y)

    X = np.stack(X)
    np.save(f'minimusic/audio-{split}.npy', X)
    Y = np.stack(Y)
    np.save(f'minimusic/labels-{split}.npy', Y)

    print('\nProcessed {} files in {} seconds'.format(i+1, time.time() - t0))
