import os
import argparse
import numpy as np

from encoder import *
from utils import traverse_dir

def load_events_per_bar(path_infile, bar_token):
    bars = []
    
    with open(path_infile) as f:    
        b = []
        for idx in f.read().split():
            b.append(int(idx))
            if int(idx) == bar_token:
                bars.append(b)
                b = []
   
        # Add end event to last bar
        bars[-1] += b

    return bars

def load_events(path_infile):
    events_idx = []
    with open(path_infile) as f:
        events_idx = [int(idx) for idx in f.read().split()]
    return events_idx

def compile(path_indir, max_len, task='language_modeling'):
    # list files
    txtfiles = traverse_dir(
        path_indir,
        is_pure=True,
        is_sort=True,
        extension=("txt"))
    n_files = len(txtfiles)
    print('num files:', n_files)

    # Get pad token
    pad_token = Event(event_type='control', value=3).to_int()
    bar_token = Event(event_type='control', value=1).to_int()

    pieces = []
    labels = []
    for fidx in range(n_files):
        path_txt = txtfiles[fidx]
        print('{}/{}'.format(fidx, path_txt))

        # The 'path_txt' start with 'songNum_phraseNum_'.
        # Each piece whose name started with "songNum_phraseNum_0" will be used as theme, 
        # which will be concated with all other pieces with the same song number and phrase number.

        # Split tokens in 'path_txt' to check if the current piece is a theme or not. 
        theme_split_name = path_txt.split("_")
        if(theme_split_name[2] == '0'):
            theme_path_infile = os.path.join(path_indir, path_txt)
            # Load the theme piece.
            theme = load_events(theme_path_infile)
            # Loop to find all variations.
            for varidx in range(n_files):
                var_path_txt = txtfiles[varidx]
                var_split_name = var_path_txt.split("_")
                if(var_split_name[0] == theme_split_name[0] and var_split_name[1] == theme_split_name[1] and var_split_name[2] != theme_split_name[2]):
                    # Prevent theme and variation pair with different key and tempo transposing.
                    suffix_theme_name = ''
                    for i in range (3, len(theme_split_name)):
                        if(i == 3):
                            suffix_theme_name += theme_split_name[i]
                        else:
                            suffix_theme_name += ('_' + theme_split_name[i])
                    suffix_var_name = ''
                    for i in range (3, len(var_split_name)):
                        if(i == 3):
                            suffix_var_name += var_split_name[i]
                        else:
                            suffix_var_name += ('_' + var_split_name[i])
                    # if((var_split_name[3] == 'original' and var_split_name[3] == theme_split_name[3]) or (var_split_name[3] != 'original' and var_split_name[3] == theme_split_name[3] and var_split_name[4] == theme_split_name[4])):
                    if(suffix_var_name == suffix_theme_name):
                        var_path_infile = os.path.join(path_indir, var_path_txt)
                        # Load variation piece.
                        variation = load_events(var_path_infile)
                        # Concat theme, '520' (sep token), and variation.
                        # We set max_len = 512.
                        sequence_theme = theme[:max_len]
                        sequence_theme += [pad_token] * (max_len - len(theme))
                        sequence_variation = variation[:max_len]
                        sequence_variation += [pad_token] * (max_len - len(variation))
                        sequence = sequence_theme + [520] + sequence_variation
                        pieces.append(sequence)

    pieces = np.vstack(pieces)

    if task != 'language_modeling':
        labels = np.vstack(labels)
        assert pieces.shape[0] == labels.shape[0]
        return pieces, labels
    
    return pieces

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='compile.py')
    parser.add_argument('--path_train_indir', type=str, required=True)
    parser.add_argument('--path_test_indir', type=str, required=True)
    parser.add_argument('--path_outdir', type=str, required=True)
    parser.add_argument('--max_len', type=int, required=True)
    parser.add_argument('--task', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.path_outdir, exist_ok=True)

    # Load datasets
    if args.task == 'emotion_classification' or args.task == 'discriminator':
        train_pieces, train_labels = compile(args.path_train_indir, args.max_len, args.task)
        test_pieces, test_labels = compile(args.path_test_indir, args.max_len, args.task)

        print('---')
        print(' > train x:', train_pieces.shape)
        print(' > train y:', train_labels.shape)
        print(' >  test x:', test_pieces.shape)
        print(' >  test y:', test_labels.shape)

        # Save datasets
        path_train_outfile = os.path.join(args.path_outdir, args.task + '_train.npz')
        path_test_outfile = os.path.join(args.path_outdir, args.task + '_test.npz')

        np.savez(path_train_outfile, x=train_pieces, y=train_labels)
        np.savez(path_test_outfile, x=test_pieces, y=test_labels)
    elif args.task == 'language_modeling':
        train_pieces = compile(args.path_train_indir, args.max_len, args.task)
        test_pieces = compile(args.path_test_indir, args.max_len, args.task)

        print('---')
        print(' > train x:', train_pieces.shape)
        print(' >  test x:', test_pieces.shape)

        # Save datasets
        path_train_outfile = os.path.join(args.path_outdir, args.task + '_train.npz')
        path_test_outfile = os.path.join(args.path_outdir, args.task + '_test.npz')

        np.savez(path_train_outfile, x=train_pieces)
        np.savez(path_test_outfile, x=test_pieces)
    else:
        raise ValueError('Invalid task.')