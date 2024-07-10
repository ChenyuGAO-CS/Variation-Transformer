import torch
import numpy as np

class Emopia(torch.utils.data.Dataset):
    def __init__(self, path_data, pad_token):
        data = np.load(path_data)
        self.pieces = torch.from_numpy(data['x']).long()
        self.pad_token = pad_token

    def __getitem__(self, idx):
        x = self.pieces[idx][:-1]
        y = self.pieces[idx][1:]
        theme_var_tokens_per_bar = self.obtain_theme_var_tokens_per_bar(x)
        theme_length = self.theme_length(x)
        theme_var_encoding = self.theme_var_encoding(x, theme_var_tokens_per_bar, theme_length)
        # import pdb; pdb.set_trace()
        return x, y, theme_var_encoding

    def __len__(self):
        return len(self.pieces)

    def obtain_theme_var_tokens_per_bar(self, x):
        bar_array = []
        cnt_tokens = 0
        for i in range (len(x)):
            if x[i] == 520:
                # '520' is the seperate token.
                cnt_tokens = 0
                bar_array.append(-1)
            elif x[i] == 440:
                # ‘440’ is the bar token, which appears at the begining of a new bar.
                bar_array.append(cnt_tokens)
                cnt_tokens = 0
            elif x[i] == 442:
                # '442' is the pad token.
                continue
            cnt_tokens = cnt_tokens + 1
        return bar_array

    def theme_length(self, x):
        cnt_tokens = 0
        for i in range (len(x)):
            if x[i] == 520:
                # '520' is the seperate token.
                break
            cnt_tokens = cnt_tokens + 1
        return cnt_tokens

    # We assume a bar of generation content has the largest similarity 
    #    with the music of the corresponding bar in the theme. 
    # According to this assumption, we could add an additional signal in attention calculation 
    #    to force the model to pay more attention to the corresponding measure in the theme 
    #    when generating the current measure of variation.
    def theme_var_encoding(self, x, theme_var_tokens_per_bar, theme_length):
        # Init a blank 2-D array.
        length_x = len(x)
        theme_var_array = np.zeros((length_x, length_x), dtype = np.int32)
        
        theme_idx = 0
        var_idx = 0
        # Get the exact index for the first bar of variation in the 'theme_var_tokens_per_bar' array.
        for i in range(len(theme_var_tokens_per_bar)):
            if theme_var_tokens_per_bar[i] == -1:
                var_idx = i + 1
                break
        # Fill the theme_var_array encoding according to 
        #     the position of each bar of the variation and that of the corresponding bar in the theme.
        theme_position = 0
        var_position = theme_length + 1
        for i in range(var_idx, len(theme_var_tokens_per_bar)):
            if theme_idx > var_idx - 1:
                break
            tmp_var_end_position = var_position + theme_var_tokens_per_bar[i]
            tmp_theme_end_position = theme_position + theme_var_tokens_per_bar[theme_idx]
            theme_var_array[var_position : tmp_var_end_position, theme_position: tmp_theme_end_position] = 1
            
            # Update the beginning position of theme and variation
            theme_position = tmp_theme_end_position
            var_position = tmp_var_end_position
            theme_idx = theme_idx + 1
        return theme_var_array