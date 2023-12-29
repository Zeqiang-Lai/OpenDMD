import torch.utils.data
import json
import os
import imageio


def cycle(dl):
    while True:
        for data in dl:
            yield data


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        super().__init__()
        with open(path, 'r') as f:
            self.captions = f.readlines()

    def __getitem__(self, index):
        return self.captions[index]

    def __len__(self):
        return len(self.captions)


class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, data_root):
        super().__init__()
        self.data_root = data_root
        with open(os.path.join(data_root, 'meta.json'), 'r') as f:
            self.meta_list = json.load(f)

    def __getitem__(self, index):
        meta = self.meta_list[index]
        caption = meta['caption']
        latents = torch.load(os.path.join(self.data_root, meta['latents_path']), map_location='cpu')
        latents = latents[0].float().numpy()
        images = imageio.imread(os.path.join(self.data_root, meta['images_path'])).transpose(2, 0, 1)
        images = images.astype('float32') / 255.0
        return latents, images, caption

    def __len__(self):
        return len(self.meta)
