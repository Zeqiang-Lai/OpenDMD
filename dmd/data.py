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
        with open(path, "r") as f:
            self.captions = f.readlines()

    def __getitem__(self, index):
        return self.captions[index]

    def __len__(self):
        return len(self.captions)


class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, data_root):
        super().__init__()
        self.data_root = data_root

        self.meta_list = self.collect_meta(data_root)

    def collect_meta(self, data_root):
        meta = []
        for name in os.listdir(data_root):
            if name.endswith(".json"):
                with open(os.path.join(data_root, name), "r") as f:
                    for line in f.readlines():
                        meta.append(json.loads(line))
        return meta

    def __getitem__(self, index):
        meta = self.meta_list[index]
        prompt = meta["prompt"]
        latents = torch.load(os.path.join(self.data_root, meta["latent_path"]), map_location="cpu")
        latents = latents[0].float().numpy()
        images = imageio.imread(os.path.join(self.data_root, meta["image_path"])).transpose(2, 0, 1)
        images = images.astype("float32") / 255.0
        return latents, images, prompt

    def __len__(self):
        return len(self.meta_list)
