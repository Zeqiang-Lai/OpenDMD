import torch.utils.data


def cycle(dl):
    while True:
        for data in dl:
            yield data


class TextDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __getitem__(self, index):
        return super().__getitem__(index)

    def __len__(self):
        return 0


class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __getitem__(self, index):
        return super().__getitem__(index)

    def __len__(self):
        return 0
