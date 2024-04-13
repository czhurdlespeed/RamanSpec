from functools import wraps
import torch.utils.data as data_util

__all__ = ['RandomSplit']

class RandomSplit:
    def __init__(self, splitRatios: list = (4, 1), isRandomSplit=True, batchSize=8) -> None:
        self.isRandomSplit = isRandomSplit
        self.batchSize = batchSize
        self.splitRatios = splitRatios

    def __call__(self, func):
        @wraps(func)
        def randomSplitIt(*args, **kwargs):
            sizeList = []
            dataset = func(*args, **kwargs)
            if self.isRandomSplit:
                dataset_length = len(dataset)
                for i in range(len(self.splitRatios) - 1):
                    sizeList.append(int((self.splitRatios[i] / sum(self.splitRatios)) * dataset_length))
                sizeList.append(dataset_length - sum(sizeList))
                subDatasets = data_util.random_split(dataset, sizeList)
                return [data_util.DataLoader(subset, shuffle=True, batch_size=self.batchSize) for subset in subDatasets]
            else:
                return data_util.DataLoader(dataset, shuffle=False, batch_size=self.batchSize)

        return randomSplitIt
