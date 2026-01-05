class TemporalDataLoader:
    def __init__(self, dataset):
        self.dataset = dataset
        self.snapshot_count = dataset.snapshot_count

    def __len__(self):
        return self.snapshot_count

    def __iter__(self):
        for i in range(len(self)):
            yield self.dataset[i]
