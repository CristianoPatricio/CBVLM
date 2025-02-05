import random

class RANDOM:
    def __init__(self, cfg, valid_ids, annotations=None) -> None:
        self.cfg = cfg
        self.valid_ids = valid_ids
        self.annotations = annotations

    def get_context_keys(self, key, n, data_column=None):
        ids = self.valid_ids.copy()

        if data_column is not None:
            data = self.annotations[self.annotations["image_id"].isin(ids)]

            n_per_class = (
                data.groupby(data_column)
                .head(n)
                .sample(frac=1)
            )
            return n_per_class["image_id"].tolist()
        else:
            # random
            random.shuffle(ids)
            return ids[:n]
