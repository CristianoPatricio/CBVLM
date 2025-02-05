import numpy as np
import pickle
import os
import pandas as pd

class RICES:
    def __init__(self, cfg, valid_ids=None, annotations=None, mode=None) -> None:
        self.cfg = cfg
        
        with open(os.path.join("data", "image_features", f"{self.cfg.data.name}", f"{self.cfg.data.name}_{self.cfg.feature_extractor}_image_features_train.pkl"), "rb") as f:
            self.image_train_feature = pickle.load(f)
        
        with open(os.path.join("data", "image_features", f"{self.cfg.data.name}", f"{self.cfg.data.name}_{self.cfg.feature_extractor}_image_features_test.pkl"), "rb") as f:
            self.image_query_feature = pickle.load(f)
        
        self.valid_ids = valid_ids
        self.annotations = annotations
        if mode is not None:
            assert mode in ["global", "max", "mean"]
        self.mode = mode


    def get_context_keys(self, key, n, data_column=None):
        similarity = np.matmul(np.stack(list(self.image_train_feature.values()), axis=0), np.array(self.image_query_feature[key])).astype(np.float32)
        similarity_dict = {k: s for (k, s) in zip(self.image_train_feature.keys(), similarity)}
        sorted_similarity_dict = sorted(similarity_dict.items(), key=lambda x:x[1], reverse=True)
        
        ids = [x for (x, _) in sorted_similarity_dict]

        if len(self.valid_ids) > 0:
            ids = [x for x in ids if x in self.valid_ids]

        if data_column is not None:
            data = self.annotations[["image_id", data_column]].set_index("image_id")
            sim = pd.DataFrame(sorted_similarity_dict, columns=["image_id", "similarity"]).set_index("image_id")
            data = data.join(sim, on="image_id")
            data = data.reset_index()
            top_n_per_class = (
                data.sort_values(by=[data_column, "similarity"], ascending=[True, False])
                .groupby(data_column)
                .head(n)
            )
            if self.mode == "global":
                top_n_per_class = top_n_per_class.sort_values(by="similarity", ascending=False)
            else:
                top_n_per_class["order_sim"] = top_n_per_class.groupby(data_column)["similarity"].transform(self.mode)
                top_n_per_class = top_n_per_class.sort_values(by=["order_sim", "similarity"], ascending=[False, False])
            
            return top_n_per_class["image_id"].tolist()
        else:
            # normal RICES
            return ids[:n]
