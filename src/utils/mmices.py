import numpy as np
import pickle
import os

class MMICES:
    def __init__(self, cfg, valid_ids=None, K=100, train_concepts=None) -> None:
        self.cfg = cfg
        
        with open(os.path.join("data", "image_features", f"{self.cfg.data.name}", f"{self.cfg.data.name}_{self.cfg.feature_extractor}_image_features_train.pkl"), "rb") as f:
            self.image_train_feature = pickle.load(f)
        
        with open(os.path.join("data", "image_features", f"{self.cfg.data.name}", f"{self.cfg.data.name}_{self.cfg.feature_extractor}_image_features_test.pkl"), "rb") as f:
            self.image_query_feature = pickle.load(f)

        if self.cfg.mmices_text_features == "descriptions":
            with open(os.path.join("data", f"{cfg.mmices_text_features}_features", f"{self.cfg.data.name}", f"{self.cfg.data.name}_{self.cfg.feature_extractor}_{self.cfg.name}_{self.cfg.mmices_text_features}_features_train.pkl"), "rb") as f:
                self.text_train_feature = pickle.load(f)
            
            with open(os.path.join("data", f"{cfg.mmices_text_features}_features", f"{self.cfg.data.name}", f"{self.cfg.data.name}_{self.cfg.feature_extractor}_{self.cfg.name}_{self.cfg.mmices_text_features}_features_test.pkl"), "rb") as f:
                self.text_query_feature = pickle.load(f)
        else:
            assert train_concepts is not None, "When using 'concepts' as mmices_text_features, need to provide the concepts for the whole training set."
            self.train_concepts = train_concepts

        self.valid_ids = valid_ids
        self.K = K

    def compute_similarity(self, demos, query):
        similarity = np.matmul(np.stack(list(demos.values()), axis=0), np.array(query))
        similarity_dict = {k: s for (k, s) in zip(demos.keys(), similarity)}
        return similarity_dict
    
    def get_context_keys(self, key, n, query_concepts=None):
        """Select top n examples with highest similarity with the image feature of query image.

        Args:
            key (int): Query image ID.
            n (int): The number of examples to select based on text similarity from the K examples selected by image similarity.

        Returns:
            list: top n examples
        """
        img_similarity_dict = self.compute_similarity(self.image_train_feature, self.image_query_feature[key])

        if self.cfg.mmices_mode == "normal":
            sorted_img_similarity_dict = sorted(img_similarity_dict.items(), key=lambda x:x[1], reverse=True)
    
            img_ids = [x for (x, _) in sorted_img_similarity_dict]

            if len(self.valid_ids) > 0:
                img_ids = [x for x in img_ids if x in self.valid_ids]

            # select K examples based on image similarity
            img_ids = img_ids[:self.K]

            # select n examples based on text similarity
            if self.cfg.mmices_text_features == "descriptions":
                text_train_features = {key: self.text_train_feature[key] for key in img_ids}
                text_similarity_dict = self.compute_similarity(text_train_features, self.text_query_feature[key])
                sorted_text_similarity_dict = sorted(text_similarity_dict.items(), key=lambda x:x[1], reverse=True)
            else:
                if query_concepts is None:
                    return img_ids[:n]

                a = np.stack(list(self.train_concepts.values()), axis=0)
                b = np.array(query_concepts)
                text_similarity = np.linalg.norm(b - a, axis=1)
                text_similarity_dict = {k: s for (k, s) in zip(self.train_concepts.keys(), text_similarity)}
                sorted_text_similarity_dict = sorted(text_similarity_dict.items(), key=lambda x:x[1], reverse=False)

        else:
            a = np.stack(list(self.train_concepts.values()), axis=0)
            b = np.array(query_concepts)
            text_similarity = np.exp(-np.linalg.norm(b - a, axis=1))
            text_similarity_dict = {k: s for (k, s) in zip(self.train_concepts.keys(), text_similarity)}

            final_dict = {}
            for k in img_similarity_dict.keys():
                final_dict[k] = img_similarity_dict[k] * text_similarity_dict[k] 
            sorted_text_similarity_dict = sorted(final_dict.items(), key=lambda x:x[1], reverse=True)

        text_ids = [x for (x, _) in sorted_text_similarity_dict]

        return text_ids[:n]

        
        
