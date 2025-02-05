from src.data.fitzpatrick17k_dataset import Fitzpatrick_Dataset
from src.data.derm7pt_dataset import Derm7ptDataset
from src.data.CORDA_dataset import CORDADataset
from src.data.DDR_dataset import DDRDataset
from torch.utils.data import Sampler

class CustomIndexSampler(Sampler):
    def __init__(self, dataset, ids):
        """
        Args:
            indices (list): A list of indices to sample from.
        """
        self.dataset = dataset
        self.ids = ids[::-1] # we want the most similar image to be closest to the query
        
        self.indices_dict = self._generate_indices()

    def _generate_indices(self):
        """
        Generates a dictionary where keys are class labels and values are lists of corresponding sample indices.    
        """
        
        indices = dict()
        for idx, batch in enumerate(self.dataset):
            indices[batch["img_id"]] = idx
        return indices

    def __iter__(self):
        """Returns an iterator over the indices."""
        
        batch = [self.indices_dict[id] for id in self.ids]

        return iter(batch)

    def __len__(self):
        """Returns the length of the indices."""
        return len(self.indices)
    
def get_dataset(cfg):
    """Creates datasets and dataloaders for the specified dataset
    """
    if (cfg.data.get("name") == 'fitzpatrick17k'):
        """Fitzpatrick17k"""
        dataset_test = Fitzpatrick_Dataset(csv_path=cfg.data.test_csv, root=cfg.data.pathBase)
        dataset_train = Fitzpatrick_Dataset(csv_path=cfg.data.train_csv, root=cfg.data.pathBase, filter=cfg.filter)
    elif (cfg.data.get("name") == 'derm7pt'):
        """Derm7pt"""
        dataset_test = Derm7ptDataset(csv_file=cfg.data.test_csv, clinical_concepts_path=cfg.data.clinical_concepts_path, img_extension=cfg.data.file_extension, path_to_images=cfg.data.pathBase)
        dataset_train = Derm7ptDataset(csv_file=cfg.data.train_csv, clinical_concepts_path=cfg.data.clinical_concepts_path, img_extension=cfg.data.file_extension, path_to_images=cfg.data.pathBase, filter=cfg.filter)
    elif (cfg.data.get("name") == 'CORDA'):
        """CORDA"""
        dataset_test = CORDADataset(csv_file=cfg.data.test_csv, img_extension=cfg.data.file_extension, path_to_images=cfg.data.pathBase)
        dataset_train = CORDADataset(csv_file=cfg.data.train_csv, img_extension=cfg.data.file_extension, path_to_images=cfg.data.pathBase, filter=cfg.filter)
    elif (cfg.data.get("name") == 'DDR'):
        """DDR"""
        dataset_test = DDRDataset(csv_file=cfg.data.test_csv, img_extension=cfg.data.file_extension, path_to_images=cfg.data.pathBase)
        dataset_train = DDRDataset(csv_file=cfg.data.train_csv, img_extension=cfg.data.file_extension, path_to_images=cfg.data.pathBase, filter=cfg.filter)
    else:
        raise ValueError(f"The dataset {cfg.data.get('name')} has not a valid implementation.")
    
    return dataset_train, dataset_test
    