import data_processing as dp
import numpy as np
import torch
from torch import tensor
from torch.utils.data import Dataset

class PrivacyPoliciesDataset(Dataset):
    
    def __init__(self, segments_array, labels_list, labels):
        
        self.segments_array = segments_array
       
        self.labels_tensor = tensor(labels_list)
        
        self.labels = labels
        
    def __len__(self):
        
        if self.segments_array.shape[0] == self.labels_tensor.shape[0]: 
            
            return self.segments_array.shape[0]
        
        else:
            
            print("Warning: number of segments don't match number of annotations")
            
            return self.segments_array.shape[0]     
    
    def __getitem__(self, idx):
        
        segment = self.segments_array[idx]
        
        label = self.labels_tensor[idx]
        
        return (segment, label)
    
    def split_dataset_randomly(self, ratio = 0.1):
    
        """

        This function randomly splits the dataset in two parts using the split ratio provided

        Args:
            dataset: torch.utils.data.Dataset, dataset containing the data to split
            ratio: double, percentage of data that will be retrieved inside s_dataset
        Returns:
            s_dataset: torch.utils.data.Dataset, dataset with length = len(dataset) * ratio
            b_dataset: torch.utils.data.Dataset, dataset with length = len(dataset) * (1 - ratio)

        """

        from random import sample

        labels = self.labels

        num_samples = int(ratio * len(self))

        s_dataset_idx_set = set(sample(range(len(self)), num_samples))

        b_dataset_idx_set = set(range(len(self))).difference(s_dataset_idx_set)

        s_dataset_idx_tensor = tensor(list(s_dataset_idx_set))

        b_dataset_idx_tensor = tensor(list(b_dataset_idx_set))

        s_dataset_data = self[s_dataset_idx_tensor]

        b_dataset_data = self[b_dataset_idx_tensor]

        s_dataset = PrivacyPoliciesDataset(s_dataset_data[0], s_dataset_data[1], labels)

        b_dataset = PrivacyPoliciesDataset(b_dataset_data[0], b_dataset_data[1], labels)

        return s_dataset, b_dataset
    
    def pickle_dataset(self, path):

        import pickle

        with open(path, "wb") as dataset_file:

            pickle.dump(self, dataset_file)
            
    def labels_stats(self):
        
        p_labels =  self.labels_tensor.sum(0)
        
        total_labels = int(p_labels.sum())
        
        num_segments = len(self)
        
        print('Num of segments: {}'.format(num_segments))
        
        print('Num of labels: {}'.format(total_labels))
        
        print('Percentages with respect to number of labels ... ')
        
        for label, idx in self.labels.items():
            
            num_p = int(p_labels[idx])
            
            # pct = round((100 * p_labels[idx] / num_segments), 2)
            pct = 100 * p_labels[idx] / total_labels
            print('{}. {} : {} ({}%)'.format(idx, label, num_p, pct))
            # print('{}. {} : ({}%)'.format(idx, label, num_p))
    
    @staticmethod
    def unpickle_dataset(path):
    
        import pickle

        with open(path, "rb") as dataset_file:

            dataset = pickle.load(dataset_file)

            return dataset
    
    @staticmethod
    def collate_data(batch):
        
        def stack_segments(segments, clearance = 2):

            import numpy as np

            segments_len = map(len, segments)
            max_len = max(segments_len)

            segments_list = []

            output_len = max_len + clearance * 2

            for i, segment in enumerate(segments):

                segment_array = np.array(segment)

                zeros_to_prepend = int((output_len - len(segment_array))/2)

                zeros_to_append = output_len - len(segment_array) - zeros_to_prepend

                resized_array = np.append(np.zeros(zeros_to_prepend), segment_array)

                resized_array = np.append(resized_array, np.zeros(zeros_to_append))

                segments_list.append(torch.tensor(resized_array, dtype = torch.int64))

                segments_tensor = torch.stack(segments_list).unsqueeze(1)

            return segments_tensor                         

        segments = [item[0] for item in batch]

        labels = [item[1] for item in batch]

        segments_tensor = stack_segments(segments)

        labels_tensor = torch.stack(labels)

        return [segments_tensor, labels_tensor]