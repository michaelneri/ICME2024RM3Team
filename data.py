from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import torch
import librosa as lb
import pandas as pd
from os.path import join
from sklearn.model_selection import train_test_split

LABELS_GC = ["Bus", "Airport", "Metro", "Restaurant", "Shopping mall", "Public square", "Urban park", "Traffic street", "Construction site", "Bar"]
LABELS_TUT = ["airport", "shopping_mall", "metro_station", "street_pedestrian", "public_square", "street_traffic", "tram", "bus", "metro", "park"]

def map_int_labels(string_label):
    return LABELS_GC.index(string_label)

def map_int_labels_TUT(string_label):
    return LABELS_TUT.index(string_label)


class ICME2024Dataset(Dataset):

    def __init__(self, pd_db:pd.DataFrame, path_audios:str) -> list[torch.tensor, torch.tensor, torch.tensor]:
        self.pd_db = pd_db
        self.path_audios = path_audios

    def __getitem__(self, index):
        filename = self.pd_db.iloc[index, 0] + ".wav"
        audio, sr = lb.load(join(self.path_audios, filename), mono = True, sr = 16000)
        # print("Sample rate of the audio file {}".format(sr))
        label_string = self.pd_db.iloc[index, 1]
        label_int = map_int_labels(label_string)
        confidence = self.pd_db.iloc[index, 2]
        return torch.tensor(audio, dtype = torch.float), torch.tensor(label_int, dtype = torch.long), torch.tensor(confidence, dtype = torch.float)

    def __len__(self):
        return self.pd_db.shape[0]

class ICME2024Datamodule(LightningDataModule):

    def __init__(self, db_training, db_val, db_test, path_audios, batch_size):
        super().__init__()
        self.db_training = db_training
        self.db_val = db_val
        self.db_test = db_test
        self.path_audios = path_audios
        self.batch_size = batch_size

    def prepare_data(self):
        pass
    
    def train_dataloader(self):
        return DataLoader(ICME2024Dataset(self.db_training, self.path_audios), batch_size = self.batch_size, shuffle = True, drop_last = True)
    
    def val_dataloader(self):
        return DataLoader(ICME2024Dataset(self.db_val, self.path_audios), batch_size = self.batch_size, drop_last = False)
    
    def test_dataloader(self):
        return DataLoader(ICME2024Dataset(self.db_test, self.path_audios), batch_size = self.batch_size, drop_last = False)
    
############################ TUT DATASET #################################################
class TUT2020Dataset(Dataset):

    def __init__(self, pd_db:pd.DataFrame, testing:bool = False) -> list[torch.tensor, torch.tensor, torch.tensor]:
        self.pd_db = pd_db
        self.testing = testing

    def __getitem__(self, index):
        filename = self.pd_db.iloc[index, 0]
        audio, sr = lb.load(join("TUT 2020 UAS", filename), mono = True, sr = 16000)
        # print("Sample rate of the audio file {}".format(sr))
        if self.testing:
            label_string = (filename.split("/")[1]).split("-")[0]
            confidence = self.pd_db.iloc[index, 1]
        else:
            label_string = self.pd_db.iloc[index, 1]
            confidence = self.pd_db.iloc[index, 2]
        label_int = map_int_labels_TUT(label_string)
        return torch.tensor(audio, dtype = torch.float), torch.tensor(label_int, dtype = torch.long), torch.tensor(confidence, dtype = torch.float)

    def __len__(self):
        return self.pd_db.shape[0]

class TUT2020Datamodule(LightningDataModule):

    def __init__(self, db_training, db_val, db_test, batch_size):
        super().__init__()
        self.db_training = db_training
        self.db_val = db_val
        self.db_test = db_test
        self.batch_size = batch_size

    def prepare_data(self):
        pass
    
    def train_dataloader(self):
        return DataLoader(TUT2020Dataset(self.db_training), batch_size = self.batch_size, shuffle = True, drop_last = True)
    
    def val_dataloader(self):
        return DataLoader(TUT2020Dataset(self.db_val), batch_size = self.batch_size, drop_last = False)
    
    def test_dataloader(self):
        return DataLoader(TUT2020Dataset(self.db_test, testing = True), batch_size = self.batch_size, drop_last = False)
    
######### TEST ###############
if __name__ == "__main__":
    path_csv = "dev_label.csv"
    path_audios = "dev"
    random_seed = 200 # for reproducibility of dataset split
    print("ICME 2024 GRAND CHALLENGE DATASET TEST")
    print("**********************************************")
    print("All files: {}".format(pd.read_csv(path_csv).shape))

    ## READING STRONG LABELS AND SPLITTING
    # if the label is not available, pandas insert NaN
    strong_files = pd.read_csv(path_csv).dropna(ignore_index = True).assign(confidence = 1)

    # split in 80-10-10
    trainingSet, testSet = train_test_split(strong_files, test_size=0.1, random_state = random_seed)
    trainingSet, valSet = train_test_split(trainingSet, test_size=0.1, random_state = random_seed)
    print("***Strong labels split***")
    print("Train shape: {}".format(trainingSet.shape))
    print("Val shape: {}".format(valSet.shape))
    print("Test shape: {}".format(testSet.shape))

    ## READING NON-LABELLED DATA
    unlabelled_files = pd.read_csv(path_csv).assign(confidence = 0)
    unlabelled_files = unlabelled_files.drop(unlabelled_files.dropna().index)
    print("Unlabelled shape: {}".format(unlabelled_files.shape))

    # TESTING DATASET
    dataset = ICME2024Dataset(trainingSet, path_audios)
    single_sample = dataset[0]
    audio_data, label, confidence = single_sample
    print("Audio data length {} with label {} and confidence {}".format(audio_data.shape, label, confidence))

    # TESTING DATAMODULE / DATALOADER
    batch_size = 16
    datamodule = ICME2024Datamodule(trainingSet, valSet, testSet, path_audios, batch_size) 
    training_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()
    print("Training dataloader length {} Val dataloader length {} test dataloader length {}".format(len(training_dataloader), len(val_dataloader), len(test_dataloader)))

    for batch in iter(val_dataloader):
        tensor_audios, tensor_labels, tensor_confidences = batch
        print(tensor_audios.shape)
        print(tensor_labels)
        print(tensor_confidences)


    print("TUT 2020 Mobile DATASET TEST")
    print("**********************************************")
    path_csv_training = "TUT 2020 UAS/evaluation_setup/fold1_train.csv"
    path_csv_eval = "TUT 2020 UAS/evaluation_setup/fold1_evaluate.csv"
    path_csv_test = "TUT 2020 UAS/evaluation_setup/fold1_test.csv"
    training_pd = pd.read_csv(path_csv_training, delimiter="\t").assign(confidence = 1)
    val_pd = pd.read_csv(path_csv_eval, delimiter="\t").assign(confidence = 1)
    test_pd = pd.read_csv(path_csv_test, delimiter="\t").assign(confidence = 1)

    # TESTING DATASET
    dataset = TUT2020Dataset(test_pd, testing = True)
    single_sample = dataset[0]
    audio_data, label, confidence = single_sample
    print("Audio data length {} with label {} and confidence {}".format(audio_data.shape, label, confidence))

    # TESTING DATAMODULE / DATALOADER
    batch_size = 16
    datamodule = TUT2020Datamodule(training_pd, val_pd, test_pd, batch_size) 
    training_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()
    print("Training dataloader length {} Val dataloader length {} test dataloader length {}".format(len(training_dataloader), len(val_dataloader), len(test_dataloader)))

    for batch in iter(val_dataloader):
        tensor_audios, tensor_labels, tensor_confidences = batch
        print(tensor_audios.shape)
        print(tensor_labels)
        print(tensor_confidences)

    