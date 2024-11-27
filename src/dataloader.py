from dataclasses import dataclass
from typing import Dict, Any, Tuple
import pandas as pd
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from sklearn.model_selection import train_test_split
import torch


@dataclass
class DataLoader:
    dataframe: pd.DataFrame
    batch_size: int
    shuffle: bool = True
    test_size: float = 0.2
    random_state: int = 42

    def __post_init__(self):
        self.train_df, self.test_df = self.split_dataframe()
        self.train_loader = self.create_dataloader(self.train_df)
        self.test_loader = self.create_dataloader(self.test_df)

    def split_dataframe(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df, test_df = train_test_split(
            self.dataframe, test_size=self.test_size, random_state=self.random_state
        )
        return train_df, test_df

    def create_dataloader(self, dataframe: pd.DataFrame) -> TorchDataLoader:
        dataset = OrganDataset(dataframe)
        return TorchDataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def get_train_dataloader(self) -> TorchDataLoader:
        return self.train_loader

    def get_test_dataloader(self) -> TorchDataLoader:
        return self.test_loader


class OrganDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = self.preprocess_dataframe(dataframe)

    def preprocess_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        for col in dataframe.columns:
            if pd.api.types.is_datetime64_any_dtype(dataframe[col]):
                dataframe[col] = dataframe[col].astype(
                    'int64') // 10**9  # Convert to Unix timestamp
            else:
                dataframe[col] = pd.to_numeric(
                    dataframe[col], errors='coerce').fillna(0)
        return dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.dataframe.iloc[idx]
        return {col: torch.tensor(val, dtype=torch.float32) for col, val in row.items()}
