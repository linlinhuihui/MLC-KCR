import os
import pandas as pd
from torch.utils.data import Dataset
from typing import Union, List, SupportsIndex, Sequence, Callable, Optional, Any, Dict
from multimethod import multimethod as singledispatchmethod
from utils.features import *
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _load_config(config_path: Path | str | None = None) -> Dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_feature_fn(name: str) -> Callable:
    if not name:
        return iFunction.load_onehot_blosum62_esm2
    if hasattr(iFunction, name):
        return getattr(iFunction, name)
    raise ValueError(f"未知的特征编码函数: {name}")


class iDataset(Dataset):
    def __init__(self, argc: Sequence[Any], y: Any = None, fe: Optional[Callable] = None) -> None:
        super().__init__()
        self.argc = list(map(lambda x: np.array(x), argc))
        self.X = self.argc[0]
        self.y = np.array(y) if y is not None else np.ones(shape=(len(self.X),))
        self.fe = fe if fe else lambda x: x

    @singledispatchmethod
    def __getitem__(self, index):
        raise NotImplementedError(f"only for int or slice or List[int]")

    @__getitem__.register
    def _(self, index: SupportsIndex):
        if len(self.argc) == 1:
            return self.fe(self.X[index]), self.y[index]
        else:
            return (self.fe(self.X[index]), *tuple(map(lambda x: x[index], self.argc[1:]))), self.y[index]

    @__getitem__.register
    def _(self, index: Union[slice, List[int]]):
        if len(self.argc) == 1:
            return (
                np.stack(list(map(self.fe, self.X[index])), axis=0),
                self.y[index],
            )
        else:
            return (
                (
                    np.stack(list(map(self.fe, self.X[index])), axis=0),
                    *tuple(map(lambda x: x[index], self.argc[1:])),
                ),
                self.y[index],
            )

    def subset(self, indices: Sequence[int]):
        return iDataset(tuple(map(lambda x: x[indices], self.argc)), y=self.y[indices], fe=self.fe)

    def __len__(self):
        return len(self.argc[0])

def get_kcr(
    path: str | Path | None = None,
    train_filename: str = "train.csv",
    test_filename: str = "test.csv",
    fe: Callable | None = None,
    config: Dict | None = None,
    config_path: str | Path | None = None,
):

    cfg = config if config is not None else _load_config(config_path)
    feature_name = cfg.get("feature_extractor", "load_onehot_blosum62_esm2")
    fe_fn = fe or _resolve_feature_fn(feature_name)

    data_root = Path(path) if path is not None else Path(r"E:\git-kcr\data\plant\data")
    train_path = Path(train_filename) if os.path.isabs(train_filename) else data_root / train_filename
    test_path = Path(test_filename) if os.path.isabs(test_filename) else data_root / test_filename

    trainset = pd.read_csv(train_path)
    testset = pd.read_csv(test_path)
    return iDataset([trainset['sequence']], y=trainset['label'], fe=fe_fn), iDataset(
        [testset['sequence']], y=testset['label'], fe=fe_fn)
