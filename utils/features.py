import os
import pickle
from typing import Dict
import numpy as np
import torch
from functools import lru_cache
import yaml


try:
    with open(r"E:\git-kcr\config.yaml", "r", encoding="utf-8") as _f:
        _global_cfg = yaml.safe_load(_f)
        _SEQLEN = int(_global_cfg.get("seqlen", 29))
except FileNotFoundError:
    _SEQLEN = 29


class iFunction:
    _feature_caches: Dict[str, Dict[str, np.ndarray]] = {}
    _cache_loaded: Dict[str, bool] = {}
    _feature_paths = {
        'esm2': 'data/plant/data/esm2_features.pkl',
        'onehot': 'data/plant/data/onehot_features.pkl',
        'blosum62': 'data/plant/data/blosum62_features.pkl',
    }

    @classmethod
    def _ensure_cache_loaded(cls, feature_type: str, pkl_path: str = None) -> None:
        if cls._cache_loaded.get(feature_type, False):
            return
        pkl_path = pkl_path or cls._feature_paths.get(feature_type)
        if not pkl_path or not os.path.exists(pkl_path):
            cls._feature_caches[feature_type] = {}
            cls._cache_loaded[feature_type] = True
            return
        try:
            with open(pkl_path, 'rb') as f:
                cls._feature_caches[feature_type] = pickle.load(f)
            cls._cache_loaded[feature_type] = True
        except Exception:
            cls._feature_caches[feature_type] = {}
            cls._cache_loaded[feature_type] = True

    @classmethod
    def _get_feature(cls, seq: str, feature_type: str, expected_dim: int, pkl_path: str = None) -> torch.Tensor:
        cls._ensure_cache_loaded(feature_type, pkl_path)
        cache = cls._feature_caches.get(feature_type, {})
        if seq in cache:
            return torch.from_numpy(cache[seq]).float()
        return torch.empty((0, expected_dim), dtype=torch.float32)

    @staticmethod
    def to_onehot(seq: str, pkl_path: str = None) -> torch.Tensor:
        return iFunction._get_feature(seq, 'onehot', 21, pkl_path)

    @staticmethod
    def to_protein_blosum62(seq: str, pkl_path: str = None) -> torch.Tensor:
        return iFunction._get_feature(seq, 'blosum62', 20, pkl_path)

    @staticmethod
    def load_esm2_from_file(seq: str, pkl_path: str = None) -> np.ndarray:
        iFunction._ensure_cache_loaded('esm2', pkl_path)
        cache = iFunction._feature_caches.get('esm2', {})
        if seq in cache:
            return cache[seq]
        if cache:
            any_feature = next(iter(cache.values()))
            return np.zeros_like(any_feature)
        return np.zeros((_SEQLEN, 320), dtype=np.float32)

    @staticmethod
    @lru_cache(maxsize=1000)
    def load_onehot_blosum62_esm2(seq: str, esm2_pkl_path: str = None) -> torch.Tensor:
        esm2_feat_np = iFunction.load_esm2_from_file(seq, esm2_pkl_path)
        esm2_feat = torch.from_numpy(esm2_feat_np).float()
        onehot_feat = iFunction.to_onehot(seq)
        blosum_feat = iFunction.to_protein_blosum62(seq)
        max_len = esm2_feat.shape[0]

        def pad_to_len(feat: torch.Tensor, target_len: int) -> torch.Tensor:
            if feat.shape[0] < target_len:
                padding = torch.zeros((target_len - feat.shape[0], feat.shape[1]), dtype=feat.dtype)
                return torch.cat([feat, padding], dim=0)
            return feat[:target_len, :]

        onehot_padded = pad_to_len(onehot_feat, max_len)
        blosum_padded = pad_to_len(blosum_feat, max_len)
        return torch.cat([esm2_feat, onehot_padded, blosum_padded], dim=1)
