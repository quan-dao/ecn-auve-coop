from typing import List, Union
import numpy as np
from pyquaternion import Quaternion


Vector = Union[List[float], List[int], np.ndarray]


def to_numpy(l: Vector) -> np.ndarray:
    if isinstance(l, list):
        return np.array(l)
    elif isinstance(l, np.ndarray):
        return l
    else:
        raise NotImplementedError(f"{type(l)} is not supported")


def to_quaternion(q: Union[Vector, Quaternion]) -> Quaternion:
    if isinstance(q, list):
        assert len(q) == 4, f"{len(q)} != 4"
        return Quaternion(q)
    elif isinstance(q, np.ndarray):
        if q.shape == (4,):
            return Quaternion(q)
        elif q.shape == (3, 3) or q.shape == (4, 4):
            return Quaternion(matrix=q)
        else:
            raise ValueError(f"{q.shape} is neither a quaternion nor a rotation matrix")
    elif isinstance(q, Quaternion):
        return q
    else:
        raise NotImplementedError(f"{type(q)} is not supported")

