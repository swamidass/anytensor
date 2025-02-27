from typing import TypeAlias, TypeVar, Optional, overload
import numpy.typing as npt
import numpy as np


Int = np.integer

Float = np.floating

Tensor: TypeAlias = np.ndarray

IntTensor: TypeAlias = np.ndarray[Int]

FloatTensor: TypeAlias = np.ndarray[Float]





