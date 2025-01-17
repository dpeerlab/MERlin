from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
from skimage import transform

"""
This module contains tools for measuring and correcting chromatic aberrations.
"""


class ChromaticCorrector(ABC):
    """An abstract class for color-specific image transformation."""

    @abstractmethod
    def transform_image(self, inputImage: np.ndarray, imageColor: str) -> np.ndarray:
        """Transform inputImage to the reference color.

        Args:
        ----
            inputImage: The image to transform. If inputImage has two
                dimensions, it is transformed as a single image. If inputImage
                has three dimensions, each element in the first dimension is
                transformed as an image as (z, x, y).
            imageColor: The color of the input image as a string. If the color
                of the input image is not in the set of transformations for
                this corrector, no transformation is applied.
        """


class IdentityChromaticCorrector(ChromaticCorrector):
    """A class for correcting chromatic aberration that performs no transformation."""

    def __init__(self) -> None:
        pass

    def transform_image(self, inputImage: np.ndarray, imageColor: str) -> np.ndarray:
        return inputImage


class RigidChromaticCorrector(ChromaticCorrector):
    """A class for correcting chromatic aberration using rigid transformation
    matrices.
    """

    def __init__(
        self,
        transformations: Dict[str, Dict[str, transform.EuclideanTransform]],
        referenceColor: str = None,
    ) -> None:
        """Creates a new RigidChromaticCorrector that transforms images
        using the specified transformations.

        Args:
        ----
              transformations: A dictionary of transformations
              referenceColor: the name of the color to transform the images to
        """
        self.transformations = transformations
        if referenceColor is None:
            self.referenceColor = min(transformations.keys())
        else:
            self.referenceColor = referenceColor

    def transform_image(self, inputImage: np.ndarray, imageColor: str) -> np.ndarray:
        if imageColor not in self.transformations[self.referenceColor]:
            return inputImage

        if imageColor == self.referenceColor:
            return inputImage

        if len(inputImage.shape) == 3:
            return np.array([self.transform_image(x, imageColor) for x in inputImage])

        return transform.warp(
            inputImage,
            self.transformations[self.referenceColor][imageColor],
            preserve_range=True,
        )
