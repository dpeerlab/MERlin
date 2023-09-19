from typing import List, Union

import cv2
import numpy as np
from skimage import registration, transform

from merlin.core import analysistask
from merlin.util import aberration


class Warp(analysistask.ParallelAnalysisTask):
    """An abstract class for warping a set of images so that the corresponding
    pixels align between images taken in different imaging rounds.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None) -> None:
        super().__init__(dataSet, parameters, analysisName)

        if "write_fiducial_images" not in self.parameters:
            self.parameters["write_fiducial_images"] = False
        if "write_aligned_images" not in self.parameters:
            self.parameters["write_aligned_images"] = False

        self.writeAlignedFiducialImages = self.parameters["write_fiducial_images"]

    def get_aligned_image_set(
        self, fov: int, chromaticCorrector: aberration.ChromaticCorrector = None
    ) -> np.ndarray:
        """Get the set of transformed images for the specified fov.

        Args:
        ----
            fov: index of the field of view
            chromaticCorrector: the ChromaticCorrector to use to chromatically
                correct the images. If not supplied, no correction is
                performed.

        Returns:
        -------
            a 4-dimensional numpy array containing the aligned images. The
                images are arranged as [channel, zIndex, x, y]
        """
        dataChannels = self.dataSet.get_data_organization().get_data_channels()
        zIndexes = range(len(self.dataSet.get_z_positions()))
        return np.array(
            [
                [
                    self.get_aligned_image(fov, d, z, chromaticCorrector)
                    for z in zIndexes
                ]
                for d in dataChannels
            ]
        )

    def get_aligned_image(
        self,
        fov: int,
        dataChannel: int,
        zIndex: int,
        chromaticCorrector: aberration.ChromaticCorrector = None,
    ) -> np.ndarray:
        """Get the specified transformed image.

        Args:
        ----
            fov: index of the field of view
            dataChannel: index of the data channel
            zIndex: index of the z position
            chromaticCorrector: the ChromaticCorrector to use to chromatically
                correct the images. If not supplied, no correction is
                performed.

        Returns:
        -------
            a 2-dimensional numpy array containing the specified image
        """
        inputImage = self.dataSet.get_raw_image(
            dataChannel, fov, self.dataSet.z_index_to_position(zIndex)
        )
        transformation = self.get_transformation(fov, dataChannel)
        if chromaticCorrector is not None:
            imageColor = self.dataSet.get_data_organization().get_data_channel_color(
                dataChannel
            )
            return transform.warp(
                chromaticCorrector.transform_image(inputImage, imageColor),
                transformation,
                preserve_range=True,
            ).astype(inputImage.dtype)
        else:
            return transform.warp(
                inputImage, transformation, preserve_range=True
            ).astype(inputImage.dtype)

    def _process_transformations(
        self, transformations: List[transform.EuclideanTransform], fov: int
    ) -> None:
        """Process the transformations determined for a given fov.

        Writes registered images and archives the transformations.

        Args:
            transformations (list[transform.EuclideanTransform]):
                Transformations for each channel.
            fov (int):
                The fov to transform.
        """
        channels = self.dataSet.get_data_organization().get_data_channels()

        if self.parameters["write_aligned_images"]:
            zPositions = self.dataSet.get_z_positions()

            metadata = self.dataSet.analysis_tiff_description(
                len(zPositions), len(channels)
            )

            with self.dataSet.writer_for_analysis_images(
                self, "aligned_images", fov
            ) as tiff:
                for transformation, channel in zip(transformations, channels):
                    for z in zPositions:
                        image = self.dataSet.get_raw_image(channel, fov, z)
                        transformed = transform.warp(
                            image, transformation, preserve_range=True
                        ).astype(image.dtype)
                        tiff.save(
                            transformed,
                            photometric="MINISBLACK",
                            metadata=metadata,
                        )

        if self.writeAlignedFiducialImages:
            metadata = self.dataSet.analysis_tiff_description(1, len(channels))

            with self.dataSet.writer_for_analysis_images(
                self, "aligned_fiducial_images", fov
            ) as tiff:
                for transformation, channel in zip(transformations, channels):
                    image = self._get_fiducial_image(channel, fov)
                    transformed = transform.warp(
                        image, transformation, preserve_range=True
                    ).astype(image.dtype)
                    tiff.save(
                        transformed,
                        photometric="MINISBLACK",
                        metadata=metadata,
                    )

        self._save_transformations(transformations, fov)

    def _save_transformations(
        self, transformations: List[transform.EuclideanTransform], fov: int
    ) -> None:
        self.dataSet.save_numpy_analysis_result(
            np.array([np.array(t) for t in transformations]),
            "offsets",
            self.get_analysis_name(),
            resultIndex=fov,
            subdirectory="transformations",
        )

    def get_transformation(
        self, fov: int, dataChannel: int = None
    ) -> Union[transform.EuclideanTransform, List[transform.EuclideanTransform]]:
        """Get the transformations for aligning images for the specified field
        of view.

        Args:
        ----
            fov: the fov to get the transformations for.
            dataChannel: the index of the data channel to get the transformation
                for. If None, then all data channels are returned.

        Returns:
        -------
            a EuclideanTransform if dataChannel is specified or a list of
                EuclideanTransforms for all dataChannels if dataChannel is
                not specified.
        """
        transformationMatrices = self.dataSet.load_numpy_analysis_result(
            "offsets", self, resultIndex=fov, subdirectory="transformations"
        )
        if dataChannel is not None:
            return transformationMatrices[dataChannel]
        else:
            return transformationMatrices


class FiducialCorrelationWarp(Warp):
    """An analysis task that warps a set of images taken in different imaging
    rounds based on the crosscorrelation between fiducial images.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None) -> None:
        super().__init__(dataSet, parameters, analysisName)

        if "highpass_sigma" not in self.parameters:
            self.parameters["highpass_sigma"] = 3

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        return (
            [self.parameters["fiducial_task"]]
            if "fiducial_task" in self.parameters
            else []
        )

    def _filter(self, inputImage: np.ndarray) -> np.ndarray:
        highPassSigma = self.parameters["highpass_sigma"]
        highPassFilterSize = int(2 * np.ceil(2 * highPassSigma) + 1)

        return inputImage.astype(float) - cv2.GaussianBlur(
            inputImage,
            (highPassFilterSize, highPassFilterSize),
            highPassSigma,
            borderType=cv2.BORDER_REPLICATE,
        )

    def _get_fiducial_image(self, channel: int, fov: int):
        if "fiducial_task" in self.parameters:
            task = self.dataSet.load_analysis_task(self.parameters["fiducial_task"])
            return task.get_fiducial_image(channel, fov)
        return self.dataSet.get_fiducial_image(channel, fov)

    def _run_analysis(self, fragmentIndex: int):
        # TODO - this can be more efficient since some images should
        # use the same alignment if they are from the same imaging round
        fixedImage = self._filter(self._get_fiducial_image(0, fragmentIndex))
        offsets = [
            registration.phase_cross_correlation(
                fixedImage,
                self._filter(self._get_fiducial_image(channel, fragmentIndex)),
                upsample_factor=100,
            )[0]
            for channel in self.dataSet.get_data_organization().get_data_channels()
        ]
        transformations = [
            transform.SimilarityTransform(translation=[-x[1], -x[0]]) for x in offsets
        ]
        self._process_transformations(transformations, fragmentIndex)
