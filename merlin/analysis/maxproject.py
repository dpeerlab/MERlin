import re
from typing import List

import numpy as np

from merlin.core import analysistask


class MaxProject(analysistask.ParallelAnalysisTask):
    """An abstract class to max project a set of channel images into a new channel."""

    def __init__(self, dataSet, parameters=None, analysisName=None) -> None:
        super().__init__(dataSet, parameters, analysisName)
        if "write_images" not in self.parameters:
            self.parameters["write_images"] = False

        self.write_images = self.parameters["write_images"]

    @property
    def z_positions(self) -> List[float]:
        return self.dataSet.get_z_positions()

    @property
    def channels(self) -> np.ndarray:
        return self.dataSet.get_data_organization().get_data_channels()

    @property
    def channel_names(self) -> np.ndarray:
        return self.dataSet.get_data_organization().data["channelName"]

    def _write_images(self, fov, images):
        metadata = self.dataSet.analysis_tiff_description(
            len(self.z_positions), len(self.channels)
        )
        with self.dataSet.writer_for_analysis_images(self, "max_projected", fov) as f:
            for c in range(len(self.channels)):
                for z in range(len(self.z_positions)):
                    f.save(images[c, z], photometric="MINISBLACK", metadata=metadata)

    def get_image(self, channel, fov, z):
        """Get an image for a specific channel, fov, and z position."""
        return self.dataSet.get_raw_image(
            channel,
            fov,
            # TODO: make z_index_to_position internal to dataSet?
            self.dataSet.z_index_to_position(z),
        )

    def get_images(self, fov: int) -> np.ndarray:
        """Get a set of images for the specified fov.

        Args:
            fov: index of the field of view

        Returns:
            A 4-dimensional numpy array containing the images for the fov.
            The images are arranged as [channel, z, x, y].
        """
        return np.array(
            [
                [self.get_image(c, fov, z) for z in self.z_positions]
                for c in self.channels
            ]
        )

    def _run_analysis(self, fov):
        # This analysis task does not need computation
        pass


class MaxProjectFiducial(MaxProject):
    """Max projects all z-slices of a set of fiducial images into a single image."""

    def __init__(self, dataSet, parameters=None, analysisName=None) -> None:
        super().__init__(dataSet, parameters, analysisName)

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        return []

    def get_image(self, channel: int, fov: int, z: int) -> np.ndarray:
        """Get an image for a specific channel, fov, and z position."""
        return self.dataSet.get_fiducial_image(
            channel,
            fov,
            # TODO: make z_index_to_position internal to dataSet?
            self.dataSet.z_index_to_position(z),
        )

    def get_images(self, fov: int) -> np.ndarray:
        images = super().get_images(fov)
        projected = np.max(images, axis=1)
        if self.write_images:
            metadata = self.dataSet.analysis_tiff_description(1, len(self.channels))

            with self.dataSet.writer_for_analysis_images(
                self, "max_projected_fiducial_beads", fov
            ) as f:
                for channel in range(len(self.channels)):
                    f.save(
                        projected[channel], photometric="MINISBLACK", metadata=metadata
                    )

        return projected


class MaxProjectBits(MaxProject):
    """Combines all bit channels into a polyT channel."""

    def __init__(self, dataSet, parameters=None, analysisName=None) -> None:
        super().__init__(dataSet, parameters, analysisName)

        if "channel_regex" not in self.parameters:
            self.parameters["channel_regex"] = ""

        self.channel_regex = self.parameters["channel_regex"]

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 1

    def get_dependencies(self):
        return [self.parameters["warp_task"]]

    def get_image(self, channel: int, fov: int, z: int) -> np.ndarray:
        warp_task = self.dataSet.load_analysis_task(self.parameters["warp_task"])
        z_index = self.dataSet.position_to_z_index(z)
        return warp_task.get_aligned_image(fov, channel, z_index)

    def get_images(self, fov: int) -> np.ndarray:
        """Get the max projected bit images for the specified fov.

        Args:
            fov (int): index of the field of view

        Returns:
            np.ndarray: Image array of shape (z, x, y) with the projected bits.
        """
        images = super().get_images(fov)
        print(self.channel_names)
        print(re.match(self.channel_regex, self.channel_names.iloc[0]))
        print(re.match(self.channel_regex, self.channel_names.iloc[-1]))
        is_bit = np.array(
            [re.match(self.channel_regex, name) for name in self.channel_names],
            dtype=bool,
        )
        bit_images = images[is_bit]
        projected = np.max(bit_images, axis=0)
        if self.write_images:
            metadata = self.dataSet.analysis_tiff_description(len(self.z_positions), 1)
            with self.dataSet.writer_for_analysis_images(self, "polyT", fov) as f:
                for z in range(len(self.z_positions)):
                    f.save(projected[z], photometric="MINISBLACK", metadata=metadata)
        return projected
