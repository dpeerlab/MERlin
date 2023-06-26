import os
import re
from typing import List, Tuple

import numpy as np
import pandas as pd

import merlin
from merlin.core import dataset


def _parse_list(inputString: str, dtype=float):
    if "," in inputString:
        return np.fromstring(inputString.strip("[] "), dtype=dtype, sep=",")
    else:
        return np.fromstring(inputString.strip("[] "), dtype=dtype, sep=" ")


def _parse_int_list(inputString: str):
    return _parse_list(inputString, dtype=int)


class InputDataError(Exception):
    pass


class DataOrganization:
    """A class to specify the organization of raw images in the original
    image files.

    Attributes:
        data: DataFrame of the provided data organization .csv
        fileMap: DataFrame of the image files of channels to process
        fiducialFileMap: DataFrame of the image files of fiducial bead channels
    """

    def __init__(self, dataSet, filePath: str = None) -> None:
        """Create a new DataOrganization for the data in the specified data set.

        If filePath is not specified, a previously stored DataOrganization
        is loaded from the dataSet if it exists. If filePath is specified,
        the DataOrganization at the specified filePath is loaded and
        stored in the dataSet, overwriting any previously stored
        DataOrganization.

        Raises
        ------
            InputDataError: If the set of raw data is incomplete or the
                    format of the raw data deviates from expectations.
        """
        self._dataSet = dataSet

        if filePath is not None:
            if not os.path.exists(filePath):
                filePath = os.sep.join([merlin.DATA_ORGANIZATION_HOME, filePath])

            self.data = pd.read_csv(
                filePath, converters={"frame": _parse_int_list, "zPos": _parse_list}
            )
            self.data["readoutName"] = self.data["readoutName"].str.strip()
            self._dataSet.save_dataframe_to_csv(
                self.data, "dataorganization", index=False
            )

        else:
            self.data = self._dataSet.load_dataframe_from_csv(
                "dataorganization",
                converters={"frame": _parse_int_list, "zPos": _parse_list},
            )

        stringColumns = [
            "readoutName",
            "channelName",
            "imageType",
            "imageRegExp",
            "fiducialImageType",
            "fiducialRegExp",
        ]
        self.data[stringColumns] = self.data[stringColumns].astype("str")
        self.fileMap = self._map_image_files()
        self.fiducialFileMap = self._map_image_files(fiducial=True)
        self._validate_file_map()
        self._validate_file_map(fiducial=True)

    def get_data_channels(self) -> np.array:
        """Get the data channels for the MERFISH data set.

        Returns
        -------
            A list of the data channel indexes
        """
        return np.array(self.data.index)

    def get_data_channel_readout_name(self, dataChannelIndex: int) -> str:
        """Get the name for the data channel with the specified index.

        Args:
        ----
            dataChannelIndex: The index of the data channel
        Returns:
            The name of the specified data channel
        """
        return self.data.iloc[dataChannelIndex]["readoutName"]

    def get_data_channel_name(self, dataChannelIndex: int) -> str:
        """Get the name for the data channel with the specified index.

        Args:
        ----
            dataChannelIndex: The index of the data channel
        Returns:
            The name of the specified data channel,
            primarily relevant for non-multiplex measurements
        """
        return self.data.iloc[dataChannelIndex]["channelName"]

    def get_data_channel_index(self, dataChannelName: str) -> int:
        """Get the index for the data channel with the specified name.

        Args:
        ----
            dataChannelName: the name of the data channel. The data channel
                name is not case sensitive.

        Returns:
        -------
            the index of the data channel where the data channel name is
                dataChannelName
        Raises:
            # TODO this should raise a meaningful exception if the data channel
            # is not found
        """
        return self.data[
            self.data["channelName"].apply(lambda x: str(x).lower())
            == str(dataChannelName).lower()
        ].index.values.tolist()[0]

    def get_data_channel_color(self, dataChannel: int) -> str:
        """Get the color used for measuring the specified data channel.

        Args:
        ----
            dataChannel: the data channel index
        Returns:
            the color for this data channel as a string
        """
        return str(self.data.at[dataChannel, "color"])

    def get_data_channel_for_bit(self, bitName: str) -> int:
        """Get the data channel associated with the specified bit.

        Args:
        ----
            bitName: the name of the bit to search for
        Returns:
            The index of the associated data channel
        """
        return self.data[self.data["readoutName"] == bitName].index.values.item()

    def get_data_channel_with_name(self, channelName: str) -> int:
        """Get the data channel associated with a gene name.

        Args:
        ----
            channelName: the name of the gene to search for
        Returns:
            The index of the associated data channel
        """
        return self.data[self.data["channelName"] == channelName].index.values.item()

    def get_fiducial_filename(self, dataChannel: int, fov: int) -> str:
        """Get the path for the image file that contains the fiducial
        image for the specified dataChannel and fov.

        Args:
        ----
            dataChannel: index of the data channel
            fov: index of the field of view
        Returns:
            The full path to the image file containing the fiducials
        """
        return self._get_image_path(dataChannel, fov, fiducial=True)

    def get_fiducial_frame_index(self, dataChannel: int) -> int:
        """Get the index of the frame containing the fiducial image
        for the specified data channel.

        Args:
        ----
            dataChannel: index of the data channel
        Returns:
            The index of the fiducial frame in the corresponding image file
        """
        return self.data.iloc[dataChannel]["fiducialFrame"]

    def get_image_filename(self, dataChannel: int, fov: int) -> str:
        """Get the path for the image file that contains the
        images for the specified dataChannel and fov.

        Args:
        ----
            dataChannel: index of the data channel
            fov: index of the field of view
        Returns:
            The full path to the image file containing the fiducials
        """
        return self._get_image_path(dataChannel, fov)

    def get_image_frame_index(self, dataChannel: int, zPosition: float) -> int:
        """Get the index of the frame containing the image
        for the specified data channel and z position.

        Args:
        ----
            dataChannel: index of the data channel
            zPosition: the z position
        Returns:
            The index of the frame in the corresponding image file
        """
        channelInfo = self.data.iloc[dataChannel]
        channelZ = channelInfo["zPos"]
        if isinstance(channelZ, np.ndarray):
            zIndex = np.where(channelZ == zPosition)[0]
            if len(zIndex) == 0:
                raise Exception(
                    "Requested z position not found. Position "
                    + "z=%0.2f not found for channel %i" % (zPosition, dataChannel)
                )
            else:
                frameIndex = zIndex[0]
        else:
            frameIndex = 0

        frames = channelInfo["frame"]
        frame = frames[frameIndex] if isinstance(frames, np.ndarray) else frames

        return frame

    def get_z_positions(self) -> List[float]:
        """Get the z positions present in this data organization.

        Returns
        -------
            A sorted list of all unique z positions
        """
        return sorted(np.unique([y for x in self.data["zPos"] for y in x]))

    def get_fovs(self, fiducial=False) -> np.ndarray:
        fileMap = self.fiducialFileMap if fiducial else self.fileMap
        return np.unique(fileMap["fov"])

    def get_sequential_rounds(self) -> Tuple[List[int], List[str]]:
        """Get the rounds that are not present in your codebook
        for all the rounds not contained in the codebook, the second list
        contains the name associated with that channel in the data.
        """
        multiplexBits = {
            b for x in self._dataSet.get_codebooks() for b in x.get_bit_names()
        }
        sequentialChannels = [
            i
            for i in self.get_data_channels()
            if self.get_data_channel_readout_name(i) not in multiplexBits
        ]
        sequentialGeneNames = [
            self.get_data_channel_name(x) for x in sequentialChannels
        ]
        return sequentialChannels, sequentialGeneNames

    def _get_image_path(
        self,
        dataChannel: int,
        fov: int,
        fiducial: bool = False,
    ) -> str:
        """Get the path for the image file for the provided data channel and fov.

        Raises:
            InputDataError: if 0 or >1 files are found for the data channel and fov
        """
        fileMap = self.fiducialFileMap if fiducial else self.fileMap
        channelName, imageType, imagingRound, bitNumber = self.data.iloc[dataChannel][
            [
                "channelName",
                "fiducialImageType" if fiducial else "imageType",
                "fiducialImagingRound" if fiducial else "imagingRound",
                "bitNumber",
            ]
        ]
        selection = fileMap[
            (fileMap["fov"] == fov)
            & (fileMap["imageType"] == imageType)
            & (fileMap["imagingRound"] == imagingRound)
            & (fileMap["bitNumber"] == bitNumber)
        ]
        if len(selection) != 1:
            msg = (
                f"Found {len(selection)} image files for {channelName} channel "
                f"for image type {imageType}, fov {fov}, and "
                f"imagingRound {imagingRound} but expected one image file."
            )
            raise InputDataError(msg)
        filemapPath = selection["imagePath"].values[0]
        return os.path.join(
            self._dataSet.dataHome, self._dataSet.dataSetName, filemapPath
        )

    def _truncate_file_path(self, path) -> None:
        head, tail = os.path.split(path)
        return tail

    def _map_image_files(self, fiducial=False) -> None:
        regExpKey = "fiducialRegExp" if fiducial else "imageRegExp"
        filemap_name = "fiducial_filemap" if fiducial else "filemap"

        try:
            fileMap = self._dataSet.load_dataframe_from_csv(filemap_name)
            fileMap["imagePath"] = fileMap["imagePath"].apply(self._truncate_file_path)

        except FileNotFoundError:
            filenames = self._dataSet.get_image_file_names()
            if len(filenames) == 0:
                msg = f"No image files found at {self._dataSet.rawDataPath}"
                raise dataset.DataFormatException(msg)
            fileData = []
            for _, channel in self.data.iterrows():
                channelName, imageType, bitNumber, regExp = channel[
                    ["channelName", "imageType", "bitNumber", regExpKey]
                ]
                matchRE = re.compile(regExp)
                foundMatch = False
                for filename in filenames:
                    match = matchRE.match(os.path.split(filename)[-1])
                    if match is not None:
                        match = match.groupdict()
                        if match["imageType"] == imageType:
                            if "imagingRound" not in match:
                                match["imagingRound"] = -1
                            match["imagePath"] = filename
                            match["bitNumber"] = bitNumber
                            foundMatch = True
                            fileData.append(match)

                if not foundMatch:
                    msg = (
                        "Unable to find image files matching regular "
                        f"expression {regExp} "
                        f"for channel {channelName} and image type {imageType}."
                    )
                    raise dataset.DataFormatException(msg)

            fileMap = pd.DataFrame(fileData)
            convert_to_int = ["imagingRound", "fov", "bitNumber"]
            fileMap[convert_to_int] = fileMap[convert_to_int].astype(int)
            fileMap["imagePath"] = fileMap["imagePath"].apply(self._truncate_file_path)

            self._dataSet.save_dataframe_to_csv(fileMap, filemap_name, index=False)
        return fileMap

    def _validate_file_map(self, fiducial=False) -> None:
        """Checks a unique file is present and valid for each data channel and fov.

        Raises:
            InputDataError: If image files are missing, non-unique, or invalid.
        """
        expectedSize = None
        for dataChannel in self.get_data_channels():
            for fov in self.get_fovs(fiducial=fiducial):
                path = self._get_image_path(dataChannel, fov, fiducial=fiducial)

                if not self._dataSet.rawDataPortal.open_file(path).exists():
                    msg = (
                        f"Image data for channel {dataChannel} and fov {fov} "
                        f"not found. Expected at {path}"
                    )
                    raise InputDataError(msg)

                try:
                    size = self._dataSet.image_stack_size(path)
                except Exception:
                    msg = (
                        f"Unable to determine image stack size for fov {fov} "
                        f"from data channel {dataChannel} at {path}"
                    )
                    raise InputDataError(msg)

                frames = self.data.iloc[dataChannel][
                    "fiducialFrame" if fiducial else "frame"
                ]
                if np.max(frames) >= size[2]:
                    msg = (
                        f"Insufficient frames in data for channel {dataChannel} and "
                        f"fov {fov}. Expected {np.max(frames)} frames but only found "
                        f"{size[2]} in file {path}"
                    )
                    raise InputDataError(msg)

                if expectedSize is None:
                    expectedSize = [size[0], size[1]]
                else:
                    if expectedSize[0] != size[0] or expectedSize[1] != size[1]:
                        msg = (
                            f"Image data for channel {dataChannel} and fov {fov}"
                            " has unexpected dimensions. Expected "
                            f"{expectedSize[0]}x{expectedSize[1]} but found "
                            f"{size[0]}x{size[1]} in image file {path}"
                        )
                        raise InputDataError(msg)
