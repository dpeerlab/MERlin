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
        imageType = self.data.loc[dataChannel, "fiducialImageType"]
        imagingRound = self.data.loc[dataChannel, "fiducialImagingRound"]
        return self._get_image_path(self.fiducialFileMap, imageType, fov, imagingRound)

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
        channelInfo = self.data.iloc[dataChannel]
        imagePath = self._get_image_path(
            self.fileMap, channelInfo["imageType"], fov, channelInfo["imagingRound"]
        )
        return imagePath

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

    def get_fovs(self, fileMap=None) -> np.ndarray:
        if fileMap is None:
            fileMap = self.fileMap

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
        self, fileMap, imageType: str, fov: int, imagingRound: int
    ) -> str:
        selection = fileMap[
            (fileMap["imageType"] == imageType)
            & (fileMap["fov"] == fov)
            & (fileMap["imagingRound"] == imagingRound)
        ]
        filemapPath = selection["imagePath"].values[0]
        return os.path.join(
            self._dataSet.dataHome, self._dataSet.dataSetName, filemapPath
        )

    def _truncate_file_path(self, path) -> None:
        head, tail = os.path.split(path)
        return tail

    def _map_image_files(self, fiducial=False) -> None:
        regExpKey = "fiducialRegExp" if fiducial else "imageRegExp"
        filename = "fiducial_filemap" if fiducial else "filemap"

        try:
            fileMap = self._dataSet.load_dataframe_from_csv(filename)
            fileMap["imagePath"] = fileMap["imagePath"].apply(self._truncate_file_path)

        except FileNotFoundError:
            uniqueEntries = self.data.drop_duplicates(
                subset=["imageType", regExpKey], keep="first"
            )

            uniqueTypes = uniqueEntries["imageType"]
            uniqueIndexes = uniqueEntries.index.values.tolist()

            fileNames = self._dataSet.get_image_file_names()
            if len(fileNames) == 0:
                raise dataset.DataFormatException(
                    "No image files found at %s." % self._dataSet.rawDataPath
                )
            fileData = []
            for currentType, currentIndex in zip(uniqueTypes, uniqueIndexes):
                matchRE = re.compile(self.data[regExpKey][currentIndex])

                matchingFiles = False
                for currentFile in fileNames:
                    matchedName = matchRE.match(os.path.split(currentFile)[-1])
                    if matchedName is not None:
                        transformedName = matchedName.groupdict()
                        if transformedName["imageType"] == currentType:
                            if "imagingRound" not in transformedName:
                                transformedName["imagingRound"] = -1
                            transformedName["imagePath"] = currentFile
                            matchingFiles = True
                            fileData.append(transformedName)

                if not matchingFiles:
                    raise dataset.DataFormatException(
                        "Unable to identify image files matching regular "
                        f"expression {self.data[regExpKey][currentIndex]} "
                        f"for image type {currentType}."
                    )

            fileMap = pd.DataFrame(fileData)
            fileMap[["imagingRound", "fov"]] = fileMap[["imagingRound", "fov"]].astype(
                int
            )
            fileMap["imagePath"] = fileMap["imagePath"].apply(self._truncate_file_path)

            self._validate_file_map(fileMap, fiducial)

            self._dataSet.save_dataframe_to_csv(fileMap, filename, index=False)
        return fileMap

    def _validate_file_map(self, fileMap, fiducial) -> None:
        """Checks all files specified in the file map are present.

        Raises
        ------
            InputDataError: If the set of raw data is incomplete or the
                    format of the raw data deviates from expectations.
        """
        expectedImageSize = None
        for dataChannel in self.get_data_channels():
            for fov in self.get_fovs(fileMap):
                channelInfo = self.data.iloc[dataChannel]
                try:
                    imagePath = self._get_image_path(
                        fileMap,
                        channelInfo["imageType"],
                        fov,
                        channelInfo["imagingRound"],
                    )
                except IndexError:
                    raise FileNotFoundError(
                        f"Unable to find image path for {channelInfo['imageType']}, "
                        f"fov={fov}, round={channelInfo['imagingRound']}"
                    )

                if not self._dataSet.rawDataPortal.open_file(imagePath).exists():
                    msg = (
                        f"Image data for channel {dataChannel} and fov {fov} "
                        f"not found. Expected at {imagePath}"
                    )
                    raise InputDataError(msg)

                try:
                    imageSize = self._dataSet.image_stack_size(imagePath)
                except Exception:
                    msg = (
                        f"Unable to determine image stack size for fov {fov} "
                        f"from data channel {dataChannel} at {imagePath}"
                    )
                    raise InputDataError(msg)

                frames = (
                    channelInfo["fiducialFrame"] if fiducial else channelInfo["frame"]
                )
                if np.max(frames) >= imageSize[2]:
                    msg = (
                        f"Insufficient frames in data for channel {dataChannel} and "
                        f"fov {fov}. Expected {np.max(frames)} frames but only found "
                        f"{imageSize[2]} in file {imagePath}"
                    )
                    raise InputDataError(msg)

                if expectedImageSize is None:
                    expectedImageSize = [imageSize[0], imageSize[1]]
                else:
                    if (
                        expectedImageSize[0] != imageSize[0]
                        or expectedImageSize[1] != imageSize[1]
                    ):
                        msg = (
                            f"Image data for channel {dataChannel} and fov {fov}"
                            " has unexpected dimensions. Expected "
                            f"{expectedImageSize[0]}x{expectedImageSize[1]} but found "
                            f"{imageSize[0]}x{imageSize[1]} in image file {imagePath}"
                        )
                        raise InputDataError(msg)
