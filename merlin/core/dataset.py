import contextlib
import datetime
import importlib
import json
import logging
import os
import pickle
import shutil
import time
from typing import Dict, List, Optional, Tuple, Union

import h5py
import networkx as nx
import numpy as np
import pandas as pd
import tables
import tifffile
import xmltodict
from matplotlib import pyplot as plt

import merlin
from merlin.core import analysistask
from merlin.data import codebook, dataorganization
from merlin.util import dataportal, imagereader

TaskOrName = Union[analysistask.AnalysisTask, str]


class DataFormatException(Exception):
    pass


class DataSet:
    def __init__(
        self, dataDirectoryName: str, dataHome: str = None, analysisHome: str = None
    ) -> None:
        """Create a dataset for the specified raw data.

        Args:
        ----
            dataDirectoryName: the relative directory to the raw data
            dataHome: the base path to the data. The data is expected
                    to be in dataHome/dataDirectoryName. If dataHome
                    is not specified, DATA_HOME is read from the
                    .env file.
            analysisHome: the base path for storing analysis results. Analysis
                    results for this DataSet will be stored in
                    analysisHome/dataDirectoryName. If analysisHome is not
                    specified, ANALYSIS_HOME is read from the .env file.
        """
        if dataHome is None:
            dataHome = merlin.DATA_HOME
        if analysisHome is None:
            analysisHome = merlin.ANALYSIS_HOME

        self.dataSetName = dataDirectoryName
        self.dataHome = dataHome
        self.analysisHome = analysisHome

        self.rawDataPath = os.sep.join([dataHome, dataDirectoryName])
        self.rawDataPortal = dataportal.DataPortal.create_portal(self.rawDataPath)
        if not self.rawDataPortal.is_available():
            print(f"The raw data is not available at {self.rawDataPath}")

        self.analysisPath = os.sep.join([analysisHome, dataDirectoryName])
        os.makedirs(self.analysisPath, exist_ok=True)

        self.logPath = os.sep.join([self.analysisPath, "logs"])
        os.makedirs(self.logPath, exist_ok=True)

        self._store_dataset_metadata()

    def _store_dataset_metadata(self) -> None:
        try:
            oldMetadata = self.load_json_analysis_result("dataset", None)
            if not merlin.is_compatible(oldMetadata["merlin_version"]):
                raise merlin.IncompatibleVersionException(
                    (
                        "Analysis was performed on dataset %s with MERlin "
                        + "version %s, which is not compatible with the current "
                        + "MERlin version %s"
                    )
                    % (self.dataSetName, oldMetadata["version"], merlin.version())
                )
        except FileNotFoundError:
            newMetadata = {
                "merlin_version": merlin.version(),
                "module": type(self).__module__,
                "class": type(self).__name__,
                "dataset_name": self.dataSetName,
                "creation_date": str(datetime.datetime.now()),
            }
            self.save_json_analysis_result(newMetadata, "dataset", None)

    def save_workflow(self, workflowString: str) -> str:
        """Save a snakemake workflow for analysis of this dataset.

        Args:
        ----
            workflowString: a string containing the snakemake workflow
                to save

        Returns: the path to the saved workflow
        """
        snakemakePath = self.get_snakemake_path()
        os.makedirs(snakemakePath, exist_ok=True)

        workflowPath = (
            os.sep.join(
                [snakemakePath, datetime.datetime.now().strftime("%y%m%d_%H%M%S")]
            )
            + ".Snakefile"
        )
        with open(workflowPath, "w") as outFile:
            outFile.write(workflowString)

        return workflowPath

    def get_snakemake_path(self) -> str:
        """Get the directory for storing files related to snakemake.

        Returns: the snakemake path as a string
        """
        return os.sep.join([self.analysisPath, "snakemake"])

    def save_figure(
        self,
        analysisTask: TaskOrName,
        figure: plt.Figure,
        figureName: str,
        subdirectory: str = "figures",
    ) -> None:
        """Save the figure into the analysis results for this DataSet.

        This function will save the figure in both png and pdf formats.

        Args:
        ----
            analysisTask: the analysis task that generated this figure.
            figure: the figure handle for the figure to save
            figureName: the name of the file to store the figure in, excluding
                    extension
            subdirectory: the name of the subdirectory within the specified
                    analysis task to save the figures.
        """
        savePath = os.sep.join(
            [self.get_analysis_subdirectory(analysisTask, subdirectory), figureName]
        )

        figure.savefig(savePath + ".png", pad_inches=0)
        figure.savefig(savePath + ".pdf", transparent=True, pad_inches=0)

    def figure_exists(
        self, analysisTask: TaskOrName, figureName: str, subdirectory: str = "figures"
    ) -> bool:
        """Determine if a figure with the specified name has been
        saved within the results for the specified analysis task.

        This function only checks for the png formats.

        Args:
        ----
            analysisTask: the analysis task that generated this figure.
            figureName: the name of the file to store the figure in, excluding
                    extension
            subdirectory: the name of the subdirectory within the specified
                    analysis task to save the figures.
        """
        savePath = (
            os.sep.join(
                [self.get_analysis_subdirectory(analysisTask, subdirectory), figureName]
            )
            + ".png"
        )

        return os.path.exists(savePath)

    def get_analysis_image_set(
        self, analysisTask: TaskOrName, imageBaseName: str, imageIndex: int = None
    ) -> np.ndarray:
        """Get an analysis image set saved in the analysis for this data set.

        Args:
        ----
            analysisTask: the analysis task that generated and stored the
                image set.
            imageBaseName: the base name of the image
            imageIndex: index of the image set to retrieve
        """
        return tifffile.imread(
            self._analysis_image_name(analysisTask, imageBaseName, imageIndex)
        )

    def get_analysis_image(
        self,
        analysisTask: TaskOrName,
        imageBaseName: str,
        imageIndex: int,
        imagesPerSlice: int,
        sliceIndex: int,
        frameIndex: int,
    ) -> np.ndarray:
        """Get an image from an image set save in the analysis for this
        data set.

        Args:
        ----
            analysisTask: the analysis task that generated and stored the
                image set.
            imageBaseName: the base name of the image
            imageIndex: index of the image set to retrieve
            imagesPerSlice: the number of images in each slice of the image
                file
            sliceIndex: the index of the slice to get the image
            frameIndex: the index of the frame in the specified slice
        """
        # TODO - It may be useful to add a function that gets all
        # frames in a slice
        imageFile = tifffile.TiffFile(
            self._analysis_image_name(analysisTask, imageBaseName, imageIndex)
        )
        indexInFile = sliceIndex * imagesPerSlice + frameIndex
        return imageFile.asarray(key=int(indexInFile))

    def writer_for_analysis_images(
        self,
        analysisTask: TaskOrName,
        imageBaseName: str,
        imageIndex: int = None,
        imagej: bool = False,
    ) -> tifffile.TiffWriter:
        """Get a writer for writing tiff files from an analysis task.

        Args:
        ----
            analysisTask:
            imageBaseName:
            imageIndex:
            imagej:
        Returns:

        """
        return tifffile.TiffWriter(
            self._analysis_image_name(analysisTask, imageBaseName, imageIndex),
            imagej=imagej,
        )

    # TODO: make slice vs. frame clearer
    # believe that sliceCount is the number of z-slices in a stack
    # and frameCount is the number of channels?
    @staticmethod
    def analysis_tiff_description(sliceCount: int, frameCount: int) -> Dict:
        imageDescription = {
            "ImageJ": "1.47a\n",
            "images": sliceCount * frameCount,
            "channels": 1,
            "slices": sliceCount,
            "frames": frameCount,
            "hyperstack": True,
            "loop": False,
        }
        return imageDescription

    def _analysis_image_name(
        self, analysisTask: TaskOrName, imageBaseName: str, imageIndex: int
    ) -> str:
        destPath = self.get_analysis_subdirectory(analysisTask, subdirectory="images")
        if imageIndex is None:
            return os.sep.join([destPath, imageBaseName + ".tif"])
        else:
            return os.sep.join([destPath, imageBaseName + str(imageIndex) + ".tif"])

    def _analysis_result_save_path(
        self,
        resultName: str,
        analysisTask: TaskOrName,
        resultIndex: int = None,
        subdirectory: str = None,
        fileExtension: str = None,
    ) -> str:
        saveName = resultName
        if resultIndex is not None:
            saveName += "_" + str(resultIndex)
        if fileExtension is not None:
            saveName += fileExtension

        if analysisTask is None:
            return os.sep.join([self.analysisPath, saveName])
        else:
            return os.sep.join(
                [self.get_analysis_subdirectory(analysisTask, subdirectory), saveName]
            )

    def list_analysis_files(
        self,
        analysisTask: TaskOrName = None,
        subdirectory: str = None,
        extension: str = None,
        fullPath: bool = True,
    ) -> List[str]:
        basePath = self._analysis_result_save_path(
            "", analysisTask, subdirectory=subdirectory
        )
        fileList = os.listdir(basePath)
        if extension:
            fileList = [x for x in fileList if x.endswith(extension)]
        if fullPath:
            fileList = [os.path.join(basePath, x) for x in fileList]
        return fileList

    def save_graph_as_gpickle(
        self,
        graph: nx.Graph,
        resultName: str,
        analysisTask: TaskOrName = None,
        resultIndex: int = None,
        subdirectory: str = None,
    ):
        """Save a networkx graph as a gpickle into the analysis results.

        Args:
        ----
            graph: the networkx graph to save
            resultName: the base name of the output file
            analysisTask: the analysis task that the graph should be
                saved under. If None, the graph is saved to the
                data set root.
            resultIndex: index of the graph to save or None if no index
                should be specified
            subdirectory: subdirectory of the analysis task that the graph
                should be saved to or None if the graph should be
                saved to the root directory for the analysis task.
        """
        savePath = self._analysis_result_save_path(
            resultName, analysisTask, resultIndex, subdirectory, ".gpickle"
        )
        with open(savePath, "wb") as f:
            pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)

    def load_graph_from_gpickle(
        self,
        resultName: str,
        analysisTask: TaskOrName = None,
        resultIndex: int = None,
        subdirectory: str = None,
    ):
        """Load a networkx graph from a gpickle objective saved in the analysis
        results.

        Args:
        ----
            resultName: the base name of the output file
            analysisTask: the analysis task that the graph should be
                saved under. If None, the graph is saved to the
                data set root.
            resultIndex: index of the graph to save or None if no index
                should be specified
            subdirectory: subdirectory of the analysis task that the graph
                should be saved to or None if the graph should be
                saved to the root directory for the analysis task.
        """
        savePath = self._analysis_result_save_path(
            resultName, analysisTask, resultIndex, subdirectory, ".gpickle"
        )
        with open(savePath, "rb") as f:
            G = pickle.load(f)
        return G

    def save_dataframe_to_csv(
        self,
        dataframe: pd.DataFrame,
        resultName: str,
        analysisTask: TaskOrName = None,
        resultIndex: int = None,
        subdirectory: str = None,
        **kwargs,
    ) -> None:
        """Save a DataFrame to a csv file stored in this dataset.

        If a previous DataFrame has been save with the same resultName,
        it will be overwritten

        Args:
        ----
            dataframe: the data frame to save
            resultName: the name of the output file
            analysisTask: the analysis task that the dataframe should be
                saved under. If None, the dataframe is saved to the
                data set root.
            resultIndex: index of the dataframe to save or None if no index
                should be specified
            subdirectory: subdirectory of the analysis task that the dataframe
                should be saved to or None if the dataframe should be
                saved to the root directory for the analysis task.
            **kwargs: arguments to pass on to pd.to_csv
        """
        savePath = self._analysis_result_save_path(
            resultName, analysisTask, resultIndex, subdirectory, ".csv"
        )

        with open(savePath, "w") as f:
            dataframe.to_csv(f, **kwargs)

    def load_dataframe_from_csv(
        self,
        resultName: str,
        analysisTask: TaskOrName = None,
        resultIndex: int = None,
        subdirectory: str = None,
        **kwargs,
    ) -> Union[pd.DataFrame, None]:
        """Load a DataFrame from a csv file stored in this data set.

        Args:
        ----
            resultName:
            analysisTask:
            resultIndex:
            subdirectory:
            **kwargs:
        Returns:
            DataFrame
        Raises:
              FileNotFoundError: if the file does not exist
        """
        savePath = self._analysis_result_save_path(
            resultName, analysisTask, resultIndex, subdirectory, ".csv"
        )
        with open(savePath) as f:
            return pd.read_csv(f, **kwargs)

    def open_pandas_hdfstore(
        self,
        mode: str,
        resultName: str,
        analysisName: str,
        resultIndex: int = None,
        subdirectory: str = None,
    ) -> pd.HDFStore:
        savePath = self._analysis_result_save_path(
            resultName, analysisName, resultIndex, subdirectory, ".h5"
        )
        return pd.HDFStore(savePath, mode=mode)

    def delete_pandas_hdfstore(
        self,
        resultName: str,
        analysisTask: TaskOrName = None,
        resultIndex: int = None,
        subdirectory: str = None,
    ) -> None:
        hPath = self._analysis_result_save_path(
            resultName, analysisTask, resultIndex, subdirectory, ".h5"
        )
        if os.path.exists(hPath):
            os.remove(hPath)

    def open_table(
        self,
        mode: str,
        resultName: str,
        analysisName: str,
        resultIndex: int = None,
        subdirectory: str = None,
    ) -> tables.file:
        savePath = self._analysis_result_save_path(
            resultName, analysisName, resultIndex, subdirectory, ".h5"
        )
        return tables.open_file(savePath, mode=mode)

    def delete_table(
        self,
        resultName: str,
        analysisTask: TaskOrName = None,
        resultIndex: int = None,
        subdirectory: str = None,
    ) -> None:
        """Delete an hdf5 file stored in this data set if it exists.

        Args:
        ----
            resultName: the name of the output file
            analysisTask: the analysis task that should be associated with this
                hdf5 file. If None, the file is assumed to be in the
                data set root.
            resultIndex: index of the dataframe to save or None if no index
                should be specified
            subdirectory: subdirectory of the analysis task that the dataframe
                should be saved to or None if the dataframe should be
                saved to the root directory for the analysis task.
        """
        hPath = self._analysis_result_save_path(
            resultName, analysisTask, resultIndex, subdirectory, ".h5"
        )

        if os.path.exists(hPath):
            os.remove(hPath)

    def open_hdf5_file(
        self,
        mode: str,
        resultName: str,
        analysisTask: TaskOrName = None,
        resultIndex: int = None,
        subdirectory: str = None,
    ) -> h5py.File:
        """Open an hdf5 file stored in this data set.

        Args:
        ----
            mode: the mode for opening the file, either 'r', 'r+', 'w', 'w-',
                or 'a'.
            resultName: the name of the output file
            analysisTask: the analysis task that should be associated with this
                hdf5 file. If None, the file is assumed to be in the
                data set root.
            resultIndex: index of the dataframe to save or None if no index
                should be specified
            subdirectory: subdirectory of the analysis task that the dataframe
                should be saved to or None if the dataframe should be
                saved to the root directory for the analysis task.

        Returns:
        -------
            a h5py file object connected to the hdf5 file
        Raise:
            FileNotFoundError: if the mode is 'r' and the specified hdf5 file
                does not exist
        """
        hPath = self._analysis_result_save_path(
            resultName, analysisTask, resultIndex, subdirectory, ".hdf5"
        )
        if mode == "r" and not os.path.exists(hPath):
            raise FileNotFoundError(
                ("Unable to open %s for reading since " + "it does not exist.") % hPath
            )

        return h5py.File(hPath, mode)

    def delete_hdf5_file(
        self,
        resultName: str,
        analysisTask: TaskOrName = None,
        resultIndex: int = None,
        subdirectory: str = None,
    ) -> None:
        """Delete an hdf5 file stored in this data set if it exists.

        Args:
        ----
            resultName: the name of the output file
            analysisTask: the analysis task that should be associated with this
                hdf5 file. If None, the file is assumed to be in the
                data set root.
            resultIndex: index of the dataframe to save or None if no index
                should be specified
            subdirectory: subdirectory of the analysis task that the dataframe
                should be saved to or None if the dataframe should be
                saved to the root directory for the analysis task.
        """
        hPath = self._analysis_result_save_path(
            resultName, analysisTask, resultIndex, subdirectory, ".hdf5"
        )
        if os.path.exists(hPath):
            os.remove(hPath)

    def save_json_analysis_result(
        self,
        analysisResult: Dict,
        resultName: str,
        analysisName: str,
        resultIndex: int = None,
        subdirectory: str = None,
    ) -> None:
        savePath = self._analysis_result_save_path(
            resultName, analysisName, resultIndex, subdirectory, ".json"
        )
        with open(savePath, "w") as f:
            json.dump(analysisResult, f)

    def load_json_analysis_result(
        self,
        resultName: str,
        analysisName: str,
        resultIndex: int = None,
        subdirectory: str = None,
    ) -> Dict:
        savePath = self._analysis_result_save_path(
            resultName, analysisName, resultIndex, subdirectory, ".json"
        )
        with open(savePath) as f:
            return json.load(f)

    def load_pickle_analysis_result(
        self,
        resultName: str,
        analysisName: str,
        resultIndex: int = None,
        subdirectory: str = None,
    ) -> Dict:
        savePath = self._analysis_result_save_path(
            resultName, analysisName, resultIndex, subdirectory, ".pkl"
        )
        with open(savePath, "rb") as f:
            return pickle.load(f)

    def save_pickle_analysis_result(
        self,
        analysisResult,
        resultName: str,
        analysisName: str,
        resultIndex: int = None,
        subdirectory: str = None,
    ):
        savePath = self._analysis_result_save_path(
            resultName, analysisName, resultIndex, subdirectory, ".pkl"
        )
        with open(savePath, "wb") as f:
            pickle.dump(analysisResult, f)

    def save_numpy_analysis_result(
        self,
        analysisResult: np.ndarray,
        resultName: str,
        analysisName: str,
        resultIndex: int = None,
        subdirectory: str = None,
    ) -> None:
        savePath = self._analysis_result_save_path(
            resultName, analysisName, resultIndex, subdirectory
        )
        np.save(savePath, analysisResult)

    def save_numpy_txt_analysis_result(
        self,
        analysisResult: np.ndarray,
        resultName: str,
        analysisName: str,
        resultIndex: int = None,
        subdirectory: str = None,
    ) -> None:
        savePath = self._analysis_result_save_path(
            resultName, analysisName, resultIndex, subdirectory
        )
        np.savetxt(savePath + ".csv", analysisResult)

    def load_numpy_analysis_result(
        self,
        resultName: str,
        analysisName: str,
        resultIndex: int = None,
        subdirectory: str = None,
    ) -> np.array:
        savePath = self._analysis_result_save_path(
            resultName, analysisName, resultIndex, subdirectory, ".npy"
        )
        return np.load(savePath, allow_pickle=True)

    def load_numpy_analysis_result_if_available(
        self,
        resultName: str,
        analysisName: str,
        defaultValue,
        resultIndex: int = None,
        subdirectory: str = None,
    ) -> np.array:
        """Load the specified analysis result or return the specified default
        value if the analysis result does not exist.

        Args:
        ----
            resultName: The name of the analysis result
            analysisName: The name of the analysis task the result is saved in
            defaultValue: The value to return if the specified analysis result
                does not exist
            resultIndex: The index of the analysi result
            subdirectory: The subdirectory within the analysis task that the
                result is saved in
        Returns: The analysis result or defaultValue if the analysis result
            doesn't exist.
        """
        try:
            return self.load_numpy_analysis_result(
                resultName, analysisName, resultIndex, subdirectory
            )
        except OSError:
            return defaultValue

    def get_analysis_subdirectory(
        self, analysisTask: TaskOrName, subdirectory: str = None, create: bool = True
    ) -> str:
        """AnalysisTask can either be the class or a string containing the
        class name.

        create - Flag indicating if the analysis subdirectory should be
            created if it does not already exist.
        """
        if isinstance(analysisTask, analysistask.AnalysisTask):
            analysisName = analysisTask.get_analysis_name()
        else:
            analysisName = analysisTask

        if subdirectory is None:
            subdirectoryPath = os.sep.join([self.analysisPath, analysisName])
        else:
            subdirectoryPath = os.sep.join(
                [self.analysisPath, analysisName, subdirectory]
            )

        if create:
            os.makedirs(subdirectoryPath, exist_ok=True)

        return subdirectoryPath

    def get_task_subdirectory(self, analysisTask: TaskOrName):
        return self.get_analysis_subdirectory(analysisTask, subdirectory="tasks")

    def get_log_subdirectory(self, analysisTask: TaskOrName):
        return self.get_analysis_subdirectory(analysisTask, subdirectory="log")

    def save_analysis_task(
        self, analysisTask: analysistask.AnalysisTask, overwrite: bool = False
    ):
        saveName = os.sep.join([self.get_task_subdirectory(analysisTask), "task.json"])

        try:
            existingTask = self.load_analysis_task(analysisTask.get_analysis_name())

            existingParameters = existingTask.get_parameters().copy()
            existingVersion = existingParameters["merlin_version"]
            newParameters = analysisTask.get_parameters().copy()
            newVersion = newParameters["merlin_version"]

            if not merlin.is_compatible(existingVersion, newVersion):
                raise merlin.IncompatibleVersionException(
                    (
                        "Analysis task with name %s has been previously created "
                        + "with MERlin version %s, which is incompatible with "
                        + "the current MERlin version, %s. Please remove the "
                        + "old analysis folder to continue."
                    )
                    % (analysisTask.analysisName, existingVersion, newVersion)
                )

            existingParameters.pop("merlin_version")
            newParameters.pop("merlin_version")

            if not overwrite and not existingParameters == newParameters:
                raise analysistask.AnalysisAlreadyExistsException(
                    (
                        "Analysis task with name %s already exists in this "
                        + "data set with different parameters."
                    )
                    % analysisTask.get_analysis_name()
                )

        except FileNotFoundError:
            pass

        with open(saveName, "w") as outFile:
            json.dump(analysisTask.get_parameters(), outFile, indent=4)

    def load_analysis_task(self, analysisTaskName: str) -> analysistask.AnalysisTask:
        loadName = os.sep.join(
            [self.get_task_subdirectory(analysisTaskName), "task.json"]
        )

        with open(loadName) as inFile:
            parameters = json.load(inFile)
            analysisModule = importlib.import_module(parameters["module"])
            analysisTask = getattr(analysisModule, parameters["class"])
            return analysisTask(self, parameters, analysisTaskName)

    def delete_analysis(self, analysisTask: TaskOrName) -> None:
        """Remove all files associated with the provided analysis
        from this data set.

        Before deleting an analysis task, it must be verified that the
        analysis task is not running.
        """
        analysisDirectory = self.get_analysis_subdirectory(analysisTask)
        shutil.rmtree(analysisDirectory)

    def get_analysis_tasks(self) -> List[str]:
        """Get a list of the analysis tasks within this dataset.

        Returns: A list of the analysis task names.
        """
        analysisList = []
        for a in os.listdir(self.analysisPath):
            if os.path.isdir(os.path.join(self.analysisPath, a)):
                if os.path.exists(os.path.join(self.analysisPath, a, "tasks")):
                    analysisList.append(a)

        analysisList.sort()
        return analysisList

    def analysis_exists(self, analysisTaskName: str) -> bool:
        """Determine if an analysis task with the specified name exists in this
        dataset.
        """
        analysisPath = self.get_analysis_subdirectory(analysisTaskName, create=False)
        return os.path.exists(analysisPath)

    def get_logger(
        self, analysisTask: analysistask.AnalysisTask, fragmentIndex: int = None
    ) -> logging.Logger:
        loggerName = analysisTask.get_analysis_name()
        if fragmentIndex is not None:
            loggerName += "." + str(fragmentIndex)

        logger = logging.getLogger(loggerName)
        logger.setLevel(logging.DEBUG)
        fileHandler = logging.FileHandler(self._log_path(analysisTask, fragmentIndex))
        fileHandler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

        return logger

    def close_logger(
        self, analysisTask: analysistask.AnalysisTask, fragmentIndex: int = None
    ) -> None:
        loggerName = analysisTask.get_analysis_name()
        if fragmentIndex is not None:
            loggerName += "." + str(fragmentIndex)

        logger = logging.getLogger(loggerName)

        handlerList = list(logger.handlers)
        for handler in handlerList:
            logger.removeHandler(handler)
            handler.flush()
            handler.close()

    def _log_path(
        self, analysisTask: analysistask.AnalysisTask, fragmentIndex: int = None
    ) -> str:
        logName = analysisTask.get_analysis_name()
        if fragmentIndex is not None:
            logName += "_" + str(fragmentIndex)
        logName += ".log"

        return os.sep.join([self.get_log_subdirectory(analysisTask), logName])

    def _analysis_status_file(
        self,
        analysisTask: analysistask.AnalysisTask,
        eventName: str,
        fragmentIndex: int = None,
    ) -> str:
        if isinstance(analysisTask, str):
            analysisTask = self.load_analysis_task(analysisTask)

        if fragmentIndex is None:
            fileName = analysisTask.get_analysis_name() + "." + eventName
        else:
            fileName = (
                analysisTask.get_analysis_name()
                + "_"
                + str(fragmentIndex)
                + "."
                + eventName
            )
        return os.sep.join([self.get_task_subdirectory(analysisTask), fileName])

    def get_analysis_environment(
        self, analysisTask: analysistask.AnalysisTask, fragmentIndex: int = None
    ) -> None:
        """Get the environment variables for the system used to run the
        specified analysis task.

        Args:
        ----
            analysisTask: The completed analysis task to get the environment
                variables for.
            fragmentIndex: The fragment index of the analysis task to
                get the environment variables for.

        Returns: A dictionary of the environment variables. If the job has not
            yet run, then None is returned.
        """
        if not self.check_analysis_done(analysisTask, fragmentIndex):
            return None

        fileName = self._analysis_status_file(
            analysisTask, "environment", fragmentIndex
        )
        with open(fileName) as inFile:
            envDict = json.load(inFile)
        return envDict

    def _record_analysis_environment(
        self, analysisTask: analysistask.AnalysisTask, fragmentIndex: int = None
    ) -> None:
        fileName = self._analysis_status_file(
            analysisTask, "environment", fragmentIndex
        )
        with open(fileName, "w") as outFile:
            json.dump(dict(os.environ), outFile, indent=4)

    def record_analysis_started(
        self, analysisTask: analysistask.AnalysisTask, fragmentIndex: int = None
    ) -> None:
        self._record_analysis_event(analysisTask, "start", fragmentIndex)
        self._record_analysis_environment(analysisTask, fragmentIndex)

    def record_analysis_running(
        self, analysisTask: analysistask.AnalysisTask, fragmentIndex: int = None
    ) -> None:
        self._record_analysis_event(analysisTask, "run", fragmentIndex)

    def record_analysis_complete(
        self, analysisTask: analysistask.AnalysisTask, fragmentIndex: int = None
    ) -> None:
        self._record_analysis_event(analysisTask, "done", fragmentIndex)

    def record_analysis_error(
        self, analysisTask: analysistask.AnalysisTask, fragmentIndex: int = None
    ) -> None:
        self._record_analysis_event(analysisTask, "error", fragmentIndex)

    def get_analysis_start_time(
        self, analysisTask: analysistask.AnalysisTask, fragmentIndex: int = None
    ) -> float:
        """Get the time that this analysis task started.

        Returns
        -------
            The start time for the analysis task execution in seconds since
            the epoch in UTC.
        """
        with open(
            self._analysis_status_file(analysisTask, "start", fragmentIndex)
        ) as f:
            return float(f.read())

    def get_analysis_complete_time(
        self, analysisTask: analysistask.AnalysisTask, fragmentIndex: int = None
    ) -> float:
        """Get the time that this analysis task completed.

        Returns
        -------
            The completion time for the analysis task execution in seconds since
            the epoch in UTC.
        """
        with open(self._analysis_status_file(analysisTask, "done", fragmentIndex)) as f:
            return float(f.read())

    def get_analysis_elapsed_time(
        self, analysisTask: analysistask.AnalysisTask, fragmentIndex: int = None
    ) -> float:
        """Get the time that this analysis took to complete.

        Returns
        -------
            The elapsed time for the analysis task execution in seconds.
            Returns None if the analysis task has not yet completed.
        """
        return self.get_analysis_complete_time(
            analysisTask, fragmentIndex
        ) - self.get_analysis_start_time(analysisTask, fragmentIndex)

    def _record_analysis_event(
        self,
        analysisTask: analysistask.AnalysisTask,
        eventName: str,
        fragmentIndex: int = None,
    ) -> None:
        fileName = self._analysis_status_file(analysisTask, eventName, fragmentIndex)
        with open(fileName, "w") as f:
            f.write("%s" % time.time())

    def _check_analysis_event(
        self,
        analysisTask: analysistask.AnalysisTask,
        eventName: str,
        fragmentIndex: int = None,
    ) -> bool:
        fileName = self._analysis_status_file(analysisTask, eventName, fragmentIndex)
        return os.path.exists(fileName)

    def _reset_analysis_event(
        self,
        analysisTask: analysistask.AnalysisTask,
        eventName: str,
        fragmentIndex: int = None,
    ):
        fileName = self._analysis_status_file(analysisTask, eventName, fragmentIndex)

        with contextlib.suppress(FileNotFoundError):
            os.remove(fileName)

    def is_analysis_idle(
        self, analysisTask: analysistask.AnalysisTask, fragmentIndex: int = None
    ) -> bool:
        fileName = self._analysis_status_file(analysisTask, "run", fragmentIndex)
        try:
            return time.time() - os.path.getmtime(fileName) > 1
        except FileNotFoundError:
            return True

    def check_analysis_started(
        self, analysisTask: analysistask.AnalysisTask, fragmentIndex: int = None
    ) -> bool:
        return self._check_analysis_event(analysisTask, "start", fragmentIndex)

    def check_analysis_done(
        self, analysisTask: analysistask.AnalysisTask, fragmentIndex: int = None
    ) -> bool:
        return self._check_analysis_event(analysisTask, "done", fragmentIndex)

    def analysis_done_filename(
        self, analysisTask: analysistask.AnalysisTask, fragmentIndex: int = None
    ) -> str:
        return self._analysis_status_file(analysisTask, "done", fragmentIndex)

    def check_analysis_error(
        self, analysisTask: analysistask.AnalysisTask, fragmentIndex: int = None
    ) -> bool:
        return self._check_analysis_event(analysisTask, "error", fragmentIndex)

    def reset_analysis_status(
        self, analysisTask: analysistask.AnalysisTask, fragmentIndex: int = None
    ):
        if analysisTask.is_running():
            raise analysistask.AnalysisAlreadyStartedException()

        self._reset_analysis_event(analysisTask, "start", fragmentIndex)
        self._reset_analysis_event(analysisTask, "run", fragmentIndex)
        self._reset_analysis_event(analysisTask, "done", fragmentIndex)
        self._reset_analysis_event(analysisTask, "error", fragmentIndex)
        self._reset_analysis_event(analysisTask, "done")


class ImageDataSet(DataSet):
    """
    Attributes:
        flipHorizontal (bool): Whether to flip the image horizontally when loading.
        flipVertical (bool): Whether to flip the image vertically when loading.
        transpose (bool): Whether to transpose the image when loading.
        micronsPexPixel (float): Conversion factor to convert pixels to microns.
        imageDimensions (tuple[int, int]): width and height of each image in pixels
    """

    def __init__(
        self,
        dataDirectoryName: str,
        dataHome: str = None,
        analysisHome: str = None,
        microscopeParametersName: str = None,
    ) -> None:
        """Create a dataset for the specified raw data.

        Args:
            dataDirectoryName: the relative directory to the raw data
            dataHome: the base path to the data. The data is expected
                    to be in dataHome/dataDirectoryName. If dataHome
                    is not specified, DATA_HOME is read from the
                    .env file.
            analysisHome: the base path for storing analysis results. Analysis
                    results for this DataSet will be stored in
                    analysisHome/dataDirectoryName. If analysisHome is not
                    specified, ANALYSIS_HOME is read from the .env file.
            microscopeParametersName: the name of the microscope parameters
                    file that specifies properties of the microscope used
                    to acquire the images represented by this ImageDataSet
        """
        super().__init__(dataDirectoryName, dataHome, analysisHome)

        if microscopeParametersName is not None:
            self._import_microscope_parameters(microscopeParametersName)

        self._load_microscope_parameters()

    def get_image_file_names(self):
        return sorted(
            self.rawDataPortal.list_files(extensionList=[".dax", ".tif", ".tiff"])
        )

    def load_image(self, path: str, frame: int) -> np.ndarray:
        """Loads a frame from an image file.

        Args:
            path (str): Path to image.
            frame (int): Frame within the image to load.

        Returns:
            np.ndarray: Image data for given frame.
        """
        with imagereader.infer_reader(self.rawDataPortal.open_file(path)) as reader:
            imageIn = reader.load_frame(int(frame))
            if self.transpose:
                imageIn = np.transpose(imageIn)
            if self.flipHorizontal:
                imageIn = np.flip(imageIn, axis=1)
            if self.flipVertical:
                imageIn = np.flip(imageIn, axis=0)
            return imageIn

    def image_stack_size(self, path: str) -> Tuple[int, int, int]:
        """Get the size of the image stack stored in the specified image path.

        Args:
            path (str): path to the image file

        Returns:
            tuple[int, int, int]: width, height, and n_frames of images
        """
        with imagereader.infer_reader(self.rawDataPortal.open_file(path)) as reader:
            return reader.size

    def _import_microscope_parameters(self, microscopeParametersName):
        sourcePath = os.sep.join(
            [merlin.MICROSCOPE_PARAMETERS_HOME, microscopeParametersName]
        )
        destPath = os.sep.join([self.analysisPath, "microscope_parameters.json"])

        shutil.copyfile(sourcePath, destPath)

    def _load_microscope_parameters(self):
        path = os.sep.join([self.analysisPath, "microscope_parameters.json"])

        if os.path.exists(path):
            with open(path) as inputFile:
                self.microscopeParameters = json.load(inputFile)
        else:
            self.microscopeParameters = {}

        self.flipHorizontal = self.microscopeParameters.get("flip_horizontal", True)
        self.flipVertical = self.microscopeParameters.get("flip_vertical", False)
        self.transpose = self.microscopeParameters.get("transpose", True)
        self.micronsPerPixel = self.microscopeParameters.get("microns_per_pixel", 0.108)
        self.imageDimensions = self.microscopeParameters.get(
            "image_dimensions", [2048, 2048]
        )

    def get_image_xml_metadata(self, imagePath: str) -> Dict:
        """Get the xml metadata stored for the specified image.

        Args:
        ----
            imagePath: the path to the image file (.dax or .tif)
        Returns: the metadata from the associated xml file
        """
        filePortal = self.rawDataPortal.open_file(imagePath).get_sibling_with_extension(
            ".xml"
        )
        return xmltodict.parse(filePortal.read_as_text())


class MERFISHDataSet(ImageDataSet):
    def __init__(
        self,
        dataDirectoryName: str,
        codebookNames: List[str] = None,
        dataOrganizationName: str = None,
        positionFileName: str = None,
        dataHome: str = None,
        analysisHome: str = None,
        microscopeParametersName: str = None,
    ) -> None:
        """Create a MERFISH dataset for the specified raw data.

        Args:
        ----
            dataDirectoryName: the relative directory to the raw data
            codebookNames: A list of the names of codebooks to use. The codebook
                    should be present in the analysis parameters
                    directory. Full paths can be provided for codebooks
                    present other directories.
            dataOrganizationName: the name of the data organization to use.
                    The data organization should be present in the analysis
                    parameters directory. A full path can be provided for
                    a codebook present in another directory.
            positionFileName: the name of the position file to use.
            dataHome: the base path to the data. The data is expected
                    to be in dataHome/dataDirectoryName. If dataHome
                    is not specified, DATA_HOME is read from the
                    .env file.
            analysisHome: the base path for storing analysis results. Analysis
                    results for this DataSet will be stored in
                    analysisHome/dataDirectoryName. If analysisHome is not
                    specified, ANALYSIS_HOME is read from the .env file.
            microscopeParametersName: the name of the microscope parameters
                    file that specifies properties of the microscope used
                    to acquire the images represented by this ImageDataSet
        """
        super().__init__(
            dataDirectoryName, dataHome, analysisHome, microscopeParametersName
        )

        self.dataOrganization = dataorganization.DataOrganization(
            self, dataOrganizationName
        )
        if codebookNames:
            self.codebooks = [
                codebook.Codebook(self, name, i) for i, name in enumerate(codebookNames)
            ]
        else:
            self.codebooks = self.load_codebooks()

        if positionFileName is not None:
            self._import_positions(positionFileName)
        try:
            self._load_positions()
        except FileNotFoundError:
            print("No positions file found")

    def save_codebook(self, codebook: codebook.Codebook) -> None:
        """Store the specified codebook in this dataset.

        If a codebook with the same codebook index and codebook name as the
        specified codebook already exists in this dataset, it is not
        overwritten.

        Args:
        ----
            codebook: the codebook to store
        Raises:
            FileExistsError: If a codebook with the same codebook index but
                a different codebook name is already save within this dataset.
        """
        existingCodebookName = self.get_stored_codebook_name(
            codebook.get_codebook_index()
        )
        if (
            existingCodebookName
            and existingCodebookName != codebook.get_codebook_name()
        ):
            raise FileExistsError(
                (
                    "Unable to save codebook %s with index %i "
                    + " since codebook %s already exists with "
                    + "the same index"
                )
                % (
                    codebook.get_codebook_name(),
                    codebook.get_codebook_index(),
                    existingCodebookName,
                )
            )

        if not existingCodebookName:
            self.save_dataframe_to_csv(
                codebook.get_data(),
                "_".join(
                    [
                        "codebook",
                        str(codebook.get_codebook_index()),
                        codebook.get_codebook_name(),
                    ]
                ),
                index=False,
            )

    def load_codebooks(self) -> List[codebook.Codebook]:
        """Get all the codebooks stored within this dataset.

        Returns
        -------
            A list of all the stored codebooks.
        """
        codebookList = []

        currentIndex = 0
        currentCodebook = self.load_codebook(currentIndex)
        while currentCodebook is not None:
            codebookList.append(currentCodebook)
            currentIndex += 1
            currentCodebook = self.load_codebook(currentIndex)

        return codebookList

    def load_codebook(self, codebookIndex: int = 0) -> Optional[codebook.Codebook]:
        """Load the codebook stored within this dataset with the specified
        index.

        Args:
        ----
            codebookIndex: the index of the codebook to load.

        Returns:
        -------
            The codebook stored with the specified codebook index. If no
            codebook exists with the specified index then None is returned.
        """
        codebookFile = [
            x
            for x in self.list_analysis_files(extension=".csv")
            if ("codebook_%i_" % codebookIndex) in x
        ]
        if len(codebookFile) < 1:
            return None
        codebookName = "_".join(
            os.path.splitext(os.path.basename(codebookFile[0]))[0].split("_")[2:]
        )
        return codebook.Codebook(self, codebookFile[0], codebookIndex, codebookName)

    def get_stored_codebook_name(self, codebookIndex: int = 0) -> Optional[str]:
        """Get the name of the codebook stored within this dataset with the
        specified index.

        Args:
        ----
            codebookIndex: the index of the codebook to load to find the name
                of.

        Returns:
        -------
            The name of the codebook stored with the specified codebook index.
            If no codebook exists with the specified index then None is
            returned.
        """
        codebookFile = [
            x
            for x in self.list_analysis_files(extension=".csv")
            if ("codebook_%i_" % codebookIndex) in x
        ]
        if len(codebookFile) < 1:
            return None
        return "_".join(
            os.path.splitext(os.path.basename(codebookFile[0]))[0].split("_")[2:]
        )

    def get_codebooks(self) -> List[codebook.Codebook]:
        """Get the codebooks associated with this dataset.

        Returns
        -------
            A list containing the codebooks for this dataset.
        """
        return self.codebooks

    def get_codebook(self, codebookIndex: int = 0) -> codebook.Codebook:
        return self.codebooks[codebookIndex]

    def get_data_organization(self) -> dataorganization.DataOrganization:
        return self.dataOrganization

    def get_stage_positions(self) -> List[List[float]]:
        return self.positions

    def get_fov_offset(self, fov: int) -> Tuple[float, float]:
        """Get the offset of the specified fov in the global coordinate system.
        This offset is based on the anticipated stage position.

        Args:
        ----
            fov: index of the field of view
        Returns:
            A tuple specifying the x and y offset of the top right corner
            of the specified fov in pixels.
        """
        # TODO - this should be implemented using the position of the fov.
        return self.positions.loc[fov]["X"], self.positions.loc[fov]["Y"]

    def z_index_to_position(self, z: int) -> float:
        """Get the z position associated with the provided z index."""
        return self.get_z_positions()[z]

    def position_to_z_index(self, zPosition: float) -> int:
        """Get the z index associated with the specified z position.

        Raises
        ------
             Exception: If the provided z position is not specified in this
                dataset
        """
        zIndex = np.where(self.get_z_positions() == zPosition)[0]
        if len(zIndex) == 0:
            raise Exception("Requested z=%0.2f position not found." % zPosition)

        return zIndex[0]

    def get_z_positions(self) -> List[float]:
        """Get the z positions present in this dataset.

        Returns:
            A sorted list of all unique z positions
        """
        return self.dataOrganization.get_z_positions()

    def get_fovs(self) -> List[int]:
        return self.dataOrganization.get_fovs()

    def get_imaging_rounds(self) -> List[int]:
        # TODO - check this function
        return np.unique(self.dataOrganization.fileMap["imagingRound"])

    def get_raw_image(self, dataChannel, fov, zPosition):
        return self.load_image(
            self.dataOrganization.get_image_filename(dataChannel, fov),
            self.dataOrganization.get_image_frame_index(dataChannel, zPosition),
        )

    def get_fiducial_image(self, channel, fov, z=None) -> np.ndarray:
        """Get the fiducial image for the specified channel, fov, and z-slice.

        Args:
            channel (_type_): Channel of image.
            fov (_type_): Field of view of image.
            z (int, optional): Z-slice within fiducial bead file of image.
                Uses fiducialFrame from data organization .csv when not specified.

        Returns:
            np.ndarray with image of fiducial beads
        """
        if z is None:
            z = self.dataOrganization.get_fiducial_frame_index(channel)
        filename = self.dataOrganization.get_fiducial_filename(channel, fov)
        return self.load_image(filename, z)

    def _import_positions_from_metadata(self):
        positionData = []
        for f in self.get_fovs():
            metadata = self.get_image_xml_metadata(
                self.dataOrganization.get_image_filename(0, f)
            )
            currentPositions = metadata["settings"]["acquisition"]["stage_position"][
                "#text"
            ].split(",")
            positionData.append([float(x) for x in currentPositions])
        positionPath = os.sep.join([self.analysisPath, "positions.csv"])
        np.savetxt(positionPath, np.array(positionData), delimiter=",")

    def _load_positions(self):
        positionPath = os.sep.join([self.analysisPath, "positions.csv"])
        if not os.path.exists(positionPath):
            self._import_positions_from_metadata()
        self.positions = pd.read_csv(positionPath, header=None, names=["X", "Y"])

    def _import_positions(self, positionFileName):
        sourcePath = os.sep.join([merlin.POSITION_HOME, positionFileName])
        destPath = os.sep.join([self.analysisPath, "positions.csv"])

        shutil.copyfile(sourcePath, destPath)

    def _convert_parameter_list(self, listIn, castFunction, delimiter=";"):
        return [castFunction(x) for x in listIn.split(delimiter) if len(x) > 0]
