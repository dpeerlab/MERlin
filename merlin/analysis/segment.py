from typing import List, Tuple

import cv2
import networkx as nx
import numpy as np
import rtree
from skimage import measure, segmentation

from merlin.core import analysistask, dataset
from merlin.util import spatialfeature, watershed


class FeatureSavingAnalysisTask(analysistask.ParallelAnalysisTask):
    """An abstract analysis class that saves features into a spatial feature
    database.
    """

    def __init__(
        self, dataSet: dataset.DataSet, parameters=None, analysisName=None
    ) -> None:
        super().__init__(dataSet, parameters, analysisName)

    def _reset_analysis(self, fov: int = None) -> None:
        super()._reset_analysis(fov)
        self.get_feature_database().empty_database(fov)

    def get_feature_database(self) -> spatialfeature.SpatialFeatureDB:
        """Get the spatial feature database this analysis task saves
        features into.

        Returns: The spatial feature database reference.
        """
        return spatialfeature.HDF5SpatialFeatureDB(self.dataSet, self)


class WatershedSegment(FeatureSavingAnalysisTask):
    """An analysis task that determines the boundaries of features in the
    image data in each field of view using a watershed algorithm.

    Since each field of view is analyzed individually, the segmentation results
    should be cleaned in order to merge cells that cross the field of
    view boundary.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None) -> None:
        super().__init__(dataSet, parameters, analysisName)

        if "seed_channel_name" not in self.parameters:
            self.parameters["seed_channel_name"] = "DAPI"
        if "watershed_channel_name" not in self.parameters:
            self.parameters["watershed_channel_name"] = "polyT"

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        # TODO - refine estimate
        return 2048

    def get_estimated_time(self):
        # TODO - refine estimate
        return 5

    def get_dependencies(self):
        dependencies = [
            self.parameters["warp_task"],
            self.parameters["global_align_task"],
        ]
        if "watershed_channel_task" in self.parameters:
            dependencies.append(self.parameters["watershed_channel_task"])
        return dependencies

    def get_cell_boundaries(self) -> List[spatialfeature.SpatialFeature]:
        featureDB = self.get_feature_database()
        return featureDB.read_features()

    def _get_seeds(self, fov: int) -> np.ndarray:
        index = self.dataSet.get_data_organization().get_data_channel_index(
            self.parameters["seed_channel_name"]
        )
        images = self._read_image_stack(fov, index)
        images = self._filter_image_stack(images, 5)
        seeds = watershed.separate_merged_seeds(watershed.extract_seeds(images))
        return seeds

    def _get_watershed_images(self, fov: int) -> Tuple[np.ndarray, np.ndarray]:
        if "watershed_channel_task" in self.parameters:
            task = self.dataSet.load_analysis_task(
                self.parameters["watershed_channel_task"]
            )
            images = task.get_images(fov)

        else:
            watershedIndex = (
                self.dataSet.get_data_organization().get_data_channel_index(
                    self.parameters["watershed_channel_name"]
                )
            )
            images = self._read_image_stack(fov, watershedIndex)
        images = self._filter_image_stack(images, 5)
        normalized_images, mask = watershed.prepare_watershed_images(images)
        return normalized_images, mask

    def _run_analysis(self, fov: int) -> None:
        globalTask = self.dataSet.load_analysis_task(
            self.parameters["global_align_task"]
        )

        seeds = self._get_seeds(fov)
        images, mask = self._get_watershed_images(fov)

        seeds[np.invert(mask)] = 0
        watershed = segmentation.watershed(
            images,
            measure.label(seeds),
            mask=mask,
            connectivity=np.ones((3, 3, 3)),
            watershed_line=True,
        )

        zPos = np.array(self.dataSet.get_data_organization().get_z_positions())
        featureList = [
            spatialfeature.SpatialFeature.feature_from_label_matrix(
                (watershed == i),
                fov,
                globalTask.fov_to_global_transform(fov),
                zPos,
            )
            for i in np.unique(watershed)
            if i != 0
        ]

        featureDB = self.get_feature_database()
        featureDB.write_features(featureList, fov)

    def _read_image_stack(self, fov: int, channel: int) -> np.ndarray:
        warpTask = self.dataSet.load_analysis_task(self.parameters["warp_task"])
        z_positions = self.dataSet.get_z_positions()
        images = [
            warpTask.get_aligned_image(fov, channel, z) for z in range(len(z_positions))
        ]
        return np.array(images)

    def _filter_image_stack(self, images: np.ndarray, filterSigma: float) -> np.ndarray:
        filterSize = int(2 * np.ceil(2 * filterSigma) + 1)
        n_slices = images.shape[0]
        filtered = [
            cv2.GaussianBlur(
                images[z],
                (filterSize, filterSize),
                filterSigma,
            )
            for z in range(n_slices)
        ]
        return np.array(filtered)


class CleanCellBoundaries(analysistask.ParallelAnalysisTask):
    """A task to construct a network graph where each cell is a node, and overlaps
    are represented by edges. This graph is then refined to assign cells to the
    fov they are closest to (in terms of centroid). This graph is then refined
    to eliminate overlapping cells to leave a single cell occupying a given
    position.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None) -> None:
        super().__init__(dataSet, parameters, analysisName)

        self.segmentTask = self.dataSet.load_analysis_task(
            self.parameters["segment_task"]
        )
        self.alignTask = self.dataSet.load_analysis_task(
            self.parameters["global_align_task"]
        )

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters["segment_task"], self.parameters["global_align_task"]]

    def return_exported_data(self, fov: int) -> nx.Graph:
        return self.dataSet.load_graph_from_gpickle("cleaned_cells", self, fov)

    def _run_analysis(self, fov: int) -> None:
        fovs = np.array(self.dataSet.get_fovs())
        boxes = self.alignTask.get_fov_boxes()
        intersections = sorted(
            [i for i, box in enumerate(boxes) if boxes[fov].intersects(box)]
        )
        intersecting_fovs = list(fovs[np.array(intersections)])

        spatialTree = rtree.index.Index()
        count = 0
        idToNum = {}
        for fov_ in intersecting_fovs:
            cells = self.segmentTask.get_feature_database().read_features(fov_)
            cells = spatialfeature.simple_clean_cells(cells)

            spatialTree, count, idToNum = spatialfeature.construct_tree(
                cells, spatialTree, count, idToNum
            )

        graph = nx.Graph()
        cells = self.segmentTask.get_feature_database().read_features(fov)
        cells = spatialfeature.simple_clean_cells(cells)
        graph = spatialfeature.construct_graph(
            graph, cells, spatialTree, fov, fovs, boxes
        )

        self.dataSet.save_graph_as_gpickle(graph, "cleaned_cells", self, fov)


class CombineCleanedBoundaries(analysistask.AnalysisTask):
    """A task to construct a network graph where each cell is a node, and overlaps
    are represented by edges. This graph is then refined to assign cells to the
    fov they are closest to (in terms of centroid). This graph is then refined
    to eliminate overlapping cells to leave a single cell occupying a given
    position.

    """

    def __init__(self, dataSet, parameters=None, analysisName=None) -> None:
        super().__init__(dataSet, parameters, analysisName)

        self.cleaningTask = self.dataSet.load_analysis_task(
            self.parameters["cleaning_task"]
        )

    def get_estimated_memory(self):
        # TODO - refine estimate
        return 2048

    def get_estimated_time(self):
        # TODO - refine estimate
        return 5

    def get_dependencies(self):
        return [self.parameters["cleaning_task"]]

    def return_exported_data(self):
        kwargs = {"index_col": 0}
        return self.dataSet.load_dataframe_from_csv(
            "all_cleaned_cells", analysisTask=self.analysisName, **kwargs
        )

    def _run_analysis(self):
        allFOVs = self.dataSet.get_fovs()
        graph = nx.Graph()
        for currentFOV in allFOVs:
            subGraph = self.cleaningTask.return_exported_data(currentFOV)
            graph = nx.compose(graph, subGraph)

        cleanedCells = spatialfeature.remove_overlapping_cells(graph)

        self.dataSet.save_dataframe_to_csv(
            cleanedCells, "all_cleaned_cells", analysisTask=self
        )


class RefineCellDatabases(FeatureSavingAnalysisTask):
    def __init__(self, dataSet, parameters=None, analysisName=None) -> None:
        super().__init__(dataSet, parameters, analysisName)

        self.segmentTask = self.dataSet.load_analysis_task(
            self.parameters["segment_task"]
        )
        self.cleaningTask = self.dataSet.load_analysis_task(
            self.parameters["combine_cleaning_task"]
        )

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        # TODO - refine estimate
        return 2048

    def get_estimated_time(self):
        # TODO - refine estimate
        return 5

    def get_dependencies(self):
        return [
            self.parameters["segment_task"],
            self.parameters["combine_cleaning_task"],
        ]

    def _run_analysis(self, fov: int) -> None:
        cleanedCells = self.cleaningTask.return_exported_data()
        originalCells = self.segmentTask.get_feature_database().read_features(fov)
        feature_database = self.get_feature_database()
        cleanedC = cleanedCells[cleanedCells["originalFOV"] == fov]
        cleanedGroups = cleanedC.groupby("assignedFOV")
        for k, g in cleanedGroups:
            cellsToConsider = g["cell_id"].values.tolist()
            features = [
                x for x in originalCells if str(x.get_feature_id()) in cellsToConsider
            ]
            feature_database.write_features(features, fov)


class ExportCellMetadata(analysistask.AnalysisTask):
    """An analysis task exports cell metadata."""

    def __init__(self, dataSet, parameters=None, analysisName=None) -> None:
        super().__init__(dataSet, parameters, analysisName)

        self.segmentTask = self.dataSet.load_analysis_task(
            self.parameters["segment_task"]
        )

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters["segment_task"]]

    def _run_analysis(self):
        df = self.segmentTask.get_feature_database().read_feature_metadata()

        self.dataSet.save_dataframe_to_csv(df, "feature_metadata", self.analysisName)
