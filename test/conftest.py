import glob
import os
import shutil

import pytest

import merlin
import test.simple_tasks as simple_tasks
from merlin.core import dataset

root = os.path.join(os.path.dirname(merlin.__file__), "..", "test")
merlin.DATA_HOME = os.path.abspath("test_data")
merlin.ANALYSIS_HOME = os.path.abspath("test_analysis")
merlin.ANALYSIS_PARAMETERS_HOME = os.path.abspath("test_analysis_parameters")
merlin.CODEBOOK_HOME = os.path.abspath("test_codebooks")
merlin.DATA_ORGANIZATION_HOME = os.path.abspath("test_dataorganization")
merlin.POSITION_HOME = os.path.abspath("test_positions")
merlin.MICROSCOPE_PARAMETERS_HOME = os.path.abspath("test_microcope_parameters")


dataDirectory = os.sep.join([merlin.DATA_HOME, "test"])
merfishDataDirectory = os.sep.join([merlin.DATA_HOME, "merfish_test"])


def copy_test_files(folder):
    to_copy = [
        (merlin.DATA_ORGANIZATION_HOME, "test_data_organization.csv"),
        (merlin.CODEBOOK_HOME, "test_codebook.csv"),
        (merlin.CODEBOOK_HOME, "test_codebook2.csv"),
        (merlin.POSITION_HOME, "test_positions.csv"),
        (merlin.ANALYSIS_PARAMETERS_HOME, "test_analysis_parameters.json"),
        (merlin.ANALYSIS_PARAMETERS_HOME, "test_max_project.json"),
        (merlin.MICROSCOPE_PARAMETERS_HOME, "test_microscope_parameters.json"),
    ]
    for home_folder, filename in to_copy:
        shutil.copyfile(
            os.sep.join([root, folder, filename]),
            os.sep.join([home_folder, filename]),
        )


@pytest.fixture(scope="session")
def base_files():
    folderList = [
        merlin.DATA_HOME,
        merlin.ANALYSIS_HOME,
        merlin.ANALYSIS_PARAMETERS_HOME,
        merlin.CODEBOOK_HOME,
        merlin.DATA_ORGANIZATION_HOME,
        merlin.POSITION_HOME,
        merlin.MICROSCOPE_PARAMETERS_HOME,
    ]
    for folder in folderList:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

    copy_test_files("auxiliary_files")

    yield

    for folder in folderList:
        shutil.rmtree(folder)


@pytest.fixture(scope="session")
def merfish_files(base_files):
    os.mkdir(merfishDataDirectory)

    for imageFile in glob.iglob(os.sep.join([root, "auxiliary_files", "*.tif"])):
        if os.path.isfile(imageFile):
            shutil.copy(imageFile, merfishDataDirectory)

    yield

    shutil.rmtree(merfishDataDirectory)


@pytest.fixture(scope="session")
def simple_data(base_files):
    os.mkdir(dataDirectory)
    testData = dataset.DataSet("test")

    yield testData

    shutil.rmtree(dataDirectory)


@pytest.fixture(scope="session")
def simple_merfish_data(merfish_files):
    testMERFISHData = dataset.MERFISHDataSet(
        "merfish_test",
        dataOrganizationName="test_data_organization.csv",
        codebookNames=["test_codebook.csv"],
        positionFileName="test_positions.csv",
        microscopeParametersName="test_microscope_parameters.json",
    )
    return testMERFISHData


@pytest.fixture(scope="session")
def two_codebook_merfish_data(merfish_files):
    testMERFISHData = dataset.MERFISHDataSet(
        "merfish_test",
        dataOrganizationName="test_data_organization.csv",
        codebookNames=["test_codebook2.csv", "test_codebook.csv"],
        positionFileName="test_positions.csv",
        analysisHome=os.path.join(
            merlin.ANALYSIS_HOME, "..", "test_analysis_two_codebook"
        ),
        microscopeParametersName="test_microscope_parameters.json",
    )
    yield testMERFISHData

    shutil.rmtree("test_analysis_two_codebook")


@pytest.fixture()
def single_task(simple_data):
    task = simple_tasks.SimpleAnalysisTask(
        simple_data, parameters={"a": 5, "b": "b_string"}
    )
    yield task
    simple_data.delete_analysis(task)


@pytest.fixture(
    params=[
        simple_tasks.SimpleAnalysisTask,
        simple_tasks.SimpleParallelAnalysisTask,
        simple_tasks.SimpleInternallyParallelAnalysisTask,
    ],
)
def simple_task(simple_data, request):
    task = request.param(simple_data, parameters={"a": 5, "b": "b_string"})
    yield task
    simple_data.delete_analysis(task)


@pytest.fixture(
    params=[
        simple_tasks.SimpleAnalysisTask,
        simple_tasks.SimpleParallelAnalysisTask,
        simple_tasks.SimpleInternallyParallelAnalysisTask,
    ],
)
def simple_merfish_task(simple_merfish_data, request):
    task = request.param(simple_merfish_data, parameters={"a": 5, "b": "b_string"})
    yield task
    simple_merfish_data.delete_analysis(task)
