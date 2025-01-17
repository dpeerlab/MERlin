import contextlib
import io
import json
import subprocess
import time

import numpy as np
import pandas as pd
import requests
from matplotlib import pyplot as plt

from merlin.core import analysistask


class SlurmReport(analysistask.AnalysisTask):
    """An analysis task that generates reports on previously completed analysis
    tasks using Slurm.

    This analysis task only works when Merlin is run through Slurm
    with every analysis task fragment run as a separate job.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None) -> None:
        super().__init__(dataSet, parameters, analysisName)

        if "codebook_index" not in self.parameters:
            self.parameters["codebook_index"] = 0

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        return [self.parameters["run_after_task"]]

    def _generate_slurm_report(self, task: analysistask.AnalysisTask):
        if isinstance(task, analysistask.ParallelAnalysisTask):
            idList = [
                self.dataSet.get_analysis_environment(task, i)["SLURM_JOB_ID"]
                for i in range(task.fragment_count())
            ]
        else:
            idList = [self.dataSet.get_analysis_environment(task)["SLURM_JOB_ID"]]

        queryResult = subprocess.run(
            [
                "sacct",
                "--format=AssocID,Account,Cluster,User,JobID,JobName,"
                + "NodeList,AveCPU,AveCPUFreq,MaxPages,MaxDiskRead,MaxDiskWrite,"
                + "MaxRSS,ReqMem,CPUTime,Elapsed,Submit,Start,End,Timelimit",
                "--units=M",
                "-P",
                "-j",
                ",".join(idList),
            ],
            stdout=subprocess.PIPE,
        )

        slurmJobDF = pd.read_csv(
            io.StringIO(queryResult.stdout.decode("utf-8")), sep="|"
        )

        return self._clean_slurm_dataframe(slurmJobDF)

    @staticmethod
    def _clean_slurm_dataframe(inputDataFrame):
        outputDF = inputDataFrame[~inputDataFrame["JobID"].str.contains(".extern")]
        outputDF = outputDF.assign(JobID=outputDF["JobID"].str.partition(".")[0])

        def get_not_nan(listIn):
            return listIn.dropna().iloc[0]

        outputDF = outputDF.groupby("JobID").aggregate(get_not_nan)

        def reformat_timedelta(elapsedIn):
            splitElapsed = elapsedIn.split("-")
            if len(splitElapsed) > 1:
                return splitElapsed[0] + " days " + splitElapsed[1]
            else:
                return splitElapsed[0]

        outputDF = outputDF.assign(
            Elapsed=pd.to_timedelta(
                outputDF["Elapsed"].apply(reformat_timedelta), unit="s"
            )
        )
        outputDF = outputDF.assign(
            Timelimit=pd.to_timedelta(
                outputDF["Timelimit"].apply(reformat_timedelta), unit="s"
            )
        )
        outputDF = outputDF.assign(
            Queued=pd.to_timedelta(
                pd.to_datetime(outputDF["Start"]) - pd.to_datetime(outputDF["Submit"]),
                unit="s",
            )
        )

        return outputDF.reindex()

    def _plot_slurm_report(self, slurmDF, analysisName):
        fig = plt.figure(figsize=(15, 4))

        plt.subplot(1, 4, 1)
        plt.boxplot(
            [
                slurmDF["MaxRSS"].str[:-1].astype(float),
                slurmDF["ReqMem"].str[:-2].astype(int),
            ],
            widths=0.5,
        )
        plt.xticks([1, 2], ["Max used", "Requested"])
        plt.ylabel("Memory (mb)")
        plt.title("RAM")
        plt.subplot(1, 4, 2)
        plt.boxplot(
            [
                slurmDF["Queued"] / np.timedelta64(1, "m"),
                slurmDF["Elapsed"] / np.timedelta64(1, "m"),
                slurmDF["Timelimit"] / np.timedelta64(1, "m"),
            ],
            widths=0.5,
        )
        plt.xticks([1, 2, 3], ["Queued", "Elapsed", "Requested"])
        plt.ylabel("Time (min)")
        plt.title("Run time")
        plt.subplot(1, 4, 3)
        plt.boxplot([slurmDF["MaxDiskRead"].str[:-1].astype(float)], widths=0.25)
        plt.xticks([1], ["MaxDiskRead"])
        plt.ylabel("Number of mb read")
        plt.title("Disk usage")
        plt.subplot(1, 4, 4)
        plt.boxplot([slurmDF["MaxDiskWrite"].str[:-1].astype(float)], widths=0.25)
        plt.xticks([1], ["MaxDiskWrite"])
        plt.ylabel("Number of mb written")
        plt.suptitle(analysisName)
        plt.tight_layout(pad=1)
        self.dataSet.save_figure(self, fig, analysisName)

    def _plot_slurm_summary(self, reportDict):
        def setBoxColors(bPlot, c):
            for element in ["boxes", "whiskers", "fliers", "means", "medians", "caps"]:
                plt.setp(bPlot[element], color=c)

        # Plot memory requested and used for each task
        fig = plt.figure(figsize=(15, 12))

        bp = plt.boxplot(
            [d["MaxRSS"].str[:-1].astype(float) for d in reportDict.values()],
            positions=np.arange(len(reportDict)) - 0.15,
            widths=0.25,
        )
        setBoxColors(bp, "r")
        bp = plt.boxplot(
            [d["ReqMem"].str[:-2].astype(float) for d in reportDict.values()],
            positions=np.arange(len(reportDict)) + 0.15,
            widths=0.25,
        )
        setBoxColors(bp, "b")
        plt.xticks(
            np.arange(len(reportDict)), list(reportDict.keys()), rotation="vertical"
        )
        plt.yscale("log")
        (hB,) = plt.plot([1, 1], "b-")
        (hR,) = plt.plot([1, 1], "r-")
        plt.legend((hB, hR), ("Requested", "Max used"))
        hB.set_visible(False)
        hR.set_visible(False)
        plt.ylabel("Memory per job (mb)")
        plt.title("Memory summary")
        plt.ylim([100, plt.ylim()[1]])
        plt.xlim([-0.5, len(reportDict) - 0.5])
        plt.vlines(
            np.arange(0.5, len(reportDict)),
            ymin=plt.ylim()[0],
            ymax=plt.ylim()[1],
            linestyles="dashed",
        )
        plt.tight_layout(pad=1)
        self.dataSet.save_figure(self, fig, "memory_summary")

        # Plot time requested, queued and used for each task
        fig = plt.figure(figsize=(15, 12))
        bp = plt.boxplot(
            [d["Elapsed"] / np.timedelta64(1, "m") for d in reportDict.values()],
            positions=np.arange(len(reportDict)) - 0.15,
            widths=0.25,
        )
        setBoxColors(bp, "r")
        bp = plt.boxplot(
            [d["Timelimit"] / np.timedelta64(1, "m") for d in reportDict.values()],
            positions=np.arange(len(reportDict)) + 0.15,
            widths=0.25,
        )
        setBoxColors(bp, "b")
        bp = plt.boxplot(
            [d["Queued"] / np.timedelta64(1, "m") for d in reportDict.values()],
            positions=np.arange(len(reportDict)) + 0.15,
            widths=0.25,
        )
        setBoxColors(bp, "g")
        plt.xticks(
            np.arange(len(reportDict)), list(reportDict.keys()), rotation="vertical"
        )
        plt.yscale("log")
        (hB,) = plt.plot([1, 1], "b-")
        (hR,) = plt.plot([1, 1], "r-")
        (hG,) = plt.plot([1, 1], "g-")
        plt.legend((hB, hR, hG), ("Requested", "Used", "Queued"))
        hB.set_visible(False)
        hR.set_visible(False)
        hG.set_visible(False)
        plt.ylabel("Time per job (min)")
        plt.title("Time summary")
        plt.xlim([-0.5, len(reportDict) + 0.5])
        plt.vlines(
            np.arange(0.5, len(reportDict)),
            ymin=plt.ylim()[0],
            ymax=plt.ylim()[1],
            linestyles="dashed",
        )
        plt.tight_layout(pad=1)
        self.dataSet.save_figure(self, fig, "time_summary")

    def _run_analysis(self):
        taskList = self.dataSet.get_analysis_tasks()

        reportTime = int(time.time())
        reportDict = {}
        analysisParameters = {}
        for t in taskList:
            currentTask = self.dataSet.load_analysis_task(t)
            try:
                if currentTask.is_complete():
                    slurmDF = self._generate_slurm_report(currentTask)
                    self.dataSet.save_dataframe_to_csv(slurmDF, t, self, "reports")
                    dfStream = io.StringIO()
                    slurmDF.to_csv(dfStream, sep="|")
                    self._plot_slurm_report(slurmDF, t)
                    reportDict[t] = slurmDF
                    analysisParameters[t] = currentTask.get_parameters()

                    with contextlib.suppress(requests.exceptions.RequestException):
                        requests.post(
                            "http://merlin.georgeemanuel.com/post",
                            files={
                                "file": (
                                    ".".join(
                                        [t, self.dataSet.dataSetName, str(reportTime)]
                                    )
                                    + ".csv",
                                    dfStream.getvalue(),
                                )
                            },
                            timeout=10,
                        )

            except Exception:
                pass

        self._plot_slurm_summary(reportDict)

        datasetMeta = {
            "image_width": self.dataSet.imageDimensions[0],
            "image_height": self.dataSet.imageDimensions[1],
            "barcode_length": self.dataSet.get_codebook(
                self.parameters["codebook_index"]
            ).get_bit_count(),
            "barcode_count": self.dataSet.get_codebook(
                self.parameters["codebook_index"]
            ).get_barcode_count(),
            "fov_count": len(self.dataSet.get_fovs()),
            "z_count": len(self.dataSet.get_z_positions()),
            "sequential_count": len(
                self.dataSet.get_data_organization().get_sequential_rounds()
            ),
            "dataset_name": self.dataSet.dataSetName,
            "report_time": reportTime,
            "analysis_parameters": analysisParameters,
        }
        with contextlib.suppress(requests.exceptions.RequestException):
            requests.post(
                "http://merlin.georgeemanuel.com/post",
                files={
                    "file": (
                        ".".join([self.dataSet.dataSetName, str(reportTime)]) + ".json",
                        json.dumps(datasetMeta),
                    )
                },
                timeout=10,
            )
