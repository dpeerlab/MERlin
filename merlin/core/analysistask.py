import os
import copy
from abc import ABC, abstractmethod
import time

import numpy as np


class AnalysisAlreadyStartedException(Exception):
    pass


class AnalysisTask(ABC):

    '''
    An abstract class for performing analysis on a DataSet. Subclasses
    should implement the analysis to perform in the run_analysis() function.
    '''

    def __init__(self, dataSet, parameters=None, analysisName=None):
        '''Creates an AnalysisTask object that performs analysis on the
        specified DataSet.

        Args:
            dataSet: the DataSet to run analysis on.
            parameters: a dictionary containing parameters used to run the
                analysis.
            analysisName: specifies a unique identifier for this
                AnalysisTask. If analysisName is not set, the analysis name
                will default to the name of the class.
        '''
        self.dataSet = dataSet
        if parameters is None:
            self.parameters = {}
        else:
            self.parameters = copy.deepcopy(parameters)

        if analysisName is None:
            self.analysisName = type(self).__name__
        else:
            self.analysisName = analysisName

        self.parameters['module'] = type(self).__module__
        self.parameters['class'] = type(self).__name__

        self.dataSet.save_analysis_task(self)

    def run(self):
        '''Run this AnalysisTask.
        
        Upon completion of the analysis, this function informs the DataSet
        that analysis is complete.
        '''

        if self.is_complete() or self.is_running():
            raise AnalysisAlreadyStartedException

        self.dataSet.record_analysis_running(self)
        self.run_analysis()
        self.dataSet.record_analysis_complete(self)

    @abstractmethod
    def run_analysis(self):
        '''Perform the analysis for this AnalysisTask.

        This function should be implemented in all subclasses with the
        logic to complete the analysis.
        '''
        pass

    @abstractmethod
    def get_estimated_memory(self):
        '''Get an estimate of how much memory is required for this
        AnalysisTask.

        Returns:
            a memory estimate in megabytes.
        '''
        pass

    @abstractmethod
    def get_estimated_time(self):
        '''Get an estimate for the amount of time required to complete
        this AnalysisTask.

        Returns:
            a time estimate in minutes.
        '''
        pass

    @abstractmethod
    def get_dependencies(self):
        '''Get the analysis tasks that must be completed before this 
        analysis task can proceed.

        Returns:
            a list containing the names of the analysis tasks that 
                this analysis task depends on
        '''
        pass

    def get_parameters(self):
        '''Get the parameters for this analysis task.

        Returns:
            the parameter dictionary
        '''
        return self.parameters

    def is_complete(self):
        '''Determines if this analysis has completed successfully
        
        Returns:
            True if the analysis is complete and otherwise False.
        '''
        return self.dataSet.check_analysis_done(self)

    def is_running(self):
        '''Determines if this analysis is currently running
        
        Returns:
            True if the analysis is complete and otherwise False.
        '''
        return self.dataSet.check_analysis_running(self) and not \
                self.is_complete()

    def get_analysis_name(self):
        '''Get the name for this AnalysisTask.

        Returns:
            the name of this AnalysisTask
        '''
        return self.analysisName


class ParallelAnalysisTask(AnalysisTask):

    '''
    An abstract class for analysis that can be run in multiple parts 
    independently. Subclasses should implement the analysis to perform in 
    the run_analysis() function
    '''

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    @abstractmethod
    def fragment_count(self):
        pass

    def run(self, fragmentIndex=None):
        if fragmentIndex is None:
            for i in range(self.fragment_count()):
                self.run(i)
        else:
            if self.is_complete(fragmentIndex) \
                    or self.is_running(fragmentIndex):
                raise AnalysisAlreadyStartedException    
            self.dataSet.record_analysis_running(self, fragmentIndex)
            self.run_analysis(fragmentIndex)
            self.dataSet.record_analysis_complete(self, fragmentIndex) 

    @abstractmethod
    def run_analysis(self, fragmentIndex):
        pass

    def is_complete(self, fragmentIndex=None):
        if fragmentIndex is None:
            for i in range(self.fragment_count()):
                if not self.is_complete(i):
                    return False

            return True

        else:
            return self.dataSet.check_analysis_done(self, fragmentIndex)

    def is_running(self, fragmentIndex=None):
        if fragmentIndex is None:
            for i in range(self.fragment_count()):
                if self.is_running(i):
                    return True 

            return False

        else:
            return self.dataSet.check_analysis_running(self, fragmentIndex) \
                    and not self.is_complete(fragmentIndex)

