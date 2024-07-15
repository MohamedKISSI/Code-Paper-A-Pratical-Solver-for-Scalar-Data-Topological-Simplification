# trace generated using paraview version 5.11.0-headless
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 11

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

datavti = XMLImageDataReader(registrationName='aneurism.vti', FileName=['./../aneurism.vti'])
datavti.PointArrayStatus = ['ImageFile']
# Properties modified on datavti
datavti.TimeArray = 'None'

# UpdatePipeline(time=0.0, proxy=datavti)

# create a new 'Calculator'
calculator1 = Calculator(registrationName='Calculator1', Input=datavti)
calculator1.Function = 'ImageFile'

UpdatePipeline(time=0.0, proxy=calculator1)

# set active source
SetActiveSource(calculator1)

# create a new 'TTK ScalarFieldNormalizer'
tTKScalarFieldNormalizer1 = TTKScalarFieldNormalizer(registrationName='TTKScalarFieldNormalizer1', Input=calculator1)
tTKScalarFieldNormalizer1.ScalarField = ['POINTS', 'Result']

UpdatePipeline(time=0.0, proxy=tTKScalarFieldNormalizer1)

# create a new 'TTK PersistenceDiagram'
tTKPersistenceDiagram1 = TTKPersistenceDiagram(registrationName='TTKPersistenceDiagram1', Input=tTKScalarFieldNormalizer1)
tTKPersistenceDiagram1.ScalarField = ['POINTS', 'Result']
tTKPersistenceDiagram1.InputOffsetField = ['POINTS', 'Result']

UpdatePipeline(time=0.0, proxy=tTKPersistenceDiagram1)

# create a new 'Threshold'
threshold1 = Threshold(registrationName='Threshold1', Input=tTKPersistenceDiagram1)
threshold1.Scalars = ['CELLS', 'Persistence']
threshold1.LowerThreshold = 0.01
threshold1.UpperThreshold = 2

UpdatePipeline(time=0.0, proxy=threshold1)

# set active source
SetActiveSource(tTKScalarFieldNormalizer1)

# create a new 'TTK TopologicalOptimization'
tTKTopologicalOptimization1 = TTKTopologicalOptimization(registrationName='TTKTopologicalOptimization1', Domain=tTKScalarFieldNormalizer1,
    Constraints=threshold1)
tTKTopologicalOptimization1.ScalarField = ['POINTS', 'Result']

# Properties modified on tTKTopologicalOptimization1
tTKTopologicalOptimization1.Method = 'Adam'
tTKTopologicalOptimization1.Epochnumber = 3000
tTKTopologicalOptimization1.ChooseLearningRate = 1
tTKTopologicalOptimization1.LearningRate = 0.0001
tTKTopologicalOptimization1.Forceminimumprecisiononmatchings = 1
tTKTopologicalOptimization1.Minimalrelativeprecision = 0.01
tTKTopologicalOptimization1.CoefStopCondition = 0.01

UpdatePipeline(time=0.0, proxy=tTKTopologicalOptimization1)

# save data
SaveData(f'./../results/baselineOptimizationApproach.vti', proxy=tTKTopologicalOptimization1, ChooseArraysToWrite=1, PointDataArrays=['Id Block', 'Result', 'Last Change', 'Modification Number'])

