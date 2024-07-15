# trace generated using paraview version 5.11.0-headless
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 11

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'XML Image Data Reader'
data = XMLImageDataReader(registrationName='baselineOptimizationApproach.vti', FileName=['./../results/baselineOptimizationApproach.vti'])
data.PointArrayStatus = ['Result', 'Modification Number', 'Last Change', 'Id Block']

# Properties modified on data
data.TimeArray = 'None'

UpdatePipeline(time=0.0, proxy=data)

# create a new 'Calculator'
calculator1 = Calculator(registrationName='Calculator1', Input=data)
calculator1.Function = ''

# Properties modified on calculator1
calculator1.Function = 'Result'

UpdatePipeline(time=0.0, proxy=calculator1)

# create a new 'XML Image Data Reader'
aneurismvti = XMLImageDataReader(registrationName='aneurism.vti', FileName=['./../aneurism.vti'])
aneurismvti.PointArrayStatus = ['ImageFile']

# Properties modified on aneurismvti
aneurismvti.TimeArray = 'None'

UpdatePipeline(time=0.0, proxy=aneurismvti)

# create a new 'Calculator'
calculator2 = Calculator(registrationName='Calculator2', Input=aneurismvti)
calculator2.Function = ''

# Properties modified on calculator2
calculator2.ResultArrayName = 'initialData'
calculator2.Function = 'ImageFile'

UpdatePipeline(time=0.0, proxy=calculator2)

# create a new 'TTK ScalarFieldNormalizer'
tTKScalarFieldNormalizer1 = TTKScalarFieldNormalizer(registrationName='TTKScalarFieldNormalizer1', Input=calculator2)
tTKScalarFieldNormalizer1.ScalarField = ['POINTS', 'initialData']

UpdatePipeline(time=0.0, proxy=tTKScalarFieldNormalizer1)

# set active source
SetActiveSource(calculator1)

# create a new 'Append Attributes'
appendAttributes1 = AppendAttributes(registrationName='AppendAttributes1', Input=[tTKScalarFieldNormalizer1, calculator1])

UpdatePipeline(time=0.0, proxy=appendAttributes1)

# create a new 'TTK LDistance'
tTKLDistance1 = TTKLDistance(registrationName='TTKLDistance1', Input=appendAttributes1)
tTKLDistance1.ScalarField1 = ['POINTS', 'Result']
tTKLDistance1.ScalarField2 = ['POINTS', 'initialData']

# Properties modified on tTKLDistance1
tTKLDistance1.Outputname = 'L2-distance'

UpdatePipeline(time=0.0, proxy=tTKLDistance1)
