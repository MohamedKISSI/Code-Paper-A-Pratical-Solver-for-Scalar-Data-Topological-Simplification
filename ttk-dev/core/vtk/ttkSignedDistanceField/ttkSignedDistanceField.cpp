#include <vtkFloatArray.h>
#include <vtkImageData.h>
#include <vtkInformation.h>
#include <vtkInformationVector.h>
#include <vtkIntArray.h>
#include <vtkPointData.h>
#include <vtkStreamingDemandDrivenPipeline.h>

#include <ttkMacros.h>
#include <ttkSignedDistanceField.h>
#include <ttkUtils.h>

// vtkObjectFactoryNewMacro(vtkResampleToImage);
vtkStandardNewMacro(ttkSignedDistanceField);

//------------------------------------------------------------------------------
ttkSignedDistanceField::ttkSignedDistanceField() {
  this->SetNumberOfInputPorts(1);
  this->SetNumberOfOutputPorts(1);
}

//------------------------------------------------------------------------------
vtkImageData *ttkSignedDistanceField::GetOutput() {
  return vtkImageData::SafeDownCast(this->GetOutputDataObject(0));
}

//------------------------------------------------------------------------------
vtkTypeBool
  ttkSignedDistanceField::ProcessRequest(vtkInformation *request,
                                         vtkInformationVector **inputVector,
                                         vtkInformationVector *outputVector) {
  // generate the data
  if(request->Has(vtkDemandDrivenPipeline::REQUEST_DATA())) {
    return this->RequestData(request, inputVector, outputVector);
  }

  // execute information
  if(request->Has(vtkDemandDrivenPipeline::REQUEST_INFORMATION())) {
    return this->RequestInformation(request, inputVector, outputVector);
  }

  // propagate update extent
  if(request->Has(vtkStreamingDemandDrivenPipeline::REQUEST_UPDATE_EXTENT())) {
    return this->RequestUpdateExtent(request, inputVector, outputVector);
  }

  return this->Superclass::ProcessRequest(request, inputVector, outputVector);
}

//------------------------------------------------------------------------------
int ttkSignedDistanceField::RequestInformation(
  vtkInformation *,
  vtkInformationVector **,
  vtkInformationVector *outputVector) {
  int wholeExtent[6]
    = {0, this->SamplingDimensions[0] - 1, 0, this->SamplingDimensions[1] - 1,
       0, this->SamplingDimensions[2] - 1};

  vtkInformation *outInfo = outputVector->GetInformationObject(0);
  outInfo->Set(
    vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(), wholeExtent, 6);

  return 1;
}

//------------------------------------------------------------------------------
int ttkSignedDistanceField::RequestUpdateExtent(
  vtkInformation *,
  vtkInformationVector **inputVector,
  vtkInformationVector *) {
  // This filter always asks for whole extent downstream. To resample
  // a subset of a structured input, you need to use ExtractVOI.
  vtkInformation *inInfo = inputVector[0]->GetInformationObject(0);
  inInfo->Remove(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT());
  if(inInfo->Has(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT())) {
    inInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(),
                inInfo->Get(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT()),
                6);
  }

  return 1;
}

//------------------------------------------------------------------------------
int ttkSignedDistanceField::FillInputPortInformation(int port,
                                                     vtkInformation *info) {
  if(port == 0) {
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkDataSet");
    //  info->Append(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(),
    //  "vtkCompositeDataSet");
    return 1;
  }
  return 0;
}

//------------------------------------------------------------------------------
int ttkSignedDistanceField::FillOutputPortInformation(int port,
                                                      vtkInformation *info) {
  if(port == 0) {
    info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkImageData");
  } else
    return 0;
  return 1;
}

//------------------------------------------------------------------------------
void ttkSignedDistanceField::computeOutputInformation(
  vtkInformationVector **inputVector) {
  xResolution_ = SamplingDimensions[0];
  yResolution_ = SamplingDimensions[1];
  zResolution_ = SamplingDimensions[2];
  const auto domain = vtkDataSet::GetData(inputVector[0]);
  double bounds[6];
  domain->GetBounds(bounds);
  DataExtent[1] = xResolution_;
  DataExtent[3] = yResolution_;
  DataExtent[5] = zResolution_;
  int shift = (ExpandBox ? 2 : 0);
  spacing_[0] = (bounds[1] - bounds[0]) / (DataExtent[1] - 1 - shift);
  spacing_[1] = (bounds[3] - bounds[2]) / (DataExtent[3] - 1 - shift);
  spacing_[2] = (bounds[5] - bounds[4]) / (DataExtent[5] - 1 - shift);
  for(unsigned int i = 0; i < spacing_.size(); ++i)
    invSpacingSquared_[i] = 1.0 / spacing_[i] / spacing_[i];
  std::cout << DataExtent[1] << ", " << DataExtent[3] << ", " << DataExtent[5]
            << std::endl;
  std::cout << spacing_[0] << ", " << spacing_[1] << ", " << spacing_[2]
            << std::endl;
  std::cout << spacing_[0] * xResolution_ << ", " << spacing_[1] * yResolution_
            << ", " << spacing_[2] * zResolution_ << std::endl;
  Origin[0] = bounds[0];
  Origin[1] = bounds[2];
  Origin[2] = bounds[4];
  if(ExpandBox) {
    Origin[0] -= spacing_[0];
    Origin[1] -= spacing_[1];
    Origin[2] -= spacing_[2];
  }
}

int ttkSignedDistanceField::RequestData(vtkInformation *ttkNotUsed(request),
                                        vtkInformationVector **inputVector,
                                        vtkInformationVector *outputVector) {
  using ttk::SimplexId;

  const auto domain = vtkDataSet::GetData(inputVector[0]);

  if(!domain)
    return !this->printErr("Unable to retrieve required input data object.");

  // triangulation Domain
  auto triangulation = ttkAlgorithm::GetTriangulation(domain);
  if(!triangulation) {
    this->printErr("Input triangulation pointer is NULL.");
    return -1;
  }

  this->preconditionTriangulation(triangulation);

  if(triangulation->isEmpty()) {
    this->printErr("Triangulation allocation problem.");
    return -1;
  }

  if(!domain) {
    this->printErr("Input pointer is NULL.");
    return -1;
  }

  const auto numberOfVertices = domain->GetNumberOfPoints();
  if(numberOfVertices <= 0) {
    this->printErr("Domain has no points.");
    return -5;
  }

  std::cout << "number of vertices = " << numberOfVertices << std::endl;
  std::cout << "triangulation->getNumberOfVertices() = "
            << triangulation->getNumberOfVertices() << std::endl;
  std::cout << "triangulation->getNumberOfTriangles() = "
            << triangulation->getNumberOfTriangles() << std::endl;

  /*for(int i = 0; i < numberOfVertices; i++) {
    if(triangulation->isVertexOnBoundary(i)) {
      std::cout << "yes !!" << std::endl;
    }
  }*/

  // Create encompassing triangulation
  computeOutputInformation(inputVector);
  vtkSmartPointer<vtkImageData> imageData
    = vtkSmartPointer<vtkImageData>::New();
  imageData->SetOrigin(Origin[0], Origin[1], Origin[2]);
  imageData->SetSpacing(spacing_[0], spacing_[1], spacing_[2]);
  imageData->SetDimensions(DataExtent[1], DataExtent[3], DataExtent[5]);
  /*imageData->AllocateScalars(VTK_FLOAT, 1);
  imageData->GetPointData()->SetNumberOfTuples(imageData->GetNumberOfPoints());*/

  auto encompassingTriangulation = ttkAlgorithm::GetTriangulation(imageData);
  if(!encompassingTriangulation) {
    this->printErr("Input encompassing triangulation pointer is NULL.");
    return -1;
  }

  this->preconditionTriangulation(encompassingTriangulation);

  if(encompassingTriangulation->isEmpty()) {
    this->printErr("Encompassing triangulation allocation problem.");
    return -1;
  }

  std::cout << "encompassingTriangulation->getNumberOfVertices() = "
            << encompassingTriangulation->getNumberOfVertices() << std::endl;
  std::cout << "encompassingTriangulation->getNumberOfEdges() = "
            << encompassingTriangulation->getNumberOfEdges() << std::endl;

  // create output arrays
  auto outputNoPoints = imageData->GetNumberOfPoints();

  vtkNew<vtkFloatArray> outputScalars{};
  outputScalars->SetNumberOfTuples(outputNoPoints);
  outputScalars->SetName("signedDistanceField");

  vtkNew<vtkIntArray> edgeCrossing{};
  edgeCrossing->SetNumberOfTuples(outputNoPoints);
  edgeCrossing->SetName("edgeCrossing");

  vtkNew<vtkIntArray> isInterior{};
  isInterior->SetNumberOfTuples(outputNoPoints);
  isInterior->SetName("isInterior");

  int ret{};

  naiveMethod_ = NaiveMethod;
  naiveMethodAllEdges_ = NaiveMethodAllEdges;
  fastMarching_ = FastMarching;
  fastMarchingOrder_ = FastMarchingOrder;
  fastMarchingIterativeBand_ = FastMarchingIterativeBand;
  fastMarchingIterativeBandRatio_ = FastMarchingIterativeBandRatio;
  ttkVtkTemplate2Macro(
    outputScalars->GetDataType(), triangulation->getType(),
    encompassingTriangulation->getType(),
    (ret = this->execute(
       ttkUtils::GetPointer<VTK_TT>(outputScalars),
       (static_cast<TTK_TT1 *>(triangulation->getData())),
       (static_cast<TTK_TT2 *>(encompassingTriangulation->getData())),
       ttkUtils::GetPointer<int>(edgeCrossing),
       ttkUtils::GetPointer<int>(isInterior))));

  // something wrong in baseCode
  if(ret) {
    this->printErr("SignedDistanceField.execute() error code: "
                   + std::to_string(ret));
    return -12;
  }

  imageData->GetPointData()->AddArray(outputScalars);
  imageData->GetPointData()->AddArray(edgeCrossing);
  imageData->GetPointData()->AddArray(isInterior);

  // Set the output
  auto output = vtkImageData::GetData(outputVector);
  output->ShallowCopy(imageData);

  return 1;
}
