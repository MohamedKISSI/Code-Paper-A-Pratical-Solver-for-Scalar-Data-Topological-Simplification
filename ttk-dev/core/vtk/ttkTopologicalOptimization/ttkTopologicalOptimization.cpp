#include "Debug.h"
#include "Triangulation.h"
#include <vtkDataArray.h>
#include <vtkDataSet.h>
#include <vtkUnstructuredGrid.h>
#include <vtkIdTypeArray.h>
#include <vtkInformation.h>
#include <vtkInformationVector.h>
#include <vtkIntArray.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkPointSet.h>
#include <vtkSmartPointer.h>

#include <vtkImageData.h>
#include <vtkStructuredGrid.h>
#include <vtkImageResize.h>
#include <vtkImageDataGeometryFilter.h>

#include <ttkMacros.h>
#include <ttkTopologicalOptimization.h>
#include <ttkUtils.h>

#include <LocalizedTopologicalOptimization.h>

vtkStandardNewMacro(ttkTopologicalOptimization);

ttkTopologicalOptimization::ttkTopologicalOptimization() {
  this->SetNumberOfInputPorts(2);
  this->SetNumberOfOutputPorts(1);
}

int ttkTopologicalOptimization::FillInputPortInformation(
  int port, vtkInformation *info) {
  if(port == 0) {
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkDataSet");
    return 1;
  } else if(port == 1) {
    // modifier vtkUnstructuredGrid
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkUnstructuredGrid");
    return 1;
  }
  return 0;
}

int ttkTopologicalOptimization::FillOutputPortInformation(
  int port, vtkInformation *info) {
  if(port == 0) {
    info->Set(ttkAlgorithm::SAME_DATA_TYPE_AS_INPUT_PORT(), 0);
    return 1;
  }
  return 0;
}

int ttkTopologicalOptimization::RequestData(
  vtkInformation *ttkNotUsed(request),
  vtkInformationVector **inputVector,
  vtkInformationVector *outputVector) {

  using ttk::SimplexId;

  const auto domain = vtkDataSet::GetData(inputVector[0]);
  // const auto constraints = vtkPointSet::GetData(inputVector[1]);
  const auto constraints = vtkUnstructuredGrid::GetData(inputVector[1]);

  if(!domain || !constraints)
    return !this->printErr("Unable to retrieve required input data objects.");


  auto output = vtkDataSet::GetData(outputVector);

  // DOMAIN 

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

  // domain offset field
  const auto inputOrder
    = this->GetOrderArray(domain, 0, triangulation, false,  2, ForceInputOffsetScalarField);
  if(!inputOrder) {
    this->printErr("Wrong input offset scalar field.");
    return -1;
  }

  // Constraints 

  ttk::DiagramType constraintDiagram; 
  const ttk::Debug dbg; 
  VTUToDiagram(constraintDiagram,
                constraints,
                dbg);
  

  // domain scalar field
  const auto inputScalars = this->GetInputArrayToProcess(0, domain);
  if(!inputScalars) {
    this->printErr("Input scalar field pointer is null.");
    return -3;
  }

  // create output arrays
  auto outputScalars
    = vtkSmartPointer<vtkDataArray>::Take(inputScalars->NewInstance());
  outputScalars->DeepCopy(inputScalars);

  // auto modificationNumber = vtkSmartPointer<vtkDataArray>::Take(inputScalars->NewInstance());
  // modificationNumber->DeepCopy(inputScalars);

  vtkNew<vtkIntArray> modificationNumber{};
  modificationNumber->SetNumberOfComponents(1);
  modificationNumber->SetNumberOfTuples(triangulation->getNumberOfVertices());

  vtkNew<vtkIntArray> idBlock{};
  idBlock->SetNumberOfComponents(1);
  idBlock->SetNumberOfTuples(triangulation->getNumberOfVertices());

  // auto lastChange = vtkSmartPointer<vtkDataArray>::Take(inputScalars->NewInstance());
  // lastChange->DeepCopy(inputScalars);

  vtkNew<vtkIntArray> lastChange{};
  lastChange->SetNumberOfComponents(1);
  lastChange->SetNumberOfTuples(triangulation->getNumberOfVertices());
  
  int ret{};
 
 
  ttkVtkTemplateMacro(
    inputScalars->GetDataType(), triangulation->getType(), 
    (
    ret = this->execute(ttkUtils::GetPointer<VTK_TT>(inputScalars),
                        ttkUtils::GetPointer<VTK_TT>(outputScalars),
                        ttkUtils::GetPointer<SimplexId>(inputOrder),
                        (static_cast<TTK_TT*>(triangulation->getData())),
                        constraintDiagram, 
                        ttkUtils::GetPointer<int>(modificationNumber),
                        ttkUtils::GetPointer<int>(lastChange), 
                        ttkUtils::GetPointer<int>(idBlock)
                        )));

  // something wrong in baseCode
  if(ret) {
    this->printErr("TopologicalOptimization.execute() error code: "
                   + std::to_string(ret));
    return -12;
  }

  modificationNumber->SetName("Modification Number");
  lastChange->SetName("Last Change"); 
  idBlock->SetName("Id Block"); 

  output->ShallowCopy(domain);
  output->GetPointData()->RemoveArray(inputOrder->GetName());
  output->GetPointData()->AddArray(outputScalars);
  output->GetPointData()->AddArray(modificationNumber);
  output->GetPointData()->AddArray(lastChange);
  output->GetPointData()->AddArray(idBlock);

  return 1;
}
