#include <ttkContourAroundPoint.h>

#include <ttkMacros.h>
#include <ttkUtils.h>

#include <vtkInformation.h>
#include <vtkInformationVector.h>
#include <vtkVersion.h>

#include <vtkCellData.h>
#include <vtkDataSet.h>
#include <vtkUnstructuredGrid.h>

#include <vtkFloatArray.h>
#include <vtkIdTypeArray.h>

#include <cassert>

vtkStandardNewMacro(ttkContourAroundPoint);

//----------------------------------------------------------------------------//

int ttkContourAroundPoint::FillInputPortInformation(int port,
                                                    vtkInformation *info) {
  switch(port) {
    case 0:
      info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkDataSet");
      return 1;
    case 1:
    case 2:
      info->Set(
        vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkUnstructuredGrid");
      return 1;
  }
  return 0;
}

int ttkContourAroundPoint::FillOutputPortInformation(int port,
                                                     vtkInformation *info) {
  switch(port) {
    case 0:
    case 1:
      info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkUnstructuredGrid");
      return 1;
  }
  return 0;
}

//----------------------------------------------------------------------------//

int ttkContourAroundPoint::RequestData(vtkInformation *ttkNotUsed(request),
                                       vtkInformationVector **iVec,
                                       vtkInformationVector *oVec) {

  _outFld = vtkUnstructuredGrid::GetData(oVec, 0);
  _outPts = vtkUnstructuredGrid::GetData(oVec, 1);

  if(!preprocessFld(vtkDataSet::GetData(iVec[0])))
    return 0;
  if(!preprocessPts(vtkUnstructuredGrid::GetData(iVec[1]),
                    vtkUnstructuredGrid::GetData(iVec[2])))
    return 0;
  if(!process())
    return 0;
  if(!postprocess())
    return 0;

  return 1;
}

//----------------------------------------------------------------------------//

bool ttkContourAroundPoint::preprocessFld(vtkDataSet *dataset) {
  ttk::Triangulation *triangulation = ttkAlgorithm::GetTriangulation(dataset);
  if(!triangulation)
    return false;

  auto scalars = GetInputArrayToProcess(0, dataset);
  if(!scalars)
    return false;

  const double radius = ui_spherical ? -1. : 0.;

  const auto errorCode = this->setInputField(
    triangulation, ttkUtils::GetVoidPointer(scalars), ui_sizeFilter, radius);
  if(errorCode < 0) {
    printErr("super->setInputField failed with code "
             + std::to_string(errorCode));
    return false;
  }

  _triangTypeCode = triangulation->getType();
  _scalarTypeCode = scalars->GetDataType();
  _scalarsName = scalars->GetName();

  std::ostringstream stream;
  stream << "Scalar type: " << scalars->GetDataTypeAsString() << " (code "
         << _scalarTypeCode << ")";
  printMsg(stream.str(), ttk::debug::Priority::VERBOSE);

  return true;
}

//----------------------------------------------------------------------------//

bool ttkContourAroundPoint::preprocessPts(vtkUnstructuredGrid *nodes,
                                          vtkUnstructuredGrid *arcs) {
  // ---- Point data ---- //

  auto points = nodes->GetPoints();
  if(points->GetDataType() != VTK_FLOAT) {
    printErr("The point coordinates must be of type float");
    return false;
  }
  auto coords
    = reinterpret_cast<float *>(ttkUtils::GetVoidPointer(points->GetData()));

  auto pData = nodes->GetPointData();
  auto scalarBuf = getBuffer<float>(pData, "Scalar", VTK_FLOAT, "float");
  auto codeBuf = getBuffer<int>(pData, "CriticalType", VTK_INT, "int");
  if(!scalarBuf || !codeBuf)
    return false;

    // ---- Cell data ---- //

#ifndef NDEBUG // each arc should of course be defined by exactly two vertices
  auto cells = arcs->GetCells();
  const auto maxNvPerC = cells->GetMaxCellSize();
  if(maxNvPerC != 2) {
    printErr(
      "The points must come in pairs but there is at least one cell with "
      + std::to_string(maxNvPerC) + " points");
    return false;
  }
  // NOTE Ideally check for minNvPerC != 2
#endif

  auto cData = arcs->GetCellData();
  auto c2p = getBuffer<int>(cData, "upNodeId", VTK_INT, "int");
  auto c2q = getBuffer<int>(cData, "downNodeId", VTK_INT, "int");
  if(!c2p || !c2q)
    return false;

  // ---- Loop over pairs ---- //

  static constexpr int minCode = 0;
  static constexpr int maxCode = 3;
  const double sadFac = ui_extension * 0.01; // saddle or mean of min and max
  const double extFac = 1 - sadFac; // factor for the extreme point

  _coords.resize(0);
  _scalars.resize(0);
  _isovals.resize(0);
  _flags.resize(0);

  auto addPoint = [this, coords, scalarBuf](int p, float isoval, int code) {
    const auto point = &coords[p * 3];
    _coords.push_back(point[0]);
    _coords.push_back(point[1]);
    _coords.push_back(point[2]);
    _scalars.push_back(scalarBuf[p]);
    _isovals.push_back(isoval);
    _flags.push_back(code == minCode ? 0 : 1);
  };

  const vtkIdType nc = arcs->GetNumberOfCells();
  for(vtkIdType c = 0; c < nc; ++c) {
    const auto p = c2p[c];
    const auto q = c2q[c];
    const auto pCode = codeBuf[p];
    const auto qCode = codeBuf[q];

    const bool pIsSad = pCode != minCode && pCode != maxCode;
    const bool qIsSad = qCode != minCode && qCode != maxCode;
    if(pIsSad || qIsSad) {
      if(pIsSad && qIsSad) // two saddles
        continue;
      // extremum and saddle
      const auto ext = pIsSad ? q : p;
      const auto sad = pIsSad ? p : q;
      const float isoval = scalarBuf[ext] * extFac + scalarBuf[sad] * sadFac;
      addPoint(ext, isoval, codeBuf[ext]);
    } else { // min-max pair
      printWrn("Arc " + std::to_string(c) + " joins a minimum and a maximum");
      const auto pVal = scalarBuf[p];
      const auto qVal = scalarBuf[q];
      const auto cVal = (pVal + qVal) / 2;
      addPoint(p, pVal * extFac + cVal * sadFac, pCode);
      addPoint(q, qVal * extFac + cVal * sadFac, qCode);
    }
  }

  auto np = _scalars.size();
  const auto errorCode = this->setInputPoints(
    _coords.data(), _scalars.data(), _isovals.data(), _flags.data(), np);
  if(errorCode < 0) {
    printErr("setInputPoints failed with code " + std::to_string(errorCode));
    return false;
  }
  return true;
}

//----------------------------------------------------------------------------//

bool ttkContourAroundPoint::process() {
  int errorCode = 0;
  switch(_scalarTypeCode) {
    vtkTemplateMacro((errorCode = this->execute<VTK_TT>()));
  }
  //   ttkVtkTemplateMacro(
  //     _scalarTypeCode, _triangTypeCode,
  //     (errorCode = this->execute<VTK_TT, TTK_TT>())
  //   )
  if(errorCode < 0) { // In TTK, negative is bad.
    printErr("super->execute failed with code " + std::to_string(errorCode));
    return false;
  }
  return true;
}

//----------------------------------------------------------------------------//

bool ttkContourAroundPoint::postprocess() {
  ttk::SimplexId const nc = _outContoursNc;
  if(nc == 0) // very fine area filter
    return true;

  // ---- Cell data (output 0) ---- //

  std::vector<int> ctypes(nc);

  vtkIdType cinfoCounter = 0;
  for(ttk::SimplexId c = 0; c < nc; ++c) {
    const auto nvOfCell = _outContoursCinfos[cinfoCounter];
    assert(nvOfCell >= 2 && nvOfCell <= 3); // ensured in super class
    ctypes[c] = nvOfCell == 2 ? VTK_LINE : VTK_TRIANGLE;
    cinfoCounter += nvOfCell + 1;
  }

  auto cells = vtkSmartPointer<vtkCellArray>::New();
  auto cinfoArr = vtkSmartPointer<vtkIdTypeArray>::New();
  ttkUtils::SetVoidArray(
    cinfoArr, _outContoursCinfos.data(), _outContoursCinfos.size(), 1);
  cells->SetCells(nc, cinfoArr);
  _outFld->SetCells(ctypes.data(), cells);

  // ---- Point data (output 0) ---- //

  if(vtkSmartPointer<vtkPoints>::New()->GetDataType() != VTK_FLOAT) {
    printErr("The API has changed! We have expected the default "
             "coordinate type to be float");
    return false;
  }

  auto points = vtkSmartPointer<vtkPoints>::New();
  auto coordArr = vtkSmartPointer<vtkFloatArray>::New();
  coordArr->SetNumberOfComponents(3);
  ttkUtils::SetVoidArray(
    coordArr, _outContoursCoords.data(), _outContoursCoords.size(), 1);
  points->SetData(coordArr);
  _outFld->SetPoints(points);

  auto scalarArr = vtkFloatArray::New();
  ttkUtils::SetVoidArray(
    scalarArr, _outContoursScalars.data(), _outContoursScalars.size(), 1);
  scalarArr->SetName(_scalarsName);
  _outFld->GetPointData()->AddArray(scalarArr);

  auto flagArr = vtkIntArray::New();
  ttkUtils::SetVoidArray(
    flagArr, _outContoursFlags.data(), _outContoursFlags.size(), 1);
  flagArr->SetName("isMax");
  _outFld->GetPointData()->AddArray(flagArr);

  // ---- Output 1 (added in a later revision of the algo) ---- //

  points = vtkSmartPointer<vtkPoints>::New();
  coordArr = vtkSmartPointer<vtkFloatArray>::New();
  coordArr->SetNumberOfComponents(3);
  ttkUtils::SetVoidArray(
    coordArr, _outCentroidsCoords.data(), _outCentroidsCoords.size(), 1);
  points->SetData(coordArr);
  _outPts->SetPoints(points);

  scalarArr = vtkFloatArray::New();
  ttkUtils::SetVoidArray(
    scalarArr, _outCentroidsScalars.data(), _outCentroidsScalars.size(), 1);
  scalarArr->SetName(_scalarsName);
  _outPts->GetPointData()->AddArray(scalarArr);

  flagArr = vtkIntArray::New();
  ttkUtils::SetVoidArray(
    flagArr, _outCentroidsFlags.data(), _outCentroidsFlags.size(), 1);
  flagArr->SetName("isMax");
  _outPts->GetPointData()->AddArray(flagArr);

  return true;
}
