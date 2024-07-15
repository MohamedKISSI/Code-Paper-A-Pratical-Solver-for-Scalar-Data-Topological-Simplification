/// \ingroup vtk
/// \class ttkDataSetToTable
/// \author Guillaume Favelier <guillaume.favelier@lip6.fr>
/// \date September 2018
///
/// \brief TTK VTK-filter that creates a vtkTable from a vtkDataSet.
///
/// \param Input vtkDataSet and scalar fields
/// \param Output vtkTable and fields as columns
///
/// This filter can be used as any other VTK filter (for instance, by using the
/// sequence of calls SetInputData(), Update(), GetOutput()).
///
/// See the related ParaView example state files for usage examples within a
/// VTK pipeline.
///
/// \b Online \b examples: \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/mergeTreePGA/">Merge
///   Tree Principal Geodesic Analysis example</a> \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/persistentGenerators_periodicPicture/">
///   Persistent Generators Periodic Picture example</a> \n

#pragma once

// VTK Module
#include <ttkDataSetToTableModule.h>

#include <ttkAlgorithm.h>

class TTKDATASETTOTABLE_EXPORT ttkDataSetToTable : public ttkAlgorithm {

private:
  int DataAssociation;

public:
  vtkSetMacro(DataAssociation, int);
  vtkGetMacro(DataAssociation, int);

  static ttkDataSetToTable *New();
  vtkTypeMacro(ttkDataSetToTable, ttkAlgorithm);

protected:
  ttkDataSetToTable();
  ~ttkDataSetToTable() override;

  int FillInputPortInformation(int port, vtkInformation *info) override;
  int FillOutputPortInformation(int port, vtkInformation *info) override;

  int RequestData(vtkInformation *request,
                  vtkInformationVector **inputVector,
                  vtkInformationVector *outputVector) override;
};
