/// \ingroup vtk
/// \class ttkMergeBlockTables
/// \author Pierre Guillou <pierre.guillou@lip6.fr>
/// \date July 2021
///
/// \brief TTK processing package for merging vtkTables from a
/// vtkMultiBlockDataSet.
///
/// \b Online \b examples: \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/mergeTreePGA/">Merge
///   Tree Principal Geodesic Analysis example</a> \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/persistentGenerators_periodicPicture/">Persistent
///   Generators Periodic Picture example</a> \n

#pragma once

#include <ttkMergeBlockTablesModule.h>

#include <ttkAlgorithm.h>

class TTKMERGEBLOCKTABLES_EXPORT ttkMergeBlockTables : public ttkAlgorithm {

public:
  static ttkMergeBlockTables *New();

  vtkTypeMacro(ttkMergeBlockTables, ttkAlgorithm);

protected:
  ttkMergeBlockTables();

  int FillInputPortInformation(int port, vtkInformation *info) override;
  int FillOutputPortInformation(int port, vtkInformation *info) override;

  int RequestData(vtkInformation *request,
                  vtkInformationVector **inputVector,
                  vtkInformationVector *outputVector) override;
};
