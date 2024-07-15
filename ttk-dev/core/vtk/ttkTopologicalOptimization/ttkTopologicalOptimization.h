/// \ingroup vtk
/// \class ttkTopologicalOptimization
/// \author Guillaume Favelier <guillaume.favelier@lip6.fr>
/// \date February 2016
///
/// \brief TTK VTK-filter for the topological simplification of scalar
/// data.
///
/// Given an input scalar field and a list of critical points to remove, this
/// filter minimally edits the scalar field such that the listed critical points
/// disappear. This procedure is useful to speedup subsequent topological data
/// analysis when outlier critical points can be easily identified. It is
/// also useful for data simplification.
///
/// The list of critical points to remove must be associated with a point data
/// scalar field that represent the vertex global identifiers in the input
/// geometry.
///
/// Note that this filter will also produce an output vertex offset scalar field
/// that can be used for further topological data analysis tasks to disambiguate
/// vertices on flat plateaus. For instance, this output vertex offset field
/// can specified to the ttkFTMTree, vtkIntegralLines, or
/// vtkScalarFieldCriticalPoints filters.
///
/// Also, this filter can be given a specific input vertex offset.
///
/// \param Input0 Input scalar field, either 2D or 3D, either regular grid or
/// triangulation (vtkDataSet)
/// \param Input1 List of critical point constraints (vtkPointSet)
/// \param Output Output simplified scalar field (vtkDataSet)
///
/// This filter can be used as any other VTK filter (for instance, by using the
/// sequence of calls SetInputData(), Update(), GetOutput()).
///
/// See the related ParaView example state files for usage examples within a
/// VTK pipeline.
///
/// \b Related \b publications \n
/// "Generalized Topological Simplification of Scalar Fields on Surfaces" \n
/// Julien Tierny, Valerio Pascucci \n
/// Proc. of IEEE VIS 2012.\n
/// IEEE Transactions on Visualization and Computer Graphics, 2012.
///
/// "Localized Topological Simplification of Scalar Data"
/// Jonas Lukasczyk, Christoph Garth, Ross Maciejewski, Julien Tierny
/// Proc. of IEEE VIS 2020.
/// IEEE Transactions on Visualization and Computer Graphics
///
/// \sa ttkTopologicalOptimizationByPersistence
/// \sa ttkScalarFieldCriticalPoints
/// \sa ttkIntegralLines
/// \sa ttkFTMTree
/// \sa ttkMorseSmaleComplex
/// \sa ttkIdentifiers
/// \sa ttk::TopologicalOptimization
///
/// \b Online \b examples: \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/1manifoldLearning/">1-Manifold
///   Learning example</a> \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/1manifoldLearningCircles/">1-Manifold
///   Learning Circles example</a> \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/2manifoldLearning/">
///   2-Manifold Learning example</a> \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/BuiltInExample1/">BuiltInExample1
///   example</a> \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/contourTreeAlignment/">Contour
///   Tree Alignment example</a> \n
///   - <a href="https://topology-tool-kit.github.io/examples/ctBones/">CT Bones
///   example</a> \n
///   - <a href="https://topology-tool-kit.github.io/examples/dragon/">Dragon
///   example</a> \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/harmonicSkeleton/">
///   Harmonic Skeleton example</a> \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/imageProcessing/">Image
///   Processing example</a> \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/interactionSites/">
///   Interaction sites</a> \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/karhunenLoveDigits64Dimensions/">Karhunen-Love
///   Digits 64-Dimensions example</a> \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/morsePersistence/">Morse
///   Persistence example</a> \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/morseSmaleQuadrangulation/">Morse-Smale
///   Quadrangulation example</a> \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/persistenceClustering0/">Persistence
///   clustering 0 example</a> \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/persistenceClustering0/">Persistence
///   clustering 1 example</a> \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/persistenceClustering0/">Persistence
///   clustering 2 example</a> \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/persistenceClustering0/">Persistence
///   clustering 3 example</a> \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/persistenceClustering0/">Persistence
///   clustering 4 example</a> \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/tectonicPuzzle/">Tectonic
///   Puzzle example</a> \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/tribute/">Tribute
///   example</a> \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/uncertainStartingVortex/">
///   Uncertain Starting Vortex example</a> \n
///

#pragma once

// VTK Module
#include <ttkTopologicalOptimizationModule.h>

// ttk code includes
#include <TopologicalOptimization.h>
#include <ttkAlgorithm.h>
#include <ttkPersistenceDiagramUtils.h>

class vtkDataArray;
class vtkUnstructuredGrid; 

class TTKTOPOLOGICALOPTIMIZATION_EXPORT ttkTopologicalOptimization
  : public ttkAlgorithm,
    protected ttk::TopologicalOptimization {

public:
  static ttkTopologicalOptimization *New();
  vtkTypeMacro(ttkTopologicalOptimization, ttkAlgorithm);

  vtkSetMacro(ForceInputOffsetScalarField, bool);
  vtkGetMacro(ForceInputOffsetScalarField, bool);

  vtkSetMacro(ConsiderIdentifierAsBlackList, bool);
  vtkGetMacro(ConsiderIdentifierAsBlackList, bool);

  vtkSetMacro(AddPerturbation, bool);
  vtkGetMacro(AddPerturbation, bool);

  vtkSetMacro(UseLTS, bool);
  vtkGetMacro(UseLTS, bool);

  vtkSetMacro(PersistenceThreshold, double);
  vtkGetMacro(PersistenceThreshold, double);

  vtkSetMacro(UseTimeThreshold, bool);
  vtkGetMacro(UseTimeThreshold, bool);

  vtkSetMacro(TimeThreshold, double);
  vtkGetMacro(TimeThreshold, double);

  vtkSetMacro(EpochNumber, int);
  vtkGetMacro(EpochNumber, int);

  vtkSetMacro(UseLazyDiscretGradientUpdate, bool);
  vtkGetMacro(UseLazyDiscretGradientUpdate, bool);

  vtkSetMacro(UseTheMultiBlocksApproach, bool);
  vtkGetMacro(UseTheMultiBlocksApproach, bool);
   
  vtkSetMacro(NumberOfBlocksPerThread, double);
  vtkGetMacro(NumberOfBlocksPerThread, double);

  vtkSetMacro(NumberEpochMultiBlock, int);
  vtkGetMacro(NumberEpochMultiBlock, int);

  vtkSetMacro(MultiBlockOneLevel, bool);
  vtkGetMacro(MultiBlockOneLevel, bool);

  vtkSetMacro(EpsilonPenalisation, double);
  vtkGetMacro(EpsilonPenalisation, double);

  vtkSetMacro(UseTopologicalSimplification, bool);
  vtkGetMacro(UseTopologicalSimplification, bool);

  vtkSetMacro(ChooseLearningRate, bool);
  vtkGetMacro(ChooseLearningRate, bool);

  vtkSetMacro(LearningRate, double);
  vtkGetMacro(LearningRate, double);

  vtkSetMacro(SemiDirectMatching, bool);
  vtkGetMacro(SemiDirectMatching, bool);

  vtkSetMacro(PDCMethod, int);
  vtkGetMacro(PDCMethod, int);

  vtkSetMacro(Method, int);
  vtkGetMacro(Method, int);

  vtkSetMacro(CoefStopCondition, double);
  vtkGetMacro(CoefStopCondition, double);

  vtkSetMacro(FinePairManagement, int);
  vtkGetMacro(FinePairManagement, int);

  vtkSetMacro(UseAdditionalPrecisionPDC, bool);
  vtkGetMacro(UseAdditionalPrecisionPDC, bool);

  vtkSetMacro(DeltaLim, double);
  vtkGetMacro(DeltaLim, double);

  vtkSetMacro(SaveData, bool);
  vtkGetMacro(SaveData, bool);

protected:
  ttkTopologicalOptimization();

  int FillInputPortInformation(int port, vtkInformation *info) override;
  int FillOutputPortInformation(int port, vtkInformation *info) override;
  int RequestData(vtkInformation *request,
                  vtkInformationVector **inputVector,
                  vtkInformationVector *outputVector) override;

private:
  // bool UseLaplacien{false};
  // bool UseDataTether{false};
  bool ForceInputOffsetScalarField{false};
  bool ConsiderIdentifierAsBlackList{false};
  bool AddPerturbation{false};
  bool UseLTS{true};
  double PersistenceThreshold{0};
};
