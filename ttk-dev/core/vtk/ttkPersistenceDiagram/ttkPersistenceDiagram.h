/// \ingroup vtk
/// \class ttkPersistenceDiagram
/// \author Guillaume Favelier <guillaume.favelier@lip6.fr>
/// \author Julien Tierny <julien.tierny@lip6.fr>
/// \date September 2016.
///
/// \brief TTK VTK-filter for the computation of persistence diagrams.
///
/// This filter computes the persistence diagram of the extremum-saddle pairs
/// of an input scalar field. The X-coordinate of each pair corresponds to its
/// birth, while its smallest and highest Y-coordinates correspond to its birth
/// and death respectively.
///
/// In practice, the diagram is represented by a vtkUnstructuredGrid. Each
/// vertex of this mesh represent a critical point of the input data. It is
/// associated with point data (vertexId, critical type). Each vertical edge
/// of this mesh represent a persistence pair. It is associated with cell data
/// (persistence of the pair, critical index of the extremum of the pair).
///
/// Persistence diagrams are useful and stable concise representations of the
/// topological features of a data-set. It is useful to fine-tune persistence
/// thresholds for topological simplification or for fast similarity
/// estimations for instance.
///
/// \param Input Input scalar field, either 2D or 3D, regular grid or
/// triangulation (vtkDataSet)
/// \param Output Output persistence diagram (vtkUnstructuredGrid)
///
/// This filter can be used as any other VTK filter (for instance, by using the
/// sequence of calls SetInputData(), Update(), GetOutput()).
///
/// See the related ParaView example state files for usage examples within a
/// VTK pipeline.
///
/// \b Related \b publication \n
/// "Computational Topology: An Introduction" \n
/// Herbert Edelsbrunner and John Harer \n
/// American Mathematical Society, 2010
///
/// Five backends are available for the computation:
///
///  1) FTM \n
/// \b Related \b publication \n
/// "Task-based Augmented Contour Trees with Fibonacci Heaps"
/// Charles Gueunet, Pierre Fortin, Julien Jomier, Julien Tierny
/// IEEE Transactions on Parallel and Distributed Systems, 2019
///
///  2) Progressive Approach \n
/// \b Related \b publication \n
/// "A Progressive Approach to Scalar Field Topology" \n
/// Jules Vidal, Pierre Guillou, Julien Tierny\n
/// IEEE Transactions on Visualization and Computer Graphics, 2021
///
/// 3) Discrete Morse Sandwich (default) \n
/// \b Related \b publication \n
/// "Discrete Morse Sandwich: Fast Computation of Persistence Diagrams for
/// Scalar Data -- An Algorithm and A Benchmark" \n
/// Pierre Guillou, Jules Vidal, Julien Tierny \n
/// IEEE Transactions on Visualization and Computer Graphics, 2023.\n
/// arXiv:2206.13932, 2023.\n
/// Fast and versatile algorithm for persistence diagram computation.
///
/// 4) Approximate Approach \n
/// \b Related \b publication \n
/// "Fast Approximation of Persistence Diagrams with Guarantees" \n
/// Jules Vidal, Julien Tierny\n
/// IEEE Symposium on Large Data Visualization and Analysis (LDAV), 2021
///
/// 5) Persistent Simplex \n
/// This is a textbook (and very slow) algorithm, described in
/// "Algorithm and Theory of Computation Handbook (Second Edition)
/// - Special Topics and Techniques" by Atallah and Blanton on page 97.
///
/// \sa ttkMergeTreePP
/// \sa ttkPersistenceCurve
/// \sa ttkScalarFieldCriticalPoints
/// \sa ttkTopologicalSimplification
/// \sa ttk::PersistenceDiagram
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
///   </a> \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/clusteringKelvinHelmholtzInstabilities/">
///   Clustering Kelvin Helmholtz Instabilities example</a> \n
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
///   href="https://topology-tool-kit.github.io/examples/karhunenLoveDigits64Dimensions//">Karhunen-Love
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
///   href="https://topology-tool-kit.github.io/examples/persistenceDiagramClustering/">Persistence
///   Diagram Clustering example</a> \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/persistenceDiagramDistance/">Persistence
///   Diagram Distance example</a> \n
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

// VTK includes
#include <vtkDataArray.h>
#include <vtkUnstructuredGrid.h>

// VTK Module
#include <ttkPersistenceDiagramModule.h>

// ttk code includes
#include <PersistenceDiagram.h>
#include <ttkAlgorithm.h>
#include <ttkMacros.h>

class TTKPERSISTENCEDIAGRAM_EXPORT ttkPersistenceDiagram
  : public ttkAlgorithm,
    protected ttk::PersistenceDiagram {

public:
  static ttkPersistenceDiagram *New();

  vtkTypeMacro(ttkPersistenceDiagram, ttkAlgorithm);

  vtkSetMacro(ForceInputOffsetScalarField, bool);
  vtkGetMacro(ForceInputOffsetScalarField, bool);

  vtkSetMacro(ShowInsideDomain, bool);
  vtkGetMacro(ShowInsideDomain, bool);

  ttkSetEnumMacro(BackEnd, BACKEND);
  vtkGetEnumMacro(BackEnd, BACKEND);

  vtkGetMacro(StartingResolutionLevel, int);
  vtkSetMacro(StartingResolutionLevel, int);

  vtkGetMacro(StoppingResolutionLevel, int);
  vtkSetMacro(StoppingResolutionLevel, int);

  vtkGetMacro(IsResumable, bool);
  vtkSetMacro(IsResumable, bool);

  vtkGetMacro(TimeLimit, double);
  vtkSetMacro(TimeLimit, double);

  vtkGetMacro(Epsilon, double);
  vtkSetMacro(Epsilon, double);

  vtkSetMacro(IgnoreBoundary, bool);
  vtkGetMacro(IgnoreBoundary, bool);

  inline void SetComputeMinSad(const bool data) {
    this->setComputeMinSad(data);
    this->dmsDimsCache[0] = data;
    this->Modified();
  }
  inline void SetComputeSadSad(const bool data) {
    this->setComputeSadSad(data);
    this->dmsDimsCache[1] = data;
    this->Modified();
  }
  inline void SetComputeSadMax(const bool data) {
    this->setComputeSadMax(data);
    this->dmsDimsCache[2] = data;
    this->Modified();
  }
  inline void SetDMSDimensions(const int data) {
    this->setComputeMinSad(data == 0 ? true : this->dmsDimsCache[0]);
    this->setComputeSadSad(data == 0 ? true : this->dmsDimsCache[1]);
    this->setComputeSadMax(data == 0 ? true : this->dmsDimsCache[2]);
    this->Modified();
  }

  vtkSetMacro(ClearDGCache, bool);
  vtkGetMacro(ClearDGCache, bool);

protected:
  ttkPersistenceDiagram();

  int RequestData(vtkInformation *request,
                  vtkInformationVector **inputVector,
                  vtkInformationVector *outputVector) override;

  int FillInputPortInformation(int port, vtkInformation *info) override;
  int FillOutputPortInformation(int port, vtkInformation *info) override;

private:
  template <typename scalarType, typename triangulationType>
  int dispatch(vtkUnstructuredGrid *outputCTPersistenceDiagram,
               vtkDataArray *const inputScalarsArray,
               const scalarType *const inputScalars,
               scalarType *outputScalars,
               SimplexId *outputOffsets,
               int *outputMonotonyOffsets,
               const SimplexId *const inputOrder,
               const triangulationType *triangulation);

  bool ForceInputOffsetScalarField{false};
  bool ShowInsideDomain{false};
  // stores the values of Compute[Min|Sad][Sad|Max] GUI checkboxes
  // when "All Dimensions" is selected
  std::array<bool, 3> dmsDimsCache{true, true, true};
  // clear DiscreteGradient cache after computation
  bool ClearDGCache{false};
};
