/// \ingroup base
/// \class ttk::DataTypes
/// \author Guillaume Favelier <guillaume.favelier@lip6.fr>
/// \date May 2018.
///
///\brief TTK base package defining the standard types.

#pragma once

namespace ttk {
  /// \brief Identifier type for simplices of any dimension.
#ifdef TTK_HW_IS_32BITS // i386
  using LongSimplexId = int;
#else // amd64
  using LongSimplexId = long long int;
#endif // TTK_HW_IS_32BITS

  /// \brief Identifier type for simplices of any dimension.
#ifdef TTK_ENABLE_64BIT_IDS
  using SimplexId = long long int;
#else
  using SimplexId = int;
#endif

  /// \brief Identifier type for threads (i.e. with OpenMP).
  using ThreadId = int;

  /// \brief Identifier type for tasks (i.e. with OpenMP).
  using TaskId = int;

  /// default name for mask scalar field
  const char MaskScalarFieldName[] = "ttkMaskScalarField";

  /// default name for vertex scalar field
  const char VertexScalarFieldName[] = "ttkVertexScalarField";

  /// default name for cell scalar field
  const char CellScalarFieldName[] = "ttkCellScalarField";

  /// default name for offset scalar field
  const char OffsetScalarFieldName[] = "ttkOffsetScalarField";

  /// default name for bivariate offset fields
  const char OffsetFieldUName[] = "ttkOffsetFieldU";
  const char OffsetFieldVName[] = "ttkOffsetFieldV";

  // default names for the Morse-Smale complex
  const char MorseSmaleCellDimensionName[] = "CellDimension";
  const char MorseSmaleCellIdName[] = "CellId";
  const char MorseSmaleBoundaryName[] = "IsOnBoundary";
  const char MorseSmaleManifoldSizeName[] = "ManifoldSize";
  const char MorseSmaleSourceIdName[] = "SourceId";
  const char MorseSmaleDestinationIdName[] = "DestinationId";
  const char MorseSmaleSeparatrixIdName[] = "SeparatrixId";
  const char MorseSmaleSeparatrixTypeName[] = "SeparatrixType";
  const char MorseSmaleSeparatrixMaximumName[] = "SeparatrixFunctionMaximum";
  const char MorseSmaleSeparatrixMinimumName[] = "SeparatrixFunctionMinimum";
  const char MorseSmaleSeparatrixDifferenceName[]
    = "SeparatrixFunctionDifference";
  const char MorseSmaleCriticalPointsOnBoundaryName[]
    = "NumberOfCriticalPointsOnBoundary";
  const char MorseSmaleAscendingName[] = "AscendingManifold";
  const char MorseSmaleDescendingName[] = "DescendingManifold";
  const char MorseSmaleManifoldName[] = "MorseSmaleManifold";

  // default names for persistence diagram meta data
  const char PersistenceCriticalTypeName[] = "CriticalType";
  const char PersistenceBirthName[] = "Birth";
  const char PersistenceDeathName[] = "Death";
  const char PersistenceCoordinatesName[] = "Coordinates";
  const char PersistencePairIdentifierName[] = "PairIdentifier";
  const char PersistenceName[] = "Persistence";
  const char PersistencePairTypeName[] = "PairType";
  const char PersistenceIsFinite[] = "IsFinite";

  // default name for compact triangulation index
  const char compactTriangulationIndex[] = "ttkCompactTriangulationIndex";

  /// default value for critical index
  enum class CriticalType {
    Local_minimum = 0,
    Saddle1,
    Saddle2,
    Local_maximum,
    Degenerate,
    Regular
  };
  /// number of different critical types
  const int CriticalTypeNumber = 6;

} // namespace ttk
