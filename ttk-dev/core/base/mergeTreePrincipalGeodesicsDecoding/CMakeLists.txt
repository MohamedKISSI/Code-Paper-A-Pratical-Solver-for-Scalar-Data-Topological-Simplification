ttk_add_base_library(mergeTreePrincipalGeodesicsDecoding
  SOURCES
    MergeTreePrincipalGeodesicsDecoding.cpp
  HEADERS
    MergeTreePrincipalGeodesicsDecoding.h
  DEPENDS
    triangulation
    mergeTreePrincipalGeodesics
)
