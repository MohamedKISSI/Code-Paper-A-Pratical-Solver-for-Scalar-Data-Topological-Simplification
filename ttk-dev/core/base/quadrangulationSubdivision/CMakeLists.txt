ttk_add_base_library(quadrangulationSubdivision
  SOURCES
   QuadrangulationSubdivision.cpp
   Quadrangulation.cpp
  HEADERS
   QuadrangulationSubdivision.h
   Quadrangulation.h
  DEPENDS
   dijkstra
   geometry
   triangulation
   surfaceGeometrySmoother
  )
