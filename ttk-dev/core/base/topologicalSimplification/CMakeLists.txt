ttk_add_base_library(topologicalSimplification
  SOURCES
    TopologicalSimplification.cpp
  HEADERS
    TopologicalSimplification.h
  DEPENDS
    triangulation
    legacyTopologicalSimplification
    localizedTopologicalSimplification
    )

#target_compile_options(topologicalSimplification PRIVATE -ggdb -O0 -fsanitize=address)
#target_link_options(topologicalSimplification PRIVATE -fsanitize=address)
