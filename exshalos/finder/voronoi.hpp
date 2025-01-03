#ifndef VORONOI_H 
#define VORONOI_H

#include "finder_h.h"
// #include "voro++/voro++.hh"
#include <voro++/c_loops.hh>
#include <voro++/cell.hh>
#include <voro++/container.hh>

/*Export the cpp functions to c*/
extern "C"{

/*Function that computs the voronoi tesselation using voro++*/
void Compute_Voronoi(fft_real *pos, size_t np, fft_real *vol, int get_vol);

}

#endif
