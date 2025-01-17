#ifndef VORONOI_H 
#define VORONOI_H

#include <voro++/voro++.hh>
#include "finder_h.hpp"

/*Function that computs the voronoi tesselation using voro++*/
void Compute_Voronoi(fft_real *pos, size_t np, fft_real *vol, int get_vol);

#endif
