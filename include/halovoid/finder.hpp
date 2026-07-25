/* Decalarations of the 3D halo/void finder. */
#ifndef FINDER_H
#define FINDER_H

#include "cell.hh"
#include "halovoid_h.hpp"
#include <algorithm> // For std::sort

// Global compile-time constant array
constexpr std::array<std::array<std::array<int, 3>, 27>, 8>
compute_neighbor_order();

// Compute the total volume of all cells
double total_volume(const Container &con);

// Find the halos and voids in 3D
void compute_voronoi_3d(const Container &con, HaloVoid &halos,
                        fft_real rho_halo, HaloVoid &voids, fft_real rho_void,
                        bool save_volume, fft_real *volume, double dist_tol);

#endif
