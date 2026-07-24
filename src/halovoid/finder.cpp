#include "finder.hpp"
#include "halovoid_h.hpp"
#include <cmath>
//#include <pdqsort.h>
#include <array>
#include <print>

// Generates the lookup table at compile-time using standard C++23 features
constexpr std::array<std::array<std::array<int, 3>, 27>, 8>
compute_neighbor_order() {
    std::array<std::array<std::array<int, 3>, 27>, 8> result{};

    for (int octant = 0; octant < 8; ++octant) {
        // Octant center coordinates (scaled by 4 to keep integer math)
        int ox = (octant & 1) ? 1 : -1;
        int oy = (octant & 2) ? 1 : -1;
        int oz = (octant & 4) ? 1 : -1;

        std::array<std::array<int, 3>, 27> neighbors{};
        int n_idx = 0;

        // 1. Populate the 27 adjacent blocks
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    neighbors[n_idx++] = {dx, dy, dz};
                }
            }
        }

        // 2. Sort the array at compile-time using std::sort and a lambda
        std::sort(neighbors.begin(), neighbors.end(),
                  [ox, oy, oz](const std::array<int, 3> &a,
                               const std::array<int, 3> &b) {
                      auto dist_sq = [ox, oy, oz](const std::array<int, 3> &n) {
                          int rx = 4 * n[0] - ox;
                          int ry = 4 * n[1] - oy;
                          int rz = 4 * n[2] - oz;
                          return rx * rx + ry * ry + rz * rz;
                      };
                      return dist_sq(a) < dist_sq(b);
                  });

        // 3. Write sorted values directly to output
        for (int i = 0; i < 27; ++i) {
            result[octant][i] = {neighbors[i][0], neighbors[i][1],
                                 neighbors[i][2]};
        }
    }

    return result;
}

// Compile-time constant array with the neighbors order
constexpr auto NEIGHBOR_ORDER = compute_neighbor_order();

// Compute the total volume of all cells
double total_volume(const Container &con) {
    // Compute some constants
    const fft_real L = con.get_L();
    const std::size_t np = con.get_np();
    const std::size_t nd = con.get_nd();
    const fft_real Lcell = con.get_Lcell();
    const fft_real scale = con.get_scale();
    const fft_real shift_cell = 0.5 * Lcell;

    // Run over all particles
    voro::voronoicell cell;
    double total_volume = 0.0;
    for (std::size_t index = 0; index < np; index++) {
        // Initialize the Voronoi cell
        cell.init(-L / 2.0, L / 2.0, -L / 2.0, L / 2.0, -L / 2.0, L / 2.0);

        // Get the position of the particle
        fft_real px = con[index, 0];
        fft_real py = con[index, 1];
        fft_real pz = con[index, 2];

        // Get the cell index
        std::size_t idx = (std::size_t)(px * scale);
        if (idx >= nd)
            idx = nd - 1;
        std::size_t idy = (std::size_t)(py * scale);
        if (idy >= nd)
            idy = nd - 1;
        std::size_t idz = (std::size_t)(pz * scale);
        if (idz >= nd)
            idz = nd - 1;

        // Compute the local coordinates
        fft_real dx_local = px - idx * Lcell;
        fft_real dy_local = py - idy * Lcell;
        fft_real dz_local = pz - idz * Lcell;

        // Get the offsets table
        const int octant = (dx_local - shift_cell > 0 ? 1 : 0) |
                           (dy_local - shift_cell > 0 ? 2 : 0) |
                           (dz_local - shift_cell > 0 ? 4 : 0);
        static const auto offsets = NEIGHBOR_ORDER[octant];

        // Compute the Voronoi volume of this particle
        double r2_max = cell.max_radius_squared();
        for (int o = 0; o < 27; o++) {
            int i = offsets[o][0];
            int j = offsets[o][1];
            int k = offsets[o][2];

            // x-direction
            int d_idx = idx + i;
            int wrap_x = 0;
            if (d_idx < 0) {
                d_idx += nd;
                wrap_x = -1;
            } else if (d_idx >= (int)nd) {
                d_idx -= nd;
                wrap_x = 1;
            }
            fft_real dist_x =
                (i == -1) ? dx_local : ((i == 1) ? Lcell - dx_local : 0.0);

            // y-direction
            int d_idy = idy + j;
            int wrap_y = 0;
            if (d_idy < 0) {
                d_idy += nd;
                wrap_y = -1;
            } else if (d_idy >= (int)nd) {
                d_idy -= nd;
                wrap_y = 1;
            }
            fft_real dist_y =
                (j == -1) ? dy_local : ((j == 1) ? Lcell - dy_local : 0.0);

            // z-direction
            int d_idz = idz + k;
            int wrap_z = 0;
            if (d_idz < 0) {
                d_idz += nd;
                wrap_z = -1;
            } else if (d_idz >= (int)nd) {
                d_idz -= nd;
                wrap_z = 1;
            }
            fft_real dist_z =
                (k == -1) ? dz_local : ((k == 1) ? Lcell - dz_local : 0.0);

            // If the block is too far away, skip all particles inside
            fft_real block_dist2 =
                dist_x * dist_x + dist_y * dist_y + dist_z * dist_z;
            if (block_dist2 > 4.0 * r2_max) {
                continue;
            }

            // Run over all particles in this block
            bool cell_cut = false;
            std::size_t np_in_block = con.get_n_in_cell(d_idx, d_idy, d_idz);
            for (std::size_t part = 0; part < np_in_block; part++) {
                fft_real dx =
                    con[d_idx, d_idy, d_idz, part, 0] - px + wrap_x * L;
                fft_real dy =
                    con[d_idx, d_idy, d_idz, part, 1] - py + wrap_y * L;
                fft_real dz =
                    con[d_idx, d_idy, d_idz, part, 2] - pz + wrap_z * L;

                // Avoid self-particles and particles too far away
                fft_real d2 = dx * dx + dy * dy + dz * dz;
                if (d2 < 1e-12 || d2 > r2_max)
                    continue;

                // Compute the plane
                if (cell.plane(dx, dy, dz)) {
                    cell_cut = true;
                }
            }

            // If the plane was cut, update the maximum radius
            if (cell_cut) {
                r2_max = cell.max_radius_squared();
            }
        }
        // Add the cell volume to the total volume
        total_volume += cell.volume();
    }

    return total_volume;
}

// Find the halos and voids in 3D
void compute_voronoi_3d(const Container &con, HaloVoid &halos,
                        fft_real rho_halos, HaloVoid &voids, fft_real rho_voids,
                        bool save_volume, fft_real *volume) {
    // Compute some constants
    const fft_real L = con.get_L();
    const std::size_t np = con.get_np();
    const std::size_t nd = con.get_nd();
    const fft_real Lcell = con.get_Lcell();
    const fft_real scale = con.get_scale();
    const fft_real shift_cell = 0.5 * Lcell;
    const double three_over_pi = 3.0 / M_PI;

    // Create the vector for the vertices of the Vornoi cells
    std::vector<double> verts;

    // Get the number of halos and voids (for the output)
    std::size_t n_halos = halos.n;
    std::size_t n_voids = voids.n;

    // Run over all particles
    voro::voronoicell cell;
    for (std::size_t index = 0; index < np; index++) {
        // Initialize the Voronoi cell
        cell.init(-L / 2.0, L / 2.0, -L / 2.0, L / 2.0, -L / 2.0, L / 2.0);

        // Get the position of the particle
        fft_real px = con[index, 0];
        fft_real py = con[index, 1];
        fft_real pz = con[index, 2];

        // Get the cell index
        std::size_t idx = (std::size_t)(px * scale);
        if (idx >= nd)
            idx = nd - 1;
        std::size_t idy = (std::size_t)(py * scale);
        if (idy >= nd)
            idy = nd - 1;
        std::size_t idz = (std::size_t)(pz * scale);
        if (idz >= nd)
            idz = nd - 1;

        // Compute the local coordinates
        fft_real dx_local = px - idx * Lcell;
        fft_real dy_local = py - idy * Lcell;
        fft_real dz_local = pz - idz * Lcell;

        // Get the offsets table
        const int octant = (dx_local - shift_cell > 0 ? 1 : 0) |
                           (dy_local - shift_cell > 0 ? 2 : 0) |
                           (dz_local - shift_cell > 0 ? 4 : 0);
        static const auto offsets = NEIGHBOR_ORDER[octant];

        // Compute the Voronoi volume of this particle
        double r2_max = cell.max_radius_squared();
        for (int o = 0; o < 27; o++) {
            int i = offsets[o][0];
            int j = offsets[o][1];
            int k = offsets[o][2];

            // x-direction
            int d_idx = idx + i;
            int wrap_x = 0;
            if (d_idx < 0) {
                d_idx += nd;
                wrap_x = -1;
            } else if (d_idx >= (int)nd) {
                d_idx -= nd;
                wrap_x = 1;
            }
            fft_real dist_x =
                (i == -1) ? dx_local : ((i == 1) ? Lcell - dx_local : 0.0);

            // y-direction
            int d_idy = idy + j;
            int wrap_y = 0;
            if (d_idy < 0) {
                d_idy += nd;
                wrap_y = -1;
            } else if (d_idy >= (int)nd) {
                d_idy -= nd;
                wrap_y = 1;
            }
            fft_real dist_y =
                (j == -1) ? dy_local : ((j == 1) ? Lcell - dy_local : 0.0);

            // z-direction
            int d_idz = idz + k;
            int wrap_z = 0;
            if (d_idz < 0) {
                d_idz += nd;
                wrap_z = -1;
            } else if (d_idz >= (int)nd) {
                d_idz -= nd;
                wrap_z = 1;
            }
            fft_real dist_z =
                (k == -1) ? dz_local : ((k == 1) ? Lcell - dz_local : 0.0);

            // If the block is too far away, skip all particles inside
            fft_real block_dist2 =
                dist_x * dist_x + dist_y * dist_y + dist_z * dist_z;
            if (block_dist2 > 4.0 * r2_max) {
                continue;
            }

            // Run over all particles in this block
            bool cell_cut = false;
            const std::size_t np_in_block =
                con.get_n_in_cell(d_idx, d_idy, d_idz);
            for (std::size_t part = 0; part < np_in_block; part++) {
                fft_real dx =
                    con[d_idx, d_idy, d_idz, part, 0] - px + wrap_x * L;
                fft_real dy =
                    con[d_idx, d_idy, d_idz, part, 1] - py + wrap_y * L;
                fft_real dz =
                    con[d_idx, d_idy, d_idz, part, 2] - pz + wrap_z * L;

                // Avoid self-particles and particles too far away
                fft_real d2 = dx * dx + dy * dy + dz * dz;
                if (d2 < 1e-12 || d2 > r2_max)
                    continue;

                // Compute the plane
                if (cell.plane(dx, dy, dz)) {
                    cell_cut = true;
                }
            }

            // If the plane was cut, update the maximum radius
            if (cell_cut) {
                r2_max = cell.max_radius_squared();
            }
        }

        // Check if the particle is a possible halo
        if (rho_halos > 0.0) {
            double den = 1.0 / cell.volume();
            // std::println("den: {} / {} / {}", den, rho_halos, np / (L * L * L));
            if (den > rho_halos) {
                halos.pos[3 * n_halos] = px;
                halos.pos[3 * n_halos + 1] = py;
                halos.pos[3 * n_halos + 2] = pz;
                halos.den[n_halos] = static_cast<fft_real>(den);

                n_halos++;
            }
        }

        // Check if a vertex is a possible void
        if (rho_voids > 0.0) {
            double radius2 = 0.0;
            double den = 10.0 * rho_voids;
            cell.vertices(verts);
            for (int v = 0; v < cell.p; v++) {
                double dx = verts[3 * v];
                double dy = verts[3 * v + 1];
                double dz = verts[3 * v + 2];
                double r2 = dx * dx + dy * dy + dz * dz;

                if (r2 > radius2) {
                    radius2 = r2;
                    den = static_cast<fft_real>(three_over_pi /
                                                std::pow(radius2, 1.5));

                    if (den < rho_voids) {
                        voids.pos[3 * n_voids] = px + static_cast<fft_real>(dx);
                        voids.pos[3 * n_voids + 1] =
                            py + static_cast<fft_real>(dy);
                        voids.pos[3 * n_voids + 2] =
                            pz + static_cast<fft_real>(dz);
                        voids.den[n_voids] = static_cast<fft_real>(den);
                    }
                }
            }
            if (den < rho_voids) {
                n_voids++;
            }
        }

        // Check if the arrays for halos and voids must be resized
        if (index % (np / 10) == 0 && index > 0) {
            double frac = static_cast<double>(index) / static_cast<double>(np);
            halos.resize(static_cast<std::size_t>(
                5.0f * static_cast<double>(n_halos) / frac));
            voids.resize(static_cast<std::size_t>(
                5.0f * static_cast<double>(n_voids) / frac));
        }

        // Save the volume if requested
        if (save_volume) {
            volume[index] = (fft_real)cell.volume();
        }
    }

    // Remove the duplicated void centter
    // if (n_voids > 0) {
    //     // Sort the voids by x
    //     std::size_t *v_idx = new std::size_t[n_voids];
    //     for (std::size_t i = 0; i < n_voids; i++) {
    //         v_idx[i] = i;
    //     }
    //     pdqsort(v_idx, v_idx + n_voids, [&voids](std::size_t a, std::size_t b) {
    //         return voids.pos[3 * a] < voids.pos[3 * b];
    //     });

    //     // Remove the duplicated voids
    //     std::size_t n_unique = 1;
    //     for (std::size_t i = 1; i < n_voids; i++) {
    //         double dx = voids.pos[3 * v_idx[i]] - voids.pos[3 * v_idx[i - 1]];
    //         double dy = voids.pos[3 * v_idx[i] + 1] - voids.pos[3 * v_idx[i - 1] + 1];
    //         double dz = voids.pos[3 * v_idx[i] + 2] - voids.pos[3 * v_idx[i - 1] + 2];
    //         if (dx * dx + dy * dy + dz * dz > 1e-3) {
    //             v_idx[n_unique++] = v_idx[i];
    //         }
    //     }

    //     // Filter the voids
    //     voids.filter(v_idx, n_unique);

    //     //Free the memory
    //     delete[] v_idx;
    // }
}
