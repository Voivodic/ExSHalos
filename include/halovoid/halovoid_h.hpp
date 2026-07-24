/* Auxiliary types and functions shared by the 2D and 3D halo/void finders. */
#ifndef HALOVOID_H
#define HALOVOID_H

#ifdef DOUBLEPRECISION_FFTW
using fft_real = double;
#define NP_OUfft_real_fft_realYPE NPY_DOUBLE
#else
using fft_real = float;
#define NP_OUT_TYPE NPY_FLOAT
#endif

// Include headers for the Voronoi computations
#include <cstddef>

// Struct for the halos and voids
struct HaloVoid {
    fft_real *pos;
    fft_real *den;
    std::size_t n;
    std::size_t capacity;

    // Constructor without capacity
    HaloVoid() : capacity(0), n(0) {
        pos = new fft_real[3 * capacity];
        den = new fft_real[capacity];
    }

    // Constructor with capacity
    HaloVoid(std::size_t _capacity) : capacity(_capacity), n(0) {
        pos = new fft_real[3 * capacity];
        den = new fft_real[capacity];
    }

    // Destructor
    ~HaloVoid() {
        delete[] pos;
        delete[] den;
    }

    // Resize the arrays
    void resize(std::size_t capacity_new) {
        if (capacity_new <= capacity) {
            return;
        }

        // Allocate the new arrays
        fft_real *pos_new = new fft_real[3 * capacity_new];
        fft_real *den_new = new fft_real[capacity_new];

        // Copy the old arrays to the new arrays
        for (std::size_t i = 0; i < n; i++) {
            pos_new[3 * i] = pos[3 * i];
            pos_new[3 * i + 1] = pos[3 * i + 1];
            pos_new[3 * i + 2] = pos[3 * i + 2];
            den_new[i] = den[i];
        }

        // Delete the old arrays
        delete[] pos;
        delete[] den;

        // Assign the new arrays
        pos = pos_new;
        den = den_new;
        capacity = capacity_new;
    }

    // Filter the arrays 
    void filter(const std::size_t *indices, std::size_t n_indices) {
        if (n_indices == n) {
            return;
        }

        // Allocate the new arrays
        fft_real *pos_new = new fft_real[3 * n_indices];
        fft_real *den_new = new fft_real[n_indices];

        // Copy the old arrays to the new arrays
        for (std::size_t i = 0; i < n_indices; i++) {
            pos_new[3 * i] = pos[3 * indices[i]];
            pos_new[3 * i + 1] = pos[3 * indices[i] + 1];
            pos_new[3 * i + 2] = pos[3 * indices[i] + 2];
            den_new[i] = den[indices[i]];
        }

        // Delete the old arrays
        delete[] pos;
        delete[] den;

        // Assign the new arrays
        pos = pos_new;
        den = den_new;
        n = n_indices;
        capacity = n_indices;
    }
};

// Get the cell index of a particle
inline std::size_t get_cell_index(const fft_real p[3], std::size_t Nd,
                                  fft_real scale) {
    std::size_t idx = static_cast<std::size_t>(p[0] * scale);
    if (idx >= Nd)
        idx = Nd - 1;
    std::size_t idy = static_cast<std::size_t>(p[1] * scale);
    if (idy >= Nd)
        idy = Nd - 1;
    std::size_t idz = static_cast<std::size_t>(p[2] * scale);
    if (idz >= Nd)
        idz = Nd - 1;

    return idx * Nd * Nd + idy * Nd + idz;
}

// Struct with the container for the particles
class Container {
    fft_real *pos;
    std::size_t *offset;
    std::size_t np;
    std::size_t nd;
    fft_real L;

  public:
    // Constructor
    Container(fft_real *_pos, std::size_t _np, std::size_t _nd, fft_real _L)
        : pos(_pos), np(_np), nd(_nd), L(_L) {
        // Compute the total number of cells
        const std::size_t num_cells = nd * nd * nd;

        // Allocate the offset array and initialize it
        offset = new std::size_t[num_cells + 1];
        for (std::size_t i = 0; i < num_cells + 1; i++)
            offset[i] = 0;

        // Count the number of particles in each cell
        const fft_real scale = static_cast<fft_real>(nd) / L;
        for (std::size_t i = 0; i < np; i++) {
            fft_real p[3] = {pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]};
            std::size_t id = get_cell_index(p, nd, scale);

            offset[id + 1]++;
        }

        // Compute the offset of each cell
        std::size_t current_offset = 0;
        std::size_t *write_ptr = new std::size_t[num_cells];
        for (std::size_t i = 0; i < num_cells; i++) {
            offset[i] = current_offset;
            write_ptr[i] = current_offset;
            current_offset += offset[i + 1];
        }
        offset[num_cells] = current_offset;

        // Fill the array of particles using in-place binning
        for (std::size_t b = 0; b < num_cells; b++) {
            // Loop while the current block still has unsorted slots
            while (write_ptr[b] < offset[b + 1]) {
                std::size_t i = write_ptr[b];
                fft_real p[3] = {pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]};
                std::size_t target_b = get_cell_index(p, nd, scale);

                // Check if the particle belongs to the current block
                if (target_b == b) {
                    write_ptr[b]++;
                } else {
                    std::size_t dest = write_ptr[target_b];

                    // Swap the particles
                    for (int dim = 0; dim < 3; dim++) {
                        pos[3 * i + dim] = pos[3 * dest + dim];
                        pos[3 * dest + dim] = p[dim];
                    }

                    // Increment the write pointer
                    write_ptr[target_b]++;
                }
            }
        }

        delete[] write_ptr;
    }

    // Destructor
    ~Container() { delete[] offset; }

    // Indexing operator for cell, particle, and dimension
    fft_real operator[](std::size_t i, std::size_t j, std::size_t k,
                              std::size_t n, std::size_t dim) const {
        std::size_t idx = i * nd * nd + j * nd + k;
        return pos[3 * (offset[idx] + n) + dim];
    }

    // Indexing operator for index, dimension
    fft_real operator[](std::size_t idx, std::size_t dim) const {
        return pos[3 * idx + dim];
    }

    // Get the number of particles
    std::size_t get_np() const { return np; }

    // Get the number of cells per dimension
    std::size_t get_nd() const { return nd; }

    // Get the number of cells
    std::size_t get_n_cells() const { return nd * nd * nd; }

    // Get the box size
    fft_real get_L() const { return L; }

    // Get the size of each cell
    fft_real get_Lcell() const { return L / static_cast<fft_real>(nd); }

    // Get the scale factor
    fft_real get_scale() const { return 1.0 / get_Lcell(); }

    // Get the number of particles in a cell
    std::size_t get_n_in_cell(std::size_t i, std::size_t j,
                              std::size_t k) const {
        std::size_t idx = i * nd * nd + j * nd + k;
        return offset[idx + 1] - offset[idx];
    }
};

#endif
