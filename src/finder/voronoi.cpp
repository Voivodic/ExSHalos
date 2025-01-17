#include "voronoi.hpp"

/*Function that computs the voronoi tesselation using voro++*/
void Compute_Voronoi(fft_real *pos, size_t np, fft_real *vol, int get_vol) {
    int i;

    // Create the container to put the particles
    voro::container con(0.0, box.L[0], 0.0, box.L[1], 0.0, box.L[2], box.nd[0],
                        box.nd[1], box.nd[2], true, true, true,
                        PART_ALLOC_PER_BLOCK);
    
    // Add the particles to the container
    for(i=0;i<np;i++)
        con.put(i, (double) pos[3*i], (double) pos[3*i+1], (double) pos[3*i+2]);

    // Create and initialize the loop class
    voro::c_loop_all loop(con);
    loop.start();

    // Define the voronoi cell used to get the properties of each cell
    voro::voronoicell c;

    // Loop over all cells in the container
    i = 0;
    do{
        con.compute_cell(c, loop);
        
        // Compute the volume of this cell
        if(get_vol == TRUE)
            vol[i] = (fft_real) c.volume();

        i ++;
    } while(loop.inc());
}
