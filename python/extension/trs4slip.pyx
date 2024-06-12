#distutils: language = c++
#cython: language_level=3

from libc.stdint cimport int32_t
import numpy as np
cimport numpy as np

cdef extern from "trs4slip.h":
    void trs4slip_astar(
        int32_t * x_next_out,
        const double *c,
        const int32_t *x,
        const int32_t *bangs,
        const int delta,
        const int N,
        const int M,
        double * vert_costs_buffer,
        int32_t * vert_layer_buffer,
        int32_t * vert_value_buffer,
        int32_t * vert_prev_buffer,
        int32_t * vert_remcap_buffer,
        const int boundcon,
        double lbound,
        double rbound  
    );

def run(np.int32_t[::1] x_next not None, # output argument
        double[::1] c not None,
        np.int32_t[::1] x not None,
        np.int32_t[::1] bangs not None,
        int delta,
        double[::1] vert_costs_buffer not None,
        np.int32_t[::1] vert_layer_buffer not None,
        np.int32_t[::1] vert_value_buffer not None,
        np.int32_t[::1] vert_prev_buffer not None,
        np.int32_t[::1] vert_remcap_buffer not None,
        int boundcon,
        double lbound,
        double rbound
        ):        
    trs4slip_astar(&x_next[0],
                   &c[0],
                   &x[0],
                   &bangs[0],
                   delta,
                   x.shape[0],
                   bangs.shape[0],
                   &vert_costs_buffer[0],
                   &vert_layer_buffer[0],
                   &vert_value_buffer[0],
                   &vert_prev_buffer[0],
                   &vert_remcap_buffer[0],
                   boundcon,
                   lbound,
                   rbound);
