from libc.stdlib cimport rand, RAND_MAX

cpdef _c_sample_neighbor(
    const long[:] adjacency,
    const long[:] offsets,
    long node
):
    cdef:
        long min_id, max_id
        const long[:] neighbors
    min_id = offsets[node]
    max_id = offsets[node + 1]

    if min_id == max_id:
        return -1
    neighbors = adjacency[min_id:max_id]
    return neighbors[<int>(<double>rand()/RAND_MAX * (max_id - min_id))]
