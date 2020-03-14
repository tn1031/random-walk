# distutils: language = c++
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

from libc.math cimport sqrt
from libc.stdlib cimport rand, RAND_MAX
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
cimport cython
from cython.operator cimport dereference, postincrement
from cython.parallel import prange


cdef extern from "<random>" namespace "std":
    cdef cppclass mt19937:
        mt19937()
        mt19937(unsigned int seed)

    cdef cppclass uniform_real_distribution[T]:
        uniform_real_distribution()
        uniform_real_distribution(T a, T b)
        T operator()(mt19937 gen)

    cdef cppclass geometric_distribution[T]:
        geometric_distribution()
        geometric_distribution(double p)
        T operator()(mt19937 gen)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef long _c_sample_neighbor(
    const long[:] adjacency,
    const long[:] offsets,
    long node
) nogil:
    cdef:
        long min_id, max_id
        const long[:] neighbors
    min_id = offsets[node]
    max_id = offsets[node + 1]

    if min_id == max_id:
        return -1
    neighbors = adjacency[min_id:max_id]
    return neighbors[<int>(<double>rand()/RAND_MAX * (max_id - min_id))]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef long c_pixie_random_walk(
    const long[:] adjacency,
    const long[:] offsets,
    long q,
    long max_steps,
    long _n_least_candidate_nodes,
    long _n_least_visited_cnt,
    long _n_total_steps,
    long[:] sample_steps,
    long steps_cnt,
    unordered_map[long, long]& visit_count
) nogil:
    cdef:
        long total_steps = 0
        long n_high_visited = 0
        long curr_item, curr_user
        int i

    while (
        total_steps < max_steps
        and n_high_visited < _n_least_candidate_nodes
    ):
        # Restart to the query track
        curr_item = q
        for i in range(sample_steps[steps_cnt]):
            curr_user = _c_sample_neighbor(
                adjacency, offsets, curr_item
            )
            # Reached a dead end
            if curr_user == -1:
                if total_steps == 0:
                    # The query node does not have any connections
                    steps_cnt += 1
                    return steps_cnt
                else:
                    break
    
            curr_item = _c_sample_neighbor(
                adjacency, offsets, curr_user
            )
            # Reached a dead end
            if curr_item == -1:
                break
    
            visit_count[curr_item] += 1
    
            if visit_count[curr_item] == _n_least_visited_cnt:
                n_high_visited += 1
    
            total_steps += 1
            if (
                total_steps >= _n_total_steps
                or n_high_visited >= _n_least_candidate_nodes
            ):
                break

        steps_cnt += 1
    
    return steps_cnt

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef wrap_c_pixie_random_walk(
    const long[:] adjacency,
    const long[:] offsets,
    const long[:] query_items,
    long[:] steps_per_query,
    long _n_least_candidate_nodes,
    long _n_least_visited_cnt,
    long _n_total_steps,
    long[:] sample_steps,
    bint _use_boosting
):
    cdef:
        long total_random_walks = <long>(query_items.nbytes / 8)
        long i, q, n_q, item, vc
        double s
        unordered_map[long, long] v, visit_count
        vector[unordered_map[long, long]] visit_counts = <vector[unordered_map[long, long]]>[]
        unordered_map[long, long].iterator it
        unordered_map[long, double] boosted_visit_counts
        unordered_map[long, double].iterator it_bc
        long steps_cnt = 0

    visit_counts.resize(total_random_walks)
    for i in range(total_random_walks):
        q = query_items[i]
        n_q = steps_per_query[i]
        
        visit_count.clear()
        steps_cnt = c_pixie_random_walk(
            adjacency, offsets,
            q, n_q, _n_least_candidate_nodes, _n_least_visited_cnt,
            _n_total_steps, sample_steps, steps_cnt, visit_count
        )
        visit_counts[i] = visit_count

    for i in range(total_random_walks):
        v = visit_counts[i]
        it = v.begin()
        while it != v.end():
            item = dereference(it).first
            vc = dereference(it).second
            if _use_boosting:
                if boosted_visit_counts.find(item) != boosted_visit_counts.end():
                    boosted_visit_counts[item] += sqrt(vc)
                else:
                    boosted_visit_counts[item] = sqrt(vc)
            else:
                if boosted_visit_counts.find(item) != boosted_visit_counts.end():
                    boosted_visit_counts[item] += vc
                else:
                    boosted_visit_counts[item] = vc
            postincrement(it)

    if _use_boosting:
        it_bc = boosted_visit_counts.begin()
        while it_bc != boosted_visit_counts.end():
            boosted_visit_counts[dereference(it_bc).first] = dereference(it_bc).second ** 2
            postincrement(it_bc)

    return boosted_visit_counts