import numpy as np
from c_randomwalk import wrap_c_pixie_random_walk


class RandomWalk:
    def __init__(
        self,
        alpha=0.1,
        n_total_steps=100000,
        n_least_candidate_nodes=500,
        n_least_visited_cnt=4,
        use_boosting=False,
        dynamic_allocation=False,
        use_pixie_weighting=False,
        n_jobs=0,
    ):
        self._alpha = alpha
        self._n_total_steps = n_total_steps  # N
        self._n_least_candidate_nodes = n_least_candidate_nodes  # n_p
        self._n_least_visited_cnt = n_least_visited_cnt  # n_v
        self._use_boosting = use_boosting
        self._dynamic_allocation = dynamic_allocation
        self._use_pixie_weighting = use_pixie_weighting
        self._n_jobs = n_jobs

    def load_graph(self, adjacency, offsets, n_users, n_items):
        self._adjacency = adjacency
        self._offsets = offsets
        assert len(self._offsets) == n_users + n_items + 1
        self._offsets.setflags(write=False)
        self._adjacency.setflags(write=False)
        self.n_users = n_users
        self.n_items = n_items
        self.n_nodes = n_users + n_items
        self.max_item_degree = np.max(
            [self.get_degree(n) for n in range(n_users, self.n_nodes)]
        )
        return self

    def get_degree(self, node):
        return self._offsets[node + 1] - self._offsets[node]

    def get_neighbors(self, node):
        min_id = self._offsets[node]
        max_id = self._offsets[node + 1]
        return self._adjacency[min_id:max_id]

    def run_random_walk(self, query_items):
        self._all_steps_cnt = 0

        # the number of steps for a query
        if self._dynamic_allocation:
            scaling_factors = [self.scaling_factor(q) for q in query_items]
            summed_scaling_factors = np.sum(scaling_factors)
            steps_per_query = (
                self._n_total_steps
                * np.array(scaling_factors)
                / summed_scaling_factors
            ).astype(np.int)
        else:
            total_random_walks = len(query_items)
            steps_per_query = np.array(
                [int(self._n_total_steps / total_random_walks)]
                * total_random_walks,
                dtype=np.int,
            )

        sample_steps = np.random.geometric(
            self._alpha, size=(self._n_total_steps,)
        )

        return wrap_c_pixie_random_walk(
            self._adjacency,
            self._offsets,
            query_items,
            steps_per_query,
            self._n_least_candidate_nodes,
            self._n_least_visited_cnt,
            self._n_total_steps,
            sample_steps,
            self._use_boosting,
        )

    def scaling_factor(self, query_item):
        item_degree = self.get_degree(query_item)
        if item_degree == 0:
            return 0
        else:
            if self._use_pixie_weighting:
                return item_degree * (
                    self.max_item_degree - np.log(item_degree)
                )
            else:
                return item_degree * (
                    1 - np.log(item_degree) + np.log(self.max_item_degree)
                )
