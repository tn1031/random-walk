import numpy as np
import random
from collections import Counter
from c_sample_neighbor import _c_sample_neighbor


class RandomWalk:
    def __init__(
        self,
        alpha=0.1,
        n_total_steps=100000,
        n_least_candidate_nodes=500,
        n_least_visited_cnt=4,
        use_boosting=True,
        dynamic_allocation=True,
        use_pixie_weighting=False,
    ):
        self._alpha = alpha
        self._n_total_steps = n_total_steps  # N
        self._n_least_candidate_nodes = n_least_candidate_nodes  # n_p
        self._n_least_visited_cnt = n_least_visited_cnt  # n_v
        self._use_boosting = use_boosting
        self._dynamic_allocation = dynamic_allocation
        self._use_pixie_weighting = use_pixie_weighting

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

    def sample_neighbor(self, node):
        n = self.get_neighbors(node)
        return n[int(random.random() * len(n))]

    def run_random_walk(self, query_items):
        self._all_steps_cnt = 0
        visit_counts = []
        boosted_visit_counts = {}

        # the number of steps for a query
        if self._dynamic_allocation:
            scaling_factors = [self.scaling_factor(q) for q in query_items]
            summed_scaling_factors = np.sum(scaling_factors)
            steps_per_query = (
                self._n_total_steps
                * np.array(scaling_factors)
                / summed_scaling_factors
            )
        else:
            total_random_walks = len(query_items)
            steps_per_query = [
                int(self._n_total_steps / total_random_walks)
            ] * total_random_walks

        for q, n_q in zip(query_items, steps_per_query):
            visit_q = self.pixie_random_walk(q, n_q)
            visit_counts.append(visit_q)

        for v in visit_counts:  # list of Counter
            for item, visit_count in v.items():
                if self._use_boosting:
                    if item in boosted_visit_counts:
                        boosted_visit_counts[item] += np.sqrt(visit_count)
                    else:
                        boosted_visit_counts[item] = np.sqrt(visit_count)
                else:
                    if item in boosted_visit_counts:
                        boosted_visit_counts[item] += visit_count
                    else:
                        boosted_visit_counts[item] = visit_count

        if self._use_boosting:
            boosted_visit_counts = {
                k: v ** 2 for k, v in boosted_visit_counts.items()
            }

        return boosted_visit_counts

    def pixie_random_walk(self, q, steps):
        visit_count = Counter()
        total_steps = 0
        n_high_visited = 0

        while (
            total_steps < steps
            and n_high_visited < self._n_least_candidate_nodes
        ):
            # Restart to the query track
            curr_item = q
            sample_steps = np.random.geometric(self._alpha)
            for i in range(sample_steps):
                curr_user = _c_sample_neighbor(
                    self._adjacency, self._offsets, curr_item
                )
                if curr_user == -1:
                    # Reached a dead end
                    if total_steps == 0:
                        # The query node does not have any connections
                        return Counter()
                    else:
                        break

                curr_user = _c_sample_neighbor(
                    self._adjacency, self._offsets, curr_item
                )
                if curr_user == -1:
                    # Reached a dead end
                    break

                visit_count[curr_item] += 1

                if visit_count[curr_item] == self._n_least_visited_cnt:
                    n_high_visited += 1

                total_steps += 1
                self._all_steps_cnt += 2
                if (
                    total_steps >= self._n_total_steps
                    or n_high_visited >= self._n_least_candidate_nodes
                ):
                    break
        return visit_count

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

