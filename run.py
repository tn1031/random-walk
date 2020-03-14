import inspect
import numpy as np
from time import time
from tqdm import tqdm
from build_graph import build_graph
from randomwalk.rw import RandomWalk as RW
from randomwalk.rw_cache import RandomWalk as RWcache
from randomwalk.rw_parallel import RandomWalk as RWparallel
from randomwalk.rw_cython import RandomWalk as RWcython


def run_evaluate(model, latest_prefs, top_k):
    scores = []
    elapsed_times = []

    for u, prefs in tqdm(latest_prefs.items()):
        if len(prefs) == 0:
            continue

        known_prefs = model.get_neighbors(u)
        if len(known_prefs) == 0:
            # new user
            continue

        t1 = time()
        visit_count = model.run_random_walk(query_items=known_prefs)
        t2 = time()

        recommended_items = [
            key
            for (key, value) in sorted(
                iter(visit_count.items()), key=lambda k_v: (k_v[1], k_v[0])
            )[::-1]
            if key not in known_prefs
        ][:top_k]

        # calc ndcg
        ind = np.array([1 if i in prefs else 0 for i in recommended_items])
        if len(ind) < top_k:
            ind = np.concatenate((ind, np.zeros(top_k - len(ind))))
        denom = np.log(np.arange(2, top_k + 2))
        numer = 2 ** ind - 1
        s = np.sum(numer / denom)
        scores.append(s)
        elapsed_times.append(t2 - t1)

    print("model: {}".format(inspect.getfile(model.__class__)))
    print(
        "avg.elapsed time: {}[s]".format(
            sum(elapsed_times) / len(elapsed_times)
        )
    )
    print(
        "avg.nDCG@{}(#samples): {:.3f}({})"
        "".format(top_k, np.mean(scores), len(scores))
    )


def main(path):
    adjacency, offsets, test_df, user2uid, item2iid = build_graph(
        path, test_size=32, split="time", out_dir="model"
    )
    n_users, n_items = len(user2uid), len(item2iid)
    latest_prefs = test_df.groupby("user_id").item_id.apply(list).to_dict()

    # rw = RW(alpha=0.1, n_total_steps=100000).load_graph(
    #    adjacency, offsets, n_users, n_items
    # )
    # run_evaluate(rw, latest_prefs, 32)

    # rw = RWcache(alpha=0.1, n_total_steps=100000).load_graph(
    #    adjacency, offsets, n_users, n_items
    # )
    # run_evaluate(rw, latest_prefs, 32)

    # rw = RWparallel(alpha=0.1, n_total_steps=100000).load_graph(
    #    adjacency, offsets, n_users, n_items
    # )
    # run_evaluate(rw, latest_prefs, 32)

    rw = RWcython(alpha=0.1, n_total_steps=100000).load_graph(
        adjacency, offsets, n_users, n_items
    )
    run_evaluate(rw, latest_prefs, 32)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("inp")
    args = parser.parse_args()
    main(args.inp)
