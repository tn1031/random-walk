import os
import numpy as np
import pandas as pd
import pickle


def _random_split(df, n):
    df = df.sample(frac=1, random_state=1219)
    test_df = df.sample(n=n)
    train_df = df[~df.index.isin(test_df.index)]
    return train_df, test_df


def _timeseries_split(df, n):
    df = df.sort_values("timestamp")
    train_df = df.iloc[:-n]
    test_df = df.iloc[-n:]
    return train_df, test_df


def build_graph(
    path, min_user=5, min_item=0, test_size=0.2, split="random", out_dir=None
):
    if os.path.isdir(path):
        return _load_graph(path)

    df = load_datacsv(path)

    if "rating" in df.columns:
        # 4点以上をpositiveとする
        df = df[df.rating >= 4]
    cold_user = df.groupby("user_id").count().item_id < min_user
    cold_item = df.groupby("item_id").count().user_id < min_item
    df = df[~df.user_id.isin(cold_user[cold_user].index)]
    df = df[~df.item_id.isin(cold_item[cold_item].index)]

    # user id 採番
    user2uid = {v: i for i, v in enumerate(df.user_id.unique())}
    item2iid = {
        v: i + len(user2uid) for i, v in enumerate(df.item_id.unique())
    }
    n_users = len(user2uid)
    n_items = len(item2iid)
    df.user_id = df.user_id.map(user2uid)
    df.item_id = df.item_id.map(item2iid)

    if isinstance(test_size, float):
        n_test = int(test_size * len(df))
    elif isinstance(test_size, int):
        n_test = test_size
    else:
        n_test = 0
    if split == "random":
        train_df, test_df = _random_split(df, n_test)
    elif split == "time":
        train_df, test_df = _timeseries_split(df, n_test)
    else:
        ValueError("unknown splitting.")

    train_df = train_df[["user_id", "item_id"]].drop_duplicates()
    test_df = test_df[["user_id", "item_id"]].drop_duplicates()

    ui_adj = train_df.groupby("user_id").item_id.apply(list)
    if len(ui_adj) < n_users:
        ui_adj = ui_adj.reindex(pd.RangeIndex(n_users))
        for i in ui_adj[ui_adj.isnull()].index:
            ui_adj.loc[i] = []
    iu_adj = train_df.groupby("item_id").user_id.apply(list)
    if len(iu_adj) < n_items:
        iu_adj = iu_adj.reindex(pd.RangeIndex(n_users, n_users + n_items))
        for i in iu_adj[iu_adj.isnull()].index:
            iu_adj.loc[i] = []

    adjacency = pd.concat([ui_adj, iu_adj])
    offsets = adjacency.apply(len)
    offsets = np.cumsum([0] + offsets.tolist())
    adjacency = np.concatenate(adjacency.tolist()).astype(np.int)

    assert len(offsets) == n_users + n_items + 1
    assert (
        adjacency[offsets[0] : offsets[1]].tolist() == ui_adj.head(1).iloc[0]
    )
    assert (
        adjacency[
            offsets[n_users + n_items - 1] : offsets[n_users + n_items]
        ].tolist()
        == iu_adj.tail(1).iloc[0]
    )

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "adjacency.npy"), adjacency)
        np.save(os.path.join(out_dir, "offsets.npy"), offsets)
        test_df[["user_id", "item_id"]].to_csv(
            os.path.join(out_dir, "test_df.csv"), index=False
        )
        with open(os.path.join(out_dir, "user2uid.pkl"), "wb") as f:
            pickle.dump(user2uid, f)
        with open(os.path.join(out_dir, "item2iid.pkl"), "wb") as f:
            pickle.dump(item2iid, f)

    return adjacency, offsets, test_df, user2uid, item2iid


def load_datacsv(path):
    fname = os.path.basename(path)
    if fname == "yelp.tsv" or fname == "u.data":
        return _yelp(path)
    elif fname == "ratings.dat":
        return _ml_1m(path)


def _yelp(path):
    df = pd.read_csv(
        path,
        sep="\t",
        names="user_id,item_id,rating,timestamp".split(","),
        header=None,
    )
    return df


def _ml_1m(path):
    df = pd.read_csv(
        path,
        sep="::",
        names="user_id,item_id,rating,timestamp".split(","),
        header=None,
    )
    return df


def _load_graph(path):
    adjacency = np.load(os.path.join(path, "adjacency.npy"))
    offsets = np.load(os.path.join(path, "offsets.npy"))
    test_df = pd.read_csv(os.path.join(path, "test_df.csv"), sep=",", header=0)
    user2uid = pickle.load(open(os.path.join(path, "user2uid.pkl"), "rb"))
    item2iid = pickle.load(open(os.path.join(path, "item2iid.pkl"), "rb"))
    return adjacency, offsets, test_df, user2uid, item2iid


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("inp")
    parser.add_argument("--min_user", "-mu", type=int, default=5)
    parser.add_argument("--min_item", "-mi", type=int, default=0)
    parser.add_argument("--split", type=str, default="random")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    build_graph(args.inp, args.min_user, args.min_item, args.split, args.out)
