import numpy as np
import pandas as pd


def build_graph(path, min_user=5, min_item=0):
    df = pd.read_csv(
        path,
        sep="::",
        names="UserID::MovieID::Rating::Timestamp".split("::"),
        header=None,
    )
    # 4点以上をpositiveとする
    df = df[df.Rating >= 4]
    cold_user = df.groupby('UserID').count().MovieID < min_user
    cold_item = df.groupby('MovieID').count().UserID < min_item
    df = df[~df.UserID.isin(cold_user[cold_user].index)]
    df = df[~df.MovieID.isin(cold_item[cold_item].index)]

    # user id 採番
    user2uid = {v: i for i, v in enumerate(df.UserID.unique())}
    item2iid = {v: i + len(user2uid) for i, v in enumerate(df.MovieID.unique())}
    n_users = len(user2uid)
    n_items = len(item2iid)
    df['user_id'] = df.UserID.map(user2uid)
    df['item_id'] = df.MovieID.map(item2iid)

    # shuffle and sample test data
    df = df.sample(frac=1, random_state=1219)
    test_ratings = df.sample(n=16)
    while len(test_ratings.user_id.unique()) < 16:
        test_ratings = df.sample(n=16)
    df = df[~df.index.isin(test_ratings.index)]

    df = df[['user_id', 'item_id']].drop_duplicates()
    ui_adj = df.groupby('user_id').item_id.apply(list)
    iu_adj = df.groupby('item_id').user_id.apply(list)

    adjacency = pd.concat([ui_adj, iu_adj])
    offsets = adjacency.apply(len)
    offsets = np.cumsum([0] + offsets.tolist())
    adjacency = np.concatenate(adjacency.tolist())

    assert len(offsets) == n_users + n_items + 1
    assert (
        adjacency[offsets[0] : offsets[1]].tolist()
        == ui_adj.head(1).iloc[0]
    )
    assert (
        adjacency[
            offsets[n_users + n_items - 1] : offsets[n_users + n_items]
        ].tolist()
        == iu_adj.tail(1).iloc[0]
    )
        self._n_users = n_users
        self._n_items = n_items
        self._user2uidx = user2uidx
        self._item2iidx = item2iidx
        # reverse
        self._node2instance = {v: k for k, v in user2uidx.items()}
        self._node2instance.update({v: k for k, v in item2iidx.items()})
        # graph stracture
        self._adjacency = adjacency
        self._offsets = offsets
    
    # rating_per_item[item_id].append(rating)
    # num_reviews_per_item = [len(rating_per_item[i]) for i in n_items]
    # avg_reviews_per_item = [sum(rating_per_item[i]) / len(rating_per_item[i]) for i in n_items]
    # shuffle

    # leave out two for each user
    # test_per_user: [(-1, -1) for _ in n_users]
    # val_per_user: [(-1, -1) for _ in n_users]

    # pos_per_user: [{} for _ in n_users]
    # pos_per_item: [{} for _ in n_items]
    # for v in votes:
    #   if test_per_user[v.user] == -1:
    #     test_per_user[v.user] = (v.item, v.ts)
    #   elif val_per_user[v.user] == -1:
    #     val_per_user[v.user] = (v.item, v.ts)
    #   else:
    #     pos_per_user[v.user][v.item] = v.ts
    #     pos_per_item[v.item][v.user] = v.ts

    # most popular: return pos_per_item[i]


def auc(auc_val, auc_test):
    auc_u_val = [0.0] * n_users
    auc_u_test = [0.0] * n_users

    for u in range(n_users):
        item_test, item_val = test_per_user[u], val_per_user[u]
        x_u_test, x_u_val = prediction(u, item_test), prediction(item_val)

        count_test, count_val = 0, 0
        max = 0
        for j in range(n_items):
            if j in pos_per_user[u] or j == item_test or j == item_val:
                continue
            max += 1
            x_uj = prediction(u, j)
            if x_u_test > x_uj:
                count_test += 1
            if x_u_val > x_uj:
                count_val += 1
        auc_u_test = count_test / max
        auc_u_val = count_val / max

    auc_val, auc_test = 0, 0
    for u in range(n_users):
        auc_val += auc_u_val[u]
        auc_test += auc_u_test[u]
    auc_val /= n_users
    auc_test /= n_users

    variance = 0
    for u in range(n_users):
        variance += square(auc_u_test[u] - auc_test)
    std = sqrt(variance / n_users)


def auc_colditem(auc_test):
    auc_u_test = [-1 for u in range(n_users)]  # denote not testing

    for u in range(n_users):
        item_test = test_per_user[u].item
        item_val = val_per_user[u].item

        if len(pos_per_item[item_test]) > 5:
            continue

        x_u_test = prediction(u, item_test)

        count_test = 0
        max = 0
        for j in range(n_items):
            if j in pos_per_user[u] or item_test == j or j == item_val:
                continue
            max += 1
            x_uj = prediction(u, j)
            if x_u_test > x_uj:
                count_test += 1
        auc_u_test[u] = count_test / max

    auc_test = 0
    num_user = 0
    for u in range(n_user):
        if auc_u_test[u] != -1:
            auc_test += auc_u_test[u]
            num_user += 1
    auc_test /= num_user

    variance = 0
    for u in range(n_user):
        if auc_u_test[u] != -1:
            variance += square(auc_u_test[u] - auc_test)
    std = sqrt(variance / num_user)

