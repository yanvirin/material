import numpy as np

def borda_count_rank_merge(rankings):
    """
    Aggregates multiple rankings to produce a single overall ranking of a list
    of items.

    rankings: a list of lists, each sublist is a ranking (0 is best)

    For example, let's say we are ranking 4 items A, B, C, and D and we have
    4 sets of a rankings. The input rankings might look like this:
    rankings = [[0, 1, 2, 3],
                [3, 0, 1, 2],
                [3, 1, 0, 2],
                [3, 1, 2, 0]]

    where: 
    [0, 1, 2, 3] expresses a preference for A, B, C, D with A the best;
    [3, 0, 1, 2] expresses a preference for B, C, D, A with B the best;
    [3, 1, 0, 2] expresses a preference for C, B, D, A with C the best;
    [3, 1, 2, 0] expresses a preference for D, B, C, A with D the best.

    agg_ranking = borda_count_rank_merge(rankings) 

    print(agg_ranking)
    [3 0 1 2] which corresponds to the ranking B, C, D, A  
    """

    # trivial case
    if len(rankings) == 1: return rankings

    all_points = [0 for _ in rankings[0]]
    max_rank = len(rankings[0]) - 1
    for ranking in rankings:
        for i, rank in enumerate(ranking):
            points = max_rank - rank
            all_points[i] += points
    order = np.argsort(all_points)[::-1].tolist()
    agg_rank = [order.index(i) for i in range(max_rank + 1)]
    return agg_rank
