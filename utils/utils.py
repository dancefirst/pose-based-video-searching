import numpy

# def summarize_vptree_result(res, dictmap):
#     try:
#         return [(dictmap[int(res[i][1][-1])], res[i][0]) for i in range(len(res))]
#     except:
#         return [(dictmap[int(res[i][1][...,-1,0])], res[i][0]) for i in range(len(res))]

# def rank_vptree_result(summaries):
#     res = {}
#     for result in summaries:
#         for i, (name, dist) in enumerate(result):
#             if name in res:
#                 # res[name] += (len(result)-i) * (1/dist)
#                 res[name] += (1/dist)
#             else:
#                 # res[name] = (len(result)-i) * (1/dist)
#                 res[name] = (1/dist)
#     return sorted(res.items(), key = lambda item: item[1], reverse=True)

# def vptree_search(tree, dictmap, query, start, end, interval, topk):
#     # vptree_search(tree, idx2vidname, queries[0], 2000, 2200, 5, 5) # query0 -> aespa1 2000 ~ 2200 frames
#     summaries = []
#     for i in range(start, end, interval):
#         result = tree.get_n_nearest_neighbors(query[i], topk)
#         summary = summarize_vptree_result(result, dictmap)
#         summaries.append(summary)
#     return summaries


def summarize_vptree_result(res, dictmap):
    if len(res[0][1].shape) == 1:
        try:
            return [(dictmap[int(res[i][1][-1])], res[i][0][0]) for i in range(len(res))]
        except:
            return [(dictmap[int(res[i][1][-1])], res[i][0]) for i in range(len(res))]
    
    elif len(res[0][1].shape) == 2:
        return [(dictmap[int(res[i][1][0][-1])], res[i][0]) for i in range(len(res))]

def rank_vptree_result(summaries):
    res = {}
    for result in summaries:
        for i, (name, dist) in enumerate(result):
            if name in res:
                res[name] *= (len(result)-i) * (1/dist)
                # res[name] += (1/dist)
            else:
                res[name] = (len(result)-i) * (1/dist)
                # res[name] = (1/dist)
    return sorted(res.items(), key = lambda item: item[1], reverse=True)

def vptree_search(tree, dictmap, query, start, end, interval, topk):
    summaries = []
    for i in range(start, end, interval):
        result = tree.get_n_nearest_neighbors(query[i], topk)
        summary = summarize_vptree_result(result, dictmap)
        summaries.append(summary)

    # for name in summary:
    #     if name in freq:
    #         freq[name] += 1
    #     else:
    #         freq[name] = 1

    return rank_vptree_result(summaries)[:5]