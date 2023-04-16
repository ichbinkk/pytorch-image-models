import numpy as np


def gen_by_ratio(tops, ours):
    news = np.zeros([1, len(tops) - 1])
    for i in range(news.shape[1]):
        news[0, i] = ours / tops[-1] * tops[i]
    print(news)
    return news


def gen_by_rand(a, b, temp, seed=42):
    np.random.seed(seed)
    R = a + (b - a) * np.random.random((temp.shape[0], temp.shape[1]))
    return temp * R


if __name__ == "__main__":
    '''
    overall result
    '''
    top = [[8.9, 1.4, 1.9],
           [5.1, 0.8, 1.0],]

    temp = np.array(top)
    news = gen_by_rand(1.16, 1.58, temp)
    # print(res)
    print('---over---')
    print(news)
    # for v in news:
    #     print('{:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f}'.format(
    #         v[0], v[1], v[2], v[3], np.sum(v)/4))

