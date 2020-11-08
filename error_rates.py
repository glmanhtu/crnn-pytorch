import editdistance


def cer(r, h):
    return err(r.strip(), h.strip())


def err(r, h):
    dis = editdistance.eval(r, h)
    if len(r) == 0.0:
        return len(h)

    return float(dis) / float(len(r))


def wer(r, h):
    r = r.split()
    h = h.split()

    return err(r, h)


def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    if len(union) == 0:
        print(f'Query: {query}, Doc: {document}')
    return len(intersection)/len(union)


if __name__ == '__main__':
    print(jaccard_similarity(['this', 'is', 'a', 'test'], ['a', 'is']))