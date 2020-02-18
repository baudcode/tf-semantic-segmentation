from tf_semantic_segmentation import threading
import random


def f(i):
    return i * 2


def test_parallize():
    args = list(range(100))

    results = threading.parallize(f, args, threads=2)
    assert(all([(args[i] * 2) == results[i] for i in range(len(args))]))

    results = threading.parallize(f, args, threads=None)
    assert(all([(args[i] * 2) == results[i] for i in range(len(args))]))

    results = threading.parallize_v2(f, args)
    assert(all([(args[i] * 2) == results[i] for i in range(len(args))]))

    results = threading.parallize_v3(f, args)
    assert(all([(args[i] * 2) == results[i] for i in range(len(args))]))

    results = threading.parallize_v3(f, args, n_processes=2)
    assert(all([(args[i] * 2) == results[i] for i in range(len(args))]))
