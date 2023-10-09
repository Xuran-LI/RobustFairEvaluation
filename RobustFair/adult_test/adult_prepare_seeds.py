import numpy
from utils.utils_generate import get_retrain_seeds

if __name__ == "__main__":
    retrain_file = "../dataset/adult/data/train.npz.npy"
    search_seeds, avg_clusters = get_retrain_seeds(retrain_file, 4, 100)
    numpy.save("../dataset/adult/test/retrain_seeds.npz.npy", search_seeds)
    numpy.save("../dataset/adult/test/retrain_avg_clusters.npz.npy", avg_clusters)

    test_file = "../dataset/adult/data/test.npz.npy"
    search_seeds, avg_clusters = get_retrain_seeds(test_file, 4, 100)
    numpy.save("../dataset/adult/test/test_seeds.npz.npy", search_seeds)
    numpy.save("../dataset/adult/test/test_avg_clusters.npz.npy", avg_clusters)
