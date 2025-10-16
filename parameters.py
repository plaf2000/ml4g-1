import os

KNN = 5
N_BEDS = len(os.listdir("Data/bed"))

FEATURES_BED = ["distance", "score"]
N_FEATURES_BED = len(FEATURES_BED)
N_FEATURES = KNN * N_FEATURES_BED * N_BEDS

RANDOM_SEED = 42

if __name__ == "__main__":
    print("KNN:", KNN)
    print("N_BEDS:", N_BEDS)
    print("FEATURES_BED:", FEATURES_BED)
    print("N_FEATURES_BED:", N_FEATURES_BED)
    print("N_FEATURES:", N_FEATURES)