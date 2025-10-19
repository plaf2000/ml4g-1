import os

KNN = 5
N_BEDS = len(os.listdir("Data/bed"))

FEATURES_BED = ["signalValue",
                "rel_pos_TSS_start",
                "rel_pos_TSS_end",
                "rel_pos_gene_start",
                "rel_pos_gene_end"]
N_FEATURES_BED = len(FEATURES_BED)
N_FEATURES = KNN * N_FEATURES_BED * N_BEDS

SIGNALS_CNN = [
    "DNase",
    "H3K4me1",
    "H3K4me3",
    "H3K27ac",
    "H3K36me3",
]

N_SIGNALS_CNN = len(SIGNALS_CNN)

SIGNAL_CNN_WINDOW = 1e4
CNN_BIN_SIZE = 100
CNN_N_BINS = int(SIGNAL_CNN_WINDOW / CNN_BIN_SIZE)

RANDOM_SEED = 42

if __name__ == "__main__":
    print("KNN:", KNN)
    print("N_BEDS:", N_BEDS)
    print("FEATURES_BED:", FEATURES_BED)
    print("N_FEATURES_BED:", N_FEATURES_BED)
    print("N_FEATURES:", N_FEATURES)
