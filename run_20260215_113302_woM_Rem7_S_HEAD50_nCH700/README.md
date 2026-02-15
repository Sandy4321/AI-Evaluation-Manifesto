Removing case
DATA_FOLDER = "data"
RUNS_ROOT = "runs"



MAX_BASE_CHUNKS_TO_PROBE = 700 #2000 #900 #10 #20 #900 #300#1234

CHUNK_SIZE = 150#200
OVERLAP = 10#75#100

HNSW_M = 4
HNSW_EF_CONSTRUCTION = 100
HNSW_EF_SEARCH = 100

K = 5
INDEXING_WAIT_SEC = 20


RANDOM_SEED = 42

PERTURB_MODE = "remove"   # "remove" or "insert"

# REMOVE MODE
N_RANDOM_SETS = 10 # how many random removing happen if PERTURB_MODE = "remove"
N_WORDS_REMOVE = 7 #23
MAX_ABLATIONS_PER_CHUNK = 200
MIN_WORD_LEN = 5

# INSERT MODE
INSERT_ENABLED =  True #True  False
INSERT_TEXT = " not at all "
INSERT_AFTER_WORD_NUM = 10 #17

# Quantization: "none" | "scalar" | "binary"
QUANTIZATION_MODE = "scalar"
ENABLE_RERANK_WITH_ORIGINAL = False
OVERSAMPLING = 2.0

# Matryoshka (MRL) local scoring (prefix dims + renormalize)
USE_MATRYOSHKA = False
MATRYOSHKA_DIMS = 10

PREVIEW_CHARS = 260

BATCH_EMBED = 16
BATCH_UPLOAD = 500

# =============================
# STORAGE SWITCH 
# =============================
# "all"  -> store full chunk text in index field contentStored
# "head" -> store only first N chars in contentStored (saves Total storage)
STORE_TEXT_MODE = "head"  # "all" or "head"
STORE_TEXT_HEAD_CHARS = 50 #10 #240  # used only when STORE_TEXT_MODE="head"
