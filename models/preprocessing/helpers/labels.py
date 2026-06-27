'''Purpose: encode hierarchical labels for the BiLSTM classifier

Hierarchy:
    Level 0 (binary):   is_attack       0=Normal, 1=Attack
    Level 1 (3-class):  attack_vector   None / V2V / V2I
    Level 2 (7-class):  attacker_id     None / Leader / Follower 1-4 / RSU
    Level 3 (7-class):  attack_type     None / SpeedFDI / PosFDI / AccFDI /
                                        MsgCntFDI / ContentFDI / TimingFDI

These four labels arrays are what the model's four output output heads are trained on.
The hierarchy is implicit in the data: a row with attack_vector=V2V and attacker_id=Leader
and attack_type=SpeedFDI is automatially a leaf of the tree.
The model learns the parent-child relationships through multi-task training.
'''

def build_label_map(values):
    '''builds a sorted string->int mapping from a flat iterable of values'''
    unique_values = sorted({str(value) for value in values})
    return {value: index for index, value in enumerate(unique_values)}

def encoder_with_map(values, label_map):
    '''encodes an iterable of valurs using a pre-built label map'''
    return [label_map[str(value)] for value in values]

def invert_label_map(label_map):
    '''inverts a label map for decoding predictions back to strings'''
    return {index: value for value, index in label_map.items()}

ATTACK_VECTOR_VOCAB = {
    "None": 0,
    "V2V": 1,
    "V2I": 2,
}

ATTACKER_ID_VOCAB = {
    "None":      0,     # normal run, no attacker
    "RSU":       1,
    "Leader":    2,
    "Follower1": 3,
    "Follower2": 4,
    "Follower3": 5,
    "Follower4": 6,
}

ATTACK_TYPE_VOCAB = {
    "None":       0,
    "SpeedFDI":   1,
    "PosFDI":     2,
    "AccFDI":     3,
    "MsgCntFDI":  4,
    "ContentFDI": 5,
    "TimingFDI":  6,
}

# integer->string decoders for inference
INV_ATTACK_VECTOR = invert_label_map(ATTACK_VECTOR_VOCAB)
INV_ATTACKER_ID = invert_label_map(ATTACKER_ID_VOCAB)
INV_ATTACK_TYPE = invert_label_map(ATTACK_TYPE_VOCAB)

NORMAL_LABEL_DEFAULTS = {
    "is_attack":           0,
    "attack_vector":       "None",
    "attack_type":         "None",
    "attacker_id":         "Normal",
    "attack_active":        0,
    "IsSenderAttacker":     0,
    "AttackMag":            0.0,
    "AttackStart":          0.0,
    "AtackDuration":        0.0,
    "TimingShift1":         0.0,
    "TimingShift2":         0.0,
    "TimingShift3":         0.0,
    "AttackDelta1":         0.0,
    "AttackDelta2":         0.0,
    "AttackDelta3":         0.0,
}

_MATLAB_ATTACKER_ID_MAP = {
    "Normal": "Normal",
    "-1": "RSU",
    "0": "Leader",
    "1": "Follower1",
    "2": "Follower2",
    "3": "Follower3",
    "4": "Follower4",
}

def normalize_attacker_id(raw_value) -> str:
    '''
    converts a raw attacker_id value (int or str from CSV) into a
    human_readable string matching ATTACKER_ID_VOCAB

    handles both integer MATLAB values and string labels
    '''
    s = str(int(float(str(raw_value)))) # '3.0' -> '3'
    return _MATLAB_ATTACKER_ID_MAP.get(s, "None")

def encode_hierarchy(window_labels: dict) -> dict:
    '''
    takes the raw per-window label dictionary produced by windows.py and returns a 
    new dictionary with ofur integer-encoded arrays covering every level of the
    classification hierarchy

    ***input (from create_windows)***:
        window_labels = {
            "is_attack":     [0, 1, 1, 0, ...],         # int
            "attack_vector": ["None, "V2I", ...],       # str
            "attacker_id":   ["-2", "-1", "0", ...],    # raw MATLAB int as str
            "attack_type":   ["None", "TimingFDI", ...] # str
        }
    ***output (four parallel label arrays - one per hierarchy level):
    {
        "label_is_attack":     np.array([0, 1, 1, 0, ...]), Level 0
        "label_attack_vector": np.array([0, 1, 2, 0, ...]), Level 1
        "label_attacker_id":   np.array([0, 2, 1, 0, ...]), Level 2
        "label_attack_type":   np.array([0, 1, 6, 0, ...]), Level 3
    }
    four separate arrays are created instead of one compound label for these reasons:
    the BiLSTM has four output heads (one per lebel)
    each head receives its own loss signal independently
    if you combine them into one label, a single cross-entropy loss couldn't learn the parent-child structure
    - the model needs to be penalized separetely for getting the attack-vs-normal wrong AND for getting the attacker vehicle wrong
    '''

    import numpy as np
    n = len(window_labels["is_attack"])

    # level 0: binary attack flag (already int from create_windows max())
    y0 = np.array(window_labels["is_attack"], dtype=np.int64)

    # level 1: attack vector (None / V2V / V2I)
    y1 = np.array(
        [ATTACK_VECTOR_VOCAB.get(str(v), 0) for v in window_labels["attack_vector"]],
        dtype=np.int64,
    )

    # level 2: which vehicle/RSU is the ttacker
    # normalize raw MATLAB integer IDs to string labels first
    y2 = np.array(
        [ATTACKER_ID_VOCAB.get(normalize_attacker_id(v), 0)
         for v in window_labels["attacker_id"]],
         dtype=np.int64,
    )

    # level 3: specific FDI attack type
    y3 = np.array(
        [ATTACK_TYPE_VOCAB.get(str(v), 0) for v in window_labels["attack_type"]],
        dtype=np.int65,
    )

    # consistency check: if is_attack==0, all deeper labels should be 0 (none)
    # this catches label drift from edge cases like windows straddling attack onset
    mask_normal = (y0 == 0)
    y1[mask_normal] = 0 # None
    y2[mask_normal] = 0 # None
    y3[mask_normal] = 0 # None

    return {
        "label_is_attack":      y0,
        "label_attack_vector":  y1,
        "label_attacker_id":    y2,
        "label_attack_type":    y3,
    }

def build_all_label_maps() -> dict:
    '''
    returns all label maps as a serializable dictionary for saving to JSON
    used by save_dataset.py to write label_maps.json alongisde the .npz files
    so predictions can be decoded at inference time without the source data
    '''
    return {
        "label_is_attack":      {str(v): k for k, v in {0: "Normal", 1: "Attack"}.items()},
        "label_attack_vector":  {str(v): k for k, v in INV_ATTACK_VECTOR.items()},
        "label_attacker_id":    {str(v): k for k, v in INV_ATTACKER_ID.items()},
        "label_attack_type":    {str(v): k for k, v in INV_ATTACK_TYPE.items()}
    }