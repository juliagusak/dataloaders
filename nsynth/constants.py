NSYNTH_TRAIN = "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.tfrecord"
NSYNTH_TEST = "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.tfrecord"
NSYNTH_VAL = "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.tfrecord"

TRAIN_FILE = NSYNTH_TRAIN.split("/")[-1]
TEST_FILE = NSYNTH_TEST.split("/")[-1]
VAL_FILE = NSYNTH_VAL.split("/")[-1]

TRAIN_EXAMPLES = 289205
VAL_EXAMPLES = 12678
TEST_EXAMPLES = 4096

AUDIO_LEN = 64000
QUALITIES_LEN = 10

NOTE_STR = "note_str"
AUDIO = "audio"
PITCH = "pitch"
VELOCITY = "velocity"
INSTR_SRC = "instrument_source"
INSTR_FAMILY = "instrument_family"
QUALITIES = "qualities"
