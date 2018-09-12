from mics.basic_dataset import BasicDataset
from mics.utils import LabelsToOneHot, LabelsEncoder


class NSynthTFRecordDataset(BasicDataset):
    def __init__(self,
                 dataset_path,
                 transforms,
                 sr,
                 signal_length=2 ** 16,
                 precision=np.float32,
                 one_hot_all=False,
                 one_hot_pitch=False,
                 one_hot_velocity=False,
                 one_hot_instr_src=False,
                 one_hot_instr_family=False,
                 encode_cat=False,
                 in_memory=True):
        super(NSynthTFRecordDataset, self).__init__(transforms, sr, signal_length, precision,
                                                one_hot_all, encode_cat, in_memory)
        self.one_hot_pitch = one_hot_pitch
        self.one_hot_velocity = one_hot_velocity
        self.one_hot_instr_src = one_hot_instr_src
        self.one_hot_instr_family = one_hot_instr_family

        # self.hpy_file = None
        #
        # f = h5py.File(dataset_path, 'r')
        # self.pitch = f[PITCH][:]
        # self.velocity = f[VELOCITY][:]
        # self.instr_src = f[INSTR_SRC][:]
        # self.instr_fml = f[INSTR_FAMILY][:]
        # self.qualities = f[QUALITIES][:]
        #
        # if self.in_memory:
        #     self.audio = f[AUDIO][:]
        #     f.close()
        # else:
        #     self.hpy_file = f
        #     self.sound = f[AUDIO]

        self.n = self.pitch.shape[0]

        if self.encode_cat:
            self.pitch_encoder = LabelsEncoder(self.pitch)
            self.velocity_encoder = LabelsEncoder(self.velocity)
            self.instr_src_encoder = LabelsEncoder(self.instr_src)
            self.instr_fml_encoder = LabelsEncoder(self.instr_fml)

            self.pitch = self.pitch_encoder(self.pitch)
            self.velocity = self.velocity_encoder(self.velocity)
            self.instr_src = self.instr_src_encoder(self.instr_src)
            self.instr_fml = self.instr_fml_encoder(self.instr_fml)

        if self.one_hot_pitch or self.one_hot_all:
            self.pitch_one_hot = LabelsToOneHot(self.pitch)
        else:
            self.pitch_one_hot = None

        if self.one_hot_velocity or self.one_hot_all:
            self.velocity_one_hot = LabelsToOneHot(self.velocity)
        else:
            self.velocity_one_hot = None

        if self.one_hot_instr_src or self.one_hot_all:
            self.instr_src_one_hot = LabelsToOneHot(self.instr_src)
        else:
            self.instr_src_one_hot = None

        if self.one_hot_instr_family or self.one_hot_all:
            self.instr_fml_one_hot = LabelsToOneHot(self.instr_fml)
        else:
            self.instr_fml_one_hot = None