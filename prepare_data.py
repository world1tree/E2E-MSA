import os.path
import random
from itertools import combinations

from Bio import AlignIO

class PfamDataset(object):

    def __init__(self, pfam_path, sa_num=2):
        assert os.path.exists(pfam_path), "%s not found." % pfam_path
        self.pfam_path = pfam_path
        self.instances = list()
        self.working_dir = os.getcwd()
        self.output_dir = os.path.join(self.working_dir, "data")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.sa_num = sa_num
        output_file_template = os.path.join(self.output_dir, "##_%d.txt")
        self.fs_train = [open(output_file_template.replace("##", "train") % i, "w") for i in range(sa_num)]
        self.fs_valid = [open(output_file_template.replace("##", "valid") % i, "w") for i in range(sa_num)]
        self.fs_test = [open(output_file_template.replace("##", "test") % i, "w") for i in range(sa_num)]

    def get_instances(self):
        if len(self.instances) != 0:
            return self.instances
        with open(self.pfam_path, 'r') as pfam_file:
            # 使用AlignIO.parse()读取PFAM文件
            pfam_alignments = AlignIO.parse(pfam_file, 'stockholm')
            # 遍历每个对齐记录
            msa_instances = iter(pfam_alignments)
            while True:
                try:
                    alignment = next(msa_instances)
                    self.instances.append(alignment)
                except StopIteration:
                    break
                except Exception:
                    break
        return self.instances

    def _get_seq(self, align_seq):
        seq = align_seq.replace("-", "")
        return seq

    def _get_label(self, align_seq):
        gap_num = align_seq.count('-')
        seq_label = list()
        num = 0
        for ch in align_seq:
            if ch == '-':
                num += 1
            else:
                seq_label.append(num)
                num = 0
        seq_label.append(num) # use </s> as virtual token.
        label_str = ','.join(map(str, seq_label))
        label_num = len(seq_label)
        return label_str, label_num

    def create_dataset(self, train_num=1000, valid_num=100, test_num=100):
        ds_size = train_num + valid_num + test_num
        caches = [[] for _ in range(self.sa_num)]
        instances = self.get_instances()
        for alignment in instances:
            seq_num = len(alignment)
            indexs_list = combinations(range(seq_num), self.sa_num)
            for indexs in indexs_list:
                for cache_index, alignment_index in enumerate(indexs):
                    id = alignment[alignment_index].id.replace(" ", "")
                    seq = self._get_seq(alignment[alignment_index].seq)
                    label, label_num = self._get_label(alignment[alignment_index].seq)
                    assert len(seq) == label_num-1
                    caches[cache_index].append("%s\t%s\t%s\n" % (id, seq, label))
                if len(caches[0]) == ds_size:
                    break
            if len(caches[0]) == ds_size:
                break

        msa_list = list()
        for i in range(ds_size):
            _msa = list()
            for j in range(self.sa_num):
                _msa.append(caches[j][i])
            msa_list.append(tuple(_msa))
        random.shuffle(msa_list)

        # write train file
        for i in range(train_num):
            for j in range(self.sa_num):
                self.fs_train[j].write(msa_list[i][j])
        # write valid file
        for i in range(train_num, train_num+valid_num):
            for j in range(self.sa_num):
                self.fs_valid[j].write(msa_list[i][j])
        # write test file
        for i in range(train_num+valid_num, train_num+valid_num+test_num):
            for j in range(self.sa_num):
                self.fs_test[j].write(msa_list[i][j])

        for fs in [self.fs_train, self.fs_valid, self.fs_test]:
            for f in fs:
                f.close()


if __name__ == '__main__':
    PfamDataset("./Pfam-A.seed").create_dataset()