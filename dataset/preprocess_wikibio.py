import re, time, os
import shutil


def split_infobox():
    """
    extract box content, field type and position information from infoboxes from original_data
    *.box.val is the box content (token)
    *.box.lab is the field type for each token
    *.box.pos is the position counted from the begining of a field
    """
    bwfile = ["processed_data/train/train.value",
              "processed_data/valid/valid.value",
              "processed_data/test/test.value"]
    bffile = ["processed_data/train/train.field",
              "processed_data/valid/valid.field",
              "processed_data/test/test.field"]
    bpfile = ["processed_data/train/train.lpos",
              "processed_data/valid/valid.lpos",
              "processed_data/test/test.lpos"]
    boxes = ["original_data/train.box", "original_data/valid.box", "original_data/test.box"]

    mixb_word, mixb_label, mixb_pos = [], [], []
    for fboxes in boxes:
        box = open(fboxes, "r").read().strip().split('\n')
        box_word, box_label, box_pos = [], [], []
        for ib in box:
            item = ib.split('\t')
            box_single_word, box_single_label, box_single_pos = [], [], []
            for it in item:
                if len(it.split(':')) > 2:
                    continue
                # print it
                prefix, word = it.split(':')
                word = word.strip().split()[0]
                if '<none>' in word or word.strip() == '' or prefix.strip() == '':
                    continue
                new_label = re.sub("_[1-9]\d*$", "", prefix)
                if new_label.strip() == "":
                    continue
                box_single_word.append(word)
                box_single_label.append(new_label)
                if re.search("_[1-9]\d*$", prefix):
                    field_id = int(prefix.split('_')[-1])
                    box_single_pos.append(field_id if field_id <= 30 else 30)
                else:
                    box_single_pos.append(1)
            box_word.append(box_single_word)
            box_label.append(box_single_label)
            box_pos.append(box_single_pos)
        mixb_word.append(box_word)
        mixb_label.append(box_label)
        mixb_pos.append(box_pos)
    for k, m in enumerate(mixb_word):
        with open(bwfile[k], "w+") as h:
            for items in m:
                for sens in items:
                    h.write(str(sens) + " ")
                h.write('\n')
    for k, m in enumerate(mixb_label):
        with open(bffile[k], "w+") as h:
            for items in m:
                for sens in items:
                    h.write(str(sens) + " ")
                h.write('\n')
    for k, m in enumerate(mixb_pos):
        with open(bpfile[k], "w+") as h:
            for items in m:
                for sens in items:
                    h.write(str(sens) + " ")
                h.write('\n')


def reverse_pos():
    # get the position counted from the end of a field
    bpfile = ["processed_data/train/train.lpos", "processed_data/valid/valid.lpos",
              "processed_data/test/test.lpos"]
    bwfile = ["processed_data/train/train.rpos", "processed_data/valid/valid.rpos",
              "processed_data/test/test.rpos"]
    for k, pos in enumerate(bpfile):
        box = open(pos, "r").read().strip().split('\n')
        reverse_pos = []
        for bb in box:
            pos = bb.split()
            tmp_pos = []
            single_pos = []
            for p in pos:
                if int(p) == 1 and len(tmp_pos) != 0:
                    single_pos.extend(tmp_pos[::-1])
                    tmp_pos = []
                tmp_pos.append(p)
            single_pos.extend(tmp_pos[::-1])
            reverse_pos.append(single_pos)
        with open(bwfile[k], 'w+') as bw:
            for item in reverse_pos:
                bw.write(" ".join(item) + '\n')


def check_generated_box():
    ftrain = ["processed_data/train/train.value",
              "processed_data/train/train.field",
              "processed_data/train/train.lpos",
              "processed_data/train/train.rpos"]
    ftest = ["processed_data/test/test.value",
             "processed_data/test/test.field",
             "processed_data/test/test.lpos",
             "processed_data/test/test.rpos"]
    fvalid = ["processed_data/valid/valid.value",
              "processed_data/valid/valid.field",
              "processed_data/valid/valid.lpos",
              "processed_data/valid/valid.rpos"]
    for case in [ftrain, ftest, fvalid]:
        vals = open(case[0], 'r').read().strip().split('\n')
        labs = open(case[1], 'r').read().strip().split('\n')
        poses = open(case[2], 'r').read().strip().split('\n')
        rposes = open(case[3], 'r').read().strip().split('\n')
        assert len(vals) == len(labs)
        assert len(poses) == len(labs)
        assert len(rposes) == len(poses)
        for val, lab, pos, rpos in zip(vals, labs, poses, rposes):
            vval = val.strip().split(' ')
            llab = lab.strip().split(' ')
            ppos = pos.strip().split(' ')
            rrpos = rpos.strip().split(' ')
            if len(vval) != len(llab) or len(llab) != len(ppos) or len(ppos) != len(rrpos):
                print(case)
                print(val)
                print(len(vval))
                print(len(llab))
                print(len(ppos))
                print(len(rrpos))
            assert len(vval) == len(llab)
            assert len(llab) == len(ppos)
            assert len(ppos) == len(rrpos)


def preprocess():
    """
    We use a triple <f, p+, p-> to represent the field information of a token in the specific field.
    p+&p- are the position of the token in that field counted from the begining and the end of the field.
    For example, for a field (birthname, Jurgis Mikelatitis) in an infoboxes, we represent the field as
    (Jurgis, <birthname, 1, 2>) & (Mikelatitis, <birthname, 2, 1>)
    """
    print("extracting token, field type and position info from original data ...")
    time_start = time.time()
    split_infobox()
    reverse_pos()
    duration = time.time() - time_start
    print("extract finished in %.3f seconds" % float(duration))


def copy_summary():
    shutil.copyfile("original_data/train.summary", "processed_data/train/train.summary")
    shutil.copyfile("original_data/valid.summary", "processed_data/valid/valid.summary")
    shutil.copyfile("original_data/test.summary", "processed_data/test/test.summary")


def make_dirs():
    os.mkdir("processed_data/")
    os.mkdir("processed_data/train/")
    os.mkdir("processed_data/test/")
    os.mkdir("processed_data/valid/")


if __name__ == '__main__':
    make_dirs()
    preprocess()
    check_generated_box()
    copy_summary()
    print("check done")