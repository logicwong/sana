import os
import sys


def process(input_dir, output_dir, split):
    f_gen_skeleton = open(os.path.join(input_dir, 'generate-{}.txt'.format(split)))

    gen_skeleton_dic = {int(row.strip().split('\t')[0][2:]): row.strip().split('\t')[2]
                        for row in f_gen_skeleton if row.startswith('H-')}
    sorted_gen_skeleton = [gen_skeleton_dic[key] for key in sorted(gen_skeleton_dic)]

    gen_skeletons = []
    for row in sorted_gen_skeleton:
        word_list = row.strip().split()
        gen_skeletons.append(' '.join(word_list))

    with open(os.path.join(output_dir, '{}.gen_tgt'.format(split)), 'w') as fw:
        for gen_skeleton in gen_skeletons:
            fw.write(gen_skeleton + '\n')


if __name__ == '__main__':
    process(sys.argv[1], sys.argv[2], sys.argv[3])