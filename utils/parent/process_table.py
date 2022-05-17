from collections import defaultdict

f_field = open("../../dataset/WikiBio_fairseq/test.field").readlines()
f_value = open("../../dataset/WikiBio_fairseq/test.value").readlines()
fw = open("wb_test_tables.txt", 'w')
for field, value in zip(f_field, f_value):
    field_list = field.strip().split()
    value_list = value.strip().split()
    dic = defaultdict(list)
    for field_word, value_word in zip(field_list, value_list):
        assert '|||' not in value_word and '\t' not in value_word
        dic[field_word].append(value_word)

    res = '\t'.join(["{}|!+{}".format(k, ' '.join(v)) for k, v in dic.items()])
    fw.write(res + '\n')
