import os
import calendar
from nltk.corpus import stopwords
from tqdm import tqdm
from string import punctuation

stop_words = set(stopwords.words('english'))
stop_words.update(punctuation)
stop_words.update(['-lrb-', '-rrb-', '``', '`', '´', '´´', '\'\'', '\'', '--'])
stop_words -= set(word for word in stop_words if word.isdigit())

months = [month.lower() for month in list(calendar.month_name) if month != '']


def get_value(field, value):
    field_list = field.strip().split()
    value_list = value.strip().split()
    new_value = []
    for field, value in zip(field_list, value_list):
        if ("date" in field or "start" in field or "end" in field) and len(value) == 2 and value.startswith('0'):
            value = value[1:]
        new_value.append(value)
    return ' '.join(new_value)


def get_tgt(field, value, summary):
    field_list = field.strip().split()
    value_list = value.strip().split()
    summary_list = summary.strip().split()
    tgt = []
    for word in summary_list:
        for month in months:
            if month in word and word.replace(month, '').isdigit():
                word = word.replace(month, ' {} '.format(month))
                break
        for field, value in zip(field_list, value_list):
            if ("date" in field or "start" in field or "end" in field) and \
                    value in word and len(value) == 4 and len(word) >= 6:
                if '-' in word:
                    word = word.replace('-', ' -- ')
                    break
                elif '/' in word:
                    word = value
                    break
                elif ',' in word:
                    word = word.replace(',', ' , ')
                    break
                else:
                    word = word.replace(value, ' {} '.format(value))
                    break
        tgt += word.strip().split()
    return ' '.join(tgt)


def get_skeleton(value, tgt):
    value_set = set(value.strip().split())
    skeleton_list = []
    for word in tgt.strip().split():
        if word not in stop_words and word in value_set:
            skeleton_list.append(word)

    return ' '.join(skeleton_list)


def process(split):
    fields = open("processed_data/{}/{}.field".format(split, split)).readlines()
    lposs = open("processed_data/{}/{}.lpos".format(split, split)).readlines()
    rposs = open("processed_data/{}/{}.rpos".format(split, split)).readlines()
    values = open("processed_data/{}/{}.value".format(split, split)).readlines()
    summaries = open("processed_data/{}/{}.summary".format(split, split)).readlines()

    fw_field = open("WikiBio_fairseq/{}.field".format(split), 'w')
    fw_lpos = open("WikiBio_fairseq/{}.lpos".format(split), 'w')
    fw_rpos = open("WikiBio_fairseq/{}.rpos".format(split), 'w')
    fw_value = open("WikiBio_fairseq/{}.value".format(split), 'w')
    fw_tgt = open("WikiBio_fairseq/{}.tgt".format(split), 'w')
    fw_skeleton = open("WikiBio_fairseq/{}.skeleton".format(split), 'w')
    for field, lpos, rpos, value, summary in tqdm(list(zip(fields, lposs, rposs, values, summaries))):
        value = get_value(field, value)
        tgt = get_tgt(field, value, summary)
        skeleton = get_skeleton(value, tgt)
        fw_field.write(field.strip() + '\n')
        fw_lpos.write(lpos.strip() + '\n')
        fw_rpos.write(rpos.strip() + '\n')
        fw_value.write(value + '\n')
        fw_tgt.write(tgt + '\n')
        fw_skeleton.write(skeleton + '\n')


def mkdirs():
    os.makedirs("WikiBio_fairseq/", exist_ok=True)


if __name__ == '__main__':
    mkdirs()
    process("train")
    process("valid")
    process("test")