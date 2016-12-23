# encoding: utf-8
# 其他工具组件
import unicodecsv as csv

# 读取汉字字典数据
def loadvoc(fname='data/id-char.csv',delimiter=',',mode='rb'):
    '''returns id2char, char2id'''
    csvfile = open(fname, mode = mode)
    data = csv.reader(csvfile, delimiter=delimiter)
    id2char = {}
    char2id = {}
    for row in data:
        id2char[int(row[0])] = row[1]
        char2id[row[1]] = int(row[0])
    del data
    csvfile.close()
    del csvfile
    return id2char, char2id

