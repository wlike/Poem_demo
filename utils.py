# -*- coding: utf-8 -*-

def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False

def is_all_chinese(str):
    for char in str:
        if not is_chinese(char):
            return False
    return True
