# Percision and Recall functions

## Implement Percision and Recall functions from scratch, Then compare your results with Scikit-Learn

# nlp_data_utils


## Codes expression 


### affix

In affix part, affix  have been checked:



### nlp

In nlp part, worked with mirass texts:


### status farsi

In status farsi part, worked with twitter texts.
To creating 1gram file have to problems, but for 2gram and 3gram files have limiting ram problem, so use some sulotion to solve this problem.

x_dic2text_ngram_v0.py:
Load phrases files and merged together.

x_dic2text_ngram_v0_0.py:
Merged x_2g_clean and phrase_2gram_spectrum_freq_clean.txt and save to merged_2gram_spectrum_freq_clean_x_f20.txt.

x_dix2text_ngram_v1_0.py:
Loaling every 162 text file and merged and when ram used 80% write result and then countinue. Results in 3gram was 17 files.

x_dic2text_ngram_v1_1.py:
Load one text ngram (3gram) file and load next files line by line, if new phrase in first file, this frequency plus to this and else write to new file per each file. Finally write upper frequency 20 in new text file by suffix "_NR" (not repetition). Finally all files (17) have non repetition.

x_dic2text_ngram_v2.py:
For 2gram, meked 2 file 



