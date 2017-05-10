import sys

# The format is:
# <METHOD NAME> <SEGMENT NUMBER> <WORD INDEX> <WORD> <BINARY SCORE>.

method = sys.argv[1] # E.g. "linear".
target_file = sys.argv[2] # E.g. "test.mt"
prediction_file = sys.argv[3] # E.g. "prediction.txt"

f_target = open(target_file)
f_pred = open(prediction_file)
k = 0
for line_target, line_pred in zip(f_target, f_pred):
    line_target = line_target.rstrip('\n')
    line_pred = line_pred.rstrip('\n')
    words = line_target.split(' ')
    tags = line_pred.split(' ')
    assert len(words) == len(tags)
    for i, (word, tag) in enumerate(zip(words, tags)):
        print '\t'.join([method, str(k), str(i), word, tag])
    k += 1
f_target.close()
f_pred.close()
