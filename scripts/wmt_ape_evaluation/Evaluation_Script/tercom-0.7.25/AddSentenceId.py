#!/usr/bin/python
import sys
'''
Created on Jan 14, 2015

@author: Turchi
'''



argv=None
if argv is None:
	argv = sys.argv
	if len(argv) < 4:

		print "Wrong Number of Parameters "	
		print "Usage: "
		print "python ....py inputFile outputFile IdString" 
		sys.exit()

InFile = argv[1]
OutFile = argv[2]
IdString = argv[3]

try:
        text = open(InFile, 'r')
except IOError:
        print 'Cannot open file '+InFile+' for reading'
        sys.exit(0)

try:
        otext = open(OutFile, 'w')
except IOError:
        print 'Cannot open file '+OutFile+' for writing'
        sys.exit(0)


c = 1
for sentence in text:
	outSentence = sentence.split("\n")[0] +' (id_'+IdString+'_'+str(c)+')'
	otext.write(outSentence + '\n')
	c = c + 1
text.close()
otext.close()





