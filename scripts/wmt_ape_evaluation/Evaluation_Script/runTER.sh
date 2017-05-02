#!/bin/bash
###########################################################################################################################
#
# Created on Jan 14, 2015
#
# Author: Marco Turchi (Fondazione Bruno Kessler, Trento)
# email: turchi@fbk.eu
#
#  This script computes the HTER of an input hypothesis file
#
############################################################################################################################

# Input Parameters:
# -h <hypothesis file> 
# -r <reference file>
# -s <string identifier>
# -o <output folder>

# To Be Changed

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
scriptPath=$DIR/'tercom-0.7.25/'
#scriptPath='/Users/marcoturchi/Dropbox/APE/Evaluation/Scripts/tercom-0.7.25/'

# Loading the parameters
usage() { echo "Usage: $0 [-h <hypothesis file>] [-r <reference file>] [-s <string identifier>] [-o <output folder>]" 1>&2; exit 1; }

while getopts ":h:r:s:o:" o; do
    case "${o}" in
        h)
            hyp=${OPTARG}
            ;;
        r)
            ref=${OPTARG}
            ;;
        s)
            id=${OPTARG}
            ;;
        o)
            outputFolder=${OPTARG}
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

if [ -z "${hyp}" ] || [ -z "${ref}" ] || [ -z "${id}" ] || [ -z "${outputFolder}" ]; then
    usage
fi

echo "h = ${hyp}"
echo "r = ${ref}"
echo "string = ${id}"
echo "oF = ${outputFolder}"


# Create the output folder, if it does not exist
mkdir -p ${outputFolder}

# Adding sentence id to the text files
python ${scriptPath}/AddSentenceId.py ${hyp} ${hyp}_${id}_ter ${id}  

python ${scriptPath}/AddSentenceId.py ${ref} ${ref}_${id}_ter ${id}

# Run TER
#Case sensitive evaluation
java -jar ${scriptPath}/tercom.7.25.jar -s -r ${ref}_${id}_ter -h ${hyp}_${id}_ter -n ${outputFolder}/${id}_TER_output_caseSens -o sum

#Case insensitive evaluation
#java -jar ${scriptPath}/tercom.7.25.jar -r ${ref}_${id}_ter -h ${hyp}_${id}_ter -n ${outputFolder}/${id}_TER_output_caseInsens -o sum


