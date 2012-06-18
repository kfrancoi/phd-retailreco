#!/bin/sh
#
# SGE: the job name
#$ -N RetailReco$1$2
#
# COMMON PARAMETERS
#
# SGE: mandatory!! the requested run-time, expressed as
#$ -l h_rt=86400
#
# GREEN PARAMETERS
#
# -l mf=8G
# -l hm=false
# -pe snode 2
# -l nb=true
# -q all.q
#
# LEMAITRE PARAMETERS
#
#$ -l p=2
#$ -pe snode
#$ -q all.q
# SGE: your Email here, for job notification
# -M kevin.francoisse@uclouvain.be
# SGE: when do you want to be notified (b for begin, e for end, s for error)?
#$ -m bes
#
# SGE: ouput in the current working dir
#$ -cwd
#
# Specify that all the environement variable must be included in the jobs environement
#$ -V


echo -n "job starts at "
date

model=$1
param=$2
SCRIPT="Experiences.py "
CMD="python ".$SCRIPT." ".$model." ".$param
echo $CMD
$CMD

echo -n "job ends at "
date
# end of job
