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
#$ -l mf=8G
#$ -l hm=false
#$ -pe snode 2
#$ -l nb=true
#$ -q all.q
#
# SGE: your Email here, for job notification
# -M kevin.francoisse@uclouvain.be
# SGE: when do you want to be notified (b for begin, e for end, s for error)?
#$ -m bes
#
# SGE: ouput in the current working dir
#$ -wd /scratch
#
# Specify that all the environement variable must be included in the jobs environement
#$ -V

. /etc/profile.d/modules.sh
module purge
module add shared intel/cce intel/fce openmpi/intel gotoblas/penryn

echo "Got $NSLOTS slots."
echo "Temp dir is $TMPDIR"
echo "Node file is:"
cat $TMPDIR/machines

WDIR=/scratch/$USER/$JOB_NAME-$JOB_ID
MYSOURCE=$HOME/Research/RetailReco/src
MYLOCALSOURCE=$WDIR/mySource

# Create Working Directory on each allocated node
# and copy data & prog from homedir to /scratch of each node

for n in $(cat $TMPDIR/machines|uniq); do
    rsh $n mkdir -p $WDIR
    rcp storage01:$MYSOURCE $n:$WDIR
done

cd $WDIR
echo -n "job starts at "
date

model=$1
param=$2
SCRIPT="Experiences.py "
CMD="python" $MYLOCALSOURCE/$SCRIPT $model $param

$CMD

echo -n "job ends at "
date
# end of job

# copy data back to homedir
for n in $(cat $TMPDIR/machines|uniq); do
    rsh storage01 mkdir -p "$HOME/$JOB_NAME-$JOB_ID/$n"
    rsh $n "/bin/rm -f $MYLOCALEXEC"
    f=`echo "$n:$WDIR storage01:$HOME/$JOB_NAME-$JOB_ID/$n"`
    rcp -rp $f
    rsh $n "/bin/rm -rf $WDIR"
done