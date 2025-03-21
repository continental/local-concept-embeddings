#!/usr/bin/env bash

# Pascal VOC2012: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

set -e

if [ ! -f data/voc2012/readme.txt ]; then

   echo "Downloading Pascal VOC2012"
   mkdir -p data/voc2012
   pushd data/voc2012

   echo "Downloading VOC2012"
   wget --progress=bar http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar -O voc2012.tar
   tar -xvf voc2012.tar
   rm voc2012.tar

   echo "Pascal VOC2012 http://host.robots.ox.ac.uk/pascal/VOC/voc2012/" >> readme.txt
   popd

fi
