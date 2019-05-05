#!/bin/bash

if [ $# -ne 3 ]
	then
	    	echo "usage : basename $0 sourceDirectory outDir percentage"
		exit 0
fi

srcDir=$1
outDir=$2
percentage=$3

cd $srcDir

fileCount=`ls -1q *.png | wc -l`
filesToMove=$(( fileCount * percentage/100 ))

echo "move $filesToMove files from $srcDir to $outDir ? (y/n)"

read choice

if [ "$choice" != "y" ]; then
	echo "canceled"
	exit 0
fi

ls | shuf -n $filesToMove | xargs -i mv --backup {} "../../$outDir"
