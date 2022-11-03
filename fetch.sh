#!/bin/bash

set -xeu

if [ $# != 1 ]; then
  echo "USAGE: $0 hw_name"
  echo "e.g. $0 hw3"
  exit 1
fi

hw_name=$1
work_dir=$(cd $(dirname $0); pwd)

if [ -d ${work_dir}/${hw_name} ]; then
  echo "${hw_name} already exists!"
  exit 1
fi

tmp_folder=/tmp/${RANDOM}
mkdir ${tmp_folder}
cd ${tmp_folder}

# Step1: Clone the homework to your local, and move all contents to a subfolder.
git clone git@github.com:dlsyscourse/${hw_name}.git
cd ${hw_name}
mkdir ${hw_name}
# NOTE: You need to handle hidden files manually.
ls | grep -v "^${hw_name}$" | xargs -I {} git mv {} ${hw_name}/
git add ${hw_name}/
git commit -m 'Move to a subfolder'
hw_branch_name=`git rev-parse --abbrev-ref HEAD`

# Step2: Merge.
cd ${work_dir}
git remote add ${hw_name}Tmp ${tmp_folder}/${hw_name}
git fetch ${hw_name}Tmp
git merge ${hw_name}Tmp/${hw_branch_name} --allow-unrelated-histories -m "Fetch ${hw_name}"
git remote rm ${hw_name}Tmp
project_branch_name=`git rev-parse --abbrev-ref HEAD`
git push origin ${project_branch_name}

rm -rf ${tmp_folder}
