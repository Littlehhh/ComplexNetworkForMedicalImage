#!/bin/bash
# author:Hui Wang

current_dir=`pwd`
origin_path="Original/Siemens15"
skull_mask_path="SkullStrippingMask/Siemens15"
remove_skull_path="RemoveSkull"
reg_result_path="Registration_Siemens15"
hipp_path="Hippocampus/Siemens15"
hipp_reg_path="Registration_Siemens15_Hippocampus"
ref_T1='MNI152_T1_1mm_brain'

if [ ! -d $remove_skull_path ]; then
	mkdir $remove_skull_path
fi

if [ ! -d $reg_result_path ]; then
	mkdir $reg_result_path
fi
if [ ! -d $hipp_reg_path ]; then
	mkdir $hipp_reg_path
fi

# process number
cnt=`grep "processor" /proc/cpuinfo |  wc -l`
echo "process number: $cnt"
# receive Ctrl+c
trap "exec 10>&-;exec 10<&-;exit 0" 2
tempfifo=$$.fifo
mkfifo $tempfifo
exec 10<>$tempfifo
rm -rf $tempfifo
for ((i=1; i<=$cnt; i++))
do
    echo >&10
done

echo $current_dir
start=`date +%s`

# remove skull and registration
for file in `ls $current_dir/$origin_path | grep ".nii.gz$"`
do
    read -u10
    {   

        file_prefix=${file%.nii.gz}
        echo "Start ${file_prefix} Process"
        echo "Start ${file} Skull Mask"
        skull_mask_file="$current_dir/$skull_mask_path/${file_prefix}_staple.nii.gz"
        if [ ! -f "$skull_mask_file" ]; then
            echo "$skull_mask_file not exist"
            exit 1
        fi 
        fslmaths "$current_dir/$origin_path/$file" \
        -mul "$skull_mask_file" \
        "$current_dir/$remove_skull_path/${file_prefix}_remove_skull.nii.gz"

        echo "Start ${file} Registration"
        flirt -dof 12 -in "$current_dir/$remove_skull_path/${file_prefix}_remove_skull.nii.gz" \
        -ref "${ref_T1}" \
        -omat "${current_dir}/${reg_result_path}/${file_prefix}.mat" \
        -out "${current_dir}/${reg_result_path}/${file_prefix}_remove_skull_to_${ref_T1}"
        echo "${file} Registration Done"
        
        echo "Start Apply Matrix on ${file}_Hipp"
        affine_matrix="${current_dir}/${reg_result_path}/${file_prefix}.mat"
        flirt -in "$current_dir/$hipp_path/${file_prefix}_hippocampus_staple.nii.gz" \
        -ref "${ref_T1}" \
        -out "$current_dir/$hipp_reg_path/${file_prefix}_hippocampus_reg.nii.gz" \
        -init $affine_matrix \
        -applyxfm

        echo -e "${file_prefix} Process Done\n"

        echo >&10
    } &

done

wait
echo "All Process Done"
# print the time
echo "$[`date +%s`-$start] seconds" 
exec 10>&-
exec 10<&-
exit 0
