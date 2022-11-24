#!/bin/bash -e
export http_proxy=http://130.251.104.8:3128
export https_proxy=http://130.251.104.8:3128
#-----------------------------------------------------------------------------------------
# Script information
script_name='FP IMPACT-BASED IGAD FORECAST'
script_version="1.0.0"
script_date='2021/03/02'

system_library_folder='/home/fp/library/'
fp_folder='/DATA/fp/fp_igad/'
lock_folder='/DATA/fp/fp_igad/lock/'
#-----------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------
# Get file information
virtualenv_folder=$system_library_folder'fp_virtualenv_python3/'
virtualenv_name='fp_virtualenv_python3_hyde'

script_folder=$system_library_folder'bulletin/'
script_file=$script_folder'/hydro/bulletin_hydro_fp.py'
settings_file=$fp_folder'fp_tools_postprocessing/impact_assessment/bulletin_hydro_fp.json'

# Get information (-u to get gmt time)
hour_run=00
time_now=$(date -u +"%Y-%m-%d $hour_run:00")
time_now_folder=$(date -u +"/%Y/%m/%d")
#time_now='2022-11-01 00:00' # DEBUG
#time_now_folder='/2021/05/02' # DEBUG

# Get lock information
file_lock_start_raw='bulletin_lock_impact-based_gfs025_realtime_%YYYY%MM%DD_%HH_START.txt'
file_lock_end_raw='bulletin_lock_impact-based_gfs025_realtime_%YYYY%MM%DD_%HH_END.txt'

file_lock_init=true
folder_lock_raw=$lock_folder'postprocessing/'

# Map merging information
time_now_file="${time_now//[!0-9]/}"
out_folder_step="/home/fp/data/fp_igad/archive/fp_impact_forecast/nwp_gfs-det/${time_now_folder}/0000/"
ancillary_folder_step="/home/fp/data/fp_igad/run/fp_impact_forecast/${time_now_folder}/"
#-----------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------
# Activate virtualenv
export PATH=$virtualenv_folder/bin:$PATH
source activate $virtualenv_name

# Add path to pythonpath
export PYTHONPATH="${PYTHONPATH}:$script_folder"

# Add additional bins to path
export PATH=$cdo_folder:$PATH
#-----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# Get time information
year=${time_now:0:4}
month=${time_now:5:2}
day=${time_now:8:2}
hour=${time_now:11:2}

# Define path data
folder_lock_def=${folder_lock_raw/"%YYYY"/$year}
folder_lock_def=${folder_lock_def/"%MM"/$month}
folder_lock_def=${folder_lock_def/"%DD"/$day}

file_lock_start_def=${file_lock_start_raw/"%YYYY"/$year}
file_lock_start_def=${file_lock_start_def/"%MM"/$month}
file_lock_start_def=${file_lock_start_def/"%DD"/$day}
file_lock_start_def=${file_lock_start_def/"%HH"/$hour}

file_lock_end_def=${file_lock_end_raw/"%YYYY"/$year}
file_lock_end_def=${file_lock_end_def/"%MM"/$month}
file_lock_end_def=${file_lock_end_def/"%DD"/$day}
file_lock_end_def=${file_lock_end_def/"%HH"/$hour}

# Create folder(s)
if [ ! -d "$folder_lock_def" ]; then
	mkdir -p $folder_lock_def
fi
# ----------------------------------------------------------------------------------------	

#-----------------------------------------------------------------------------------------
# Info script start
echo " ==================================================================================="
echo " ==> "$script_name" (Version: "$script_version" Release_Date: "$script_date")"
echo " ==> START ..."
echo " ==> COMMAND LINE: " python3 $script_file -settings_file $settings_file -time $time_now

# Execution pid
execution_pid=$$

#-----------------------------------------------------------------------------------------
# File lock definition
path_file_lock_def_start=$folder_lock_def/$file_lock_start_def 
path_file_lock_def_end=$folder_lock_def/$file_lock_end_def
   
# Init lock conditions
echo " ====> INITILIZE LOCK FILES ... "
if $file_lock_init; then
    # Delete lock files
    if [ -f "$path_file_lock_def_start" ]; then
       rm "$path_file_lock_def_start"
    fi
    if [ -f "$path_file_lock_def_end" ]; then
       rm "$path_file_lock_def_end"
    fi
    echo " ====> INITILIZE LOCK FILES ... DONE!"
else
    echo " ====> INITILIZE LOCK FILES ... SKIPPED!"
fi
#-----------------------------------------------------------------------------------------  

#-----------------------------------------------------------------------------------------
# Run check
if [ -f $path_file_lock_def_start ] && [ -f $path_file_lock_def_end ]; then   

    #-----------------------------------------------------------------------------------------
    # Process completed
    echo " ===> EXECUTION ... SKIPPED! ALL DATA WERE PROCESSED DURING A PREVIOUSLY RUN"
    #-----------------------------------------------------------------------------------------

elif [ -f $path_file_lock_def_start ] && [ ! -f $path_file_lock_def_end ]; then
        
    #-----------------------------------------------------------------------------------------
    # Process running condition
    echo " ===> EXECUTION ... SKIPPED! SCRIPT IS STILL RUNNING ... WAIT FOR PROCESS END"
    #-----------------------------------------------------------------------------------------

elif [ ! -f $path_file_lock_def_start ] && [ ! -f $path_file_lock_def_end ]; then

    #-----------------------------------------------------------------------------------------
    # Lock File START
    time_step=$(date +"%Y-%m-%d %H:%S")
    echo " ================================ " >> $path_file_lock_def_start
    echo " ==== EXECUTION START REPORT ==== " >> $path_file_lock_def_start
    echo " "
    echo " ==== PID:" $execution_pid >> $path_file_lock_def_start
    echo " ==== Algorithm: $script_name" >> $path_file_lock_def_start
    echo " ==== RunTime: $time_step" >> $path_file_lock_def_start
    echo " ==== ExecutionTime: $time_now" >> $path_file_lock_def_start
    echo " ==== Status: RUNNING" >> $path_file_lock_def_start
    echo " "
    echo " ================================ " >> $path_file_lock_def_start

    # Run python script (using setting and time)
    #python3 $script_file -settings_file $settings_file -time "$time_now" || { rm $path_file_lock_def_start ; echo " ===> EXECUTION ... FAILED" ; exit 1; } 
    
    # Merge gridded alert levels 
    gdal_merge.py -o $out_folder_step/${time_now_file}_FPalert1kmNetwork.tif \
    $ancillary_folder_step/alert_fc_IGAD_D3_${time_now_file}.tif \
    $ancillary_folder_step/alert_fc_IGAD_D4_${time_now_file}.tif \
    $ancillary_folder_step/alert_fc_IGAD_D5_${time_now_file}.tif \
    $ancillary_folder_step/alert_fc_IGAD_D6_${time_now_file}.tif \
    $ancillary_folder_step/alert_fc_IGAD_D7_${time_now_file}.tif \
    $ancillary_folder_step/alert_fc_IGAD_D8_${time_now_file}.tif \
    $ancillary_folder_step/alert_fc_IGAD_D9_${time_now_file}.tif \
    $ancillary_folder_step/alert_fc_IGAD_D10_${time_now_file}.tif \
    $ancillary_folder_step/alert_fc_IGAD_D11_${time_now_file}.tif \
    $ancillary_folder_step/alert_fc_IGAD_D12_${time_now_file}.tif \
    $ancillary_folder_step/alert_fc_IGAD_D14_${time_now_file}.tif \
    -ul_lr 28.6241389 -14.9862778 51.4168056 15.5330833 -a_nodata 0 -ot Int16 || true
    
    gdal_merge.py -o $out_folder_step/${time_now_file}_FPalert3kmNetwork.tif \
    $ancillary_folder_step/alert_fc_IGAD_D1_${time_now_file}.tif \
    $ancillary_folder_step/alert_fc_IGAD_D2_${time_now_file}.tif \
    -ul_lr 21.7325278 -4.0103056 39.7960000 23.3750278 -a_nodata 0 -ot Int16 || true
    
    gdalwarp -srcnodata 0 $ancillary_folder_step/alert_fc_IGAD_D15_${time_now_file}.tif $out_folder_step/${time_now_file}_FPalert1p5kmNetwork.tif || true
    
    # Lock File END
    time_step=$(date +"%Y-%m-%d %H:%S")
    echo " ============================== " >> $path_file_lock_def_end
    echo " ==== EXECUTION END REPORT ==== " >> $path_file_lock_def_end
    echo " "
    echo " ==== PID:" $execution_pid >> $path_file_lock_def_start
    echo " ==== Algorithm: $script_name" >> $path_file_lock_def_end
    echo " ==== RunTime: $time_step" >> $path_file_lock_def_end
    echo " ==== ExecutionTime: $time_now" >> $path_file_lock_def_end
    echo " ==== Status: COMPLETED" >>  $path_file_lock_def_end
    echo " "
    echo " ============================== " >> $path_file_lock_def_end
        
    # Info script end
    echo " ===> EXECUTION ... DONE"
    
else
        
    #-----------------------------------------------------------------------------------------
    # Exit unexpected mode
    echo " ===> EXECUTION ... FAILED! SCRIPT ENDED FOR UNKNOWN LOCK FILES CONDITION!"
    #-----------------------------------------------------------------------------------------
        
fi
#-----------------------------------------------------------------------------------------

# Info script end
echo " ==> "$script_name" (Version: "$script_version" Release_Date: "$script_date")"
echo " ==> ... END"
echo " ==> Bye, Bye"
echo " ==================================================================================="
# ----------------------------------------------------------------------------------------

