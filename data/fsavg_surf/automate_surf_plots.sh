hemis="lh rh"
root=/Users/gt/Documents/GitHub/auditory_brain_dnn/data/fsavg_surf
saveroot=/Users/gt/Documents/GitHub/auditory_brain_dnn/data/fsavg_surf/SURF_PLOTS_PNG_20230504
labelroot=/Users/gt/Documents/GitHub/auditory_brain_dnn/data/fsavg_surf/labels_glasser2016
models="Kell2018word Kell2018speaker Kell2018audioset Kell2018music Kell2018multitask ResNet50audioset ResNet50music ResNet50word ResNet50speaker	ResNet50multitask AST wav2vec DCASE2020 DS2 VGGish ZeroSpeech2020 S2T metricGAN sepformer"
target="NH2015"
randnetw="false"

for model in $models ; do

  if [ "$randnetw" = "true" ]; then
    model_folder=${model}_randnetw
  else
    model_folder=$model
  fi

  for hemi in $hemis ; do

    echo $model

    if [ "$hemi" = "rh" ]; then
      azimuth=180
    else
      azimuth=0
    fi
    if [[ "$model" = "Kell2018word" || "$model" = "Kell2018speaker" || "$model" = "Kell2018audioset" || "$model" = "Kell2018music" || "$model" = "Kell2018multitask" || "$model" = "ResNet50audioset" || "$model" = "ResNet50music" || "$model" = "ResNet50multitask" ]]; then
      max=6
    fi
    if [[ "$model" = "ResNet50word" || "$model" = "ResNet50speaker" || "$model" = "wav2vec" ]]; then
      max=5
    fi
    if [ "$model" = "AST" ]; then
      max=7
    fi
    if [ "$model" = "DCASE2020" ]; then
      max=2
    fi
    if [ "$model" = "DS2" ]; then
      max=8
    fi
    if [ "$model" = "VGGish" ]; then
      max=11
    fi
    if [ "$model" = "ZeroSpeech2020" ]; then
      max=4
    fi
    if [ "$model" = "S2T" ]; then
      max=10
    fi
    if [ "$model" = "sepformer" ]; then
      max=22
    fi
    if [ "$model" = "metricGAN" ]; then
      max=4
    fi


freeview  -f $SUBJECTS_DIR/fsaverage/surf/$hemi.inflated:overlay=$root/${model_folder}/TYPE=subj-median-argmax_METRIC=median_r2_test_c_PLOTVAL=pos_$target/written_surfaces/${model}_subj-avg_$hemi.nearest.mgh:overlay_color=colorwheel,inverse,truncate:overlay_threshold=1,$max:label=$labelroot/$hemi.primary.label:label_outline=1:label_color=0,0,0 -cam azimuth $azimuth --zoom 1.4 --viewport 3d --colorscale -ss "${saveroot}/${model_folder}_${hemi}_${target}_w_label_min-1_max-$max.png"  --quit

  done
done
