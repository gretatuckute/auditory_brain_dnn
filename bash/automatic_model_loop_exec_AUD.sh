#!/bin/sh
model="$1"
target="$2"
echo $model
echo $target
randnetw="False"
overwrite=0
resultdir='/mindhive/mcdermott/u/gretatu/auditory_brain_dnn/results/'
if [ "$model" = "Kell2018" ]; then
  models="Kell2018word Kell2018speaker Kell2018music Kell2018audioset"
  layers="input_after_preproc relu0 maxpool0 relu1 maxpool1 relu2 relu3 relu4 avgpool relufc final"
fi
if [ "$model" = "Kell2018Seed2" ]; then
  models="Kell2018wordSeed2 Kell2018speakerSeed2 Kell2018audiosetSeed2"
  layers="input_after_preproc relu0 maxpool0 relu1 maxpool1 relu2 relu3 relu4 avgpool relufc"
fi
if [ "$model" = "Kell2018wordClean" ]; then
  models="Kell2018wordClean"
  layers="input_after_preproc relu0 maxpool0 relu1 maxpool1 relu2 relu3 relu4 avgpool relufc final"
fi
if [ "$model" = "Kell2018wordCleanSeed2" ]; then
  models="Kell2018wordCleanSeed2"
  layers="input_after_preproc relu0 maxpool0 relu1 maxpool1 relu2 relu3 relu4 avgpool relufc final"
fi
if [ "$model" = "Kell2018speakerClean" ]; then
  models="Kell2018speakerClean"
  layers="input_after_preproc relu0 maxpool0 relu1 maxpool1 relu2 relu3 relu4 avgpool relufc"
fi
if [ "$model" = "Kell2018speakerCleanSeed2" ]; then
  models="Kell2018speakerCleanSeed2"
  layers="input_after_preproc relu0 maxpool0 relu1 maxpool1 relu2 relu3 relu4 avgpool relufc"
fi
if [ "$model" = "ResNet50" ]; then
  models="ResNet50word ResNet50speaker ResNet50audioset ResNet50music"
  layers="input_after_preproc conv1_relu1 maxpool1 layer1 layer2 layer3 layer4 avgpool final"
fi
if [ "$model" = "ResNet50wordClean" ]; then
  models="ResNet50wordClean"
  layers="input_after_preproc conv1_relu1 maxpool1 layer1 layer2 layer3 layer4 avgpool final"
fi
if [ "$model" = "ResNet50wordCleanSeed2" ]; then
  models="ResNet50wordCleanSeed2"
  layers="input_after_preproc conv1_relu1 maxpool1 layer1 layer2 layer3 layer4 avgpool final"
fi
if [ "$model" = "ResNet50speakerClean" ]; then
  models="ResNet50speakerClean"
  layers="input_after_preproc conv1_relu1 maxpool1 layer1 layer2 layer3 layer4 avgpool"
fi
if [ "$model" = "ResNet50speakerCleanSeed2" ]; then
  models="ResNet50speakerCleanSeed2"
  layers="input_after_preproc conv1_relu1 maxpool1 layer1 layer2 layer3 layer4 avgpool"
fi
if [ "$model" = "ResNet50Seed2" ]; then
  models="ResNet50wordSeed2 ResNet50speakerSeed2 ResNet50audiosetSeed2"
  layers="input_after_preproc conv1_relu1 maxpool1 layer1 layer2 layer3 layer4 avgpool"
fi
if [ "$model" = "Kell2018multitask" ]; then
  models="Kell2018multitask"
  layers="input_after_preproc relu0 maxpool0 relu1 maxpool1 relu2 relu3 relu4 avgpool relufc"
fi
if [ "$model" = "Kell2018multitaskSeed2" ]; then
  models="Kell2018multitaskSeed2"
  layers="input_after_preproc relu0 maxpool0 relu1 maxpool1 relu2 relu3 relu4 avgpool relufc"
fi
if [ "$model" = "ResNet50multitask" ]; then
  models="ResNet50multitask"
  layers="input_after_preproc conv1_relu1 maxpool1 layer1 layer2 layer3 layer4 avgpool"
fi
if [ "$model" = "ResNet50multitaskSeed2" ]; then
  models="ResNet50multitaskSeed2"
  layers="input_after_preproc conv1_relu1 maxpool1 layer1 layer2 layer3 layer4 avgpool"
fi
if [ "$model" = "Kell2018init" ]; then
  models="Kell2018init"
  layers="input_after_preproc relu0 maxpool0 relu1 maxpool1 relu2 relu3 relu4 avgpool relufc final"
fi
if [ "$model" = "ResNet50init" ]; then
  models="ResNet50init"
  layers="input_after_preproc conv1_relu1 maxpool1 layer1 layer2 layer3 layer4 avgpool final" #fixed in main, should run w music
fi
if [ "$model" = "AST" ]; then
  models="AST"
  layers="Embedding Encoder_1 Encoder_2 Encoder_3 Encoder_4 Encoder_5 Encoder_6 Encoder_7 Encoder_8 Encoder_9 Encoder_10 Encoder_11 Encoder_12 Final"
fi
if [ "$model" = "ASTSL01" ]; then
  models="ASTSL01"
  layers="Embedding Encoder_1 Encoder_2 Encoder_3 Encoder_4 Encoder_5 Encoder_6 Encoder_7 Encoder_8 Encoder_9 Encoder_10 Encoder_11 Encoder_12 Final"
fi
if [ "$model" = "ASTSL10" ]; then
  models="ASTSL10"
  layers="Embedding Encoder_1 Encoder_2 Encoder_3 Encoder_4 Encoder_5 Encoder_6 Encoder_7 Encoder_8 Encoder_9 Encoder_10 Encoder_11 Encoder_12 Final"
fi
if [ "$model" = "DCASE2020" ]; then
  echo "hello"
  models="DCASE2020"
  layers="GRU_1 GRU_2 GRU_3 GRU_4 Linear"
fi
if [ "$model" = "DS2" ]; then
  models="DS2"
  layers="Tanh_1 Tanh_2 LSTM_1-cell LSTM_2-cell LSTM_3-cell LSTM_4-cell LSTM_5-cell Linear"
fi
if [ "$model" = "metricGAN" ]; then
  models="metricGAN"
  layers="LSTM_1-cell LSTM_2-cell Linear_1 Linear_2"
fi
if [ "$model" = "metricGANSL01" ]; then
  models="metricGANSL01"
  layers="LSTM_1-cell LSTM_2-cell Linear_1 Linear_2"
fi
if [ "$model" = "metricGANSL10" ]; then
  models="metricGANSL10"
  layers="LSTM_1-cell LSTM_2-cell Linear_1 Linear_2"
fi
if [ "$model" = "S2T" ]; then
  models="S2T"
  layers="Embedding Encoder_1 Encoder_2 Encoder_3 Encoder_4 Encoder_5 Encoder_6 Encoder_7 Encoder_8 Encoder_9 Encoder_10 Encoder_11 Encoder_12"
fi
if [ "$model" = "sepformer" ]; then
  models="sepformer"
  layers="Embedding Encoder_1 Encoder_2 Encoder_3 Encoder_4 Encoder_5 Encoder_6 Encoder_7 Encoder_8 Encoder_9 Encoder_10 Encoder_11 Encoder_12 Encoder_13 Encoder_14 Encoder_15 Encoder_16 Encoder_17 Encoder_18 Encoder_19 Encoder_20 Encoder_21 Encoder_22 Encoder_23 Encoder_24 Encoder_25 Encoder_26 Encoder_27 Encoder_28 Encoder_29 Encoder_30 Encoder_31 Encoder_32"
fi
if [ "$model" = "sepformerSL01" ]; then
  models="sepformerSL01"
  layers="Embedding Encoder_1 Encoder_2 Encoder_3 Encoder_4 Encoder_5 Encoder_6 Encoder_7 Encoder_8 Encoder_9 Encoder_10 Encoder_11 Encoder_12 Encoder_13 Encoder_14 Encoder_15 Encoder_16 Encoder_17 Encoder_18 Encoder_19 Encoder_20 Encoder_21 Encoder_22 Encoder_23 Encoder_24 Encoder_25 Encoder_26 Encoder_27 Encoder_28 Encoder_29 Encoder_30 Encoder_31 Encoder_32"
fi
if [ "$model" = "sepformerSL10" ]; then
  models="sepformerSL10"
  layers="Embedding Encoder_1 Encoder_2 Encoder_3 Encoder_4 Encoder_5 Encoder_6 Encoder_7 Encoder_8 Encoder_9 Encoder_10 Encoder_11 Encoder_12 Encoder_13 Encoder_14 Encoder_15 Encoder_16 Encoder_17 Encoder_18 Encoder_19 Encoder_20 Encoder_21 Encoder_22 Encoder_23 Encoder_24 Encoder_25 Encoder_26 Encoder_27 Encoder_28 Encoder_29 Encoder_30 Encoder_31 Encoder_32"
fi
if [ "$model" = "spectemp" ]; then
  models="spectemp"
  layers="avgpool"
fi
if [ "$model" = "VGGish" ]; then
  models="VGGish"
  layers="ReLU_1 MaxPool2d_1 ReLU_2 MaxPool2d_2 ReLU_3 ReLU_4 MaxPool2d_3 ReLU_5 ReLU_6 MaxPool2d_4 ReLU_7 ReLU_8 ReLU_9 Post-Processed_Features"
fi
if [ "$model" = "VGGishSL01" ]; then
  models="VGGishSL01"
  layers="ReLU_1 MaxPool2d_1 ReLU_2 MaxPool2d_2 ReLU_3 ReLU_4 MaxPool2d_3 ReLU_5 ReLU_6 MaxPool2d_4 ReLU_7 ReLU_8 ReLU_9 Post-Processed_Features"
fi
if [ "$model" = "VGGishSL10" ]; then
  models="VGGishSL10"
  layers="ReLU_1 MaxPool2d_1 ReLU_2 MaxPool2d_2 ReLU_3 ReLU_4 MaxPool2d_3 ReLU_5 ReLU_6 MaxPool2d_4 ReLU_7 ReLU_8 ReLU_9 Post-Processed_Features"
fi
if [ "$model" = "wav2vec" ]; then
  models="wav2vec"
  layers="Embedding Encoder_1 Encoder_2 Encoder_3 Encoder_4 Encoder_5 Encoder_6 Encoder_7 Encoder_8 Encoder_9 Encoder_10 Encoder_11 Encoder_12 Final"
fi
if [ "$model" = "ZeroSpeech2020" ]; then
  models="ZeroSpeech2020"
  layers="ReLU_1 ReLU_2 ReLU_3 ReLU_4 ReLU_5"
fi
echo "About to run models: $models"
echo "Layers of interest: $layers"
echo "Random network: $randnetw"
echo "Target: $target"
echo ""
for model in $models ; do
  echo "____________ Model $model ____________"
  echo ""
  for layer in $layers ; do
      echo "Layer: $layer"
      for flag in $randnetw ; do
        echo "Random network: $flag"

        # Check whether the file exists in resultdir/model/identifier/df_output.pkl
        identifier="AUD-MAPPING-Ridge_TARGET-${target}_SOURCE-${model}-${layer}_RANDNETW-${flag}_ALPHALIMIT-50"
        file_to_look_for="$resultdir/$model/$identifier/df_output.pkl"

        # If overwrite is set to 1, then we don't care if the file exists or not
        if [ "$overwrite" = "1" ]; then
#          echo "FILE EXISTS: $file_to_look_for"
          echo "OVERWRITING ......."
          sbatch AUD_cpu.sh $model $layer $flag $target
        else
          # If overwrite is set to 0, then we only run the script if the file doesn't exist
          if [ -f "$file_to_look_for" ]; then
#            echo "FILE EXISTS: $file_to_look_for"
            echo "SKIPPING ......."
          else
            echo "FILE DOES NOT EXIST: $file_to_look_for"
            echo "RUNNING ......."
#            sbatch AUD_cpu.sh $model $layer $flag $target
          fi
        fi
      done
  done
done



