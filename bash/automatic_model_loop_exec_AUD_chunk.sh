#!/bin/sh
model="$1"
chunks="0 1 2 3"
echo $model
randnetw="True"
if [ "$model" = "Kell2018" ]; then
  models="Kell2018word Kell2018speaker Kell2018music Kell2018audioset"
  layers="input_after_preproc relu0 maxpool0 relu1 maxpool1 relu2 relu3 relu4 avgpool relufc final"
fi
if [ "$model" = "ResNet50" ]; then
  models="ResNet50word ResNet50speaker ResNet50audioset ResNet50music"
  layers="input_after_preproc conv1_relu1 maxpool1 layer1 layer2 layer3 layer4 avgpool final" #fixed in main, should run w music
fi
if [ "$model" = "Kell2018multitask" ]; then
  models="Kell2018multitask"
  layers="input_after_preproc relu0 maxpool0 relu1 maxpool1 relu2 relu3 relu4 avgpool relufc final_word final_speaker final_audioset"
fi
if [ "$model" = "ResNet50multitask" ]; then
  models="ResNet50multitask"
  layers="input_after_preproc conv1_relu1 maxpool1 layer1 layer2 layer3 layer4 avgpool final_word final_speaker final_audioset"
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
if [ "$model" = "DCASE2020" ]; then
  echo "hello"
  models="DCASE2020"
  layers="GRU_1 GRU_2 GRU_3 GRU_4 Linear"
fi
if [ "$model" = "DS2" ]; then
  models="DS2"
  layers="Tanh_1 Tanh_2 LSTM_1 LSTM_1-cell LSTM_2 LSTM_2-cell LSTM_3 LSTM_3-cell LSTM_4 LSTM_4-cell LSTM_5 LSTM_5-cell Linear"
fi
if [ "$model" = "metricGAN" ]; then
  models="metricGAN"
  layers="LSTM_1 LSTM_1-cell LSTM_2 LSTM_2-cell Linear_1 Linear_2"
fi
if [ "$model" = "S2T" ]; then
  models="S2T"
  layers="Embedding Encoder_1 Encoder_2 Encoder_3 Encoder_4 Encoder_5 Encoder_6 Encoder_7 Encoder_8 Encoder_9 Encoder_10 Encoder_11 Encoder_12"
fi
if [ "$model" = "spectemp" ]; then
  models="spectemp"
  layers="avgpool"
fi
if [ "$model" = "sepformer" ]; then
  models="sepformer"
  layers="Embedding Encoder_1 Encoder_2 Encoder_3 Encoder_4 Encoder_5 Encoder_6 Encoder_7 Encoder_8 Encoder_9 Encoder_10 Encoder_11 Encoder_12 Encoder_13 Encoder_14 Encoder_15 Encoder_16 Encoder_17 Encoder_18 Encoder_19 Encoder_20 Encoder_21 Encoder_22 Encoder_23 Encoder_24 Encoder_25 Encoder_26 Encoder_27 Encoder_28 Encoder_29 Encoder_30 Encoder_31 Encoder_32"
fi
if [ "$model" = "VGGish" ]; then
  models="VGGish"
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
echo "Chunks: $chunksw"
echo ""
for model in $models ; do
  echo "____________ Model $model ____________"
  echo ""
  for layer in $layers ; do
      echo "Layer: $layer"
      for flag in $randnetw ; do
        echo "Random network: $flag"
        for chunk in $chunks ; do
          echo "Chunk: $chunk"
          sbatch AUD_chunk.sh $model $layer $flag $chunk
          done
      done
  done
done




