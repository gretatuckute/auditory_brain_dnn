#!/bin/sh
models="ResNet50word ResNet50speaker ResNet50music ResNet50audioset"
randnetw="False"
#layers="input_after_preproc relu0 maxpool0 relu1 maxpool1 relu2 relu3 relu4 avgpool relufc final"
layers="input_after_preproc conv1_relu1 maxpool1 layer1 layer2 layer3 layer4 avgpool final" # for resnet50music, it is relu1 and not conv1_relu1
#layers="GRU_1 GRU_2 GRU_3 GRU_4 Linear"
#layers="Tanh_1 Tanh_2 LSTM_1 LSTM_1-cell LSTM_2 LSTM_2-cell LSTM_3 LSTM_3-cell LSTM_4 LSTM_4-cell LSTM_5 LSTM_5-cell Linear"
#layers="Conv2d_1 Conv2d_2 LSTM_1 LSTM_2 LSTM_3 LSTM_4 LSTM_5 Linear"
#layers="Embedding Encoder_1 Encoder_2 Encoder_3 Encoder_4 Encoder_5 Encoder_6 Encoder_7 Encoder_8 Encoder_9 Encoder_10 Encoder_11 Encoder_12 Final"
#layers="ReLU_1 ReLU_2 ReLU_3 ReLU_4 ReLU_5 ReLU_6 Linear_1 ReLU_7 Linear_2 ReLU_8 Linear_3 ReLU_9 Post-Processed_Features"
#layers="Conv2d_1 ReLU_1 MaxPool2d_1 Conv2d_2 ReLU_2 MaxPool2d_2 Conv2d_3 ReLU_3 Conv2d_4 ReLU_4 MaxPool2d_3 Conv2d_5 ReLU_5 Conv2d_6 ReLU_6 MaxPool2d_4 Linear_1 ReLU_7 Linear_2 ReLU_8 Linear_3 ReLU_9 Post-Processed_Features"
#layers="input_after_preproc batchnorm0 conv0 relu0 maxpool0 batchnorm1 conv1 relu1 maxpool1 batchnorm2 conv2 relu2 conv3 relu3 conv4 relu4 avgpool fullyconnected relufc dropout final"
for model in $models ; do
  echo "____________ Model $model ____________"
  echo ""
  for layer in $layers ; do
      echo "Layer: $layer"
      for flag in $randnetw ; do
        echo "Random network: $flag"
        sbatch AUD_cpu.sh $model $layer $flag
      done
  done
done