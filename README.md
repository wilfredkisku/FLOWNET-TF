# Flownet 1.0 and Flownet 2.0
Flownet architecture is a simplified structure for end to end learning for predicting the optical flow. The network is trained to predict the X,Y flow. The approach is to stack the images that are to be used for prediction. The other idea for the architecture is to use a siamese network to extract features from the network and then combine them to get the higher representation. To aid the network in this matching process, there is a new architectural layer called as the correlation layer that is added.Given two multi-channel feature maps f<sup>1</sup>,f<sup>2</sup> : R<sup>2</sup> &rarr; R<sup>c</sup>, with w, h, and c being their width, height and number of channels, our correlation layer lets the network compare each patch from f<sup>1</sup> with each path from f<sup>2</sup>. 

<code>ffmpeg -s cif -r 30 -i foreman-cif.yuv -vcodec mpeg4 -g 12 -bf 2 -qscale 0 video-foreman.mp4</code>
## Flownet 1.0 algorithm

<ul>
  <li>[Images, Flows{u,v}] := preprocessing(Path)</li>
  <li>[trainImages,trainFlow{u,v}] := obtainTrainingData([Images, Flows{u,v}])</li>
  <li>[trainedModel] := train(Adam[[trainImages,trainFlow{u,v}]])</li>
  <li>[weightsParams, accuracy] := [trainedModel]</li>
  <li>[prediction{u, v}] := [trainedModel, predictImage[frame1, frame2]]</li>
</ul>

## Flownet 2.0 algorithm

<ul>
  <li></li>
  <li></li>
  <li></li>
</ul>


## Usage

## Results
