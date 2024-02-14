# My Final Year Project at Aston University

This project contains a gesture generation system including:
* a data gathering pipeline
* a proposed test model
* a blender model for the generated rig.

Overall this project was not able to generate a model which learned co-speech gestures, due to a lack of resources and poor training data.

![image](https://github.com/VanessaKostadinova/final-year-project/assets/48443752/e60c48b5-43b3-4c0c-ae78-4bc8b3bb48ba)

## Data Pipeline
The data gathering of this project consists of many components all used together to generate a BVH file and accompanying text file from a series of TED talks.
### Data Downloading
The data of the TED talks can be downloaded from YouTube using the download_yt_video.py script. This script recursively downloads TED talks from a specific CSV schema.
The downloaded TED talks are stored as MP4 and named based on their accompanying transcript can be downloaded using the download_trainscript.py script.
The data from these must then be split into usable components using the splitting scripts.
### Extracting Gestures
To extract gestures from a 2D video, a project called OpenPose was used on each split segment. The output of this is a list of 2D armature coordinates. The OpenPose gestures originally featured hands, however, as discussed later, this was scaled back to just the arms due to quality issues.

To turn the 2D armature coordinates into a 3D rig, a project called MocapNet was used to estimate an animated rig from the 2d coordinates.
MocapNet struggled to differentiate the orientation of the different joints in the hands provided by OpenPose, resulting in disjointed fingers which made the dataset worse. 

Even after cleaning using the bvh_hand_process.py script, which set constraints on the joints and attempted to reverse their rotations when incorrect, this problem persisted. So the hands were removed from the dataset.
### Transcript Mapping
The transcripts of the videos were time mapped using a WAV file extracted from the MP4 videos using a project called Gentle. This allowed transcripts to be paired with their associated BVH files for training.
## Proposed ML Model
For ease of training and implementation, a Seq2Seq model was implemented in pyTorch which took in the transcript files and was trained to output the BVH rotations.
This model only learned how to average one pose and never got past this stage.

The aim of this model was to act as an easy architecture to test the training data with before attempting the training of a Transformer Model. This was not possible due to time constraints.
## Blender 3D Model
The provided model comes pre-rigged for the skeleton provided by the rest of the pipeline. The model is low polygon and designed to roughly follow a human form. This model was produced to be clear to follow but not require many resources to render or to calculate rigged movements, this means outputs from the data pipeline or ML model can be viewed in real time.

![image](https://github.com/VanessaKostadinova/final-year-project/assets/48443752/c382b995-31e9-43c1-97f4-00cbe0497b15)

![image](https://github.com/VanessaKostadinova/final-year-project/assets/48443752/bbd395af-7fff-4170-86e8-5a29d6014ffc)

The provided model and scene also allow for scenic renders of the model, handposed or in action.
