a.Dataset Used

1.RAVDESS:

You can Download the Dataset here at this Link:

https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

Dataset information:

Files:
This portion of the RAVDESS contains 1440 files: 60 trials per actor x 24 actors = 1440. The RAVDESS contains 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent. Speech emotions includes calm, happy, sad, angry, fearful, surprise, and disgust expressions. Each expression is produced at two levels of emotional intensity (normal, strong), with an additional neutral expression.

File naming convention:
Each of the 1440 files has a unique filename. The filename consists of a 7-part numerical identifier (e.g., 03-01-06-01-02-01-12.wav). These identifiers define the stimulus characteristics:

Filename identifiers:

Modality (01 = full-AV, 02 = video-only, 03 = audio-only).

Vocal channel (01 = speech, 02 = song).

Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).

Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.

Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").

Repetition (01 = 1st repetition, 02 = 2nd repetition).

Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

Filename example: 03-01-06-01-02-01-12.wav

Audio-only (03)

Speech (01)

Fearful (06)

Normal intensity (01)

Statement "dogs" (02)

1st Repetition (01)

12th Actor (12)

Female, as the actor ID number is even.


2.CREMA-D:
You can Download the Dataset here at this Link:
https://www.kaggle.com/datasets/ejlok1/cremad

Dataset Information:

CREMA-D is a data set of 7,442 original clips from 91 actors. These clips were from 48 male and 43 female actors between the ages of 20 and 74 coming from a variety of races and ethnicities (African America, Asian, Caucasian, Hispanic, and Unspecified).

Actors spoke from a selection of 12 sentences. The sentences were presented using one of six different emotions (Anger, Disgust, Fear, Happy, Neutral and Sad) and four different emotion levels (Low, Medium, High and Unspecified).

Participants rated the emotion and emotion levels based on the combined audiovisual presentation, the video alone, and the audio alone. Due to the large number of ratings needed, this effort was crowd-sourced and a total of 2443 participants each rated 90 unique clips, 30 audio, 30 visual, and 30 audio-visual. 95% of the clips have more than 7 ratings.

Description:

Text Data Files:
SentenceFilenames.csv - list of movie files used in study
finishedEmoResponses.csv - the first emotional response with timing.
finishedResponses.csv - the final emotional Responses with emotion levels with repeated and practice responses removed, used to tabulate the votes
finisedResponsesWithRepeatWithPractice.csv - the final emotional responses with emotion levels with repeated and practice responses in tact. Used to observe repeated responses and practice responses.
processedResults/tabulatedVotes.csv - the tabulated votes for each movie file.
VideoDemographics.csv - a mapping of ActorID (the first 4 digits of each video file) to Age, Sex, Race, and Ethicity.
R Scripts
processFinishedResponses.R - converts the finisedResponses.csv to the tabulated
readTabulatedVotes.R - reads processedResults/tabulatedVotes.csv
Finished Responses Columns
(finishedResponses.csv and
finishedResponsesWithRepeatWithPractice.csv)
"localid" - a participant identifier
"pos" - the original log file order for the participant
"ans" - the emotion character with level separated by an underscore
"ttr" - the response time in milliseconds
"queryType" - a numeric value specifying the type of stimulus: 1. - voice only, 2. face only, 3. audio-visual
"numTries" - number of extra emotion clicks.
"clipNum" - the file order of the clip from SentenceFilenames.csv
"questNum" - the order of questions for the query type
"subType" - the type of response in the logs, all values are 4 for the final emotion response
"clipName" - the name of the video file
"sessionNums" - the distinct number for the session
"respEmo" - the emotion response
"respLevel" - the emotion level response
"dispEmo" - the displayed emotion
"dispVal" - the displayed value
"dispLevel" - a numeric representation of the displayed value, 20 for low, 50 for med, 80 for hi.
Finished EmoResponses Columns
(finishedEmoResponses.csv)
"localid" - a participant identifier
"sessionNums" - the distinct number for the session
"queryType" - a numeric value specifying the type of stimulus: 1. - voice only, 2. face only, 3. audio-visual
"questNum" - the order of questions for the query type
"pos" - the original log file order for the participant
"ttr" - the response time in milliseconds
"numTries" - number of extra emotion clicks.
"clipNum" - the file order of the clip from SentenceFilenames.csv
"clipName" - the name of the video file
Summary Table Columns
processedResults/summaryTable.csv
"fileName" - name of the movie file rated
"VoiceVote" - the emotion (or emotions separated by a colon) with the majority vote for Voice ratings. (A, D, F, H, N, or S)
"VoiceLevel" - the numeric rating (or ratings separated by a colon) corresponding to the emotion(s) listed in "VoiceVote"
"FaceVote" - the emotion (or emotions separated by a colon) with the majority vote for Face ratings. (A, D, F, H, N, or S)
"FaceLevel" - the numeric rating (or ratings separated by a colon) corresponding to the emotion(s) listed in "FaceVote"
"MultiModalVote" - the emotion (or emotions separated by a colon) with the majority vote for MultiModal ratings. (A, D, F, H, N, or S)
"MultiModalLevel" - the numeric rating (or ratings separated by a colon) corresponding to the emotion(s) listed in "MultiModalVote"
Tabulated Votes Columns
processedResults/tabulatedVotes.csv
"A" - count of Anger Responses
"D" - count of Disgust Responses
"F" - count of Fear Responses
"H" - count of Happy Responses
"N" - count of Neutral Responses
"S" - count of Sad Responses
"fileName" - name of the movie file rated
"numResponses" - total number of responses
"agreement" - proportion of agreement
"emoVote" - the majority vote agreement
"meanEmoResp" - the mean of all emotion levels
"meanAngerResp" - the mean of the anger levels
"meanDisgustResp" - the mean of the disgust levels
"meanFearResp" - the mean of the fear levels
"meanHappyResp" - the mean of the happy levels
"meanNeutralResp" - the mean of the neutral levels
"meanSadResp" - the mean of the sad levels
"medianEmoResp" - the median of all emotion levels
"meanEmoRespNorm" - the normalized mean of all emotion levels
"meanAngerRespNorm" - the normalized mean of anger emotion levels
"meanDisgustRespNorm" - the normalized mean of disgust emotion levels
"meanFearRespNorm" - the normalized mean of fear emotion levels
"meanHappyRespNorm" - the normalized mean of happy emotion levels
"meanNeutralRespNorm" - the normalized mean of neutral emotion levels
"meanSadRespNorm" - the normalized mean of sad emotion levels
"medianEmoRespNorm" - the normalized median of all emotion levels
Video Demographics Columns
VideoDemographics.csv
"ActorID" - the first 4 digits of the video/audio file that identifies the actor in the video.
"Age" - the age in years of the actor at the time of the recording
"Sex" - the binary sex that the actor identified
"Race" - African American, Asian, Caucasian, or Unknown
"Ethnicity" - Hispanic or Not Hispanic
Filename labeling conventions
The Actor id is a 4 digit number at the start of the file. Each subsequent identifier is separated by an underscore (_).

Actors spoke from a selection of 12 sentences (in parentheses is the three letter acronym used in the second part of the filename):
It's eleven o'clock (IEO).
That is exactly what happened (TIE).
I'm on my way to the meeting (IOM).
I wonder what this is about (IWW).
The airplane is almost full (TAI).
Maybe tomorrow it will be cold (MTI).
I would like a new alarm clock (IWL)
I think I have a doctor's appointment (ITH).
Don't forget a jacket (DFA).
I think I've seen this before (ITS).
The surface is slick (TSI).
We'll stop in a couple of minutes (WSI).

The sentences were presented using different emotion (in parentheses is the three letter code used in the third part of the filename):

Anger (ANG)

Disgust (DIS)

Fear (FEA)

Happy/Joy (HAP)

Neutral (NEU)

Sad (SAD)

and emotion level (in parentheses is the two letter code used in the fourth part of the filename):
Low (LO)

Medium (MD)

High (HI)

Unspecified (XX)

The suffix of the filename is based on the type of file, flv for flash video used for presentation of both the video only, and the audio-visual clips. mp3 is used for the audio files used for the audio-only presentation of the clips. wav is used for files used for computational audio processing.

b.Extraction Tools

1.OpenSMILE:

You can install it in Python by,
!pip install opensmile
For more information refer this link:
https://audeering.github.io/opensmile-python/

2.Librosa:

You can install it in Python by,
!pip install librosa
For more information refer this link:
https://librosa.org/doc/latest/index.html

Swarm Intelligence : it refers to the intelligent behaviors of ants, birds, and other swarm animals that achieve their goals through the interaction between individuals or with the environment in the process of migration and foraging.

IDEA: To Create Multiple High Accuracy Model and then Combining Thier Output and Do the Detection. So the accuracy will Very Constant.

FOLDERS:

Filename Format:
[DatasetUsed]_[ExtractionToolUsed]_[ModelUsedToTrain]

RAVDESS_OpenSMILE_RNN(LSTM):

01.audioTocsv.py - converting .wav file to .csv.

02.extractingInfoFromThemNamesMerging.py - labeling the Emotions and Intensity from the filename.

03.RNNmodelTrainSave.py - training the RNN(LSTM) model and saving it.

04.predictionINTE.py - predicts the Intensity of the given audio.

05.predictionEMO.py - predicts the Emotion of the given audio.

all the .wav file is just for samples audio.

CREMA-D_OpenSMILE_GRU:

-[FolderName]_training - Training and Saving the Model.

-[FolderName]_prediction - Load the Model and Predict the Emotion of the Given Audio.

CREMA-D_OpenSMILE_FNN:

-[FolderName]_training - Training and Saving the Model.

-[FolderName]_prediction - Load the Model and Predict the Emotion of the Given Audio.

CREMA-D_OpenSMILE_LSTM:

-[FolderName]_training - Training and Saving the Model.

-[FolderName]_prediction - Load the Model and Predict the Emotion of the Given Audio.

RAVDESS_OpenSMILE_wav2vec:

-[FolderName]_training - Training and Saving the Model.

-[FolderName]_prediction - Load the Model and Predict the Emotion of the Given Audio.













