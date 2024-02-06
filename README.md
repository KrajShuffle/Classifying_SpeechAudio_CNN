# Alcohol Intoxication Detection via Gender-Specific Convolutional Neural Networks trained on Human Speech Samples

This repository contains notebooks illustrating data preprocessing, feature extraction, model training & validation, and model inference workflows for male and female alcohol intoxication detection models. Overall, these notebooks illustrate how each individual workflow contributes to a specific phase of the model development & optimization process culminating in its potential deployment. 

## Overview of Entire Approach

### Inputs
1. User’s Speech Audio .WAV File with a sampling rate of 22,050 Hz
2. User's Corresponding .TextGrid file, which has the same .WAV filename except for extension (.WAV vs .TextGrid)
3. Gender (Options: Male or Female)
    * Each Gender has:
      * Unique set of data parameters used to generate Mel Spectrograms (Feature used in distinguishing between intoxicated & sober)
      * Intrinsic model & model’s hyperparameters used to generate chunk level predictions
        * Optimized Models are saved in TorchScript format, lends itself for easy deployment & excellent scalability in inference

### Workflow
1. Use instantiated Spectrify object to acquire information on when high-quality chunks or samples occur within user's speech file
   * Context:
     A chunk is essentially a sequence of phonetic sounds or utterances grouped together till the total time length of that chunk is at least 1 second long. TextGrids are files used to identify when these pre-annotated phonetic sounds & noise are occurring, so we can prevent long silences and noise being present in the chunks being used for intoxication classification. This is possible because the TextGrids provide an audio file mapping of when all phonetic sounds, long speech pauses, and noise occur in the audio file. To get a chunk of at least 1 second long, the start and stop times of phonetic sounds are sequenced together till the total time of the sequence is at least 1 second long.
   * Relevant Spectrify object parameters:
     * ***silence length*** (criteria used to determine if a person has stopped speaking or it’s a small unavoidable pause between phonetic sounds which is okay)
     * ***desired chunk length*** (in seconds). I used 1 second in this project, but could be set to whatever value you decide
   * Relevant Spectrify functions are ***Planner*** and ***Phraser***:
     * Planner extracts all phonetic sounds, noise, and pauses’ start and stop times from the TextGrid file
     * Phraser uses the Planner's insights and logic to detect noise or long pauses and outputs start and end times of identified, high-quality chunks that meets the desired chunk length of 1 second. 
   * For each speech file's high-quality chunk, the ***chunk's start and end times***, ***filename of file from which the chunk was extracted from***, and ***user's gender*** are documented.  

2. Use Spectrify Class' Spectrify function to create Mel Spectrogram with recently acquired chunk location information
    1. Create each identified, high-quality chunk by loading in the audio file along with the start and end times of that chunk
    2. With gender information, feed in gender-specific, optimized data parameters to create Mel Spectrogram for each high-quality chunk
    3. Normalize intrinsic values of Mel Spectrogram in order to be a suitable input for a Convolutional Neural Network (CNN)
    4. Normalize size of Mel Spectrogram:
        * Done because each chunk's time length is not exactly 1 second, but is slightly larger than 1 second
        * Height of Spectrogram is the number of mels which determines amount of detail in the frequency range
        * Width of Spectrogram is cut to be strictly 1 second as determined by ***int((Sampling Rate * chunk_len)/ hop length) + 1***, where chunk_len = 1
    5. Mel Spectrogram gets reshaped into an array of (num_channels, height of array, width of array) so (1, 64, 345) since 1 channel (black & white image)

All of this is accomplished by a combination of the Spectrify’s Spectrify function and a Pytorch Dataset class, which calls the Spectrify function and a function to reshape into a suitable format for the CNN

Overall, the Spectrify & PyTorch Dataset class accomplish 3 ***key*** steps:
  * Extracts information of the ***start and stop times of a chunk (sample from the audio file)*** that meets the ***minimum time threshold of 1 second***
  * Uses that information to create a predefined-size, Mel Spectrogram based on the gender-specific data parameters provided
  * Reformats Mel Spectrogram to be a suitable input for a convolutional neural network (CNN)

3. Create Predictions for each Chunk
    * Load in the male or female model (dependent on user’s gender) with optimized model weights as defined in TorchScript Documentation
    * Output of model will be logits essentially the result of the final dense layer’s weights scaled by the output of previous dense layer + bias term
      * Output = Weights * Input(Output of Prev. Dense Layer) + Bias
    * Pass logits of each chunk into sigmoid function to generate pseudo probabilities: Values between 0 and 1
    * Predicted chunk class is 1 (Intoxicated or Positive) if probability >= 0.5. Otherwise, 0 (Sober or Negative)

4. Aggregate Chunk Intoxicated or Sober predictions for User Speech WAV File or User Sober/Intoxicated Classification:
    * To reiterate, each WAV file can output 1 or more chunks so the resulting the WAV file vote is the result of the aggregation of the chunk class predictions
    * If there is at least 1 chunk for each class:
      * Majority voting for user class prediction if number of chunks for a certain class is greater than the number of chunks for the other class
    * If equal chunks for each class:
      * Predicting the intoxicated (positive) class and outputting the mean of pseudo-probabilities by sigmoid function of chunks predicted positive or 1
    * Else if there is only 1 unique chunk class (1 chunk or 1 set of chunks of all intoxicated (1) or sober (0)):
      * If sole class predicted is 1:
        * User is predicted to be class 1 with mean of pseudo-probabilities outputted by sigmoid function of chunks predicted positive. This mean is used to serve as a confidence metric that the model had when predicting the user to be intoxicated. 
      * If sole class predicted is 0:
        * User is predicted 0
    * Output is User’s Sober/Intoxicated Prediction:
      * 1 if Intoxicated along with model confidence metric and 0 if Sober

For inference, this entire workflow along with being able to adapt to user's gender (Male or Female) has been saved as separate functions within the Speech_Classify Class, which can easily be called as demonstrated in the example provided in the DynGen_WavFileClassification Jupyter Notebook. 




