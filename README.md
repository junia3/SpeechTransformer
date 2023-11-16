# SpeechTransformer

### Datasets
librispeech_100h, 360h : training
librispeech_dev : validation

[Dataset Link](https://www.openslr.org/12)

I replaced the feature extraction part (backbone) for audio data in the existing model with a mel filterbank instead of the VGG extractor.  
Additionally, I generated sequence information in a separate CSV file and trained the model. Originally, the goal was to proceed to streaming Keyword Spotting (KWS), but due to time constraints, I concluded with a simple first-stage pre-training.
