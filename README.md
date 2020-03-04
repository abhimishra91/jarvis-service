# Jarvis NLP Service


An NLP Project which entails, reading client emails and using that text to classify what task is requested by the client.
This is a rest service for the trained model.
This model will give 3 possible queues for the incoming specified email along with thier probability percentage.

This API will be further connected to other services to facilitate end to end automation. External services can be:

* MS-Teams Bot Framework

Currently the model is using - DistilBERT from the Transformers library. Further details of the Model: