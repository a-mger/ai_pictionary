# AI Pictionary
For this project we used the Google Quick Draw Dataset to train an AI to recognize user generated sketches. The complete dataset consists of 345 labels and a total number of 50 million images.

The Project consists of 3 Models namely:
- CNN with 50 Labels each 50.000 images
- CNN with 150 labels each 70.000 images
- CNN with 250 labels each 90.000 images

# Training Data
Only the compressed bitmap files were used for the training of the CNNs.

# Google Cloud Services
The training of the models has been performed with the Google cloud AI platform. 
The prediction runs via the docker image on the Google Cloud repository and can be accessed via an API

# Fron tend
The front end [ai_pictionary_web](https://github.com/a-mger/ai_pictionary_web) was built with streamlit and drawable_canvas. The drawing data gets compressed and formatted to match the training data. 
Deploymend is done via Heroku

# Result
[ai_pictionary](https://ai-pictionary.herokuapp.com/)
