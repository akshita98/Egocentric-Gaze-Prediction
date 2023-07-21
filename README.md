Run pytorch_train for training the model.

The training data must be in train folder. Names of video file and ground_truth files must be same.
It is set to train the fine gaze prediction model.
Parameters such as number of epochs or using pretreined models can be given by editing the train_fine function parameters.

If Transfer Learning is to be used set tl=True

The code generates plots for loss and accuracy and saves them along with .pth model file


Running pytorch_video file predicts the ground-truth onto the videos present in the test folder.