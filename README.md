# HandGestureDetection

Allow hand positions detection thanks to a neural network.
For now, the java applet is used to capture the video stream, process the images.
The java applet also acts as a server which send in real time an image to the python client.
The python client is the trained neural network which predict which gesture is on the real time image.
Then the prediction is sent to the Java server, which displays the result.
