"""
HandDetector.py
This script can train the neural network AND run it for predictions. 

"""

import NetLoader as NetLoader # Custom net loader script 
import DataLoader as DataLoader # Custom data loader script
import EasySocket as EasySocket # Custom scoket script
import numpy as np 
import HandNetMaker as netMaker
from keras.optimizers import SGD, RMSprop
from keras.preprocessing.image import ImageDataGenerator
import keras

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
       
if __name__ == "__main__":
    
    model_path = "hand_detection_model_3.json" #  Mode file path
    weights_path = "hand_detection_weights_v2.h5" # Weights file path

    res_x = 96
    res_y = 96
    
    # MUST BE THE SAME LOCATION AS THE PATH IN HandGesture.java
    real_time_path = "../../realTime.png"
    
    raw_input_size = 0 # there are no real input so set to 0
    raw_output_size = 4
    
    train_mode = True
    prediction_mode = False
    
    # Can be removed if there is no folder with gesture data 
    image_val_data_location_1_1 = "HandGestureData/TakeOffPos/"
    image_val_data_location_2_1 = "HandGestureData/LandPos/"
    image_val_data_location_3_1 = "HandGestureData/GoRightPos/"
    image_val_data_location_4_1 = "HandGestureData/GoLeftPos/"
    image_val_data_location_5_1 = "HandGestureData/GoDownPos/"
    image_val_data_location_6_1 = "HandGestureData/GoUpPos/"
    image_val_data_location_7_1 = "HandGestureData/GoBackPos/"
    image_val_data_location_8_1 = "HandGestureData/ComeForwardPos/"
    image_val_data_location_9_1 = "HandGestureData/NonePos/"
    image_val_data_location_1_2 = "HandGestureData/TakeOffRot/"
    image_val_data_location_2_2 = "HandGestureData/LandRot/"
    image_val_data_location_3_2 = "HandGestureData/GoRightRot/"
    image_val_data_location_4_2 = "HandGestureData/GoLeftRot/"
    image_val_data_location_5_2 = "HandGestureData/GoDownRot/"
    image_val_data_location_6_2 = "HandGestureData/GoUpRot/"
    image_val_data_location_7_2 = "HandGestureData/GoBackRot/"
    image_val_data_location_8_2 = "HandGestureData/ComeForwardRot/"
    image_val_data_location_9_2 = "HandGestureData/NoneRot/"
    
    
    """
    net = NetLoader.NetLoader(model_file=model_path, weights_file=weights_path,
                              learning_rate = 0.001,decay_rate=0.00000001,
                              create_file=True,epoch_save = 1)
   
    data_list = [ # First data set (Rotational Gestures)
                   image_val_data_location_1_1,image_val_data_location_2_1, 
                   image_val_data_location_3_1,image_val_data_location_4_1,
                   image_val_data_location_5_1,image_val_data_location_6_1,
                   image_val_data_location_7_1,image_val_data_location_8_1,
                   image_val_data_location_9_1,
                   
                   # Second data set (Positional Gestures)
                   image_val_data_location_1_2,image_val_data_location_2_2,
                   image_val_data_location_3_2,image_val_data_location_4_2,
                   image_val_data_location_5_2,image_val_data_location_6_2,
                   image_val_data_location_7_2,image_val_data_location_8_2,
                   image_val_data_location_9_2]

    # if traning mode is false, then data list can just be an empty array
    if(train_mode == False):
        data_list = []
        
    # create data loader
    # if data_list is an empty array then calling set_elements_to_train will be an error 
    


    """
    data_list = []
    data = DataLoader.DataLoader(data_list, size_x = res_x,
                                 size_y=res_y, num_inputs=raw_input_size, 
                                 num_outputs=raw_output_size,black_white=False)

    from keras.applications.mobilenetv2 import MobileNetV2

    batch_size = 32
    train_datagen = ImageDataGenerator(rescale=1/255)

    train_generator = train_datagen.flow_from_directory(
    'HandGestureData/data/train',
    batch_size=batch_size,
    target_size=(96,96),
    class_mode='categorical')

    validation_generator = train_datagen.flow_from_directory(
    'HandGestureData/data/validation',
    batch_size=batch_size,
    target_size=(96, 96),
    class_mode='categorical')

    """
    model = netMaker.custom_model_hand()
                                 
    model.compile(loss = "mean_squared_error", 
                optimizer=SGD(lr=0.0001, decay=0.00000001, 
                momentum=0.9, nesterov=True))

    model.summary()
    """

    IMG_SHAPE = (96, 96, 3)

    # Create the base model from the pre-trained model MobileNet V2
    base_model = MobileNetV2(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')

    base_model.trainable = False

    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(4, activation='softmax')
    ])

    model.compile(optimizer=RMSprop(lr=0.0001), loss="categorical_crossentropy", metrics=['categorical_accuracy'])
    
    # if in traning mode
    if(train_mode == True):
        
        #data.combine_data(random_sort= True)
        #input_element_1, output_element_1 = data.get_set_elements_to_train(0)
    
        #if(prediction_mode == True):
            #pre = net.predict(input_element_1[0])
            #for i in range(0,len(pre)):
                #pred = pre[i]*100
                #print("Max Index: "+str(np.argmax(pred))+"  Output: "+str(int(pred[0]))+" "+str(int(pred[1]))+" "+str(int(pred[2]))+" "+str(int(pred[3]))+" "+str(int(pred[4])))
        #else:    
            #for i in range(0,15):
            #    net.fit(input_element_1[0],output_element_1,verbose = 2)
            history = model.fit_generator(train_generator,
                    steps_per_epoch=train_generator.n // batch_size,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.n // batch_size,
                    epochs=30)

            model.save_weights(weights_path)
            """
            acc = history.history['acc']
            val_acc = history.history['val_acc']

            loss = history.history['loss']
            val_loss = history.history['val_loss']

            plt.figure(figsize=(8, 8))
            plt.subplot(2, 1, 1)
            plt.plot(acc, label='Training Accuracy')
            plt.plot(val_acc, label='Validation Accuracy')
            plt.legend(loc='lower right')
            plt.ylabel('Accuracy')
            plt.ylim([min(plt.ylim()), 1])
            plt.title('Training and Validation Accuracy')

            plt.subplot(2, 1, 2)
            plt.plot(loss, label='Training Loss')
            plt.plot(val_loss, label='Validation Loss')
            plt.legend(loc='upper right')
            plt.ylabel('Cross Entropy')
            plt.ylim([0, max(plt.ylim())])
            plt.title('Training and Validation Loss')
            plt.show()
            """
    # if traning mode is false
    else: 

        model.load_weights(weights_path)

        socket = EasySocket.EasySocket(preset_unpack_types = ['i']) # add a preset type of 1 integer (get that float value)
        
        socket.connect() #connect to server
        
        while True:
            if(socket.get_anything(4,0) == True):# get the integer (it does not make a difference what it gets)
                
                try:
                    raw_RGB = data.load_image(real_time_path) # loader raw image
                except (FileNotFoundError, OSError):
                    print("file not found")
                    continue


                raw_RGB = np.array(raw_RGB,dtype = np.float32) 


                raw_RGB = np.rollaxis(raw_RGB,0,2)
                raw_RGB = np.rollaxis(raw_RGB,2,1)
                                
                #print(raw_RGB.shape)
                pre = model.predict(np.array([raw_RGB])) # get prediction
                print(pre)
                
                #pre = [[0.9955657]]
                
                # create a line of message
                message = ""
                for i in range(0,len(pre[0])):
                    message +=str(pre[0][i])
                    if(i == len(pre[0])-1):
                        message+="\n"
                    else:
                        message+=" "
    
                socket.send_string_data(message) # send predictio nmessage to server
                
        socket.close() #close socket if while loop breaks....which it never will lol
