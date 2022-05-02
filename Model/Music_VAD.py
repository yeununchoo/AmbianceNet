import tensorflow as tf

class Music_VAD(tf.keras.Model):
    def __init__(self):
        super(Music_VAD, self).__init__()
        
        # variables
        
        self.music_length = 10 # 10 seconds per audio
        self.VGG_size = 128 # VGG embedding size
        self.batch_size = 128 
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001) 

        # layers
        
        self.conv01 = tf.keras.layers.Conv2D(filters = 8, 
                                             kernel_size = [4, 16], 
                                             strides = (1, 2),
                                             padding = "same", 
                                             input_shape = (self.music_length, self.VGG_size, 1), 
                                             name = "conv_01"
                                            )
        self.conv02 = tf.keras.layers.Conv2D(filters = 8, 
                                             kernel_size = [4, 16], 
                                             strides = (2, 2),
                                             padding = "same", 
                                             name = "conv_02"
                                            )
        self.conv03 = tf.keras.layers.Conv2D(filters = 8, 
                                             kernel_size = [4, 8], 
                                             strides = (2, 2),
                                             padding = "same", 
                                             activation = "leaky_relu", 
                                             name = "conv_03"
                                            )

        
        self.flatten = tf.keras.layers.Flatten()
        self.dense01 = tf.keras.layers.Dense(32, 
                                             activation = "leaky_relu", 
                                             name = "dense")
        self.dense02 = tf.keras.layers.Dense(3, 
                                             activation = "sigmoid", 
                                             name = "output") 
        
    
    def call(self, VGGish_input):
        """
        :param VGGish_input: batched input of the VGGish embeddings, [batch_size x 10 x 128 x 1]
        :return prbs: The 3d VAD coordinates as a tensor, [batch_size x 3]
        """
        
        x = self.conv01(VGGish_input)
        x = self.conv02(x)
        x = self.conv03(x)
        
        x = self.flatten(x)
        
        x = self.dense01(x)
        VAD_coordinates = self.dense02(x)

        # The final result should look like this
        # VAD_coordinates = 
        #     [[v_1, a_1, d_1],
        #      [v_2, a_2, d_2],
        #      ...
        #      [v_batch, a_batch, d_batch]]
        
        return VAD_coordinates

    def loss_function(self, VAD_true, VAD_pred, sample_weights):
        """
        Calculates the Euclidean distance between the true VAD coordinates and the predicted coordinates.

        :param VAD_true: float tensor, [batch_size x 3]
        :param VAD_pred: float tensor, [batch_size x 3]
        :param sample_weights: float tensor, [batch_size x 1]

        :return: the loss of the model as a tensor
        """
        
        delta_VAD = tf.math.square(VAD_true - VAD_pred)
        squared_distances = tf.reduce_sum(delta_VAD, axis = 1)
        
        weighted_squared_distances = sample_weights*squared_distances
        
        loss = tf.reduce_mean(weighted_squared_distances)

        return loss





