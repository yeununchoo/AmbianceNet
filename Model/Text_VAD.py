import tensorflow as tf

class Text_VAD(tf.keras.Model):
    def __init__(self, vocab_size, window_size):
        super(Text_VAD, self).__init__()
        self.vocab_size = vocab_size # The size of the English vocab
        self.window_size = window_size # The English window size

        # TODO:
        # 1) Define any hyperparameters

        # Define batch size and optimizer/learning rate
        self.batch_size = 128 # You probably should change this
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001) 
        self.embedding_size = 48 

        # 2) Define embeddings, encoder, decoder, and feed forward layers
        
        initializer = tf.keras.initializers.TruncatedNormal(mean = 0.0, stddev = 0.01)

        self.embedding = tf.keras.layers.Embedding(self.vocab_size, 
                                                   self.embedding_size, 
                                                   embeddings_initializer = initializer,
                                                   name = "embedding_matrix")
        
        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(24), 
                                                  merge_mode='concat')
        self.dense01 = tf.keras.layers.Dense(24, activation = "leaky_relu") 
        self.dense02 = tf.keras.layers.Dense(3, activation = "sigmoid") 

    def call(self, text_input):
        """
        :param text_input: batched tokenized words of the reddit comments, [batch_size x window_size]
        :return VAD_coordinates: The 3d VAD coordinates as a tensor, [batch_size x 3]
        """
    
        x = self.embedding(text_input)

        x = self.lstm(x)
        x = self.dense01(x)
        x = self.dense02(x)
        
        VAD_coordinates = x
        
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
        # weighted_squared_distances = squared_distances
        loss = tf.reduce_mean(weighted_squared_distances)
        
        return loss
    
    
