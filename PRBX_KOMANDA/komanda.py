#1
import tensorflow as tf
import numpy as np
import os
#slim = tf.contrib.slim
import tf_slim as slim
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
#abs_path = "C:\\Users\\Pratham B\\Desktop\\UNI_Y3\\PRBX\\github\\hmb1\\" #local
abs_path = "/shared/storage/cs/studentscratch/pb1028/new_venv/hmb1/"  #gpu
#ANCHOR - 2
SEQ_LEN = 5
BATCH_SIZE = 4 
LEFT_CONTEXT = 5
HEIGHT = 480
WIDTH = 640
CHANNELS = 3
RNN_SIZE = 16 #32
RNN_PROJ = 16 #32
CSV_HEADER = "index,timestamp,width,height,frame_id,filename,angle,torque,speed,lat,long,alt".split(",")
OUTPUTS = CSV_HEADER[-6:-3] # angle,torque,speed
OUTPUT_DIM = len(OUTPUTS) # predict all features: steering angle, torque and vehicle speed
#ANCHOR - 3
class BatchGenerator(object):
    def __init__(self, sequence, seq_len, batch_size):
        self.sequence = sequence
        self.seq_len = seq_len
        self.batch_size = batch_size
        chunk_size = 1 + (len(sequence) - 1) / batch_size
        self.indices = [(i*chunk_size) % len(sequence) for i in range(batch_size)]
    def next(self):
        while True:
            output = []
            for i in range(self.batch_size):
                idx = int(self.indices[i])
                left_pad = self.sequence[idx - LEFT_CONTEXT:idx]
                if len(left_pad) < LEFT_CONTEXT:
                    left_pad = [self.sequence[0]] * (LEFT_CONTEXT - len(left_pad)) + left_pad
                assert len(left_pad) == LEFT_CONTEXT
                leftover = len(self.sequence) - idx
                if leftover >= self.seq_len:
                    result = self.sequence[idx:idx + self.seq_len]
                else:
                    result = self.sequence[idx:] + self.sequence[:self.seq_len - leftover]
                assert len(result) == self.seq_len
                self.indices[i] = (idx + self.seq_len) % len(self.sequence)
                images, targets = list(zip(*result))
                images_left_pad, _ = list(zip(*left_pad))
                output.append((np.stack(images_left_pad + images), np.stack(targets)))
            output = list(zip(*output))
            output[0] = np.stack(output[0]) # batch_size x (LEFT_CONTEXT + seq_len)
            output[1] = np.stack(output[1]) # batch_size x seq_len x OUTPUT_DIM
            return output
def read_csv(filename):
    with open(filename, 'r') as f:
        lines = [ln.strip().split(",")[-7:-3] for ln in f.readlines()]
        lines = map(lambda x: (x[0], np.float32(x[1:])), lines) # imagefile, outputs
        return lines
def process_csv(filename, val=5):
    sum_f = np.float128([0.0] * OUTPUT_DIM)
    sum_sq_f = np.float128([0.0] * OUTPUT_DIM)
    lines = read_csv(filename)
    # leave val% for validation
    train_seq = []
    valid_seq = []
    cnt = 0
    for ln in lines:
        if cnt < SEQ_LEN * BATCH_SIZE * (100 - val): 
            train_seq.append(ln)
            sum_f += ln[1]
            sum_sq_f += ln[1] * ln[1]
        else:
            valid_seq.append(ln)
        cnt += 1
        cnt %= SEQ_LEN * BATCH_SIZE * 100
    mean = sum_f / len(train_seq)
    var = sum_sq_f / len(train_seq) - mean * mean
    std = np.sqrt(var)
    print(len(train_seq), len(valid_seq))
    print (mean, std) # we will need these statistics to normalize the outputs (and ground truth inputs)
    return (train_seq, valid_seq), (mean, std)
#ANCHOR - 4
(train_seq, valid_seq), (mean, std) = process_csv(filename="interpolated_train.csv", val=5) # concatenated interpolated.csv from rosbags 
test_seq = read_csv("interpolated_test.csv") # interpolated.csv for testset filled with dummy values 
#ANCHOR - 5
#layer_norm = lambda x: tf.compat.v1.estimator.layers.layer_norm(inputs=x, center=True, scale=True, activation_fn=None, trainable=True)
layer_norm = tf.keras.layers.LayerNormalization(center=True, scale=True, trainable=True)
def get_optimizer(loss, lrate):
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lrate)
    gradvars = optimizer.compute_gradients(loss)
    gradients, v = list(zip(*gradvars))
    [print(x.name) for x in v]
    gradients, _ = tf.clip_by_global_norm(gradients, 15.0)
    return optimizer.apply_gradients(list(zip(gradients, v)))
def apply_vision_simple(image, keep_prob, batch_size, seq_len, scope=None, reuse=None):
    video = tf.reshape(image, shape=[batch_size, LEFT_CONTEXT + seq_len, HEIGHT, WIDTH, CHANNELS])
    with tf.compat.v1.variable_scope(scope, 'Vision', [image], reuse=reuse):
        net = slim.convolution(video, num_outputs=32, kernel_size=[3,12,12], stride=[1,6,6], padding="VALID") #128
        net = tf.compat.v1.nn.dropout(x=net, keep_prob=keep_prob)
        aux1 = slim.fully_connected(tf.reshape(net[:, -seq_len:, :, :, :], [batch_size, seq_len, -1]), 32, activation_fn=None) #128
        net = slim.convolution(net, num_outputs=64, kernel_size=[2,5,5], stride=[1,2,2], padding="VALID")
        net = tf.compat.v1.nn.dropout(x=net, keep_prob=keep_prob)
        aux2 = slim.fully_connected(tf.reshape(net[:, -seq_len:, :, :, :], [batch_size, seq_len, -1]), 32, activation_fn=None) #128    
        net = slim.convolution(net, num_outputs=64, kernel_size=[2,5,5], stride=[1,1,1], padding="VALID")
        net = tf.compat.v1.nn.dropout(x=net, keep_prob=keep_prob)
        aux3 = slim.fully_connected(tf.reshape(net[:, -seq_len:, :, :, :], [batch_size, seq_len, -1]), 32, activation_fn=None) #128       
        net = slim.convolution(net, num_outputs=64, kernel_size=[2,5,5], stride=[1,1,1], padding="VALID")
        net = tf.compat.v1.nn.dropout(x=net, keep_prob=keep_prob)
        # at this point the tensor 'net' is of shape batch_size x seq_len x ...
        aux4 = slim.fully_connected(tf.reshape(net, [batch_size, seq_len, -1]), 32, activation_fn=None) #128       
        net = slim.fully_connected(tf.reshape(net, [batch_size, seq_len, -1]), 256, activation_fn=tf.nn.relu) #1024
        net = tf.compat.v1.nn.dropout(x=net, keep_prob=keep_prob)
        net = slim.fully_connected(net, 128, activation_fn=tf.nn.relu) #512
        net = tf.compat.v1.nn.dropout(x=net, keep_prob=keep_prob)
        net = slim.fully_connected(net, 64, activation_fn=tf.nn.relu) #256
        net = tf.compat.v1.nn.dropout(x=net, keep_prob=keep_prob)
        net = slim.fully_connected(net, 32, activation_fn=None) #128
        return layer_norm(tf.compat.v1.nn.elu(net + aux1 + aux2 + aux3 + aux4)) # aux[1-4] are residual connections (shortcuts)
class SamplingRNNCell(tf.compat.v1.nn.rnn_cell.RNNCell):
  """Simple sampling RNN cell."""
  def __init__(self, num_outputs, use_ground_truth, internal_cell):
    """
    if use_ground_truth then don't sample
    """
    self._num_outputs = num_outputs
    self._use_ground_truth = use_ground_truth # boolean
    self._internal_cell = internal_cell # may be LSTM or GRU or anything  
  @property
  def state_size(self):
    return self._num_outputs, self._internal_cell.state_size # previous output and bottleneck state
  @property
  def output_size(self):
    return self._num_outputs # steering angle, torque, vehicle speed
  def __call__(self, inputs, state, scope=None):
    (visual_feats, current_ground_truth) = inputs
    prev_output, prev_state_internal = state
    context = tf.concat([prev_output, visual_feats], axis=1)
    new_output_internal, new_state_internal = self._internal_cell(context, prev_state_internal) # here the internal cell (e.g. LSTM) is called
    #new_output = tf.compat.v1.estimator.layers.fully_connected(
    new_output = slim.layers.fully_connected(
        inputs=tf.concat([new_output_internal, prev_output, visual_feats], axis=1),
        num_outputs=self._num_outputs,
        activation_fn=None,
        scope="OutputProjection")
    # if self._use_ground_truth == True, we pass the ground truth as the state; otherwise, we use the model's predictions
    return new_output, (current_ground_truth if self._use_ground_truth else new_output, new_state_internal)
#ANCHOR - 6
graph = tf.Graph()
with graph.as_default():
    # inputs  
    learning_rate = tf.compat.v1.placeholder_with_default(input=1e-4, shape=())
    keep_prob = tf.compat.v1.placeholder_with_default(input=1.0, shape=())
    aux_cost_weight = 0.1 #tf.placeholder_with_default(input=0.1, shape=())    
    inputs = tf.compat.v1.placeholder(shape=(BATCH_SIZE,LEFT_CONTEXT+SEQ_LEN), dtype=tf.string) # pathes to png files from the central camera
    targets = tf.compat.v1.placeholder(shape=(BATCH_SIZE,SEQ_LEN,OUTPUT_DIM), dtype=tf.float32) # seq_len x batch_size x OUTPUT_DIM
    targets_normalized = (targets - mean) / std
    input_images = tf.stack([tf.io.decode_jpeg(tf.io.read_file((abs_path + x))) #tf.image.decode_png(tf.read_file(x))
                            for x in tf.unstack(tf.reshape(inputs, shape=[(LEFT_CONTEXT+SEQ_LEN) * BATCH_SIZE]))])
    input_images = -1.0 + 2.0 * tf.cast(input_images, tf.float32) / 255.0
    input_images.set_shape([(LEFT_CONTEXT+SEQ_LEN) * BATCH_SIZE, HEIGHT, WIDTH, CHANNELS])
    visual_conditions_reshaped = apply_vision_simple(image=input_images, keep_prob=keep_prob, 
                                                     batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
    visual_conditions = tf.compat.v1.reshape(visual_conditions_reshaped, [BATCH_SIZE, SEQ_LEN, -1])
    visual_conditions = tf.compat.v1.nn.dropout(x=visual_conditions, keep_prob=keep_prob)    
    rnn_inputs_with_ground_truth = (visual_conditions, targets_normalized)
    rnn_inputs_autoregressive = (visual_conditions, tf.zeros(shape=(BATCH_SIZE, SEQ_LEN, OUTPUT_DIM), dtype=tf.float32))    
    internal_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=RNN_SIZE, num_proj=RNN_PROJ)
    cell_with_ground_truth = SamplingRNNCell(num_outputs=OUTPUT_DIM, use_ground_truth=True, internal_cell=internal_cell)
    cell_autoregressive = SamplingRNNCell(num_outputs=OUTPUT_DIM, use_ground_truth=False, internal_cell=internal_cell)    
    def get_initial_state(complex_state_tuple_sizes):
        flat_sizes = tf.nest.flatten(complex_state_tuple_sizes)
        init_state_flat = [tf.tile( 
            multiples=[BATCH_SIZE, 1], 
            input=tf.compat.v1.get_variable("controller_initial_state_%d" % i, initializer=tf.zeros_initializer, shape=([1, s]), dtype=tf.float32))
         for i,s in enumerate(flat_sizes)]
        init_state = tf.nest.pack_sequence_as(complex_state_tuple_sizes, init_state_flat)
        return init_state
    def deep_copy_initial_state(complex_state_tuple):
        flat_state = tf.nest.flatten(complex_state_tuple)
        flat_copy = [tf.identity(s) for s in flat_state]
        deep_copy = tf.nest.pack_sequence_as(complex_state_tuple, flat_copy)
        return deep_copy    
    controller_initial_state_variables = get_initial_state(cell_autoregressive.state_size)
    controller_initial_state_autoregressive = deep_copy_initial_state(controller_initial_state_variables)
    controller_initial_state_gt = deep_copy_initial_state(controller_initial_state_variables)
    with tf.compat.v1.variable_scope("predictor"):
        out_gt, controller_final_state_gt = tf.compat.v1.nn.dynamic_rnn(cell=cell_with_ground_truth, inputs=rnn_inputs_with_ground_truth, 
                          sequence_length=[SEQ_LEN]*BATCH_SIZE, initial_state=controller_initial_state_gt, dtype=tf.float32,
                          swap_memory=True, time_major=False)
    with tf.compat.v1.variable_scope("predictor", reuse=True):
        out_autoregressive, controller_final_state_autoregressive = tf.compat.v1.nn.dynamic_rnn(cell=cell_autoregressive, inputs=rnn_inputs_autoregressive, 
                          sequence_length=[SEQ_LEN]*BATCH_SIZE, initial_state=controller_initial_state_autoregressive, dtype=tf.float32,
                          swap_memory=True, time_major=False)    
    mse_gt = tf.reduce_mean(tf.math.squared_difference(out_gt, targets_normalized))
    mse_autoregressive = tf.reduce_mean(tf.math.squared_difference(out_autoregressive, targets_normalized))
    mse_autoregressive_steering = tf.reduce_mean(tf.math.squared_difference(out_autoregressive[:, :, 0], targets_normalized[:, :, 0]))
    steering_predictions = (out_autoregressive[:, :, 0] * std[0]) + mean[0]    
    total_loss = mse_autoregressive_steering + aux_cost_weight * (mse_gt + mse_autoregressive)    
    optimizer = get_optimizer(total_loss, learning_rate)
    tf.compat.v1.summary.scalar("MAIN TRAIN METRIC: rmse_autoregressive_steering", tf.math.sqrt(mse_autoregressive_steering))
    tf.compat.v1.summary.scalar("rmse_gt", tf.math.sqrt(mse_gt))
    tf.compat.v1.summary.scalar("rmse_autoregressive", tf.math.sqrt(mse_autoregressive))    
    summaries = tf.compat.v1.summary.merge_all()
    train_writer = tf.compat.v1.summary.FileWriter('v3/train_summary', graph=graph)
    valid_writer = tf.compat.v1.summary.FileWriter('v3/valid_summary', graph=graph)
    saver = tf.compat.v1.train.Saver(write_version=tf.compat.v1.train.SaverDef.V2)    
#ANCHOR - 7
#tf.config.gpu.set_per_process_memory_fraction(0.25)
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
#gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1)
checkpoint_dir = os.getcwd() + "/v3"
global_train_step = 0
global_valid_step = 0
KEEP_PROB_TRAIN = 0.25
def do_epoch(session, sequences, mode):
    global global_train_step, global_valid_step
    test_predictions = {}
    valid_predictions = {}
    batch_generator = BatchGenerator(sequence=sequences, seq_len=SEQ_LEN, batch_size=BATCH_SIZE)
    total_num_steps = int(1 + (batch_generator.indices[1] - 1) / SEQ_LEN)
    controller_final_state_gt_cur, controller_final_state_autoregressive_cur = None, None
    acc_loss = np.float128(0.0)
    for step in range(total_num_steps):
        feed_inputs, feed_targets = batch_generator.next()
        feed_dict = {inputs : feed_inputs, targets : feed_targets}
        if controller_final_state_autoregressive_cur is not None:
            feed_dict.update({controller_initial_state_autoregressive : controller_final_state_autoregressive_cur})
        if controller_final_state_gt_cur is not None:
            feed_dict.update({controller_final_state_gt : controller_final_state_gt_cur})
        if mode == "train":
            feed_dict.update({keep_prob : KEEP_PROB_TRAIN})
            summary, _, loss, controller_final_state_gt_cur, controller_final_state_autoregressive_cur = \
                session.run([summaries, optimizer, mse_autoregressive_steering, controller_final_state_gt, controller_final_state_autoregressive],
                           feed_dict = feed_dict)
            train_writer.add_summary(summary, global_train_step)
            global_train_step += 1
        elif mode == "valid":
            model_predictions, summary, loss, controller_final_state_autoregressive_cur = \
                session.run([steering_predictions, summaries, mse_autoregressive_steering, controller_final_state_autoregressive],
                           feed_dict = feed_dict)
            valid_writer.add_summary(summary, global_valid_step)
            global_valid_step += 1  
            feed_inputs = feed_inputs[:, LEFT_CONTEXT:].flatten()
            steering_targets = feed_targets[:, :, 0].flatten()
            model_predictions = model_predictions.flatten()
            stats = np.stack([steering_targets, model_predictions, (steering_targets - model_predictions)**2])
            for i, img in enumerate(feed_inputs):
                valid_predictions[img] = stats[:, i]
        elif mode == "test":
            model_predictions, controller_final_state_autoregressive_cur = \
                session.run([steering_predictions, controller_final_state_autoregressive],
                           feed_dict = feed_dict)           
            feed_inputs = feed_inputs[:, LEFT_CONTEXT:].flatten()
            model_predictions = model_predictions.flatten()
            for i, img in enumerate(feed_inputs):
                test_predictions[img] = model_predictions[i]
        if mode != "test":
            acc_loss += loss
            print('\n' + str(step + 1) + "/" + str(total_num_steps) + " - " + str(np.sqrt(acc_loss / (step+1))))
    return (np.sqrt(acc_loss / total_num_steps), valid_predictions) if mode != "test" else (None, test_predictions)
NUM_EPOCHS=5
best_validation_score = None
with tf.compat.v1.Session(graph=graph, config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as session:
    session.run(tf.compat.v1.initialize_all_variables())
    print('Initialized')
    ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    if ckpt:
        print("Restoring from" + str(ckpt))
        saver.restore(sess=session, save_path=ckpt)
    for epoch in range(NUM_EPOCHS):
        print ("Starting epoch %d", epoch)
        valid_score, valid_predictions = do_epoch(session=session, sequences=valid_seq, mode="valid")
        print ("Validation Score: " + str(valid_score))
        if best_validation_score is None: 
            best_validation_score = valid_score
        if valid_score < best_validation_score:
            saver.save(session, 'v3/checkpoint-sdc-ch2')
            best_validation_score = valid_score
            print("\n SAVED at epoch %d", epoch)
            with open("v3/valid-predictions-epoch%d" % epoch, "w") as out:
                result = np.float128(0.0)
                for img, stats in valid_predictions.items():
                    #print >> out, img, stats
                    print(out)
                    print(img)
                    print(stats)
                    result += stats[-1]
            print ("Validation unnormalized RMSE:" + str(np.sqrt(result / len(valid_predictions))))
            with open("v3/test-predictions-epoch%d" % epoch, "w") as out:
                _, test_predictions = do_epoch(session=session, sequences=test_seq, mode="test")
                #print >> out,
                print(out)
                print("frame_id,steering_angle")
                for img, pred in test_predictions.items():
                    img = img.replace(abs_path + "test_center/", "")
                    #print >> out, "%s,%f" % (img, pred)
                    print(out)
                    print(img)
                    print(pred)
        if epoch != NUM_EPOCHS - 1:
            print("Training")
            do_epoch(session=session, sequences=train_seq, mode="train")


"""
Yes, in addition to doing a restored_keras_model.summary(), you can save the model architecture as a png file using the plot_model API.
from keras.utils import plot_model
plot_model(restored_keras_model, to_file='model.png')
"""