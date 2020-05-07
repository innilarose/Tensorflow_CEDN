import numpy as np
import tensorflow as tf
from model_contour import build_model
from utils import random_crop_and_pad_image_and_labels
import os
import cv2
import time
from tensorflow.python.ops import variables
slim = tf.contrib.slim
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('max_to_keep', 50,
                     'Maximium number of checkpoints to be saved.')

#flags.DEFINE_float('learning_power', 0.9,
                   'The power value used in the poly learning policy.')

flags.DEFINE_integer('Epochs', 10,
                     'The number of steps used for training')

#flags.DEFINE_float('momentum', 0.9, 'The momentum value to use')

flags.DEFINE_integer('train_crop_size', 480 ,
                           'Image crop size [height, width] during training.')

flags.DEFINE_string('tf_initial_checkpoint', None,
                    'The initial checkpoint in tensorflow format.')

flags.DEFINE_float('learning_rate', .0000001,
                   'Learning rate employed during slow start.')

flags.DEFINE_string('image_dir', None,
                    'The Image Directory.')

flags.DEFINE_string('label_dir', None,
                    'The Label Directory.')

flags.DEFINE_string('log_dir', None,
                    'The Logs Directory.')

#flags.DEFINE_float('clip_by_value', 1.0, 'The value to be used for clipping.')

 
flags.DEFINE_string('train_text', None,
                    'The Path to the text file containing names of Images and Labels')###This text file should not have extensions in their names such as 8192.png or 8192.jpg instead just the name such as 8192

Image_directory = FLAGS.image_dir
print(Image_directory)
Label_directory = FLAGS.label_dir
print(Label_directory)
my_log_dir = FLAGS.log_dir

def save(saver, sess, logdir, step):

   model_name = 'model.ckpt'
   checkpoint_path = os.path.join(logdir, model_name)
    
   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step)
   print('The checkpoint has been created.')

def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))
       
def main(unused_argv):    
    
    image_ph = tf.placeholder(tf.uint8,[1,None,None,3],name='image_placeholder')
    label_ph = tf.placeholder(tf.uint8,[1,None,None,1],name='label_placeholder')
    size = FLAGS.train_crop_size
    image,label=random_crop_and_pad_image_and_labels(tf.squeeze(image_ph),tf.squeeze(label_ph,axis=0),size,size)
    norm_image = tf.image.per_image_standardization(tf.squeeze(image))
    norm_image = tf.expand_dims(norm_image,dim=0)
    print(norm_image)
    print(label_ph)

    pred = build_model(norm_image)
    one_hot_labels = slim.one_hot_encoding(
    tf.cast(label,dtype=tf.uint8), 1, on_value=1.0, off_value=0.0)
    print(type(pred))
    total_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(tf.squeeze(one_hot_labels),dtype=tf.float32),logits =tf.squeeze(pred) )
    total_loss = tf.reduce_mean(total_loss)
    all_trainables = tf.trainable_variables()
    total_loss_scalar = tf.summary.scalar("total_cost", total_loss)
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=FLAGS.max_to_keep)
    train_summary_op = tf.summary.merge([total_loss_scalar])
    train_writer = tf.summary.FileWriter(my_log_dir+'/train',
                                        graph=tf.get_default_graph())
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)
    #train_op = optimizer.minimize(total_loss)  
    #grads = tf.gradients(total_loss, all_trainables)
    #grads_and_vars = zip(grads, all_trainables)
    #grads_and_vars = tf.compute_gradients(total_loss)
    #grads= tf.apply_gradients(grads_and_vars)
    train_op = optimizer.minimize(total_loss)
    #if FLAGS.clip_by_value:                     
        #clipped_value=[(tf.clip_by_value(grad, -FLAGS.clip_by_value, +FLAGS.clip_by_value), var) for grad, var in grads_and_vars]
        #train_op = optimizer.apply_gradients(clipped_value)
    #else:                
        #train_op = optimizer.apply_gradients(grads_and_vars)
    init = variables.global_variables_initializer()
    #opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
    #config=tf.ConfigProto(gpu_options=opts)
    with tf.Session() as sess:
        sess.run(init)
        if FLAGS.tf_initial_checkpoint==True:
                         load(loader, sess, FLAGS.tf_initial_checkpoint)
        print('Training Starts........')
        step_iter = 0
        final_time=0
        final_loss=0
        for epoch in range(FLAGS.Epochs):
            time_total=0
            loss_total=0
            save(saver, sess, my_log_dir, step_iter)
            i=0;
            f = open(FLAGS.train_text,'r')
            message = f.read()
            lines = message.split('\n')
            #print(lines)
            for l in lines:
                #print(l)
                step_iter =step_iter+1
                i=i+1;
                try :  
                    l=l.strip()
                    print(Image_directory+'/'+l+'.png')
                    input_image = cv2.imread(Image_directory+'/'+l+'.png')
                    #print("Input Read")
                    #print(input_image)
                    #cv2.imwrite("O:\\innila\\Dataset_Main\\Data_test\\Trial_original.png",input_image)
                    labs_person = cv2.imread(Label_directory+'/'+l+'.png',0)
                    #print("Label Read")
                    #print(labs_person)
                    #cv2.imwrite("O:\\innila\\Dataset_Main\\Data_test\\Label.png",labs_person)
                    labs_person = labs_person/255.0
                    labs_person = np.expand_dims(labs_person,axis=0)
                    labs_person = np.expand_dims(labs_person,axis=3)
                    input_image = np.expand_dims(input_image,axis=0)
                    feed_dict={image_ph:input_image,label_ph:labs_person}
                    start_time = time.time()
                    L,P,_,sum_op = sess.run([total_loss,pred,train_op,train_summary_op], feed_dict=feed_dict)
                    train_writer.add_summary(sum_op, step_iter)     
                    duration = time.time() - start_time
                    time_total+=duration
                    loss_total+=L
                    print('::Step::'+str(epoch)+','+str(i),'::total_loss::'+ str(L),'::time::'+str(duration))
                except:
                    print("Error")
            final_time+=time_total
            final_loss+=loss_total
            print("Total Time Duration:: {}".format(time_total))
            time_avg = time_total/200
            loss_avg = loss_total/200
            print("Average Time Duration:: {}".format(time_avg))
            print("Total Loss:: {}".format(loss_avg))
        acc,acc_op = tf.metrics.accuracy(labels=tf.squeeze(one_hot_labels),predictions=tf.squeeze(pred))
        accuracy = sess.run(acc)
        #matches = tf.equal(tf.argmax(tf.squeeze(preds), 1), tf.argmax(tf.squeeze(one_hot_labels), 1))
        #acc,acc_op = tf.reduce_mean(tf.cast(matches, tf.float32))
        # On valid/test set.
        #train_accuracy = sess.run(acc,feed_dict = {image_ph:input_image,label_ph:labs_person, keep_prob : 1.0})
        print("Accuracy: %f" %(accuracy))
        #train_accuracy = acc.eval()
        f_time = final_time
        #f_loss = final_loss/10
        print("Overall Time Duration for 10 Epochs:: {}".format(f_time))
        #print("Overall Loss for 10 Epochs:: {}".format(f_loss)) 
        
        

if __name__ == '__main__':
  flags.mark_flag_as_required('image_dir')
  flags.mark_flag_as_required('label_dir')
  flags.mark_flag_as_required('log_dir')
  flags.mark_flag_as_required('train_text')
  tf.app.run()

cv2.waitKey(0)
cv2.DestroyAllWindows()