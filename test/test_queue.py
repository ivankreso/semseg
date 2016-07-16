import tensorflow as tf

queue_size = 10
with tf.Graph().as_default():
  sess = tf.Session()
  queue = tf.FIFOQueue(capacity=queue_size, dtypes=tf.int32)
  enqueue_placeholder = tf.placeholder(dtype=tf.int32)
  enqueue_op = queue.enqueue(enqueue_placeholder)
  dequeue_op = queue.dequeue()
  for f in range(queue_size):
    sess.run([enqueue_op], feed_dict={enqueue_placeholder: f})
  sess.run(queue.close())

  dequeue_op = tf.reshape(dequeue_op, shape=[1])
  queue_batch = tf.train.batch([dequeue_op], batch_size=1, num_threads=1, capacity=64)

  #global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
  #                              trainable=False)
  #lr = tf.train.exponential_decay(0.01,
  #                                global_step,
  #                                5,
  #                                0.2,
  #                                staircase=True)
  #tf.scalar_summary('learning_rate', lr)
  #summary_op = tf.merge_all_summaries()  

  init = tf.initialize_all_variables()
  sess.run(init)

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  for i in range(queue_size):
    print(sess.run([queue_batch]))
    #print(sess.run([summary_op]))

  coord.request_stop()
  coord.join(threads, stop_grace_period_secs=5)
  sess.close()


#def create_session():
#  """Resets local session, returns new InteractiveSession"""
#
#  config = tf.ConfigProto(log_device_placement=True)
#  config.gpu_options.per_process_gpu_memory_fraction=0.3 # don't hog all vRAM
#  config.operation_timeout_in_ms=5000   # terminate on long hangs
#  sess = tf.InteractiveSession("", config=config)
#  return sess
#
#tf.reset_default_graph()
#q = tf.FIFOQueue(4, tf.string)
#enqueue_val = tf.placeholder(dtype=tf.string)
#enqueue_op = q.enqueue(enqueue_val)
#size_op = q.size()
#dequeue_op = q.dequeue()
#sess = create_session()
#def enqueueit(val):
#  sess.run([enqueue_op], feed_dict={enqueue_val:val})
#  print("queue1 size: ", sess.run(size_op))
#enqueueit("1")
#enqueueit("2")
#enqueueit("3")
##sess.run(q.close())
#
#dequeue_op.set_shape([])
#queue2 = tf.train.batch([dequeue_op], batch_size=1, num_threads=1, capacity=1)
#threads = tf.train.start_queue_runners()
#
#def dequeueit():
#  print("queue1 size: ", sess.run(size_op))
#  print("queue2 size before: ", sess.run("batch/fifo_queue_Size:0"))
#  print("result: ", sess.run(queue2))
#  print("queue2 size after: ", sess.run("batch/fifo_queue_Size:0"))
#
#coord = tf.train.Coordinator()
#dequeueit()
#dequeueit()
#dequeueit()
#coord.request_stop()
#coord.join(threads, stop_grace_period_secs=5)


#from datasets.cityscapes.cityscapes import CityscapesDataset
#from datasets.cityscapes.cityscapes_info import class_info, class_color_map
#import datasets.reader as reader

#FLAGS = tf.app.flags.FLAGS
#tf.app.flags.DEFINE_integer('img_width', 1024, '')
#tf.app.flags.DEFINE_integer('img_height', 432, '')
#tf.app.flags.DEFINE_integer('img_depth', 3, '')
#tf.app.flags.DEFINE_integer('batch_size', 1, '')
#
#dataset_dir = '/home/kivan/datasets/Cityscapes/tensorflow/records/1024x432/'
#dataset = CityscapesDataset(dataset_dir, 'train')
#queue_size = 10
#with tf.Graph().as_default():
#  sess = tf.Session()
#  #queue = tf.FIFOQueue(capacity=queue_size, dtypes=tf.int32)
#  #enqueue_placeholder = tf.placeholder(dtype=tf.int32)
#  #enqueue_op = queue.enqueue(enqueue_placeholder)
#  #dequeue_op = queue.dequeue()
#  #for f in range(queue_size):
#  #  sess.run([enqueue_op], feed_dict={enqueue_placeholder: f})
#  #sess.run(queue.close())
#  #dequeue_op = tf.reshape(dequeue_op, shape=[1])
#  #queue_batch = tf.train.batch([dequeue_op], batch_size=1, num_threads=1, capacity=64)
#
#  image, labels, weights, num_labels, img_name = reader.inputs_single_epoch(sess, dataset)
#
#  coord = tf.train.Coordinator()
#  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#
#  for i in range(dataset.num_examples()):
#    print(sess.run([img_name]))
#    #print(sess.run([queue_batch]))
#
#  coord.request_stop()
#  coord.join(threads, stop_grace_period_secs=5)
#  sess.close()


