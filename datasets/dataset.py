import os

class Dataset(object):
  def __init__(self, data_dir, subset):
    self.subset = subset
    self.data_dir = os.path.join(data_dir, subset)
    files = next(os.walk(self.data_dir))[2]
    self.filenames = [os.path.join(self.data_dir, f) for f in files]
    #self.filenames = [self.filenames[i] for i in range(30)]


  def num_classes(self):
    return self.num_classes

  def num_examples(self):
    return len(self.filenames)

  def get_filenames(self):
    return self.filenames

  #def enqueue(self, sess, enqueue_op, placeholder):
  #  for f in self.filenames:
  #    sess.run([enqueue_op], feed_dict={placeholder: f})
  #  #for i in range(10):
  #  #  sess.run([enqueue_op], feed_dict={placeholder: self.filenames[i]})
  #  #sess.run([enqueue_op], feed_dict={placeholder: self.filenames[0]})

