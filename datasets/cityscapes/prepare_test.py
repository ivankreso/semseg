




def save_predictions(sess, image, logits, softmax):
  width = FLAGS.img_width
  height = FLAGS.img_height
  img_dir = FLAGS.dataset_dir
  cities = next(os.walk(img_dir))[1]
  for city in cities:
    city_dir = join(img_dir, city)
    image_list = next(os.walk(city_dir))[2]
    print(city)
    for i in trange(len(image_list)):
      img = ski.data.load(img_dir + image_list[i])
      mask_path = join(GT_DIR, city, img_prefix + '_gtFine_labelIds.png')
      #print(gt_path)
      orig_gt_img = ski.data.load(gt_path)

      img_data = img.reshape(1, height, width, 3)
      out_logits, out_softmax = sess.run([logits, softmax], feed_dict={image : img_data})
      y = out_logits[0].argmax(2).astype(np.int32)
      p = np.amax(out_softmax, axis=2)
      #print('Over 90% = ', (p > 0.9).sum() / p.size)
      #print(p)
      eval_helper.draw_output(y, Dataset.CLASS_INFO, os.path.join(FLAGS.save_dir, image_list[i]))
      save_path = os.path.join(FLAGS.save_dir, 'softmax_' + image_list[i])
      ski.io.imsave(save_path, p)
