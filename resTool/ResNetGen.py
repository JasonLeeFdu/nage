import time
from resTool.opst import *
from resTool.utils import *

class ResNet(object):
    def __init__(self, sess, args):
        self.model_name = 'ResNet'
        self.sess = sess
        self.dataset_name = args.dataset

        if self.dataset_name == 'cifar10' :
            self.train_x, self.train_y, self.test_x, self.test_y = load_cifar10()
            self.img_size = 32
            self.c_dim = 3
            self.label_dim = 10

        if self.dataset_name == 'cifar100' :
            self.train_x, self.train_y, self.test_x, self.test_y = load_cifar100()
            self.img_size = 32
            self.c_dim = 3
            self.label_dim = 100

        if self.dataset_name == 'mnist' :
            self.train_x, self.train_y, self.test_x, self.test_y = load_mnist()
            self.img_size = 28
            self.c_dim = 1
            self.label_dim = 10

        if self.dataset_name == 'fashion-mnist' :
            self.train_x, self.train_y, self.test_x, self.test_y = load_fashion()
            self.img_size = 28
            self.c_dim = 1
            self.label_dim = 10

        if self.dataset_name == 'tiny' :
            self.train_x, self.train_y, self.test_x, self.test_y = load_tiny()
            self.img_size = 64
            self.c_dim = 3
            self.label_dim = 200


        self.checkpoint_dir = args.checkpoint_dir
        self.log_dir = args.log_dir

        self.res_n = args.res_n

        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.iteration = len(self.train_x) // self.batch_size

        self.init_lr = args.lr


    ##################################################################################
    # Generator
    ##################################################################################

    def network(self, x, is_training=True, reuse=False):
        with tf.variable_scope("Resnet", reuse=reuse):

            if self.res_n < 50 :
                residual_block = resblock
            else :
                residual_block = bottle_resblock

            residual_list = get_residual_layer(self.res_n)

            ch = 32 # paper is 64
            x = conv(x, channels=ch, kernel=3, stride=1, scope='conv')

            for i in range(residual_list[0]) :
                x = residual_block(x, channels=ch, is_training=is_training, downsample=False, scope='resblock0_' + str(i))

            ########################################################################################################

            x = residual_block(x, channels=ch*2, is_training=is_training, downsample=True, scope='resblock1_0')

            for i in range(1, residual_list[1]) :
                x = residual_block(x, channels=ch*2, is_training=is_training, downsample=False, scope='resblock1_' + str(i))

            ########################################################################################################

            x = residual_block(x, channels=ch*4, is_training=is_training, downsample=True, scope='resblock2_0')

            for i in range(1, residual_list[2]) :
                x = residual_block(x, channels=ch*4, is_training=is_training, downsample=False, scope='resblock2_' + str(i))

            ########################################################################################################

            x = residual_block(x, channels=ch*8, is_training=is_training, downsample=True, scope='resblock_3_0')

            for i in range(1, residual_list[3]) :
                x = residual_block(x, channels=ch*8, is_training=is_training, downsample=False, scope='resblock_3_' + str(i))

            ########################################################################################################


            x = batch_norm(x, is_training, scope='batch_norm')
            x = relu(x)

            x = global_avg_pooling(x)
            x = fully_conneted(x, units=self.label_dim, scope='logit')

            return x

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ Graph Input """
        self.train_inptus = tf.placeholder(tf.float32, [self.batch_size, self.img_size, self.img_size, self.c_dim], name='train_inputs')
        self.train_labels = tf.placeholder(tf.float32, [self.batch_size, self.label_dim], name='train_labels')

        self.test_inptus = tf.placeholder(tf.float32, [len(self.test_x), self.img_size, self.img_size, self.c_dim], name='test_inputs')
        self.test_labels = tf.placeholder(tf.float32, [len(self.test_y), self.label_dim], name='test_labels')

        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        """ Model """
        self.train_logits = self.network(self.train_inptus)
        self.test_logits = self.network(self.test_inptus, is_training=False, reuse=True)

        self.train_loss, self.train_accuracy = classification_loss(logit=self.train_logits, label=self.train_labels)
        self.test_loss, self.test_accuracy = classification_loss(logit=self.test_logits, label=self.test_labels)
        
        reg_loss = tf.losses.get_regularization_loss()
        self.train_loss += reg_loss
        self.test_loss += reg_loss


        """ Training """
        self.optim = tf.train.MomentumOptimizer(self.lr, momentum=0.9).minimize(self.train_loss)

        """" Summary """
        self.summary_train_loss = tf.summary.scalar("train_loss", self.train_loss)
        self.summary_train_accuracy = tf.summary.scalar("train_accuracy", self.train_accuracy)

        self.summary_test_loss = tf.summary.scalar("test_loss", self.test_loss)
        self.summary_test_accuracy = tf.summary.scalar("test_accuracy", self.test_accuracy)

        self.train_summary = tf.summary.merge([self.summary_train_loss, self.summary_train_accuracy])
        self.test_summary = tf.summary.merge([self.summary_test_loss, self.summary_test_accuracy])

    ##################################################################################
    # Train
    ##################################################################################

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            epoch_lr = self.init_lr
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter

            if start_epoch >= int(self.epoch * 0.75) :
                epoch_lr = epoch_lr * 0.01
            elif start_epoch >= int(self.epoch * 0.5) and start_epoch < int(self.epoch * 0.75) :
                epoch_lr = epoch_lr * 0.1
            print(" [*] Load SUCCESS")
        else:
            epoch_lr = self.init_lr
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):
            if epoch == int(self.epoch * 0.5) or epoch == int(self.epoch * 0.75) :
                epoch_lr = epoch_lr * 0.1

            # get batch data
            for idx in range(start_batch_id, self.iteration):
                batch_x = self.train_x[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_y = self.train_y[idx*self.batch_size:(idx+1)*self.batch_size]

                batch_x = data_augmentation(batch_x, self.img_size, self.dataset_name)

                train_feed_dict = {
                    self.train_inptus : batch_x,
                    self.train_labels : batch_y,
                    self.lr : epoch_lr
                }

                test_feed_dict = {
                    self.test_inptus : self.test_x,
                    self.test_labels : self.test_y
                }


                # update network
                _, summary_str, train_loss, train_accuracy = self.sess.run(
                    [self.optim, self.train_summary, self.train_loss, self.train_accuracy], feed_dict=train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # test
                summary_str, test_loss, test_accuracy = self.sess.run(
                    [self.test_summary, self.test_loss, self.test_accuracy], feed_dict=test_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%5d/%5d] time: %4.4f, train_accuracy: %.2f, test_accuracy: %.2f, learning_rate : %.4f" \
                      % (epoch, idx, self.iteration, time.time() - start_time, train_accuracy, test_accuracy, epoch_lr))

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        return "{}{}_{}_{}_{}".format(self.model_name, self.res_n, self.dataset_name, self.batch_size, self.init_lr)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test(self):
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        test_feed_dict = {
            self.test_inptus: self.test_x,
            self.test_labels: self.test_y
        }


        test_accuracy = self.sess.run(self.test_accuracy, feed_dict=test_feed_dict)
        print("test_accuracy: {}".format(test_accuracy))

def ConvLayer(inp,h,w,inc,outc,training,padding='SAME',strides=[1,1,1,1],name='Conv2d'):
    with tf.name_scope(name):
        weight = tf.Variable(tf.truncated_normal([h,w,inc,outc],mean=0,stddev=1e-3),name='weight')
        bias   = tf.Variable(tf.truncated_normal([outc],mean=0,stddev=1e-8),name='bias')
        out    = tf.nn.conv2d(inp,weight,padding=padding,name='conv',strides=strides) + bias
        out    = tf.layers.batch_normalization(out,training=training)
        out    = tf.nn.relu(out)
    return out

def getRes18(inp, num_class, is_training=True, reuse=False):
    with tf.variable_scope("Resnet", reuse=reuse):
        residual_block = resblock
        residual_list = get_residual_layer(18)

        ch = 32
        tmp = conv(inp, channels=ch, kernel=3, stride=1, scope='conv')

        L224 = tmp


        for i in range(residual_list[0]) :
            tmp = residual_block(tmp, channels=ch, is_training=is_training, downsample=False, scope='resblock0_' + str(i))

        ########################################################################################################

        L112 = residual_block(tmp, channels=ch*2, is_training=is_training, downsample=True, scope='resblock1_0')
        tmp = L112
        for i in range(1, residual_list[1]) :
            tmp = residual_block(tmp, channels=ch*2, is_training=is_training, downsample=False, scope='resblock1_' + str(i))

        ########################################################################################################

        L56 = residual_block(tmp, channels=ch*4, is_training=is_training, downsample=True, scope='resblock2_0')
        tmp = L56
        for i in range(1, residual_list[2]) :
            tmp = residual_block(tmp, channels=ch*4, is_training=is_training, downsample=False, scope='resblock2_' + str(i))



        '''改进二 最后一段接短点 
        ########################################################################################################

        L28 = residual_block(tmp, channels=ch*8, is_training=is_training, downsample=True, scope='resblock_3_0')
        x1 = batch_norm(L28, is_training, scope='batch_normx1')
        x2 = tf.nn.leaky_relu(x1)
        x2 = conv(x2, channels=ch*16,kernel=3,scope='resblock_3_' + str(i))

        ########################################################################################################

        x3 = batch_norm(x2, is_training, scope='batch_normx3')
        x3 = tf.nn.leaky_relu(x3)
        x3 = global_avg_pooling(x3)
        logitMid = dropout(x3, is_training)
        logitFinal = fully_conneted(logitMid, units=num_class, scope='logitFinal')
        '''

        ''' 改进一：中间直接解出来'''
        L28 = residual_block(tmp, channels=ch * 8, is_training=is_training, downsample=True, scope='resblock_3_0')

        c1 = conv(L56,ch * 4,scope='appendc1')
        c1 = batch_norm(c1,is_training,scope='appendc1')
        c1 = relu(c1)
        c1 = tf.nn.max_pool(c1,[1,2,2,1],[1,2,2,1],'SAME')
        c2 = conv(c1, ch * 8,scope='appendc2')
        c2 = batch_norm(c2, is_training,scope='appendc2')
        c2 = relu(c2)
        x3 = global_avg_pooling(c2)
        logitMid = dropout(x3, is_training)
        logitFinal = fully_conneted(logitMid, units=num_class, scope='logitFinal')


        ''' 原始设置 -- 目前最佳 周五
        ########################################################################################################

        L28 = residual_block(tmp, channels=ch*8, is_training=is_training, downsample=True, scope='resblock_3_0')
        tmp = L28
        tmp = resblockSpecial1(tmp, channels=ch*16, is_training=is_training, downsample=True, scope='resblock_3_' + str(i))

        ######################################################################################################## 

        x1 = batch_norm(tmp, is_training, scope='batch_norm')
        x2 = tf.nn.leaky_relu(x1)
        x3 = global_avg_pooling(x2)
        logitMid = dropout(x3, is_training)
        logitFinal = fully_conneted(logitMid, units=num_class, scope='logitFinal')
        '''

        '''改进三：
        L28 = residual_block(tmp, channels=ch*8, is_training=is_training, downsample=True, scope='resblock_3_0')
        c1  = ConvLayer(L28,3,3,256,512,is_training,strides=[1,2,2,1])
        c1  = tf.nn.max_pool(c1,[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        c2  = ConvLayer(c1,7,7,512,1024,is_training,strides=[1,2,2,1],padding='VALID')
        logitMid = dropout(c2, is_training)
        logitFinal = fully_conneted(logitMid, units=num_class, scope='logitFinal')
        ##
        '''


        endPoints = dict()
        endPoints['L224']  = L224
        endPoints['L112']  = L112
        endPoints['L56']  = L56
        endPoints['L28']  = L28
        endPoints['logitFinal']  = logitFinal

        return endPoints


def getRes18Dig(inp, num_class, choice,is_training=True, reuse=False):
    with tf.variable_scope("Resnet", reuse=reuse):
        residual_block = resblock
        residual_list = get_residual_layer(18)

        ch = 32
        tmp = conv(inp, channels=ch, kernel=3, stride=1, scope='conv')

        L224 = tmp


        for i in range(residual_list[0]) :
            tmp = residual_block(tmp, channels=ch, is_training=is_training, downsample=False, scope='resblock0_' + str(i))

        ########################################################################################################

        L112 = residual_block(tmp, channels=ch*2, is_training=is_training, downsample=True, scope='resblock1_0')
        tmp = L112
        for i in range(1, residual_list[1]) :
            tmp = residual_block(tmp, channels=ch*2, is_training=is_training, downsample=False, scope='resblock1_' + str(i))

        ########################################################################################################

        L56 = residual_block(tmp, channels=ch*4, is_training=is_training, downsample=True, scope='resblock2_0')
        tmp = L56
        for i in range(1, residual_list[2]) :
            tmp = residual_block(tmp, channels=ch*4, is_training=is_training, downsample=False, scope='resblock2_' + str(i))


        if choice == 1:
            '''改进二 最后一段接短点 '''
            ########################################################################################################

            L28 = residual_block(tmp, channels=ch*8, is_training=is_training, downsample=True, scope='resblock_3_0')
            x1 = batch_norm(L28, is_training, scope='batch_normx1')
            x2 = tf.nn.leaky_relu(x1)
            x2 = conv(x2, channels=ch*16,kernel=3,scope='resblock_3_' + str(i))

            ########################################################################################################

            x3 = batch_norm(x2, is_training, scope='batch_normx3')
            x3 = tf.nn.leaky_relu(x3)
            x3 = global_avg_pooling(x3)
            logitMid = dropout(x3, is_training)
            logitFinal = fully_conneted(logitMid, units=num_class, scope='logitFinal')

        elif choice == 2:
            ''' 改进一：中间直接解出来'''
            L28 = residual_block(tmp, channels=ch * 8, is_training=is_training, downsample=True, scope='resblock_3_0')

            c1 = conv(L56, ch * 4, scope='appendc1')
            c1 = batch_norm(c1, is_training, scope='appendc1')
            c1 = relu(c1)
            c1 = tf.nn.max_pool(c1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
            c2 = conv(c1, ch * 8, scope='appendc2')
            c2 = batch_norm(c2, is_training, scope='appendc2')
            c2 = relu(c2)
            x3 = global_avg_pooling(c2)
            logitMid = dropout(x3, is_training)
            logitFinal = fully_conneted(logitMid, units=num_class, scope='logitFinal')
        elif choice == 3:
            ''' 原始设置 -- 目前最佳 周五'''
            ########################################################################################################

            L28 = residual_block(tmp, channels=ch*8, is_training=is_training, downsample=True, scope='resblock_3_0')
            tmp = L28
            tmp = resblockSpecial1(tmp, channels=ch*16, is_training=is_training, downsample=True, scope='resblock_3_' + str(i))

            ########################################################################################################

            x1 = batch_norm(tmp, is_training, scope='batch_norm')
            x2 = tf.nn.leaky_relu(x1)
            x3 = global_avg_pooling(x2)
            logitMid = dropout(x3, is_training)
            logitFinal = fully_conneted(logitMid, units=num_class, scope='logitFinal')

        elif choice == 4:
            '''改进三：'''
            L28 = residual_block(tmp, channels=ch*8, is_training=is_training, downsample=True, scope='resblock_3_0')
            c1  = ConvLayer(L28,3,3,256,512,is_training,strides=[1,2,2,1])
            c1  = tf.nn.max_pool(c1,[1,2,2,1],strides=[1,2,2,1],padding='SAME')
            c2  = ConvLayer(c1,7,7,512,1024,is_training,strides=[1,2,2,1],padding='VALID')
            logitMid = dropout(c2, is_training)
            logitFinal = fully_conneted(logitMid, units=num_class, scope='logitFinal')
            ##
        endPoints = dict()
        endPoints['L224']  = L224
        endPoints['L112']  = L112
        endPoints['L56']  = L56
        endPoints['L28']  = L28
        endPoints['logitFinal']  = logitFinal

        return endPoints


def getRes18Cls(inp, num_class, is_training=True):
    residual_block = resblock
    residual_list = get_residual_layer(18)
    ch = 32
    x = conv(inp, channels=ch, kernel=3, stride=1, scope='conv')

    for i in range(residual_list[0]):
        x = residual_block(x, channels=32, is_training=is_training, downsample=False, scope='resblock0_' + str(i))

    ########################################################################################################

    x = residual_block(x, channels= 64, is_training=is_training, downsample=True, scope='resblock1_0')

    for i in range(1, residual_list[1]):
        x = residual_block(x, channels=64, is_training=is_training, downsample=False, scope='resblock1_' + str(i))

    ########################################################################################################

    x = residual_block(x, channels=128, is_training=is_training, downsample=True, scope='resblock2_0')

    for i in range(1, residual_list[2]):
        x = residual_block(x, channels=128, is_training=is_training, downsample=False, scope='resblock2_' + str(i))

    ########################################################################################################

    x = residual_block(x, channels=256, is_training=is_training, downsample=True, scope='resblock_3_0')

    ########################################################################################################

    x = batch_norm(x, is_training, scope='batch_norm1')
    x = relu(x)
    x = conv(x,channels=512,kernel=3)
    x = batch_norm(x, is_training, scope='batch_norm2')
    x = relu(x)
    x = global_avg_pooling(x)
    x = dropout(x,is_training)
    x = fully_conneted(x, units=num_class, scope='logit')

    return x