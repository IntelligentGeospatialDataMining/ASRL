
import argparse
import os
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorlayer as tl



#####################  hyper parameters  ####################

LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 10000  # size of replay buffer
BATCH_SIZE = 35  # update batchsize


###############################  DDPG  ####################################

class DDPG(object):
    """
    DDPG class
    """

    def __init__(self, a_dim, s_dim, a_bound):
        # memory are used for data storage
        # MEMORY_CAPACITY，s_dim * 2 + a_dim + 1：
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound

        W_init = tf.random_normal_initializer(mean=0, stddev=0.1)
        b_init = tf.constant_initializer(0.1)

        # W_init = tf.random_normal_initializer(mean=0, stddev=0.01)
        # b_init = tf.constant_initializer(0.01)




        # build actor net ,output s and a
        def get_actor(input_state_shape, name=''):
            """
            Build actor network
            :param input_state_shape: state
            :param name: name
            :return: act
            """
            inputs = tl.layers.Input(input_state_shape, name='A_input')
            x = tl.layers.Dense(n_units=30, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='A_l1')(inputs)
            x = tl.layers.Dense(n_units=a_dim, act=tf.nn.sigmoid, W_init=W_init, b_init=b_init, name='A_a')(x)
            # x=tl.layers.Dense(n_units=a_dim,W_init=W_init, b_init=b_init, name='A_a')(x)
            x = tl.layers.Lambda(lambda x: np.array(a_bound) * x)(x)
            return tl.models.Model(inputs=inputs, outputs=x, name='Actor' + name)

        # build critic net ,input s and a,output Q value
        def get_critic(input_state_shape, input_action_shape, name=''):
            """
            Build critic network
            :param input_state_shape: state
            :param input_action_shape: act
            :param name: name
            :return: Q value Q(s,a)
            """
            s = tl.layers.Input(input_state_shape, name='C_s_input')
            a = tl.layers.Input(input_action_shape, name='C_a_input')
            x = tl.layers.Concat(1)([s, a])
            x = tl.layers.Dense(n_units=60, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='C_l1')(x)
            x = tl.layers.Dense(n_units=1, W_init=W_init, b_init=b_init, name='C_out')(x)
            return tl.models.Model(inputs=[s, a], outputs=x, name='Critic' + name)

        self.actor = get_actor([None, s_dim])
        self.critic = get_critic([None, s_dim], [None, a_dim])
        self.actor.train()
        self.critic.train()

        # update parameters,used for first time
        def copy_para(from_model, to_model):
            """
            Copy parameters for soft updating
            :param from_model: latest model
            :param to_model: target model
            :return: None
            """
            for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
                j.assign(i)

        # build actor_target net,parameters are the same with actor,not participating in training
        self.actor_target = get_actor([None, s_dim], name='_target')
        copy_para(self.actor, self.actor_target)
        self.actor_target.eval()

        # build critic_target net,parameters are the same with critic,not participating in training
        self.critic_target = get_critic([None, s_dim], [None, a_dim], name='_target')
        copy_para(self.critic, self.critic_target)
        self.critic_target.eval()

        self.R = tl.layers.Input([None, 1], tf.float32, 'r')

        # Build ema, moving average
        self.ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # soft replacement

        self.actor_opt = tf.optimizers.Adam(LR_A)
        self.critic_opt = tf.optimizers.Adam(LR_C)

    def ema_update(self):
        """
        update
        """

        paras = self.actor.trainable_weights + self.critic.trainable_weights  # Get the parameters to be updated including actor and critic
        self.ema.apply(paras)
        for i, j in zip(self.actor_target.trainable_weights + self.critic_target.trainable_weights, paras):
            i.assign(self.ema.average(j))  # Assignment with moving average

    # choose action.input state and output action
    def choose_action(self, s):
        """
        Choose action
        :param s: state
        :return: act
        """
        return self.actor(np.array([s], dtype=np.float32))[0]

    def learn(self):
        """
        Update parameters
        :return: None
        """
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)  # random BATCH_SIZE index
        bt = self.memory[indices, :]  # choose data based on index
        bs = bt[:, :self.s_dim]  # state s
        ba = bt[:, self.s_dim:self.s_dim + self.a_dim]  # action
        br = bt[:, -self.s_dim - 1:-self.s_dim]  # reward
        bs_ = bt[:, -self.s_dim:]  # state s'
        # print("bs:",bs)
        # print("ba:",ba)
        # print("br:", br)
        # print("bs_:", bs_)
        # Critic：
        # br + GAMMA * q_
        with tf.GradientTape() as tape:
            a_ = self.actor_target(bs_)
            q_ = self.critic_target([bs_, a_])
            y = br + GAMMA * q_
            q = self.critic([bs, ba])
            # print(q)
            td_error =tf.losses.mean_squared_error(y, q)
        c_grads = tape.gradient(td_error, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(c_grads, self.critic.trainable_weights))

        # Actor：
        with tf.GradientTape() as tape:
            a = self.actor(bs)
            q = self.critic([bs, a])
            a_loss =- tf.reduce_mean(q)  # Use -loss for larger reward values
        a_grads = tape.gradient(a_loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(a_grads, self.actor.trainable_weights))

        self.ema_update()

    # store s，a，r，s_
    def store_transition(self, s, a, r, s_):
        """
        Store data in data buffer
        :param s: state
        :param a: act
        :param r: reward
        :param s_: next state
        :return: None
        """

        s = s.astype(np.float32)
        s_ = s_.astype(np.float32)


        transition = np.hstack((s, a, [r], s_))

        # pointer record the total num of memory
        # index record current data index
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        # store transition
        self.memory[index, :] = transition
        self.pointer += 1

    def save_ckpt(self,string):
        """
        save trained weights
        :return: None
        """
        if not os.path.exists(''):
            os.makedirs('')
        tl.files.save_weights_to_hdf5('model/ddpg_actor-' + string + '.hdf5', self.actor)
        tl.files.save_weights_to_hdf5('model/ddpg_actor_target-' + string + '.hdf5', self.actor_target)
        tl.files.save_weights_to_hdf5('model/ddpg_critic-' + string + '.hdf5', self.critic)
        tl.files.save_weights_to_hdf5('model/ddpg_critic_target-' + string + '.hdf5', self.critic_target)

    def load_ckpt(self,string):
        """
        load trained weights
        :return: None
        """

        tl.files.load_hdf5_to_weights_in_order('model/ddpg_actor-' + string + '.hdf5', self.actor)
        tl.files.load_hdf5_to_weights_in_order('model/ddpg_actor_target-' + string + '.hdf5', self.actor_target)
        tl.files.load_hdf5_to_weights_in_order('model/ddpg_critic-' + string + '.hdf5', self.critic)
        tl.files.load_hdf5_to_weights_in_order('model/ddpg_critic_target-' + string + '.hdf5', self.critic_target)



