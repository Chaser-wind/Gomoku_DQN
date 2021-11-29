from numpy.lib.npyio import load
from gomoku import Gomoku
import numpy as np
from dqn import DeepQNetwork
import ctypes

def run(load_episode = None, rival = None):
    if load_episode != None and load_episode != 0:
        RL.load_model(load_episode)
        RL.learn_step_counter = load_episode
    else:
        load_episode = 0
    step = 0
    for episode in range(load_episode, load_episode + 5000):
        print('episode:', episode)
        RL.epsilon = RL.epsilon_max * (1 - 0.7 ** (episode / 800 + 1))
        game.reset()
        s = game.get_state()
        game_step = 0
        if rival != None:
            rival.reset()
        # lasts = None
        # last_action_index = None
        # lastr = None
        elen = 0
        while True:
            action_index = RL.choose_action(s)
            s_, r, done = game.step(action_index)
            elen += 1
            #print('DQN play:')
            #print(game.grid[1] - game.grid[0])
            if rival != None:
                if done:
                    RL.store_memory(s.reshape([-1, ]), s_.reshape([-1, ]), action_index, r)
                    if hasattr(RL, 'memory_counter') and RL.memory_counter > 8 and step % 5 == 0:
                        RL.learn()
                    print('DQN win, len: %d' % (elen))
                    break
                else:
                    rival.play(int(action_index // RL.gsize), int(action_index % RL.gsize))
                    rival.go()
                    rx, ry = rival.getx(), rival.gety()
                    rival.play(rx, ry)
                    ns, nr, rd = game.step(rx * RL.gsize + ry)
                    elen += 1
                    #print('AI play:')
                    #print(game.grid[1] - game.grid[0])
                    #print(r - nr)
                    RL.store_memory(s.reshape([-1, ]), ns.reshape([-1, ]), action_index, r - nr)
                    if hasattr(RL, 'memory_counter') and RL.memory_counter > 8 and step % 5 == 0:
                        RL.learn()
                    s = ns
                    if rd:
                        print('AI win,  len: %d' % (elen))
                        break
            else:
                RL.store_memory(s.reshape([-1, ]), s_.reshape([-1, ]), action_index, r)
                
                if hasattr(RL, 'memory_counter') and RL.memory_counter > 32 and step % 5 == 0:
                    RL.learn()

                
                if done:
                    print('len: %d' % (elen))
                    if r != 1:
                        print('Tie')
                    break

                s = s_ 

            #game.turn()
            step += 1
            game_step += 1

        if (1 + episode) % 200 == 0:
            RL.save_model(episode + 1)
            print('model saved!')
    #print('game over')
    # RL.save_model(episode + 1)
    # print('model saved!')
    RL.writer.close()


if __name__ == '__main__':
    gsize = 15
    line = 5
    last_learn = 0
    game = Gomoku(gsize, line)
    RL = DeepQNetwork(n_actions=game.n_actions,
                      n_features=game.n_features,
                      memory_size=10000,
                      batch_size=128,
                      train_epochs=20,
                      gsize=gsize,
                      last_learn_step=last_learn,
                      logdir='log_{}x{}_{}'.format(gsize, gsize, line),
                      modeldir='data_{}x{}_{}'.format(gsize, gsize, line))
    run(last_learn, rival=None)
    
    # s = [99, 100, 101, 102, 103, 104, 99, 101]
    # for g in s:
    #     game.step(g)
    #     print(np.reshape(game.get_state(), [15, 15]))
    #     game.turn()
    #print(score_list)
    #np.savetxt('score_list.txt', np.array(score_list))
