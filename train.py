#!/home/tima/.pyenv/shims/python3
# coding=utf-8
# PYTHON_ARGCOMPLETE_OK
import torch
from tqdm import tqdm

from Agent import DQNAgent
from tetris import Tetris
from tetris_master import TetrisMaster2


if __name__ == '__main__':
    import os
    from argparse import ArgumentParser
    import argcomplete

    parser = ArgumentParser()

    game_group = parser.add_argument_group('Game', 'Params that control game behavior')
    game_group.add_argument('--width', type=int, default=10,
                            help='Width of game board. Default = %(default)d.')
    game_group.add_argument('--hight', type=int, default=20,
                            help='Hight of game board. Default = %(default)d.')
    game_group.add_argument('--speed', type=int, default=1,
                            help='Game speed (number of ticks between piece falling 1 down). Default = %(default)d.')
    game_group.add_argument('--train', action='store_true', default=False,
                            help='Train mode (I piece and hole for it). Default = %(default)d.')

    model_group = parser.add_argument_group('Model', 'Model params')
    model_group.add_argument('--depth', type=int, default=8,
                             help='Depth of convolutions. Default = %(default)d.')
    model_group.add_argument('--hidden-num', type=int, default=32,
                             help='Number of hidden layers. Default = %(default)d.')
    model_group.add_argument('--hidden-size', type=int, default=100,
                             help='Number of features in hidden layer. Default = %(default)d.')

    agent_group = parser.add_argument_group('Agent', 'Agent params')
    agent_group.add_argument('--gamma', type=float, default=0.99,
                             help='. Default = %(default).2e.')
    agent_group.add_argument('--betta', type=float, default=0.3,
                             help='. Default = %(default).2e.')
    agent_group.add_argument('--epsilon', type=float, default=1.0,
                             help='Chance to get random choice of action ' +
                             '(helps with initial learning). Default = %(default).2e.')
    agent_group.add_argument('--epsilon-min', type=float, default=0.1,
                             help='Minimum chance to get random choice of action. Default = %(default).2e.')
    agent_group.add_argument('--epsilon-decay', type=float, default=0.999,
                             help='Decay of epsilon on each iteration. Default = %(default).2e.')

    learning_group = parser.add_argument_group('Learning', 'Learning params')
    learning_group.add_argument('--epoches', type=int, default=100,
                                help='Total number of epoches to run. Default = %(default)d.')
    learning_group.add_argument('--train-episodes', type=int, default=100,
                                help='Number of train episodes in each epoch. Default = %(default)d.')
    learning_group.add_argument('--test-episodes', type=int, default=5,
                                help='Number of test episodes in each epoch. Default = %(default)d.')
    learning_group.add_argument('--lr', type=float, default=1e-5,
                                help='Learning rate. Default = %(default).2e.')
    learning_group.add_argument('--game-len', type=int, default=200,
                                help='Maximum lenght of the game in ticks. Default = %(default)d.')
    learning_group.add_argument('--batch-size', type=int, default=32,
                                help='Size of learning batch. Default = %(default)d.')

    appearance_group = parser.add_argument_group('Appearance', 'Appearance and verbose params')
    appearance_group.add_argument('--rf', '--round-factor', type=int, default=4,
                                  help='Round factor for verbosing. Default = %(default)d.')

    saving_group = parser.add_argument_group('Save/Load', 'Saving and loading params')
    saving_group.add_argument('--checkpoint', type=str,
                              help='Name of checkpoint to load. Default = %(default)s.')
    saving_group.add_argument('--checkpoint-path', type=str, default='checkpoints',
                              help='Path to checkpoints dir. Default = %(default)s.')
    saving_group.add_argument('--llc', '--load-last-checkpoint', action='store_true', default=False,
                              help='Load last saved checkpoint. Default = %(default)r.')

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    arg_gr = {}

    for g in parser._action_groups:
        gd = {a.dest: getattr(args, a.dest, None) for a in g._group_actions}
        arg_gr[g.title] = gd

    # Initialize the game
    env = Tetris(**arg_gr['Game'])

    # Initialize model
    model = TetrisMaster2(**arg_gr['Model']).double()

    if arg_gr['Save/Load']['checkpoint'] or arg_gr['Save/Load']['llc']:
        if arg_gr['Save/Load']['llc']:
            cp_path = os.sep.join([arg_gr['Save/Load']['checkpoint_path'],
                                   'test_model'])
            arg_gr['Save/Load']['checkpoint'] = cp_path
        cp = torch.load(arg_gr['Save/Load']['checkpoint'])
        model.load_state_dict(cp['model_state_dict'])

    # Initialize the agent
    state_dim = (env.board.shape, 1)
    action_dim = 4
    agent = DQNAgent(model, state_dim, action_dim,
                     learning_rate=arg_gr['Learning']['lr'],
                     **arg_gr['Agent'])

    for e in range(arg_gr['Learning']['epoches']):
        # torch.autograd.set_detect_anomaly(True)

        # Training the DQN agent
        train_iter = tqdm(range(arg_gr['Learning']['train_episodes']))
        model.train()
        for episode in train_iter:
            with torch.no_grad():
                state = env.reset()
                total_reward = 0
                done = False
                iters = 0
                while iters < arg_gr['Learning']['game_len'] and not done:
                    action = agent.select_action(state)
                    next_state, reward, done = env.step(action)
                    agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    total_reward = total_reward + reward
                    iters += 1
            loss = agent.replay(arg_gr['Learning']['batch_size'])
            train_iter.set_postfix(str=f"loss {loss.item():.5}, " + f"{iters=}, " +
                           f"rwd: {round(total_reward, arg_gr['Appearance']['rf'])}, " +
                           f"eps {round(agent.epsilon, arg_gr['Appearance']['rf'])}")
        train_iter.close()

        # Evaluate the trained agent
        test_iter = tqdm(range(arg_gr['Learning']['test_episodes']))
        model.eval()
        with torch.no_grad():
            for _ in test_iter:
                state = env.reset()
                total_reward = 0
                done = False
                iters = 0
                # while iters < arg_gr['Learning']['game_len'] and not done:
                while iters < 100 and not done:
                    action = agent.select_action(state, test=True)
                    next_state, reward, done = env.step(action)
                    state = next_state
                    total_reward += reward
                    iters += 1
                test_iter.set_postfix(str=f"Test Episode, loss {round(loss.item(), arg_gr['Appearance']['rf'])}, " +
                            f"Total Reward: {round(total_reward, arg_gr['Appearance']['rf'])}")
        test_iter.close()

        # print('saving')
        try:
            torch.save({
                'model_state_dict': agent.online_net.state_dict(),
            }, os.sep.join([arg_gr['Save/Load']['checkpoint_path'], 'test_model']))
        except KeyboardInterrupt:
            torch.save({
                'model_state_dict': agent.online_net.state_dict(),
            }, os.sep.join([arg_gr['Save/Load']['checkpoint_path'], 'test_model']))

        it.close()
