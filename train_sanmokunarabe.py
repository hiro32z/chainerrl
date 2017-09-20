# -*- coding: utf-8 -*-
"""
三目並べ強化学習

メモ：
黒石（先行）と白石（後攻）で別々のエージェントを学習させる．
"""
from __future__ import print_function
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import numpy as np
import sys

# 定数の定義
BOARD_SIZE = 3 # 盤面サイズ 3x3
NONE = 0   # 盤面にある石 なし
BLACK = 1  # 盤面にある石 黒
WHITE = 2  # 盤面にある石 白
STONE = [' ', '●', '○'] # 石の表示用

# Q-関数の定義
class QFunction(chainer.Chain):
    def __init__(self, obs_size, n_actions, n_hidden_channels=128):
        super(QFunction, self).__init__(
            l0=L.Linear(obs_size, n_hidden_channels),
            l1=L.Linear(n_hidden_channels, n_hidden_channels),
            l2=L.Linear(n_hidden_channels, n_hidden_channels),
            l3=L.Linear(n_hidden_channels, n_actions))

    def __call__(self, x, test=False):        
        h = F.leaky_relu(self.l0(x))
        h = F.leaky_relu(self.l1(h))
        h = F.leaky_relu(self.l2(h))
        return chainerrl.action_value.DiscreteActionValue(self.l3(h))

# 碁盤クラス
class GoBoard():
    # インスタンス
    def __init__(self):
        self.board_reset()

    # 盤面のリセット
    def board_reset(self):
        self.board = np.array([NONE] * (BOARD_SIZE * BOARD_SIZE), dtype=np.float32) # 碁盤は1次元配列で定義 (1x9)        
        self.winner = NONE # 勝者
        self.turn = BLACK  # 最初は黒のターン
        self.game_end = False # ゲーム終了チェックフラグ
        self.miss_end = False # エージェントのミス打ち判定フラグ        

    # 白黒のターンチェンジ
    def change_turn(self):
        self.turn = WHITE if self.turn == BLACK else BLACK

    # エージェントのアクションと勝敗判定．置けない場所に置いたら負け．
    def agent_action(self, pos):
        if self.board[pos] == NONE:
            self.board[pos] = self.turn
            self.end_check()
        else:                        
            self.winner = WHITE if self.turn == BLACK else BLACK
            self.miss_end = True
            self.game_end = True
    
    # ランダム
    def random_action(self):        
        return self.find_empty_positions()

    # 空いているマスを見つけて，座標をランダムに1つ選択する
    def find_empty_positions(self):
        pos = np.where(self.board == 0)[0]        
        if len(pos) > 0:
            return np.random.choice(pos) # 空いている場所の座標の1つをランダム返す
        else:
            return 0 # 空きなし     

    # 盤面を表示する
    def show_board(self):
        print('  ', end='')            
        for l in range(1, BOARD_SIZE + 1):
            print(' {}'.format(l), end='')
        print('')
        row = 1
        print('{0:2d} '.format(row), end='')
        row += 1
        for i in range(0, BOARD_SIZE * BOARD_SIZE):
            if i != 0 and i % BOARD_SIZE == 0: # 1行表示したら改行
                print('')
                print('{0:2d} '.format(row), end='')
                row += 1
            if self.board[i] == 0:
                ix = 0
            elif self.board[i] == 1:
                ix = 1
            else:
                ix = 2
            print('{} '.format(STONE[ix]), end='')
        print('')

    # ゲームの終了チェック
    def end_check(self):
        for i in range(0, BOARD_SIZE * BOARD_SIZE):
            self.winner = self.conjunction_check(i)            
            if self.winner != 0:
                self.game_end = True # 3連ができたらゲーム終了
                break
        if np.count_nonzero(self.board) == BOARD_SIZE * BOARD_SIZE: # 碁盤がすべて埋まった場合ゲーム終了
            self.game_end = True

    # 座標(line, row)から3連接のチェック
    def conjunction_check(self, pos):
        # 石の有無チェック
        if self.board[pos] == NONE:
            return 0 # 石がなければ0を返す
        # 縦方向のチェック
        if pos + (BOARD_SIZE * 2) < BOARD_SIZE * BOARD_SIZE:
            if self.board[pos] == self.board[pos+BOARD_SIZE] == self.board[pos+(BOARD_SIZE*2)]:
                return self.board[pos] # 縦3連が存在
        # 斜め（右下）方向のチェック
        if pos + ((BOARD_SIZE + 1) * 2) < BOARD_SIZE * BOARD_SIZE:
            if self.board[pos] == self.board[pos+BOARD_SIZE+1] == self.board[pos+((BOARD_SIZE+1)*2)]:
                return self.board[pos] # 右下斜め3連が存在
        # 斜め（左下）方向のチェック
        if pos + ((BOARD_SIZE - 1) * 2) < BOARD_SIZE * BOARD_SIZE:
            if ((pos + (BOARD_SIZE - 1) * 2) // BOARD_SIZE) - (pos // BOARD_SIZE) >= 2:
                if self.board[pos] == self.board[pos+BOARD_SIZE-1] == self.board[pos+((BOARD_SIZE-1)*2)]:
                    return self.board[pos] # 左下斜め3連が存在
        # 横方向チェック
        if pos // BOARD_SIZE == (pos + 2) // BOARD_SIZE: # 先頭と末尾が同じ行かどうか
            if self.board[pos] == self.board[pos+1] == self.board[pos+2]:
                return self.board[pos] # 横3連が存在
        return 0 # 3連は存在せずの場合は-1を返す          

# メイン関数            
def main():
    
    board = GoBoard() # 碁盤の初期化    

    obs_size = BOARD_SIZE * BOARD_SIZE # 現在の碁盤の状況
    n_actions = BOARD_SIZE * BOARD_SIZE # 行動数は9（3x3マスのどこに石を置くか）
    q_func = QFunction(obs_size, n_actions)

    # Use Adam to optimize q_func. eps=1e-2 is for stability.
    optimizer = chainer.optimizers.Adam(eps=1e-2)
    optimizer.setup(q_func)

    # Set the discount factor that discounts future rewards.
    gamma = 0.95

    # Use epsilon-greedy for exploration
    explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(start_epsilon=1.0, end_epsilon=0.1, decay_steps=10000, random_action_func=board.random_action)
    
    # DQN uses Experience Replay.
    # Specify a replay buffer and its capacity.
    replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)

    # Now create agents that will interact with the environment.
    agent_black = chainerrl.agents.DQN(
        q_func, optimizer, replay_buffer, gamma, explorer,
        replay_start_size=500, update_interval=1, target_update_interval=100)
    agent_white = chainerrl.agents.DQN(
        q_func, optimizer, replay_buffer, gamma, explorer,
        replay_start_size=500, update_interval=1, target_update_interval=100)
    agents = ['', agent_black, agent_white]

    #学習ゲーム回数
    n_episodes = 50000
    win = 0
    lose = 0
    draw = 0
    miss_counts = [0, 0, 0] # ミス打ちを行った数

    #エピソードの繰り返し実行
    for i in range(1, n_episodes + 1):
        board.board_reset()
        rewards = [0, 0, 0]    
        board.turn = BLACK # 黒石が先行        

        while not board.game_end:            
            # 石を配置する場所を取得            
            pos = agents[board.turn].act_and_train(board.board.copy(), rewards[board.turn])
            # 石を配置            
            board.agent_action(pos)

            # ゲームが終了したら            
            if board.game_end:                                
                if board.miss_end:
                    miss_counts[board.turn] += 1
                if board.winner == BLACK:
                    rewards[BLACK] = 1
                    rewards[WHITE] = -1
                    win += 1
                elif board.winner == 0:
                    draw += 1                    
                else:
                    rewards[BLACK] = -1
                    rewards[WHITE] = 1
                    lose += 1                    
                #エピソードを終了して学習                
                if not board.miss_end:                       
                    # 勝者のエージェントの学習
                    agents[board.turn].stop_episode_and_train(board.board.copy(), rewards[board.turn], True)
                    board.change_turn()
                    # 敗者のエージェントの学習
                    agents[board.turn].stop_episode_and_train(board.board.copy(), rewards[board.turn], True) 
                else:
                    # ミス打ちの場合1つ前の状態に戻す
                    agents[board.turn].stop_episode_and_train(prev, rewards[board.turn], False)
            else:
                prev = board.board.copy()                
                board.change_turn()

        # 学習の進捗表示
        if i % 100 == 0:            
            print('==== Episode {} : win {}, lose {}, draw {} ===='.format(i, win, lose, draw)) # 勝敗数は黒石基準
            print('<BLACK> miss: {}, statistics: {}, eplilon {}'.format(miss_counts[BLACK], agent_black.get_statistics(), agent_black.explorer.epsilon))
            print('<WHITE> miss: {}, statistics: {}, eplilon {}'.format(miss_counts[WHITE], agent_white.get_statistics(), agent_white.explorer.epsilon))
            # カウンタ変数の初期化            
            win = 0
            draw = 0
            lose = 0
            miss_counts = [0, 0, 0] # ミス打ちをやってしまった数
        
        if i % 10000 == 0:
            # 10000エピソードごとにモデルを保存する
            agent_black.save("result_black_" + str(i))
            agent_white.save("result_white_" + str(i))
        
    print("学習終了.\n")

if __name__ == '__main__':
    main()
