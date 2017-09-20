# -*- coding: utf-8 -*-
"""
三目並べエージェントのテスト
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

    obs_size = BOARD_SIZE * BOARD_SIZE
    n_actions = BOARD_SIZE * BOARD_SIZE
    q_func = QFunction(obs_size, n_actions)

    # Use Adam to optimize q_func. eps=1e-2 is for stability.
    optimizer = chainer.optimizers.Adam(eps=1e-2)
    optimizer.setup(q_func)
    q_func.to_cpu()

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
    
    agent_black.load('result_black_1000')
    agent_white.load('result_white_1000')

    #cmd = input('先制（黒石, 1） or 後攻（白石, 2）を選択：')
    #cmd = int(cmd)
    #assert(cmd == 1 or cmd == 2)
    print('あなたは「○」（後攻）です。ゲームスタート！')
    board.board_reset()     
    while not board.game_end:
        pos = agent_black.act(board.board)
        board.agent_action(pos)
        print('エージェントの番 -> {}'.format(pos))
        board.show_board()
        if board.game_end:
            if board.miss_end:
                print('エージェントの打ち間違い！')
            elif board.winner == BLACK:
                print('あなたの負け！')
            else:
                print('引き分け') 
            board.board_reset()
            continue

        board.change_turn() 
        pos = raw_input('どこに石を置きますか？ (行列で指定。例 "1 2")：')        
        pos = pos.split(' ')        
        # 1次元の座標に変換する
        pos = (int(pos[0]) - 1) * BOARD_SIZE + (int(pos[1]) - 1)        
        board.agent_action(pos)
        board.show_board()
        if board.game_end:
            if board.winner == WHITE:
                print('あなたの勝ち！')
            elif board.winner == BLACK:
                print('あなたの負け！')
            else:
                print('引き分け')
            board.board_reset()
            continue

        board.change_turn()  

if __name__ == '__main__':
    main()
