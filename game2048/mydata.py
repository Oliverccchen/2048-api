import numpy as np
from game2048.game import Game
from game2048.agents import ExpectiMaxAgent
from game2048.agents import MyAgent
from game2048.displays import Display
import csv
import os

game_size = 4
score_to_win = 2048
iter_num = 3000

game = Game(game_size, score_to_win)
board = game.board
agenta = ExpectiMaxAgent(game, Display())
agentb = MyAgent(game, Display())
directiona = agenta.step()
directionb = agentb.step()
board = game.move(directionb)

i = 0
dic = {}
idx = 0

# save file
filename = '/home/olivia/PycharmProjects/2048/game2048/data/traindata10.csv'
if os.path.exists(filename):
    start = True
else:
    start = False
    os.mknod(filename)

with open(filename, "a") as csvfile:
    writer = csv.writer(csvfile)
    if not start:
        writer.writerow(["R1C1", "R1C2", "R1C3", "R1C4", \
                         "R2C1", "R2C2", "R2C3", "R2C4", \
                         "R3C1", "R3C2", "R3C3", "R3C4", \
                         "R4C1", "R4C2", "R4C3", "R4C4", \
                         "direction"])

    while i < iter_num:
        game = Game(game_size, score_to_win)
        board = game.board
        print('Iter idx:', i)

        while(game.end == 0):
            agenta = ExpectiMaxAgent(game, Display())
            directiona = agenta.step()
            agentb = MyAgent(game, Display())
            directionb = agentb.step()
            board = game.board
            board[board == 0] = 1
            board = np.log2(board).flatten()

            data = np.int32(np.append(board, directiona))
            writer.writerow(data)

            idx = idx + 1
            if idx % 200 == 0:
                print(data)
                idx = 0

            game.move(directionb)

        i = i + 1


