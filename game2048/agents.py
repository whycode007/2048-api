import numpy as np
import torch
import model
import torchvision.transforms as transforms
import torch.nn as nn
import var_model



def load_checkpoint(model, checkpoint_PATH):
    if checkpoint_PATH != None:
        model_CKPT = torch.load(checkpoint_PATH,map_location=torch.device('cpu'))
        model.load_state_dict(model_CKPT['state_dict'])
        print('loading checkpoint!')
    return model

class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction



class myAgent(Agent): #for the first data_process
    def __init__(self,game,display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        self.model = load_checkpoint(var_model.Model(),'colab/epoch9.pth.tar')

    def transform_board(self):
        board = self.game.board.copy()

        new_board = []
        for row in range(board.shape[0]):
            for col in range(board.shape[1]):
                if board[row][col]==0:
                    new_board.append(board[row][col])
                else:
                    new_board.append(np.log2(board[row][col]))
        sample = np.array(new_board)
        new_board = np.expand_dims(sample, axis=0) / 11.0
        new_board = torch.from_numpy(new_board)
        return new_board


    def step(self):
        new_board = self.transform_board()
        out = self.model(new_board.float())
        print(out)
        direction = torch.max(out,1)[1].data.numpy().squeeze()
        print(direction)
        return direction


