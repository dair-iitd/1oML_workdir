#! /usr/bin/env python3
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implement solution generator for n queens."""
from IPython.core.debugger import Pdb
import jacinle.random as random
import numpy as np
import copy
__all__ = ['NQueenSolution', 'randomly_generate_family']


def get_xy(a,n):
    row = a//n
    col = a % n
    return (row,col)

class NQueenSolution(object):
    def __init__(self):
        self.solutions = []
        self.relations = {}

    def reset(self):
        self.solutions = []

    def get_relations(self,n):
        #
        #Pdb().set_trace()
        if n in self.relations:
            return self.relations[n]
        #
        board_size  = n*n
        rows = np.zeros((board_size, board_size))
        cols = np.zeros((board_size, board_size))
        diagonals = np.zeros((board_size, board_size))
        off_diagonals = np.zeros((board_size,board_size))
        for i in range(board_size):
            for j in range(board_size):
                row1, col1 = get_xy(i,n)
                row2, col2  = get_xy(j,n)
                if row1 == row2:
                    rows[i,j] = 1
                if col1 == col2:
                    cols[i,j] = 1
                if (row2 - row1) == (col2 - col1):
                    diagonals[i,j] = 1
                if (row2 - row1) == (col1 - col2):
                    off_diagonals[i,j] = 1

        self.relations[n] =  np.stack([rows,cols,diagonals,off_diagonals]).swapaxes(0,2)
        return self.relations[n]

    def solve(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
        grid = np.zeros((n,n))
        solved = self.helper(n, 0, grid)
        print (len(self.solutions))
        #if solved:
        #    return ["".join(item) for item in grid]
        #else:
        #    return None

    def helper(self, n, row, grid):
        if n == row:
            self.solutions.append((copy.deepcopy(grid),n))
            return
        for col in range(n):
            if self.is_safe(row, col, grid):
                grid[row][col] = 1
                self.helper(n, row + 1, grid)
                if len(self.solutions) >= 1000:
                    return
                grid[row][col] = 0
                
    def is_safe(self, row, col, board):
        for i in range(len(board)):
            if board[row][i] == 1 or board[i][col] == 1:
                return False
        i = 0
        while row - i >= 0 and col - i >= 0:
            if board[row - i][col - i] == 1:
                return False
            i += 1
        i = 0
        while row + i < len(board) and col + i < len(board):
            if board[row + i][col - i] == 1:
                return False
            i += 1
        i = 1
        while row + i < len(board) and col - i >= 0:
            if board[row+ i][col - i] == 1:
                return False
            i += 1
        i = 1
        while row - i >= 0 and col + i < len(board):
            if board[row - i][col + i] == 1:
                return False
            i += 1
        return True
        
"""
n=10
s = Solution()
s.solveNQueens(n)


x = []
y = []
np.random.shuffle(s.solutions)
for sol in s.solutions:
    for i in range(n):
        for j in range(n):
            if sol[i][j]==1:
                sol[i][j]=0
                x.append(copy.deepcopy(sol))
                sol[i][j]=1
                y.append(sol)

"""

