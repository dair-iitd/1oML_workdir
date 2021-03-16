from difflogic.dataset.graph.nqueens import NQueenSolution
from difflogic.thutils import is_safe
import torch

solver = NQueenSolution()
solver.solve(10)

solutions = solver.solutions
a = solutions[0][0]
#a[2,:]=0
#a[2,3]=1

print(is_safe(torch.Tensor(a.flatten())))
