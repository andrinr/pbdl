import torch

class Grid:
    def __init__(self, shape : torch.Size) -> None:
        self.shape = shape
        self.grid = torch.zeros(shape, dtype=torch.float32)

    def diff(self, dim : int) -> torch.Tensor:
        return torch.diff(self.grid, dim=dim, n=1, append=0)

    def for_diff2(self, dim : int) -> torch.Tensor:
        return torch.diff(self.grid, dim=dim, n=2, append=0)
    
    def central_diff(self, dim : int) -> torch.Tensor:
        return torch.diff(self.grid, dim=dim, n=1, prepend=0, append=0) / 2.0
    