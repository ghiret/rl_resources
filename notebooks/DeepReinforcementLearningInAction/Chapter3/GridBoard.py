import numpy as np


def randPair(s, e):
    """
    Generate a random pair of integers within a specified range.

    Args:
    ----
        s: Start of range (inclusive)
        e: End of range (exclusive)

    Returns:
    -------
        tuple: A pair of random integers (x, y) where s <= x,y < e

    """
    return np.random.randint(s, e), np.random.randint(s, e)


class BoardPiece:
    def __init__(self, name, code, pos):
        """
        Initialize a board piece.

        Args:
        ----
            name: Name of the piece
            code: ASCII character to display on the board
            pos: 2-tuple position (x, y) on the board

        """
        self.name = name  # name of the piece
        self.code = code  # an ASCII character to display on the board
        self.pos = pos  # 2-tuple e.g. (1,4)


class BoardMask:
    def __init__(self, name, mask, code):
        """
        Initialize a board mask.

        Args:
        ----
            name: Name of the mask
            mask: 2D numpy array with 1s where mask elements are present
            code: ASCII character to display for masked positions

        """
        self.name = name
        self.mask = mask
        self.code = code

    def get_positions(self):  # returns tuple of arrays
        return np.nonzero(self.mask)


def zip_positions2d(positions):
    """
    Convert a tuple of two arrays into a list of coordinate pairs.

    Args:
    ----
        positions: A tuple of two arrays (x_coords, y_coords)

    Returns:
    -------
        list: List of (x, y) coordinate pairs

    """
    x, y = positions
    return list(zip(x, y, strict=False))


class GridBoard:
    def __init__(self, size=4):
        """
        Initialize a grid board.

        Args:
        ----
            size: Board dimensions (size x size), defaults to 4

        """
        self.size = size  # Board dimensions, e.g. 4 x 4
        self.components = {}  # name : board piece
        self.masks = {}

    def addPiece(self, name, code, pos=(0, 0)):
        newPiece = BoardPiece(name, code, pos)
        self.components[name] = newPiece

    # basically a set of boundary elements
    def addMask(self, name, mask, code):
        # mask is a 2D-numpy array with 1s where the boundary elements are
        newMask = BoardMask(name, mask, code)
        self.masks[name] = newMask

    def movePiece(self, name, pos):
        move = True
        for _, mask in self.masks.items():
            if pos in zip_positions2d(mask.get_positions()):
                move = False
        if move:
            self.components[name].pos = pos

    def delPiece(self, name):
        del self.components[name]

    def render(self):
        dtype = "<U2"
        displ_board = np.zeros((self.size, self.size), dtype=dtype)
        displ_board[:] = " "

        for _, piece in self.components.items():
            displ_board[piece.pos] = piece.code

        for _, mask in self.masks.items():
            displ_board[mask.get_positions()] = mask.code

        return displ_board

    def render_np(self):
        num_pieces = len(self.components) + len(self.masks)
        displ_board = np.zeros((num_pieces, self.size, self.size), dtype=np.uint8)
        layer = 0
        for _, piece in self.components.items():
            pos = (layer,) + piece.pos
            displ_board[pos] = 1
            layer += 1

        for _, _ in self.masks.items():
            x, y = self.masks["boundary"].get_positions()
            z = np.repeat(layer, len(x))
            a = (z, x, y)
            displ_board[a] = 1
            layer += 1
        return displ_board

    def copy(self):
        """
        Create a deep copy of the GridBoard.

        Returns
        -------
            GridBoard: A new GridBoard instance with copied components and masks

        """
        # Create a new board with the same size
        new_board = GridBoard(size=self.size)

        # Copy all components (pieces)
        for name, piece in self.components.items():
            new_board.addPiece(name, piece.code, pos=piece.pos)

        # Copy all masks
        for name, mask in self.masks.items():
            # Create a copy of the mask array
            new_mask = mask.mask.copy()
            new_board.addMask(name, new_mask, mask.code)

        return new_board


def addTuple(a, b):
    """
    Add two tuples element-wise.

    Args:
    ----
        a: First tuple
        b: Second tuple of same length as a

    Returns:
    -------
        tuple: Element-wise sum of the input tuples

    """
    return tuple([sum(x) for x in zip(a, b, strict=False)])
