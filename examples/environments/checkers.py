class Piece:
    def __init__(self):
        self.position = (None, None)


class Player:
    def __init__(self):
        self.num_pieces = 12
        self.pieces = [Piece() for _ in range(self.num_pieces)]

    def available_actions(self):
        return None

    def move(self, piece, action):
        r"""
        Actions:
        0: forward left
        1: forward right
        2: backward right
        3: backward left
        """
        pass


class Checkers:
    # TODO: Issue of sparse rewards?
    # NOTE: Maybe add better reward signal

    def __init__(self):
        # TODO: Accept grid size as input?
        self.players = [Player() for _ in range(2)]

    @property
    def state(self):
        board = [[None for _ in range(8)] for _ in range(8)]
        for player in self.players:
            for piece in player.pieces:
                # Add pieces by location for each player
                a, b = piece.position
                board[a, b] = "N"
        return board

    def step(self, player_id, piece, location):
        self.players[player_id].move(piece, location)
