import chess
from pprint import pprint

board = chess.Board()


board.push(chess.Move.from_uci("e2e4"))
board.push(chess.Move.from_uci("e7e5"))
board.push(chess.Move.from_uci("g1f3"))
board.push(chess.Move.from_uci("g7g5"))

a = board.__str__()
print(a)

boardArray = []

temp = []
for i in a:
    if i != ' ' and i != '\n' and i != None:
        # print(i)
        if len(temp) < 7:
            temp.append(i)
        else:
            temp.append(i)
            boardArray.append(temp)
            temp = []


pprint(boardArray)
