from Forza4 import Forza4
import os

# Clearing the Screen


game = Forza4()

finished = False
player = 1

while not finished:
    print(game)
    print("Player " + str(player))

    col = input("Enter column [0-6]: ")
    ok = game.insert_coin(player=player, col=int(col))

    if not ok:
        finished = True

    if game.check_win():
        break


    if player == 2:
        player = 1
    else:
        player = 2

    os.system('cls')

