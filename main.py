from Forza4 import Forza4
import os

# Clearing the Screen


game = Forza4()
print(game)

player = 1

while True:

    print("Player " + str(player))

    col = input("Enter column [0-6]: ")
    correctly_inserted = game.insert_coin(player=player, col=int(col))

    while not correctly_inserted:
        print("Not possible to insert coin there, retry ")
        col = input("Enter column [0-6]: ")
        correctly_inserted = game.insert_coin(player=player, col=int(col))

    win = game.check_victory(player)

    os.system('cls')
    print(game)

    if win:
        finished = True
        print("Player "+str(player)+" WIN")
        break

    if game.check_tie():
        finished = True
        print("IT'S TIE!")
        break

    if player == 2:
        player = 1
    else:
        player = 2


