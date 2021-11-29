from tkinter import *
from gomoku import Gomoku
from dqn import DeepQNetwork
import sys

gsize = 15
line = 5
game = Gomoku(gsize, line)
RL = DeepQNetwork(n_actions=game.n_actions,
                      n_features=game.n_features,
                      memory_size=10000,
                      batch_size=128,
                      train_epochs=20,
                      gsize=gsize,
                      use_cuda=False,
                      logdir=None,
                      modeldir='data_{}x{}_{}'.format(gsize, gsize, line))
mid = int(sys.argv[1]) if len(sys.argv) > 1 else None
if mid != None:
    RL.load_model(mid)

grid_width = 80
r = 30
piece_color = ['black', 'white']  
person_flag = 0         
gamedone = False
PVP = False

window = Tk()
window.title("Gomoku")
canvas = Canvas(window, bg = "SandyBrown", width = grid_width * (gsize + 1), height = grid_width * (gsize + 1))
canvas.grid(row = 0, column = 0, rowspan = 10)
var = StringVar()
for i in range(gsize):
    canvas.create_line(grid_width, (grid_width * i + grid_width), grid_width * gsize, (grid_width * i + grid_width))
    canvas.create_line((grid_width * i + grid_width), grid_width, (grid_width * i + grid_width), grid_width * gsize)

def mouseBack(event):     
    
    global person_flag, piece_color, gamedone, var, PVP
    if gamedone:
        return
    i = round(event.x / grid_width)
    j = round(event.y / grid_width)
    if not game.act_legal([i - 1, j - 1]):
        return
    state, _, done = game.step((i - 1) * gsize + j - 1)
    canvas.create_oval(i * grid_width - r, j * grid_width - r,
                        i * grid_width + r, j * grid_width + r, 
                        fill = piece_color[person_flag], tags = ("piece"))
    person_flag = 1 - person_flag
    if done:
        gamedone = True
        var.set('Win!')
        return 
    
    if not PVP:
        action = RL.choose_action(state)
        state, _, done = game.step(action)
        i, j = action // gsize + 1, action % gsize + 1
        canvas.create_oval(i * grid_width - r, j * grid_width - r,
                            i * grid_width + r, j * grid_width + r, 
                            fill = piece_color[person_flag], tags = ("piece"))
        #print('AI place on [{},{}]'.format(i - 1, j - 1))
        person_flag = 1 - person_flag
        if done:
            var.set('Lose!')
            #print(game.grid[1] - game.grid[0])
            gamedone = True
    else:
        piece_canvas = Canvas(window, width = 200, height = 200)
        piece_canvas.grid(row = 0, column = 1)
        piece_canvas.create_oval(100 - r, 100 - r,100 + r, 100 + r,fill = piece_color[person_flag])

canvas.bind("<Button-1>",mouseBack) 

def click_resetPVP():
    global person_flag, gamedone, var, PVP
    PVP = True
    gamedone = False
    person_flag = 0
    canvas.delete("piece")
    game.reset()
    var.set("Playing")
    piece_canvas = Canvas(window, width = 200, height = 200)
    piece_canvas.grid(row = 0, column = 1)
    piece_canvas.create_oval(100 - r, 100 - r,100 + r, 100 + r,fill = 'black')

def click_resetB():
    global person_flag, gamedone, var, PVP
    PVP = False
    gamedone = False
    person_flag = 0
    canvas.delete("piece")
    game.reset()
    var.set("Playing")
    piece_canvas = Canvas(window, width = 200, height = 200)
    piece_canvas.grid(row = 0, column = 1)
    piece_canvas.create_oval(100 - r, 100 - r,100 + r, 100 + r,fill = 'black')

def click_resetW():
    global person_flag, gamedone, var, PVP
    PVP = False
    gamedone = False
    canvas.delete("piece")
    game.reset()
    action = RL.choose_action(game.get_state())
    game.step(action)
    i, j = action // gsize + 1, action % gsize + 1 
    canvas.create_oval(i * grid_width - r, j * grid_width - r,
                        i * grid_width + r, j * grid_width + r, 
                        fill = 'black', tags = ("piece"))
    person_flag = 1
    var.set("Playing")
    piece_canvas = Canvas(window, width = 200, height = 200)
    piece_canvas.grid(row = 0, column = 1)
    piece_canvas.create_oval(100 - r, 100 - r,100 + r, 100 + r,fill = 'white')

button1 = Button(window,text="开始(黑)",font=('黑体', 10),fg='blue',width=10,height=2,command = click_resetB)
button1.grid(row = 4,column = 1)
button3 = Button(window,text="开始(白)",font=('黑体', 10),fg='blue',width=10,height=2,command = click_resetW)
button3.grid(row = 5,column = 1)
button2 = Button(window,text="开始(PVP)",font=('黑体', 10),fg='blue',width=10,height=2,command = click_resetPVP)
button2.grid(row = 6,column = 1)

piece_canvas = Canvas(window, width = 200, height = 200)
piece_canvas.grid(row = 0, column = 1)
piece_canvas.create_oval(100 - r, 100 - r,100 + r, 100 + r,fill = 'black')

var.set("Playing")
label = Label(window, textvariable=var, font=("宋体",16))
label.grid(row = 1,column = 1) 
window.mainloop()