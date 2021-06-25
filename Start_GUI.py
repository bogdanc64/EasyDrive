import tkinter as tk
import sys
import os
from PIL import Image, ImageTk

window = tk.Tk()
window.title("Sign Detector - Menu")
window.columnconfigure([0,1,2], minsize=150)
window.rowconfigure([0,1,2], minsize=150)

image1 = Image.open("logo.png")
image1 = image1.resize((150, 250), Image.ANTIALIAS)
test = ImageTk.PhotoImage(image1)

label = tk.Label()
label = tk.Label(image=test)
label.image = test
label.grid(row=0, column=1)

def create_window():
    window.destroy()
    os.system('python3 Main_GUI.py')
    

button_start = tk.Button(
    text="Start",
    width=25,
    height=5,
    command=create_window
)
button_start.grid(row=1, column=1)

window.resizable(False, False) 
window.mainloop()