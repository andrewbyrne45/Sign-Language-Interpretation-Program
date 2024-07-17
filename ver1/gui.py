# imports
from modules import *

# set appearance mode and colour scheme
customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

# set element
root = customtkinter.CTk()
root.title("Sign Language Interpretation")
root.geometry("500x350")

# if user is to select option 1 within the GUI
def but1():
    import detectSymbols
    # create text box so that signs may be interpreted
    textBox = Text(root, height=2)
    textBox.pack(pady=12)
    # functionality 

# if user is to select option 1 within the GUI
def but2():
    print("Test")
    # create text box so that words may be interpreted
    textBox = Text(root, height=2)
    textBox.pack(pady=12)

# setting frame and master element and adding to root
frame = customtkinter.CTkFrame(master=root)
frame.pack(pady=20, padx=60, fill="both", expand=True)

# adding label and buttons to the frame
label = customtkinter.CTkLabel(master=frame, text="Please select an option")
label.pack(pady=12, padx=10)

button1 = customtkinter.CTkButton(master=frame, text="Option 1", command=but1)
button1.pack(pady=12, padx=10)

button2 = customtkinter.CTkButton(master=frame, text="Option 2", command=but2)
button2.pack(pady=12, padx=10)

# method which when executed, runs the application
root.mainloop()