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
    import main
    # create text box so that signs may be interpreted
    textBox = Text(root, height=2)
    textBox.pack(pady=12)
    # functionality 

# setting frame and master element and adding to root
frame = customtkinter.CTkFrame(master=root)
frame.pack(pady=20, padx=60, fill="both", expand=True)

# adding label and buttons to the frame
label = customtkinter.CTkLabel(master=frame, text="Sign Language Interpretation")
label.pack(pady=12, padx=10)

button1 = customtkinter.CTkButton(master=frame, text="Click here to start", command=but1)
button1.pack(pady=12, padx=10)

# method which when executed, runs the application
root.mainloop()