from tkinter import *
from tkinter import filedialog, messagebox, simpledialog, ttk
from PIL import ImageTk, Image
from record import *
import os
import shutil
from pathlib import Path
import simpleaudio as sa
from threading import Timer

global samples, loaded_model

def show_frame(frame):
    frame.tkraise()

def load_samples():
    global samples

    # Prompt User to Specify files to load in
    samples  = filedialog.askopenfilenames(parent=frame1, title='Select Audio Files')
    if samples:
        print('Loaded in the following files: {}'.format(samples))
        messagebox.showinfo('Samples Loaded!', 'The Selected Samples have been loaded in, click \'Store Samples\' to load them into the GUI Program')
    else:
        print('No files were selected')
        messagebox.showinfo('Samples not Loaded!', 'No Audio Files were specified to be loaded in')

def load_model():
    global loaded_model

    # Prompt User to Specify files to load in
    loaded_model  = filedialog.askopenfilenames(parent=frame3, initialdir='Models/', title='Select A Model To Use For Predictions')
    if loaded_model:
        print('Loaded the following model: {}'.format(loaded_model))
        messagebox.showinfo('Model Loaded!', 'The Selected Model has been loaded in, you can now ask VOID to make a prediction')
    else:
        print('No model was selected')
        messagebox.showinfo('Model not Loaded!', 'No Model was Selected')

def store_samples():
    global samples

    if samples:
        new_paths_list = []
        for sample in samples:
            new_path = shutil.copy(sample, 'Samples/')
            new_paths_list.append(new_path)

        if new_paths_list:
            print('Selected Files have been stored in the Samples Directory')
            print('New File Paths: {}'.format(new_paths_list))
            messagebox.showinfo('Samples Stored!', 'The Selected Samples have been stored under /Samples Directory')

        # Clear Buffer of previously selected files
        samples = []
    else:
        print('There were no loaded samples to store')
        messagebox.showinfo('Samples not Stored!', 'There were no Audio Files to store. Click \'Load Samples\' to specify files')

def playback_samples():
    # Prompt User to Specify files to load in
    playback_samples  = filedialog.askopenfilenames(parent=frame1, initialdir='Samples/', title='Select Audio Files')

    for sample in playback_samples:
        wav_obj = sa.WaveObject.from_wave_file(sample)
        play_obj = wav_obj.play()
        play_obj.wait_done()

def record_multiple_samples():
    amount = simpledialog.askinteger(title='Specify Amount', prompt='Specify Amount of Samples to Record')
    record_audio(amount)

def make_prediction():
    global samples

    missing_vars = False

    ## TODO - FIX Undefined Errors and ensure thid handling works as expected
    if not samples or loaded_model:
        missing_vars = True

    if not missing_vars:
        print('Ensure you have chosen a model and loaded a Sample before clicking \'Make Prediction\'')
    else:
        # TO DO
        print('Making Prediction on {}, using the model named {}'.format(samples[0], loaded_model[0]))

def progress_bar():
    global progress_bar
    progress_bar = ttk.Progressbar(window, orient=HORIZONTAL, length=200, mode='determinate')
    progress_bar.place(relx=0.5, rely = 0.52, anchor=CENTER)
    progress_increment(5)

def progress_increment(seconds):
    global progress_bar
    increment_amount = (100/seconds)
    print('Seconds: {}, Incrementing By: {}'.format(seconds, increment_amount))

    for x in range(seconds):
        progress_bar['value'] += increment_amount
        window.update_idletasks()
        time.sleep(1)

# Define the Window onject
window = Tk()
window.title('VOID GUI')

# Define Image
bg_im = Image.open('Resources/IndexPageImage.png')

# This Method will open the window automatically when the program starts
window.state('zoomed')

# These methods ensure it scales with the size of the window
window.rowconfigure(0, weight=1)
window.columnconfigure(0, weight=1)

# Declare the Frames (Menu Screens)
# Main Menu
frame1 = Frame(window)
# Managing Samples Screen
frame2 = Frame(window)
# Prediction Screen
frame3 = Frame(window)

# Display all the ferame
for frame in (frame1, frame2, frame3):
    frame.grid(row=0, column=0, sticky='nsew')

# -------------------- FRAME 1 CODE (HOME SCREEN) ---------------------------
# Converting to ImageTk type so it can be placed in label
photo = ImageTk.PhotoImage(bg_im)
f1_im_label = Label(frame1, image=photo)
f1_im_label.place(x=0, y=0, relwidth=1, relheight=1)

# Defining Buttons
sample_button = Button(frame1, fg='white', background='blue', activebackground='blue', font=('Candara', 20, 'bold italic'), activeforeground='blue', text='Sample Management', padx=10, pady=10, command = lambda:show_frame(frame2))
sample_button.place(relx=0.50, rely = 0.45, anchor=CENTER)

prediction_button = Button(frame1, fg='white', background='blue', activebackground='blue', font=('Candara', 20, 'bold italic'), activeforeground='blue', text='ENTER THE VOID', padx=10, pady=10, command = lambda:show_frame(frame3))
prediction_button.place(relx=0.50, rely = 0.60, anchor=CENTER)

# Exit Button
exit_button = Button(frame1, fg='white', background='blue', activebackground='blue', font=('Candara', 20, 'bold italic'), activeforeground='blue', text='Exit the App', padx=10, pady=10, command = window.destroy)
exit_button.place(relx=0.5, rely=0.9, anchor=CENTER)

frame1_title = Label(frame1, text='VOID GUI: Main Menu', bg='blue')
frame1_title.pack(fill='x')

# ---------------------------- FRAME 2 CODE-----------------------------------
# Displaying Background Image
# Converting to ImageTk type so it can be placed in label
f2_im_label = Label(frame2, image=photo)
f2_im_label.place(x=0, y=0, relwidth=1, relheight=1)

# Defining Buttons
load_samples_button = Button(frame2, fg='white', background='blue', activebackground='blue', font=('Candara', 20, 'bold italic'), activeforeground='pink', text='Load in Sample/s', padx=10, pady=10, command = load_samples)
load_samples_button.place(relx=0.50, rely=0.15, anchor=CENTER)

store_samples_button = Button(frame2, fg='white', background='blue', activebackground='blue', font=('Candara', 20, 'bold italic'), activeforeground='pink', text='Store Sample/s', padx=10, pady=10, command = store_samples)
store_samples_button.place(relx=0.50, rely=0.30, anchor=CENTER)

# Record Button when pressed will record 5 seconds of Audio and save it as a WAV file to root directory
record_one_button = Button(frame2, fg='white', background='blue', activebackground='blue', font=('Candara', 20, 'bold italic'), activeforeground='pink', text='Record a Single Sample', padx=10, pady=10, command = progress_bar) #lambda:record_audio(1))
record_one_button.place(relx=0.50, rely = 0.45, anchor=CENTER)

# Record Button when pressed will record 5 seconds of Audio and save it as a WAV file to root directory
record_mult_button = Button(frame2, fg='white', background='blue', activebackground='blue', font=('Candara', 20, 'bold italic'), activeforeground='pink', text='Record Multiple Samples', padx=10, pady=10, command = record_multiple_samples)
record_mult_button.place(relx=0.50, rely = 0.60, anchor=CENTER)

# When this Button is pressed, the User can specify a number of Audio files, which will then be played back to back sequentially
playback_samples_button = Button(frame2, fg='white', background='blue', activebackground='blue', font=('Candara', 20, 'bold italic'), activeforeground='pink', text='Playback Sample/s', padx=10, pady=10, command = playback_samples)
playback_samples_button.place(relx=0.50, rely = 0.75, anchor=CENTER)

main_menu_button = Button(frame2, fg='white', background='blue', activebackground='blue', font=('Candara', 20, 'bold italic'), activeforeground='pink', text='Main Menu', padx=10, pady=10, command = lambda:show_frame(frame1))
main_menu_button.place(relx=0.50, rely = 0.90, anchor=CENTER)

frame2_title = Label(frame2, text='Sample Management', bg='blue')
frame2_title.pack(fill='x')

# ---------------------------- FRAME 3 CODE-----------------------------------
# Converting to ImageTk type so it can be placed in label
f3_im_label = Label(frame3, image=photo)
f3_im_label.place(x=0, y=0, relwidth=1, relheight=1)

# Defining Buttons
load_model_button = Button(frame3, fg='white', background='blue', activebackground='blue', font=('Candara', 20, 'bold italic'), activeforeground='blue', text='Choose a Model/s', padx=10, pady=10, command = load_model)
load_model_button.place(relx=0.5, rely=0.45, anchor=CENTER)

f3_load_samples_button = Button(frame3, fg='white', background='blue', activebackground='blue', font=('Candara', 20, 'bold italic'), activeforeground='blue', text='Load a Sample/s', padx=10, pady=10, command = load_samples)
f3_load_samples_button.place(relx=0.50, rely = 0.55, anchor=CENTER)

make_prediction_button = Button(frame3, fg='white', background='blue', activebackground='blue', font=('Candara', 20, 'bold italic'), activeforeground='blue', text='Make Prediction', padx=10, pady=10, command = make_prediction)
make_prediction_button.place(relx=0.50, rely = 0.75, anchor=CENTER)

f3_main_menu_button = Button(frame3, fg='white', background='blue', activebackground='blue', font=('Candara', 20, 'bold italic'), activeforeground='pink', text='Main Menu', padx=10, pady=10, command = lambda:show_frame(frame1))
f3_main_menu_button.place(relx=0.50, rely = 0.90, anchor=CENTER)

frame3_title = Label(frame3, text='The VOID', bg='blue')
frame3_title.pack(fill='x')


show_frame(frame1)

window.mainloop()
