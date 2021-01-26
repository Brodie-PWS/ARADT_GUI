from tkinter import *
from tkinter import filedialog, messagebox, simpledialog, ttk
from PIL import ImageTk, Image
from record import *
from predict import *
from pathlib import Path
import threading
import os
import shutil
import simpleaudio as sa
import matplotlib.pyplot as plt
import numpy as np
import re

global samples, loaded_model

dataset_dict = {
    'Evaluation': 'Labels/ASVspoof2017_V2_eval.trl.txt',
    'Development': 'Labels/ASVspoof2017_V2_eval.trl.txt',
    'Training': 'Labels/ASVspoof2017_V2_eval.trl.txt'
    }

def show_frame(frame):
    frame.tkraise()

def load_samples():
    global samples

    # Prompt User to Specify files to load in
    samples  = filedialog.askopenfilenames(parent=frame1, initialdir='Samples/', title='Select Audio Files')
    if samples:
        print('Loaded in the following files: {}'.format(samples))
        messagebox.showinfo('Samples Loaded!', 'The Selected Samples have been loaded in, click \'Store Samples\' to load them into the GUI Program')
    else:
        print('No files were selected')
        messagebox.showinfo('Samples not Loaded!', 'No Audio Files were specified to be loaded in')

def load_model():
    global loaded_model

    # Prompt User to Specify files to load in
    loaded_model  = filedialog.askopenfilename(parent=frame3, initialdir='Models/', title='Select A Model To Use For Predictions')
    if loaded_model:
        print('Loaded the following model: {}'.format(loaded_model))
        messagebox.showinfo('Model Loaded!', 'The Selected Model has been loaded in, you can now ask VOID to make a prediction')
    else:
        print('No model was selected')
        messagebox.showinfo('Model not Loaded!', 'No Model was Selected')

def store_samples():
    global samples

    try:
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
    except Exception as ex:
        print('Error: {}'.format(ex))
        print('There were no loaded samples to store')
        messagebox.showinfo('Samples not Stored!', 'There were no Audio Files to store. Click \'Load Samples\' to specify files')
        pass

def delete_samples():
    # Prompt User to Specify Samples to be deleted from GUI
    del_samples  = filedialog.askopenfilenames(parent=frame1, initialdir='Samples/', title='Select Audio Files')
    if del_samples:
        print('Selected Following Files for Deletion: {}'.format(del_samples))
        res = messagebox.askquestion('Are You Sure?', 'Are you sure you want to delete the selected samples?')
        if res == 'yes':
            for sample_path in del_samples:
                os.remove(sample_path)
                print('Removed {}'.format(sample_path))
            print('All selected Samples have been deleted')
            messagebox.showinfo('Samples deleted!', 'The selected Samples have been deleted')
        else:
            messagebox.showinfo('Samples not deleted!', 'The selected Samples were not deleted')
    else:
        print('No files were selected')
        messagebox.showinfo('No files selected', 'No files were selected for deletion')

def playback_samples():
    # Prompt User to Specify files to load in
    playback_samples  = filedialog.askopenfilenames(parent=frame1, initialdir='Samples/', title='Select Audio Files')

    for sample in playback_samples:
        wav_obj = sa.WaveObject.from_wave_file(sample)
        play_obj = wav_obj.play()
        play_obj.wait_done()

def record_sample():
    # Calls to external function in record.py.
    global duration
    duration = simpledialog.askinteger(title='Specify Duration', prompt='Specify the Duration of the Recording')
    if not duration: return
    progress_bar_start('single')
    record_audio(duration)

def record_multiple_samples():
    # Calls to external function in record.py.
    global duration
    amount = simpledialog.askinteger(title='Specify Amount', prompt='Specify Amount of Samples to Record')
    if not amount: return
    duration = simpledialog.askinteger(title='Specify Duration', prompt='Specify the Duration of each Recording')
    if not duration: return

    for x in range(amount):
        progress_bar_start('multiple')
        record_audio(duration)

def progress_bar_start(setting):
    global progress_bar, duration
    # Progress Bar Setup
    progress_style = ttk.Style()
    progress_style.theme_use('clam')
    progress_style.configure('blue.Horizontal.TProgressbar', troughcolour='cyan', bordercolor='cyan', foreground='cyan', background='blue', lightcolor='blue', darkcolor='blue')
    progress_bar = ttk.Progressbar(window, style='blue.Horizontal.TProgressbar', orient=HORIZONTAL, length=200, mode='determinate')
    # Position Progress Bar based on which button was pressed
    if setting == 'single':
        progress_bar.place(relx=0.5, rely = 0.65, anchor=CENTER)
    elif setting == 'multiple':
        progress_bar.place(relx=0.5, rely = 0.70, anchor=CENTER)

    progress_finished = False
    for x in range(duration):
        # Increment Progress Bar
        window.after(1000, progress_increment(duration))
        # Once Progress reaches 100, display feedbback and reset progress bar
        if progress_bar['value'] == 100:
            print('Progress Bar Reset')
            messagebox.showinfo('Recording Finished!', 'Recording has finished, Now Saving, Please Wait...')
            progress_bar.destroy()

        window.update_idletasks()

def progress_increment(duration):
    global progress_bar
    # Determine the increment value based on duration of sample
    increment_amount = (100/duration)
    progress_bar['value'] += increment_amount

def make_prediction():
    pred_list = []

    # Ask User to choose a Model
    chosen_model = filedialog.askopenfilenames(parent=frame1, initialdir='Models/', title='Specify a Model to use')
    # If no Model was chosen, alert User and exit function
    if len(chosen_model) == 0:
        print('Ensure you have chosen a model before clicking \'Make Prediction\'')
        messagebox.showinfo('Error!', 'Make sure to choose a Model before clicking \'Make Prediction\'')
        return

    # Ask User to specify a Sample or list of Samples
    chosen_samples  = filedialog.askopenfilenames(parent=frame1, initialdir='Samples/', title='Select Audio Files')
    # If no samples were chosen, alert User and exit function
    if len(chosen_samples) == 0:
        print('Ensure you have specified Sample/s')
        messagebox.showinfo('Error!', 'Ensure you have selected Samples')
        return

    global predictions
    predictions = predict_pipeline(chosen_model[0], chosen_samples)
    formatted_predictions = reformat_predictions(predictions)
    messagebox.showinfo('Prediction Results', f'{formatted_predictions}')

def reformat_predictions(predictions):
    formatted_predictions = '          ----- PREDICTIONS -----'
    for prediction in predictions:
        formatted_predictions += '\n' + prediction
    return formatted_predictions

def show_last_predictions():
    if 'predictions' in globals():
        formatted_predictions = reformat_predictions(predictions)
        messagebox.showinfo('Prediction Results', f'{formatted_predictions}')
    else:
        messagebox.showinfo('No Results', 'There were no results to retrieve :(')

def verify_predictions():
    results = {}
    dataset_path = ''
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    if 'predictions' in globals():
        for prediction in predictions:
            # Extract Matching Strings from Predictions, Pulls out Filename and Classification
            sample_format = re.findall('[A-Z]\w+', prediction)
            # Assign matches to their own variables
            sample_name = sample_format[0]
            sample_predicted_label = sample_format[1]

            # Based on the starting letter, determine which Dataset Text file should be loaded
            if sample_name.startswith('E_'):
                dataset_path = dataset_dict.get('Evaluation')
            elif sample_name.startswith('T_'):
                dataset_path = dataset_dict.get('Training')
            elif sample_name.startswith('D_'):
                dataset_path = dataset_dict.get('Development')
            else:
                print(f'No Label could be found for Sample: {sample_name}. Continuing...')
                continue
            print(f'Dataset Path = {dataset_path}\n')

            if dataset_path:
                labels_txt = open(dataset_path, 'r')
                lines = labels_txt.readlines()
                # Look for the line corresponding to sample and retrieve actual label
                for line in lines:
                    if sample_name in line:
                        if 'spoof' in line:
                            actual_label = 'Spoof'
                        elif 'genuine' in line:
                            actual_label = 'Genuine'
                        print(f'Found Match in line: \n{line}')

            print(f'What the Sample was Predicted to be: {sample_predicted_label}')
            print(f'What the Sample actually is: {actual_label}\n')
            # Determine TP, TN, FP, FN values
            if actual_label == 'Genuine':
                if actual_label == sample_predicted_label:
                    TP += 1
                else:
                    FN += 1
            elif actual_label == 'Spoof':
                if actual_label == sample_predicted_label:
                    TN += 1
                else:
                    FP += 1

        try:
            ACCURACY = (TP + TN) / (TP + TN + FP + FN)
            FAR = FP / (FP + TN)
        except Exception as ex:
            messagebox.showinfo('Error', 'No labels could be found for selected Sample/s. Therefore the Predictions could not be verified')
            return
        messagebox.showinfo('Results', f'\nOverall Accuracy: {ACCURACY}\nFAR:{FAR}\n\nTP:{TP}\nTN:{TN}\nFP:{FP}\nFN:{FN}')
    else:
        messagebox.showinfo('No Results', 'There were no previous predictions to verify')

def analyse_samples():
    index = 0
    # Prompt User to select Samples for analysis
    chosen_samples  = filedialog.askopenfilenames(parent=frame1, initialdir='Samples/', title='Select Audio Files')
    # Iterate through selected samples and create a figure for each one
    for sample in chosen_samples:
        plot_graphs(index, sample)
        index += 1

    # Display all of the created figures
    plt.show()

def plot_graphs(index, sample_fname):
    sample_wav = wave.open(sample_fname, 'r')
    sample_freq = 16000
    data = np.frombuffer(sample_wav.readframes(sample_freq), dtype=np.int16)
    sig = sample_wav.readframes(-1)
    sig = np.frombuffer(sig, 'Int16')

    # Determine Filename
    filename = os.path.basename(sample_fname)
    plt.figure(index)
    plt.suptitle('Analysis of {}'.format(filename))

    # Generate Spectrogram of Signal
    c = plt.subplot(311)
    Pxx, freqs, bins, im = c.specgram(sig, NFFT=1024, Fs=16000, noverlap=900)
    c.set_xlabel('Time')
    c.set_ylabel('Frequency')
    # Generate PSD (Power Spectral Density) plot of Signal
    p = plt.subplot(312)
    plt.psd(sig, color='blue')
    # Generate Magnitude Spectrum plot of Signal
    ms = plt.subplot(313)
    plt.magnitude_spectrum(sig, color='blue')

    # Adjust spacing between Plots
    plt.subplots_adjust(hspace=0.5)

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
# When pressed will prompt the User to specify which Samples to load into the 'samples' list
load_samples_button = Button(frame2, fg='white', background='blue', activebackground='blue', font=('Candara', 20, 'bold italic'), activeforeground='pink', text='Load in Sample/s', padx=10, pady=10, command = load_samples)
load_samples_button.place(relx=0.50, rely=0.15, anchor=CENTER)

# When pressed will Store the Samples in the 'samples' list into the GUI
store_samples_button = Button(frame2, fg='white', background='blue', activebackground='blue', font=('Candara', 20, 'bold italic'), activeforeground='pink', text='Store Sample/s', padx=10, pady=10, command = store_samples)
store_samples_button.place(relx=0.50, rely=0.30, anchor=CENTER)

# When pressed will ask User to specify Samples for deletion
delete_samples_button = Button(frame2, fg='white', background='blue', activebackground='blue', font=('Candara', 20, 'bold italic'), activeforeground='pink', text='Delete Samples', padx=10, pady=10, command = delete_samples)
delete_samples_button.place(relx=0.50, rely = 0.45, anchor=CENTER)

# Record Button when pressed will record 5 seconds of Audio and save it as a WAV file to root directory
record_one_button = Button(frame2, fg='white', background='blue', activebackground='blue', font=('Candara', 20, 'bold italic'), activeforeground='pink', text='Record a Single Sample', padx=10, pady=10, command = record_sample) #lambda:record_audio(1))
record_one_button.place(relx=0.50, rely = 0.60, anchor=CENTER)

# When pressed will record multiple Samples back to back
record_mult_button = Button(frame2, fg='white', background='blue', activebackground='blue', font=('Candara', 20, 'bold italic'), activeforeground='pink', text='Record Multiple Samples', padx=10, pady=10, command = record_multiple_samples)
record_mult_button.place(relx=0.50, rely = 0.75, anchor=CENTER)

# When this Button is pressed, the User can specify a number of Audio files, which will then be played back to back sequentially
playback_samples_button = Button(frame2, fg='white', background='blue', activebackground='blue', font=('Candara', 20, 'bold italic'), activeforeground='pink', text='Playback Sample/s', padx=10, pady=10, command = playback_samples)
playback_samples_button.place(relx=0.50, rely = 0.90, anchor=CENTER)

# When pressed will take User back to the main menu
main_menu_button = Button(frame2, fg='white', background='blue', activebackground='blue', font=('Candara', 20, 'bold italic'), activeforeground='pink', text='Main Menu', padx=10, pady=10, command = lambda:show_frame(frame1))
main_menu_button.place(relx=0.20, rely = 0.50, anchor=CENTER)

frame2_title = Label(frame2, text='Sample Management', bg='blue')
frame2_title.pack(fill='x')

# ---------------------------- FRAME 3 CODE-----------------------------------
# Converting to ImageTk type so it can be placed in label
f3_im_label = Label(frame3, image=photo)
f3_im_label.place(x=0, y=0, relwidth=1, relheight=1)

# Defining Buttons
analyse_samples_button = Button(frame3, fg='white', background='blue', activebackground='blue', font=('Candara', 20, 'bold italic'), activeforeground='blue', text='Analyse Sample/s', padx=10, pady=10, command = analyse_samples)
analyse_samples_button.place(relx=0.5, rely=0.45, anchor=CENTER)

make_prediction_button = Button(frame3, fg='white', background='blue', activebackground='blue', font=('Candara', 20, 'bold italic'), activeforeground='blue', text='Make A Prediction', padx=10, pady=10, command = make_prediction)
make_prediction_button.place(relx=0.50, rely = 0.55, anchor=CENTER)

show_last_predictions_button = Button(frame3, fg='white', background='blue', activebackground='blue', font=('Candara', 20, 'bold italic'), activeforeground='blue', text='Show Last Prediction/s', padx=10, pady=10, command = show_last_predictions)
show_last_predictions_button.place(relx=0.60, rely = 0.75, anchor=CENTER)

verify_predictions_button = Button(frame3, fg='white', background='blue', activebackground='blue', font=('Candara', 20, 'bold italic'), activeforeground='blue', text='Verify Last Prediction/s', padx=10, pady=10, command = verify_predictions)
verify_predictions_button.place(relx=0.40, rely = 0.75, anchor=CENTER)

f3_main_menu_button = Button(frame3, fg='white', background='blue', activebackground='blue', font=('Candara', 20, 'bold italic'), activeforeground='pink', text='Main Menu', padx=10, pady=10, command = lambda:show_frame(frame1))
f3_main_menu_button.place(relx=0.50, rely = 0.90, anchor=CENTER)

frame3_title = Label(frame3, text='The VOID', bg='blue')
frame3_title.pack(fill='x')

show_frame(frame1)

window.mainloop()
