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
import time
import glob
import librosa
import librosa.display

global samples, loaded_model, popup_displayed
# Flag which indicates whether or not an instance of a Popup is active
# Prevents additional windows being created
popup_displayed = False

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
    formatted_predictions = reformat_predictions(predictions, chosen_model[0])
    # Store the Predictions in a Txt file under 'Predictions/'
    store_predictions(formatted_predictions)
    PopUpMsg(f'{formatted_predictions}')

def reformat_predictions(predictions, chosen_model):
    formatted_predictions = '----- PREDICTIONS -----'
    spoof_counter = 0
    genuine_counter = 0
    for prediction in predictions:
        if '[Spoof]' in prediction:
            spoof_counter += 1
        elif '[Genuine]' in prediction:
            genuine_counter += 1
        formatted_predictions += '\n' + prediction
    formatted_predictions += f'\n\nGenuine Samples: {genuine_counter}\nSpoof Samples: {spoof_counter}'
    formatted_predictions += f'\n\nModel Used: {os.path.basename(chosen_model)}'
    return formatted_predictions

def show_last_predictions():
    # Get list of all Txt files in Predictions directory
    file_list =  glob.glob('Predictions/*')
    # Determine the latest file
    try:
        latest_file = max(file_list, key=os.path.getctime)
    except Exception as ex:
        messagebox.showinfo('Error!', 'There are no previous Predictions to show')
        return

    # Call function to read txt file into a variable
    predictions_str = read_txt_file(latest_file)
    # Display the Message
    PopUpMsg(f'{predictions_str}')

def read_txt_file(txt_fpath):
    new_str = ''
    file = open(txt_fpath, 'r')
    lines = file.readlines()
    for line in lines:
        new_str += line
    return new_str

def store_predictions(predictions):
    # Get Current Timestamp for Filename Uniqueness
    current_timestamp = time.strftime("%d-%m-%Y-(%H%M)")
    prediction_filename = f'{current_timestamp} Predictions'
    # Create Directory if it doesn't already exist
    if not os.path.exists('Predictions/'):
        os.makedirs('Predictions/')
    file = open(f'Predictions/{prediction_filename}', 'w+')

    if predictions:
        for prediction in predictions:
            file.write(prediction)
        file.close()
    else:
        print('There are no Predictions to Store')

def browse_predictions():
    chosen_txt_file = filedialog.askopenfilename(parent=frame1, initialdir='Predictions/', title='Specify A Prediction File')
    if chosen_txt_file:
        predictions_str = read_txt_file(chosen_txt_file)
        PopUpMsg(f'{predictions_str}')
    else:
        messagebox.showinfo('No File Specified!', 'No Prediction file was specified')

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
                labels_txt_path = open(dataset_path, 'r')
                lines = labels_txt_path.readlines()
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
            PopUpMsg('No labels could be found for selected Sample/s. Therefore the Predictions could not be verified')
            return
        PopUpMsg(f'\nOverall Accuracy: {ACCURACY}\nFAR:{FAR}\n\nTP:{TP}\nTN:{TN}\nFP:{FP}\nFN:{FN}')
    else:
        messagebox.showinfo('No Predictions to Verify', 'There are no previous predictions to verify')

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
    c = plt.subplot(221)
    c.title.set_text('Spectrogram')
    c.set_xlabel('Time')
    c.set_ylabel('Frequency')
    Pxx, freqs, bins, im = c.specgram(sig, NFFT=1024, Fs=16000, noverlap=900)

    # Generate PSD (Power Spectral Density) plot of Signal
    p = plt.subplot(224)
    p.title.set_text('Power Spectral Density Features (PSD)')
    plt.psd(sig, color='blue')

    # Generate Magnitude Spectrum plot of Signal
    ms = plt.subplot(223)
    ms.title.set_text('Magnitude Spectrum')
    plt.magnitude_spectrum(sig, color='blue')

    # Generate MFCC Plot of the Signal
    librosa_sig, sr = librosa.load(sample_fname)
    mfcc = plt.subplot(222)
    mfcc.title.set_text('Mel-Frequency Cepstral Coefficients (MFCC)')
    mfccs = librosa.feature.mfcc(librosa_sig)
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')

    # Adjust spacing between Plots
    plt.subplots_adjust(hspace=0.5)

def create_new_model():
    valid_models = ['SVM', 'svm', 'KNN', 'knn']
    model_settings = {}
    # Prompt for Model Type, C Val and Kernel Type
    model_type = simpledialog.askstring('Input Model Type:', 'Please Specify The Type of Model (SVM)')
    if not model_type:
        messagebox.showinfo('No Model Type Specified', 'A Model Type was not specified')
        return
    elif model_type not in valid_models:
        messagebox.showinfo('Invalid Type', 'The specified Model Type is not a valid Type')
        return

    if model_type == 'svm' or 'SVM':
        c_val = simpledialog.askstring('Input C Parameter Value:', 'Please Specify Value of the C Parameter')
        if not c_val:
            return
        else:
            try:
                c_val_float = float(c_val)
            except Exception as ex:
                print(f'Error when casting User Input [{c_val}] to float value')
                return
        kernel = simpledialog.askstring('Input Kernel Type', 'Please Specify a Kernel Type from [RBF, Poly, Linear]')
        if not kernel:
            return

    model_settings['c_val'] = c_val_float
    model_settings['kernel'] = kernel

    classifier = make_model(model_type, model_settings)
    train_model(classifier)
    return

def create_dataset():
    print('\n------------CREATING DATASET------------')
    chosen_samples  = filedialog.askopenfilenames(parent=frame1, initialdir='Samples/', title='Select Audio Files')
    if not chosen_samples:
        messagebox.showinfo('No Files Selected!', 'Ensure you have selected Audio Files')
        return
    labels_txt_path = filedialog.askopenfilename(parent=frame1, initialdir='Labels/', title='Select A Text File Containing Labels')
    if not labels_txt_path:
        messagebox.showinfo('No Labels File Selected!', 'Ensure you have selected a Labels Text file')
        return

    # Create Associations between Samples and Labels
    associations_dict, missed_labels = get_associations(chosen_samples, labels_txt_path)
    print(f'Associations:{associations_dict}')
    print(f'Missed_labels:{missed_labels}')

    # Build the Association Report String to be displayed later
    message = create_association_report_str(associations_dict, missed_labels)

    # Extract Features
    extracted_features = extract_features(associations_dict)
    # Store Extracted Features in NPY File
    store_extracted_features(extracted_features)

    # Display Assocation Report
    PopUpMsg(f'{message}')
    print('----------------------------------------')
    return

def make_labels_file():
    print('Creating Labels Text File')
    filepath = create_labels_text_file_template()
    TextEditor(window, filepath)
    return

def edit_labels_file():
    print('Editing Labels Text File')
    filepath = filedialog.askopenfilename(parent=frame1, initialdir='Labels/', title='Select A Text File To Edit')
    if not filepath:
        messagebox.showinfo('No File Selected!', 'Ensure you have selected a Text file to Edit')
        return
    TextEditor(window, filepath)
    return

def create_labels_text_file_template():
    chosen_samples  = filedialog.askopenfilenames(parent=frame1, initialdir='Samples/', title='Select Audio Files')
    # Create a Template based on the selected Samples
    filename = 'Labels/TestNew.txt'
    with open(filename, 'w') as file:
        for sample in chosen_samples:
            sample_fname = os.path.basename(sample)
            file.write(f'{sample_fname} Label\n')

    print('Created Template Text File based on the selected Samples')
    return filename

def get_associations(chosen_samples, labels_txt_path):
    associations_dict = {}
    print(f'Chosen_samples:{chosen_samples}\nlabels_txt_path:{labels_txt_path}')

    # Load labels text file and read lines into variable:
    with open(labels_txt_path, 'r') as f:
        lines = f.read().splitlines()

    found_list = []
    chosen_samples_name_list = []
    # Iterate through all selected Samples and try to find labels for each of them
    # from the Labels Text file
    for sample in chosen_samples:
        sample_filename = os.path.basename(sample)
        chosen_samples_name_list.append(sample_filename)
        for line in lines:
            if sample_filename in line:
                if 'spoof' in line:
                    associations_dict[sample] = 0
                    found_list.append(sample_filename)
                elif 'genuine' in line:
                    associations_dict[sample] = 1
                    found_list.append(sample_filename)

    # Determine which Samples labels could not be found for by getting
    # difference between the list specified Samples and the list of samples
    # that labels could be found for (Used in building Report String)
    missed_labels = get_list_difference(found_list, chosen_samples_name_list)
    print(f'Labels Found: {found_list}')
    print(f'Missed Labels = {missed_labels}')
    return associations_dict, missed_labels

def get_list_difference(list1, list2):
    difference = [i for i in list1 + list2 if i not in list1 or i not in list2]
    return difference

def create_association_report_str(associations_dict, missed_labels):
    report_str = '----------- Label Association Report -----------'
    report_str += '\n------ Labels Found For ------'
    for key, value in associations_dict.items():
        # print(f'Key: {key}\nValue: {value}')
        if value == 0:
            report_str += f'\nLabel found for {os.path.basename(key)}: Spoof (0)'
        elif value == 1:
            report_str += f'\nLabel found for {os.path.basename(key)}: Genuine (1)'

    if len(missed_labels) != 0:
        report_str += '\n\n------ Not Found For ------'
        for sample_name in missed_labels:
            report_str += f'\nCould not find Label for {sample_name}'
        report_str += '\n------------------------------------------------'

    report_str += '\n\nEXTRACTED FEATURES FOR SAMPLES WITH LABELS'
    return report_str

class TextEditor():
    def __init__(self, window, txt_fpath=None):
        self.window = Toplevel(window)
        self.window.geometry('600x450')
        self.window.title('Text Editor')
        self.window.config(bg='#44DDFF')
        self.file_path = txt_fpath

        # Create Textspace
        self.textspace = Text(self.window)
        self.textspace.grid(row=0, column=0)

        # If a File Path was passed in, construct the Text Editor with the template
        if self.file_path:
            with open(self.file_path, 'r') as file:
                self.textspace.delete(0.0, END)
                self.textspace.insert(0.0, file.read() + '\n')
                file.close()

        # Object was called with a Filepath
        if self.file_path:
            Button(self.window, text='Save', command= lambda:self.updatefile(self.file_path)).grid(row=1, column=0)
        else:
            Button(self.window, text='Save', command=self.savefile).grid(row=1, column=0)
        Button(self.window, text='Save As', command=self.saveasfile).grid(row=2, column=0)

    def savefile(self):
        savewindow = Tk()
        savewindow.geometry('250x40')
        filecontents = self.textspace.get(0.0, END)

        def writefile():
            with open(file_name_entry.get() + '.txt', 'w+') as file:
                file.write(filecontents)
                file.close()
                savewindow.destroy()
            return None

        Label(savewindow, text='File Name').grid(row=0, column=0)
        file_name_entry = Entry(savewindow, width=40)
        file_name_entry.grid(row=0, column=0)

        Button(savewindow, text='Save', command=writefile).grid(row=0, column=2)

    def saveasfile(self):
        prefix = 'Labels/'
        save_name = simpledialog.askstring('Enter the Filename', 'Enter the Name of the File')
        filecontents = self.textspace.get(0.0, END)

        if save_name:
            with open(prefix + save_name + '.txt', 'w+') as file:
                file.write(filecontents)
                file.close()
        else:
            print('No name was specified, file was not saved!')
            return
        return

    def updatefile(self, file_name):
        filecontents = self.textspace.get(0.0, END)
        with open(file_name, 'w+') as file:
            file.write(filecontents)
            file.close()
            self.window.destroy()

# Class for Creating and Displaying a Custom Popup Message
class PopUpMsg():
    def __init__(self, display_str):
        global popup_displayed
        self.display_str = display_str

        create_window = self.check_displayed()
        if create_window:
            self.popup = Toplevel(window)
            self.popup.title('Popup')
            self.popup.geometry('200x150')
            self.popup.config(bg='#2f2f6d')
            self.popup.state('zoomed')

            popup_canvas = Canvas(self.popup, width=1000, height=1000)
            popup_canvas.pack()
            # Add BG Image to Canvas
            popup_canvas.create_image(500, 750, image=bg_photo, anchor=CENTER)
            # Display Message
            self.textspace = Text(self.popup, bg='#2f2f6d', fg='white', font=('Candara', 12), pady=10)
            self.textspace.insert(0.0, self.display_str)
            self.textspace.tag_configure('center', justify='center')
            self.textspace.tag_add('center', '1.0', 'end')
            self.textspace.place(relx=0.50, rely = 0.35, anchor=CENTER)

            ok_button = Button(self.popup, fg='#333276', background='#44DDFF', activebackground='#44DDFF', font=('Candara', 12, 'bold italic'), activeforeground='white', text='Ok', padx=2, pady=2, command = self.close_popup)
            ok_button.place(relx=0.50, rely = 0.85, anchor=CENTER)
            popup_displayed = True
            print(f'\nPopup Window Displayed:\n{display_str}')
        else:
            return

    def close_popup(self):
        global popup_displayed
        self.popup.destroy()
        popup_displayed = False
        return

    def check_displayed(self):
        global popup_displayed
        if not popup_displayed:
            return True
        else:
            return False

# Define the Window onject
window = Tk()
window.title('VOID GUI')

# Define Image
bg_im = Image.open('Resources/IndexPageImage.png')
home_im = Image.open('Resources/IndexPageImageTitle.png')

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
# Model Maker Screen
frame4 = Frame(window)

# Display all the ferame
for frame in (frame1, frame2, frame3, frame4):
    frame.grid(row=0, column=0, sticky='nsew')

# -------------------- FRAME 1 CODE (HOME SCREEN) ---------------------------
# Converting to ImageTk type so it can be placed in label
home_bg = ImageTk.PhotoImage(home_im)
f1_im_label = Label(frame1, image=home_bg)
f1_im_label.place(x=0, y=0, relwidth=1, relheight=1)

# Defining Buttons
sample_button = Button(frame1, fg='#333276', background='#44DDFF', activebackground='#44DDFF', font=('Candara', 20, 'bold italic'), activeforeground='white', text='Sample Management', padx=10, pady=10, command = lambda:show_frame(frame2))
sample_button.place(relx=0.50, rely = 0.40, anchor=CENTER)

model_management_button = Button(frame1, fg='#333276', background='#44DDFF', activebackground='#44DDFF', font=('Candara', 20, 'bold italic'), activeforeground='white', text='Model Management', padx=10, pady=10, command = lambda:show_frame(frame4))
model_management_button.place(relx=0.50, rely = 0.55, anchor=CENTER)

prediction_button = Button(frame1, fg='#333276', background='#44DDFF', activebackground='#44DDFF', font=('Candara', 20, 'bold italic'), activeforeground='white', text='ENTER THE VOID', padx=10, pady=10, command = lambda:show_frame(frame3))
prediction_button.place(relx=0.50, rely = 0.70, anchor=CENTER)

# Exit Button
exit_button = Button(frame1, fg='#333276', background='#44DDFF', activebackground='#44DDFF', font=('Candara', 20, 'bold italic'), activeforeground='white', text='Exit the App', padx=10, pady=10, command = window.destroy)
exit_button.place(relx=0.5, rely=0.9, anchor=CENTER)

frame1_title = Label(frame1, text='VOID GUI: Main Menu', bg='#44DDFF')
frame1_title.pack(fill='x')

# ---------------------------- FRAME 2 CODE-----------------------------------
# Displaying Background Image
# Converting to ImageTk type so it can be placed in label
bg_photo = ImageTk.PhotoImage(bg_im)
f2_im_label = Label(frame2, image=bg_photo)
f2_im_label.place(x=0, y=0, relwidth=1, relheight=1)

# Defining Buttons
# When pressed will prompt the User to specify which Samples to load into the 'samples' list
load_samples_button = Button(frame2, fg='#333276', background='#44DDFF', activebackground='#44DDFF', font=('Candara', 20, 'bold italic'), activeforeground='white', text='Load in Sample/s', padx=10, pady=10, command = load_samples)
load_samples_button.place(relx=0.50, rely=0.15, anchor=CENTER)

# When pressed will Store the Samples in the 'samples' list into the GUI
store_samples_button = Button(frame2, fg='#333276', background='#44DDFF', activebackground='#44DDFF', font=('Candara', 20, 'bold italic'), activeforeground='white', text='Store Sample/s', padx=10, pady=10, command = store_samples)
store_samples_button.place(relx=0.50, rely=0.30, anchor=CENTER)

# When pressed will ask User to specify Samples for deletion
delete_samples_button = Button(frame2, fg='#333276', background='#44DDFF', activebackground='#44DDFF', font=('Candara', 20, 'bold italic'), activeforeground='white', text='Delete Samples', padx=10, pady=10, command = delete_samples)
delete_samples_button.place(relx=0.50, rely = 0.45, anchor=CENTER)

# Record Button when pressed will record 5 seconds of Audio and save it as a WAV file to root directory
record_one_button = Button(frame2, fg='#333276', background='#44DDFF', activebackground='#44DDFF', font=('Candara', 20, 'bold italic'), activeforeground='white', text='Record a Single Sample', padx=10, pady=10, command = record_sample) #lambda:record_audio(1))
record_one_button.place(relx=0.50, rely = 0.60, anchor=CENTER)

# When pressed will record multiple Samples back to back
record_mult_button = Button(frame2, fg='#333276', background='#44DDFF', activebackground='#44DDFF', font=('Candara', 20, 'bold italic'), activeforeground='white', text='Record Multiple Samples', padx=10, pady=10, command = record_multiple_samples)
record_mult_button.place(relx=0.50, rely = 0.75, anchor=CENTER)

# When this Button is pressed, the User can specify a number of Audio files, which will then be played back to back sequentially
playback_samples_button = Button(frame2, fg='#333276', background='#44DDFF', activebackground='#44DDFF', font=('Candara', 20, 'bold italic'), activeforeground='white', text='Playback Sample/s', padx=10, pady=10, command = playback_samples)
playback_samples_button.place(relx=0.50, rely = 0.90, anchor=CENTER)

# When pressed will take User back to the main menu
main_menu_button = Button(frame2, fg='#333276', background='#44DDFF', activebackground='#44DDFF', font=('Candara', 20, 'bold italic'), activeforeground='white', text='Main Menu', padx=10, pady=10, command = lambda:show_frame(frame1))
main_menu_button.place(relx=0.20, rely = 0.50, anchor=CENTER)

frame2_title = Label(frame2, text='Sample Management', bg='#44DDFF')
frame2_title.pack(fill='x')

# ---------------------------- FRAME 3 CODE-----------------------------------
# Converting to ImageTk type so it can be placed in label
f3_im_label = Label(frame3, image=home_bg)
f3_im_label.place(x=0, y=0, relwidth=1, relheight=1)

# Defining Buttons
analyse_samples_button = Button(frame3, fg='#333276', background='#44DDFF', activebackground='#44DDFF', font=('Candara', 20, 'bold italic'), activeforeground='white', text='Analyse Sample/s', padx=10, pady=10, command = analyse_samples)
analyse_samples_button.place(relx=0.5, rely=0.35, anchor=CENTER)

make_prediction_button = Button(frame3, fg='#333276', background='#44DDFF', activebackground='#44DDFF', font=('Candara', 20, 'bold italic'), activeforeground='white', text='Make A Prediction', padx=10, pady=10, command = make_prediction)
make_prediction_button.place(relx=0.50, rely = 0.45, anchor=CENTER)

browse_predictions_button = Button(frame3, fg='#333276', background='#44DDFF', activebackground='#44DDFF', font=('Candara', 20, 'bold italic'), activeforeground='white', text='View Previous Predictions', padx=10, pady=10, command = browse_predictions)
browse_predictions_button.place(relx=0.50, rely = 0.65, anchor=CENTER)

show_last_predictions_button = Button(frame3, fg='#333276', background='#44DDFF', activebackground='#44DDFF', font=('Candara', 20, 'bold italic'), activeforeground='white', text='Show Last Prediction/s', padx=10, pady=10, command = show_last_predictions)
show_last_predictions_button.place(relx=0.60, rely = 0.75, anchor=CENTER)

verify_predictions_button = Button(frame3, fg='#333276', background='#44DDFF', activebackground='#44DDFF', font=('Candara', 20, 'bold italic'), activeforeground='white', text='Verify Last Prediction/s', padx=10, pady=10, command = verify_predictions)
verify_predictions_button.place(relx=0.40, rely = 0.75, anchor=CENTER)

f3_main_menu_button = Button(frame3, fg='#333276', background='#44DDFF', activebackground='#44DDFF', font=('Candara', 20, 'bold italic'), activeforeground='white', text='Main Menu', padx=10, pady=10, command = lambda:show_frame(frame1))
f3_main_menu_button.place(relx=0.50, rely = 0.90, anchor=CENTER)

frame3_title = Label(frame3, text='The VOID', bg='#44DDFF')
frame3_title.pack(fill='x')

# ---------------------------- FRAME 4 CODE-----------------------------------
# Converting to ImageTk type so it can be placed in label
f4_im_label = Label(frame4, image=home_bg)
f4_im_label.place(x=0, y=0, relwidth=1, relheight=1)

# Defining Buttons
create_labels_button = Button(frame4, fg='#333276', background='#44DDFF', activebackground='#44DDFF', font=('Candara', 20, 'bold italic'), activeforeground='white', text='Create A Labels File', padx=10, pady=10, command = make_labels_file)
create_labels_button.place(relx=0.5, rely=0.35, anchor=CENTER)

edit_labels_button = Button(frame4, fg='#333276', background='#44DDFF', activebackground='#44DDFF', font=('Candara', 20, 'bold italic'), activeforeground='white', text='Edit A Labels File', padx=10, pady=10, command = edit_labels_file)
edit_labels_button.place(relx=0.5, rely=0.45, anchor=CENTER)

create_dataset_button = Button(frame4, fg='#333276', background='#44DDFF', activebackground='#44DDFF', font=('Candara', 20, 'bold italic'), activeforeground='white', text='Create The Dataset', padx=10, pady=10, command = create_dataset)
create_dataset_button.place(relx=0.5, rely=0.60, anchor=CENTER)

create_model_button = Button(frame4, fg='#333276', background='#44DDFF', activebackground='#44DDFF', font=('Candara', 20, 'bold italic'), activeforeground='white', text='Create A New Model', padx=10, pady=10, command = create_new_model)
create_model_button.place(relx=0.5, rely=0.75, anchor=CENTER)

# When pressed will take User back to the main menu
f4_main_menu_button = Button(frame4, fg='#333276', background='#44DDFF', activebackground='#44DDFF', font=('Candara', 20, 'bold italic'), activeforeground='white', text='Main Menu', padx=10, pady=10, command = lambda:show_frame(frame1))
f4_main_menu_button.place(relx=0.50, rely = 0.90, anchor=CENTER)

frame4_title = Label(frame4, text='Model Management', bg='#44DDFF')
frame4_title.pack(fill='x')

show_frame(frame1)

window.mainloop()
