import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, Entry, messagebox
from tkinterdnd2 import TkinterDnD, DND_FILES
import numpy as np
import os
####
from rigid_co import *
from without_seg_fun import *
import imageio
import numpy as np
import subprocess
import os
import platform
import threading
####
def upload_all_files_raw(moving_path_raw=None,target_path_raw=None):
    pre_data = nib.load(target_path_raw).get_fdata()
    post_data = nib.load(moving_path_raw).get_fdata()
    img_data_pre = np.zeros((pre_data.shape[0], pre_data.shape[1], pre_data.shape[-1]), dtype=np.uint8)
    img_data_post = np.zeros((post_data.shape[0], post_data.shape[1], post_data.shape[-1]), dtype=np.uint8)
    for i in range(pre_data.shape[-1]):
        img_data_pre[:, :, i] = draw_two_parallel_lines(np.zeros((pre_data.shape[0], pre_data.shape[0]))) 
    
    for i in range(post_data.shape[-1]):
        img_data_post[:, :, i] = draw_wide_diagonal_line(np.zeros((post_data.shape[0], post_data.shape[0])))
    sdf_ct= distance_transform_edt(torch.tensor(img_data_post).permute(2, 0, 1))  
    sdf_oct= distance_transform_edt(torch.tensor(img_data_pre).permute(2, 0, 1))

    CT_sdf_cpr = torch.tensor(sdf_ct).unsqueeze(0)
    OCT_sdf_cpr = torch.tensor(sdf_oct).unsqueeze(0)
    Area_CT_ori = torch.sum((CT_sdf_cpr > 0), dim=(0, 2, 3))
    Area_OCT = torch.sum((OCT_sdf_cpr > 0), dim=(0, 2, 3))
    return CT_sdf_cpr,OCT_sdf_cpr,Area_CT_ori,Area_OCT,post_data,pre_data,img_data_pre,img_data_post,sdf_ct,sdf_oct
def upload_all_files_seg(moving_path_mask=None,target_path_mask=None,moving_path_raw=None,target_path_raw=None):
    pre_data = nib.load(target_path_raw).get_fdata()
    post_data = nib.load(moving_path_raw).get_fdata()
    # load segmentation data
    img = nib.load(target_path_mask)
    img_data_pre = img.get_fdata() 
    img = nib.load(moving_path_mask)
    img_data_post = img.get_fdata()
    sdf_ct= distance_transform_edt(torch.tensor(img_data_post).permute(2, 0, 1))  
    sdf_oct= distance_transform_edt(torch.tensor(img_data_pre).permute(2, 0, 1))

    CT_sdf_cpr = torch.tensor(sdf_ct).unsqueeze(0)
    OCT_sdf_cpr = torch.tensor(sdf_oct).unsqueeze(0)
    Area_CT_ori = torch.sum((CT_sdf_cpr > 0), dim=(0, 2, 3))
    Area_OCT = torch.sum((OCT_sdf_cpr > 0), dim=(0, 2, 3))
    return CT_sdf_cpr,OCT_sdf_cpr,Area_CT_ori,Area_OCT,post_data,pre_data,img_data_pre,img_data_post,sdf_ct,sdf_oct
def gui_play(Area_CT_ori,Area_OCT,CT_sdf_cpr,OCT_sdf_cpr,post_data,pre_data,only=False):
    root = tk.Tk()
    root.title("Oct Coregistration Tool")
    app = PeaksMatcherGUI(master=root, Area_CT=Area_CT_ori, Area_OCT=Area_OCT, CT_image=CT_sdf_cpr>0, OCT_image=OCT_sdf_cpr>0,or_ct=torch.tensor(post_data).permute(2, 0, 1).unsqueeze(0),or_oct=torch.tensor(pre_data).permute(2, 0, 1).unsqueeze(0),only=only)
    app.mainloop()
    # Capture organized pairs before closing
    #organized_pairs = organize_matched_pairs(app.highlighted_points)
    organized_pairs,ang_big,an = organize_matched_pairss(app.saving_orientation,app.saving_orientation_angl_show)
    return organized_pairs,ang_big,an
def gui_play_all(Area_CT_ori,Area_OCT,CT_sdf_cpr,OCT_sdf_cpr,post_data,pre_data,only=False):
    roots = tk.Tk()
    roots.title("Oct Coregistration Tool")
    app = PeaksMatcherGUI(master=roots, Area_CT=Area_CT_ori, Area_OCT=Area_OCT, CT_image=CT_sdf_cpr>0, OCT_image=OCT_sdf_cpr>0,or_ct=torch.tensor(post_data).permute(2, 0, 1).unsqueeze(0),or_oct=torch.tensor(pre_data).permute(2, 0, 1).unsqueeze(0),only=False)
    app.mainloop()
    # Capture organized pairs before closing
    #organized_pairs = organize_matched_pairs(app.highlighted_points)
    organized_pairs,ang_big,an = organize_matched_pairss(app.saving_orientation,app.saving_orientation_angl_show)
    return organized_pairs,ang_big,an
def gui_co_rec(organized_pairs,ang_big,an,post_data,pre_data,CT_sdf_cpr,OCT_sdf_cpr,patient_id=None,only=False,save_path=None):
    print("Last organized pairs:", organized_pairs)

    left_count = sum(1 for item in an if item[1] == 'left')
    right_count = sum(1 for item in an if item[1] == 'right')
    an = np.array(an)
    
    if left_count>=right_count:
        print("The orientation is left")
        print(left_count)
        z = [float(x[0])* math.pi/180 for x in an] 
        z = [-abs(value) for value in z]
        right_indices = [i for i, item in enumerate(an) if item[1] == 'right']
        for i in right_indices:
            z[i] = -6.28-z[i]
    else:
        print(right_count)
        z = [float(x[0])* math.pi/180 for x in an]
        left_indices = [i for i, item in enumerate(an) if item[1] == 'left'] 
        for i in left_indices:
            z[i] = 6.28-z[i]

    #z = [float(x[0])* math.pi/180 for x in an]
    #left_indices = [i for i, item in enumerate(an) if item[1] == 'left'] 
    #for i in left_indices:
    #    z[i] = -z[i]

    ang_big = z
    print("Anglel",ang_big)


    CT_selected_indices = torch.tensor([t[0] for t in organized_pairs])
    OCT_selected_indices = torch.tensor([t[1] for t in organized_pairs])
    idx_OCT_shift = OCT_selected_indices[0]
    idx_CT_shift = CT_selected_indices[0]

    # indices following shift based on first bifurcation
    OCT_selected_indices_shift = OCT_selected_indices - idx_OCT_shift
    print('OCT_selected_indices_shift:must start with 0', OCT_selected_indices_shift)
    CT_selected_indices_shift = CT_selected_indices - idx_CT_shift
    print('CT_selected_indices_shift:must start with 0', CT_selected_indices_shift)
    CT_sdf_cpr = CT_sdf_cpr[:,CT_selected_indices[0]:,...]
    OCT_sdf_cpr = OCT_sdf_cpr[:,OCT_selected_indices[0]:,...]

    # cat same length
    if CT_sdf_cpr.shape[1] < OCT_sdf_cpr.shape[1]:
        OCT_sdf_cpr = OCT_sdf_cpr[:, :CT_sdf_cpr.shape[1], ...]
    else:
        CT_sdf_cpr = CT_sdf_cpr[:, :OCT_sdf_cpr.shape[1], ...]
    #angl of bifurcation
    #RULES
    # 1. Moving image is CT refernce to target OCT
    # 2. Clockwise is negative, counter clockwise is positive
    # 3. Maintain the same SIGN orientation of the moving image for ALL bifurcations (ALL positive or ALL negative)
    # 4. Maximum angle is 180 degrees or 3.14 radians
    print('ang_for each bifurcation:',ang_big)
    angl  = ang_big
    theta_shift = angl


    t = torch.linspace(0, 1,CT_sdf_cpr.shape[1])
    vector = torch.full((CT_sdf_cpr.shape[1],1), float('nan'))

    for i,b in zip(CT_selected_indices_shift,torch.tensor(theta_shift)):
        vector[i] = b
    # parametrized spline
    coeffs = natural_cubic_spline_coeffs(t, vector)
    splines = NaturalCubicSpline(coeffs)
    theta_vec_cubic = splines.evaluate(t)
    mask = ~np.isnan(vector.numpy()).flatten()
    t_clean = t.numpy()[mask]
    vector_clean = vector.numpy()[mask].flatten()
    pchip = PchipInterpolator(t_clean, vector_clean)
    # add end point by cubic spline
    arr = pchip(t)  # Get the array
    arr[CT_selected_indices_shift[-1]:] = theta_vec_cubic[CT_selected_indices_shift[-1]:].reshape(-1)  # Modify the slice  

    #plt.scatter(CT_selected_indices_shift, angl, label='Original', color='red')
    #plt.plot(pchip(t)[:410], label='Pchip', color='orange')
    #plt.plot(theta_vec_cubic, label='Cubic', color='green')
    #plt.plot(arr, label='Final', color='blue')
    #plt.legend()
    ct_data_or  = torch.tensor(post_data[:,:,idx_CT_shift:CT_sdf_cpr.shape[1]+idx_CT_shift]).permute(2, 0, 1).unsqueeze(0)
    oct_data_or = torch.tensor(pre_data[:,:,idx_OCT_shift:OCT_sdf_cpr.shape[1]+idx_OCT_shift]).permute(2, 0, 1).unsqueeze(0)

    ph_or = ridgit_register(ct_data_or[0], torch.tensor(np.array(arr).reshape(-1,1)))
    ph = ridgit_register(CT_sdf_cpr[0], torch.tensor(np.array(arr).reshape(-1,1)))
    if only:
        oct_circl = detect_center(pre_data)
        ct_circl = detect_center(post_data)
    else:
        ct_circl = []
        for i in range(CT_sdf_cpr.shape[1]):
            orientation,centroid = center_circle(ph.unsqueeze(0)[0,i,...].detach().numpy()>0)
            ct_circl.append(centroid)

        oct_circl= []
        for i in range(OCT_sdf_cpr.shape[1]):
            orientation,centroid = center_circle(OCT_sdf_cpr[0,i,...].detach().numpy()>0)
            oct_circl.append(centroid)

        # Convert centers to numpy arrays for easier processing
        oct_circl = np.array([c for c in oct_circl if c[0] is not None and c[1] is not None])
        ct_circl = np.array([c for c in ct_circl if c[0] is not None and c[1] is not None])
   
        # Ensure there are enough points to apply the filter
        if len(oct_circl) > 31 and len(ct_circl) > 31:
            # Smooth the center coordinates
            oct_circl = np.copy(oct_circl)
            ct_circl = np.copy(ct_circl)
            
            for dim in range(2):  # For both x and y dimensions
                oct_circl[:, dim] = savgol_filter(oct_circl[:, dim], 31, 2)
                oct_circl[:, dim] = savgol_filter(ct_circl[:, dim], 31, 2)
   
    ph_or_circle = rotation(ph_or.unsqueeze(0),oct_circl,ct_circl)
    #ph_or_smooth_min = rotation(ph_or.unsqueeze(0),OCT_centers_smoothedmin,CT_centers_smoothedmin)
    #ph_translation_smooth_min = rotation(ph.unsqueeze(0),OCT_centers_smoothedmin,CT_centers_smoothedmin)
    ph_circle = rotation(ph.unsqueeze(0),oct_circl,ct_circl)



    patient = patient_id
    #d = torch.load('data_dict_bif_angl.pt')
    d = {}
    d[patient] = {}
    ############################ PRE-FINAL ############################

    d[patient]['bif_(PostTarget_StentMoving)'] = [organized_pairs,ang_big]
    d[patient]['bif_(PostTarget_StentMoving)'][0]
    # substract first tuple from all tuples in the list
    l = [(x[0]-d[patient]['bif_(PostTarget_StentMoving)'][0][0][0], x[1]-d[patient]['bif_(PostTarget_StentMoving)'][0][0][1]) for x in d[patient]['bif_(PostTarget_StentMoving)'][0]]
    fixed_image = oct_data_or.detach().numpy().squeeze().transpose(1, 2, 0)
    moving_image = np.array(ph_or_circle).transpose(1, 2, 0)

    bil = []
    for i in range(len(l)-1):
        
        lengthf = fixed_image[:,:,l[i][1]:(l[i+1][1])].shape[-1]
        lengthm = moving_image[:,:,l[i][0]:(l[i+1][0])].shape[-1]

        frames = moving_image[:,:,l[i][0]:(l[i+1][0])]
        print(lengthf-lengthm)
        if (lengthf-lengthm)>0 and (l[i+1][1]-l[i][1])>5:
            print('duplicate')
            num_duplicates = lengthf - lengthm 
            sq = duplicate_frames(frames, num_duplicates=num_duplicates, exclusion_fraction=0.2)
            bil.append(sq)
        if (lengthf-lengthm)<0 and (l[i+1][1]-l[i][1])>5:
            print('remove')
            num_removals = lengthm - lengthf 
            sq = remove_frames(frames, num_removals=num_removals, exclusion_fraction=0.2)
            bil.append(sq)

        if (lengthf-lengthm)==0 or (l[i+1][1]-l[i][1])<=5:
            print('normal')
            bil.append(frames)
    bil.append(moving_image[:,:,l[-1][0]:])
    br = np.concatenate((bil),axis=2)
    print(OCT_selected_indices_shift)
    show = True
    if show:
        images1 = (fixed_image * 255).astype(np.uint8)
        images2 = (br * 255).astype(np.uint8)
        # Get the dimensions of the first frame from each sequence
        channels1 = images1.shape[-1]
        channels2 = images2.shape[-1]

        # If the number of channels is different, make them the same
        if channels1 > channels2:
            images1 = images1[..., :channels2]
        elif channels2 > channels1:
            images2 = images2[..., :channels1]
        combined_frames = np.hstack((images1, images2)) 
        print(f"Combined frames shape: {combined_frames.shape}")
        # Save as an MP4 video
        fps = 30  # frames per second
        name  = patient_id + '.mp4'
        video_path = save_path+"/" + name
        with imageio.get_writer(video_path, fps=fps, macro_block_size=1) as writer:
            for frame in range(images1.shape[-1]):
                writer.append_data(combined_frames[:,:,frame])

        print("Video saved as output.mp4")

        # Path to your MP4 file
        print(save_path)
        video_path = save_path+"/" + name

        # Check the operating system
        if platform.system() == "Windows":
            os.startfile(video_path)
        elif platform.system() == "Darwin":  # macOS
            subprocess.call(["open", video_path])
        else:  # Linux and other UNIX-like systems
            subprocess.call(["xdg-open", video_path])

    patient = patient_id
    #create a dictionary to store the data pt format



    #d = torch.load('data.pt')
    d={}
    d[patient] = {}
    d[patient]['bif_(PostTarget_StentMoving)'] = [organized_pairs,ang_big]
    torch.save(d, save_path+"/"+patient+'meta_data.pt')
    #br.shape[-1]
    brs = nib.Nifti1Image(br, np.eye(4))
    nib.save(brs, save_path+"/"+patient+'moving.nii.gz')

    fixed_images = nib.Nifti1Image(fixed_image, np.eye(4))       
    nib.save(fixed_images,  save_path+"/"+patient+'target.nii.gz')
    return CT_selected_indices_shift, OCT_selected_indices_shift, fixed_image, br, organized_pairs,ang_big,ct_data_or

class FileDropApp:
    def __init__(self, mastert):
        self.mastert = mastert
        self.mastert.attributes('-fullscreen', True)
        self.result = None 
        # Variables to hold the file paths
        self.obligatory_files = []
        self.optional_files = []
        self.co_registration_thread = None
        # Store frames and labels for updating
        self.obligatory_files_frame = None
        self.optional_files_frame = None
        self.start_button = None
        self.patient_id = None
        self.task_thread = None
        self.save_path = tk.StringVar() 
        self.estimated_duration = 10  # Estimated duration of the task in seconds
        self.progress_increment = 100 / (self.estimated_duration * 10)  # Progress increment per 100ms
        self.progress_value = 0  # Initialize progress value
        # Configure the layout
        self.configure_layout()

    def configure_layout(self):
        """Set up the layout for the drag-and-drop areas and file upload buttons."""
        # Create a frame to hold the label and entry widget
        patient_frame = Frame(self.mastert)
        patient_frame.pack(pady=10)

        # Section for Patient ID label inside the frame
        patient_id_label = Label(patient_frame, text="Enter Patient ID", font=('Arial', 12, 'bold'))
        patient_id_label.pack(side='right', padx=10)

        # Entry widget for Patient ID inside the frame
        self.patient_id_entry = Entry(patient_frame, width=30)
        self.patient_id_entry.pack(side='right', padx=5)
        self.patient_id_entry.bind("<Return>", self.confirm_entry)
        # Frame to hold the entry and button side by side
        save_path_frame = tk.Frame(self.mastert)
        save_path_frame.pack(pady=30)  # Adjust the padding as needed

        # Entry widget to display the selected save path
        self.save_path_entry = Entry(save_path_frame, width=30, textvariable=self.save_path)
        self.save_path_entry.pack(side="left", padx=5)

        # Button to browse and select the save path
        browse_button = Button(save_path_frame, text="Choose Save Location", command=self.browse_save_path)
        browse_button.pack(side="left")

        # Section for Obligatory Files
        self.obligatory_files_frame = self.create_drop_section(
            "Patient Files (Required)",
            "Drag and drop files e.g. <patient_id>.nii.gz files here",
            self.browse_obligatory_file,
            self.drop_obligatory_file
        )
        
        # Section for Optional Files
        self.optional_files_frame = self.create_drop_section(
            "Segmentation Files (Optional)",
            "Drag and drop files e.g. <segmentation>.nii.gz files here",
            self.browse_optional_file,
            self.drop_optional_file
        )
 
        # Start Button (initially disabled)
        self.start_button = Button(self.mastert, text="Start", state="disabled", command=self.start_action)
        self.start_button.pack(pady=20)

        self.progress = ttk.Progressbar(self.mastert, orient="horizontal", length=200, mode="determinate")
        self.progress.pack(pady=20)

    def browse_save_path(self):
        """Open a dialog to select a folder to save the output."""
        path = filedialog.askdirectory()  # Open directory selection dialog
        if path:
            self.save_path.set(path)  # Set the selected path in the save_path variable
        self.update_start_button_state()
    # Function to confirm entry and retrieve the Patient ID
    def confirm_entry(self, event=None):
        patient_id = self.patient_id_entry.get()
        if patient_id:
            self.update_start_button_state()
            print(f"Patient ID entered: {patient_id}")
        else:
            self.update_start_button_state()
            print("No Patient ID entered.")
    def create_drop_section(self, title, drop_label, browse_command, drop_handler):
        """Creates a section with a label, drag-and-drop area, and file browse button."""
                # File upload button
        browse_btn = Button(self.mastert, text=f" {title}", command=browse_command)
        browse_btn.pack(padx=10)
        # Section label (title)
        #section_label = Label(self.mastert, text=title, font=('Arial', 12, 'bold'))
        #section_label.pack(pady=10)

        # Create a frame with a groove border for the drop area
        drop_frame = Frame(self.mastert, width=400, height=100, relief="groove", bd=2)
        drop_frame.pack(pady=10)
        drop_frame.pack_propagate(False)  # Prevent resizing based on contents

        # Drop area label
        drop_area_label = Label(drop_frame, text=drop_label, bg="#f0f0f0", fg="black")
        drop_area_label.pack(expand=True, fill="both")

        # Bind drag-and-drop event handler to the frame
        drop_frame.drop_target_register(DND_FILES)
        drop_frame.dnd_bind('<<Drop>>', drop_handler)



        # Frame to display the added files and delete buttons
        files_frame = Frame(self.mastert)
        files_frame.pack(pady=10)

        return files_frame

    # File Browse for Obligatory Files
    def browse_obligatory_file(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("NumPy Files", "*.gz")])
        if file_paths:
            for file_path in file_paths:
                self.process_file(file_path, self.obligatory_files, "Required", self.obligatory_files_frame)

    # File Drop for Obligatory Files
    def drop_obligatory_file(self, event):
        file_paths = self.split_file_paths(event.data)
        for file_path in file_paths:
            if file_path.endswith('.gz'):
                self.process_file(file_path, self.obligatory_files, "Required", self.obligatory_files_frame)
            else:
                messagebox.showerror("Invalid File", "Please drop a .nii.gz file")

    # File Browse for Optional Files
    def browse_optional_file(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("NumPy Files", "**.gz")])
        if file_paths:
            for file_path in file_paths:
                self.process_file(file_path, self.optional_files, "Optional", self.optional_files_frame)
        self.update_start_button_state()
    # File Drop for Optional Files
    def drop_optional_file(self, event):
        file_paths = self.split_file_paths(event.data)
        for file_path in file_paths:
            if file_path.endswith('.gz'):
                self.process_file(file_path, self.optional_files, "Optional", self.optional_files_frame)
            else:
                messagebox.showerror("Invalid File", "Please drop a .nii.gz file")
        self.update_start_button_state()
    def split_file_paths(self, data):
        """Split file paths when multiple files are dragged and dropped."""
        return data.split(" ")  # Split file paths by space for multi-file drag

    # Process the selected file (either via browse or drop)
    def process_file(self, file_path, file_list, file_type, frame):
        if len(file_list) < 2:
            file_list.append(file_path)
            self.update_file_list(file_list, frame, file_type)
            self.update_start_button_state()
        else:
            messagebox.showwarning(f"{file_type} Files", f"Only 2 {file_type.lower()} files are allowed!")

    def update_file_list(self, file_list, frame, file_type):
        """Update the display of files and add delete buttons."""
        # Clear the existing content in the frame
        for widget in frame.winfo_children():
            widget.destroy()

        # Create labels and delete buttons for each file
        for idx, file_path in enumerate(file_list):
            file_label = Label(frame, text=os.path.basename(file_path), fg="green")
            file_label.grid(row=idx, column=0, sticky='w', padx=5)

            # Create a delete button next to each file
            delete_button = Button(frame, text="Delete", fg="red", command=lambda f=file_path: self.delete_file(f, file_list, frame, file_type))
            delete_button.grid(row=idx, column=1, padx=5)

    def delete_file(self, file_path, file_list, frame, file_type):
        """Delete a file from the list and update the display."""
        file_list.remove(file_path)
        self.update_file_list(file_list, frame, file_type)
        self.update_start_button_state()

    def update_start_button_state(self):
        """Enable or disable the Start button based on file conditions."""
        if len(self.obligatory_files) == 2 and (len(self.optional_files) == 0 or len(self.optional_files) == 2) and self.save_path.get() and self.patient_id_entry.get():
            print("All files and information are ready")
            self.start_button.config(state="normal")


        else:
            self.start_button.config(state="disabled")

    def start_action(self):
        self.progress_value = 0  # Reset the progress bar value
        self.progress['value'] = 0  # Reset the progress bar display

        # Start the long-running task in a separate thread
        self.task_thread = threading.Thread(target=self.run_task)
        self.task_thread.start()

        # Start automatically updating the progress bar
        self.update_progress()
    def update_progress(self):
        # Continue updating the progress bar until self.result is available
        if not self.result:  # If the result is not yet available
            if self.progress['value'] < 100:
                self.progress['value'] += 3  # Increment progress bar by 2% every 500ms
            self.mastert.after(800, lambda: self.update_progress())
            #self.mastert.after(500, self.update_progress)  # Call this function again after 500ms
        else:
            # Once the result is ready, set the progress bar to 100% and continue with logic
            self.progress['value'] = 100
            print("Loading Complete")
            self.handle_result()
    def handle_result(self):
            """Handle the result once the task is done."""
            if self.result:
                CT_sdf_cpr, OCT_sdf_cpr, Area_CT_ori, Area_OCT, post_data, pre_data, img_data_pre, img_data_post, sdf_ct, sdf_oct, only = self.result
                if only:
                   organized_pairs, ang_big, an = gui_play(Area_CT_ori, Area_OCT, CT_sdf_cpr, OCT_sdf_cpr, post_data, pre_data, only)
                else:
                    organized_pairs, ang_big, an = gui_play_all(Area_CT_ori, Area_OCT, CT_sdf_cpr, OCT_sdf_cpr, post_data, pre_data, only)
                gui_co_rec(organized_pairs, ang_big, an, post_data, pre_data, CT_sdf_cpr, OCT_sdf_cpr, patient_id=self.patient_id_entry.get(), only=only, save_path=self.save_path.get())

    def run_task(self):
 
        """Action triggered when the Start button is clicked."""
        if len(self.obligatory_files) == 2 and len(self.optional_files) == 2:
            self.patient_id = self.patient_id_entry.get()
            #self.master.destroy()  # Close the current window
            patient = self.patient_id
            """
            patient = '101021'
            moving_path_mask=patient_id+'Final.nii.gz'
            target_path_mask=patient_id+'Pre.nii.gz'
            moving_path_raw=patient_id+'Final_0000.nii.gz'
            target_path_raw=patient_id+'Pre_0000.nii.gz'
            """
            for file_path in self.optional_files:
                if 'Final' in file_path:
                    moving_path_mask = file_path
                elif 'Pre' in file_path:
                    target_path_mask = file_path
            for file_path in self.obligatory_files:
                if 'Final' in file_path:
                    moving_path_raw = file_path
                elif 'Pre' in file_path:
                    target_path_raw = file_path
            CT_sdf_cpr,OCT_sdf_cpr,Area_CT_ori,Area_OCT,post_data,pre_data,img_data_pre,img_data_post,sdf_ct,sdf_oct=upload_all_files_seg(moving_path_mask=moving_path_mask,target_path_mask=target_path_mask,moving_path_raw=moving_path_raw,target_path_raw=target_path_raw)
            self.result = (CT_sdf_cpr, OCT_sdf_cpr, Area_CT_ori, Area_OCT, post_data, pre_data, img_data_pre, img_data_post, sdf_ct, sdf_oct,False)
            self.progress['value'] = 100  # Ensure the progress bar is full
        else:
            self.patient_id = self.patient_id_entry.get()
            #self.master.destroy()  # Close the current window

            patient = self.patient_id
            
            """
            patient = '101021'
            moving_path_raw=patient_id+'Final_0000.nii.gz'
            target_path_raw=patient_id+'Pre_0000.nii.gz'
            """
            for file_path in self.obligatory_files + self.optional_files:
                if 'Final_0000' in file_path:
                    moving_path_raw = file_path
                elif 'Pre_0000' in file_path:
                    target_path_raw = file_path
             
            CT_sdf_cpr,OCT_sdf_cpr,Area_CT_ori,Area_OCT,post_data,pre_data,img_data_pre,img_data_post,sdf_ct,sdf_oct=upload_all_files_raw(moving_path_raw=moving_path_raw,target_path_raw=target_path_raw)
            self.result = (CT_sdf_cpr, OCT_sdf_cpr, Area_CT_ori, Area_OCT, post_data, pre_data, img_data_pre, img_data_post, sdf_ct, sdf_oct,True)
            self.progress['value'] = 100  # Ensure the progress bar is full

if __name__ == "__main__":
    # Create TkinterDnD window
    root = TkinterDnD.Tk()
    root.title("File Drop Application")
    
    # Instantiate the app
    app = FileDropApp(mastert=root)
    root.mainloop()




    