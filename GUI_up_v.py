import numpy as np
from scipy.signal import find_peaks
import ipywidgets as widgets
import tkinter as tk
from tkinter import ttk
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from skimage.transform import resize
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec
from tkinter import messagebox
# Colors for picks
from orientation_algorith import *
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np

distinct_colors_extended = [
    '#FF0000',  # red
    '#008000',  # green
    '#0000FF',  # blue
    '#00FFFF',  # cyan
    '#FF00FF',  # magenta
    '#FFFF00',  # yellow
    '#000000',  # black
    '#FFA500',  # orange
    '#800080',  # purple
    '#A52A2A',  # brown
    '#00FF00',  # lime
    '#FFC0CB',  # pink
    '#008080',  # teal
    '#E6E6FA',  # lavender
    '#808000',  # olive
    '#800000',  # maroon
    '#000080',  # navy
    '#808080',  # gray
    '#FA8072',  # salmon
    '#FFD700',  # gold
    '#ADD8E6',  # lightblue
    '#006400',  # darkgreen
    '#EE82EE',  # violet
    '#40E0D0',  # turquoise
    '#8B0000',  # darkred
    '#4B0082',  # indigo
    '#FFDAB9',  # peach
    '#FF7F50',  # coral
    '#DDA0DD',  # plum
    '#CCCCFF',  # periwinkle
    '#F0E68C',  # khaki
    '#FFDB58',  # mustard
    '#36454F',  # charcoal
    '#800020',  # burgundy
    '#F0FFFF',  # azure
    '#F5F5DC',  # beige
    '#008000',  # emerald
    '#D2691E',  # chocolate
    '#DC143C',  # crimson
    '#FF00FF',  # fuchsia
    '#FFBF00',  # amber
    '#FFFFF0',  # ivory
    '#F5F5F5',  # pearl
    '#082567',  # sapphire
    '#E0115F',  # ruby
    '#DA70D6',  # orchid
    '#00FF7F',  # springgreen
    '#7FFFD4',  # aquamarine
    '#6A5ACD',  # slateblue
    '#1E90FF',  # dodgerblue
    '#B22222',  # firebrick
    '#228B22',  # forestgreen
    '#F0FFF0',  # honeydew
    '#FF69B4',  # hotpink
    '#F08080',  # lightcoral
    '#7B68EE',  # mediumslateblue
    '#98FB98',  # palegreen
    '#A0522D',  # sienna
    '#C0C0C0',  # silver
    '#FFFAFA',  # snow
    '#D2B48C',  # tan
    '#F5DEB3'   # wheat
]




# Custom slider class - just a simple slider with a label
class CustomSlider(tk.Frame):
    def __init__(self, parent, from_=0, to=100, resolution=1, orient=tk.HORIZONTAL, label="", command=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.value = from_
        self.from_ = from_
        self.to = to
        self.resolution = resolution
        self.command = command

        # Use grid geometry manager for more precise placement
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.canvas = tk.Canvas(self, width=280, height=50, bg='light grey')
        self.canvas.grid(row=0, column=0, sticky="ew")
        
        self.label = tk.Label(self, text=label, bg='light grey')
        self.label.grid(row=0, column=1, sticky="ns")

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.draw_slider()

    def draw_slider(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(10, 10, 270, 30, fill="#A9A9A9")  # Adjusted for the new canvas width
        position = (self.value - self.from_) / (self.to - self.from_) * 260 + 15  # Adjusted for the new canvas width
        self.canvas.create_oval(position - 10, 10, position + 10, 30, fill="white", outline="")

    def on_click(self, event):
        self.update_value(event.x)

    def on_drag(self, event):
        self.update_value(event.x)

    def update_value(self, x):
        x = max(10, min(x, 270))  # Adjusted for the new canvas width
        self.value = ((x - 10) / 260) * (self.to - self.from_) + self.from_  # Adjusted for the new canvas width
        self.value = round(self.value / self.resolution) * self.resolution
        self.draw_slider()
        if self.command:
            self.command(self.value)
def organize_matched_pairs(highlighted_points):
    # Extract CT and OCT peak values
    ct_peaks = [value for name, value in highlighted_points if name == "CT"]
    oct_peaks = [value for name, value in highlighted_points if name == "OCT"]

    # Sort both lists to ensure pairing from min to max values
    ct_peaks_sorted = sorted(ct_peaks)
    oct_peaks_sorted = sorted(oct_peaks)

    # Ensure there's a one-to-one correspondence by checking if the lengths match
    if len(ct_peaks_sorted) == len(oct_peaks_sorted):
        pairs = list(zip(ct_peaks_sorted, oct_peaks_sorted))
    else:
        pairs = []

    return pairs
# function to return the selected peaks
def organize_matched_pairss(saving_orientation,saving_orientation_angl_show):
    # Extract CT and OCT peak values
    sorted_keys_ct = sorted(saving_orientation['ct'].keys())
    print("sorted keys ct",sorted_keys_ct)
    st_ct_k = sorted(saving_orientation_angl_show['ct'].keys())
    sorted_keys_oct = sorted(saving_orientation['oct'].keys())
    print("sorted keys oct",sorted_keys_oct)
    # Sort both lists to ensure pairing from min to max values
    ct_peaks_sorted = sorted_keys_ct
    oct_peaks_sorted = sorted_keys_oct
    an = [saving_orientation_angl_show['ct'][i] for i in st_ct_k]
    ang_big = [saving_orientation['ct'][i][-1] for i in sorted_keys_ct]
    # Ensure there's a one-to-one correspondence by checking if the lengths match
    if len(ct_peaks_sorted) == len(oct_peaks_sorted):
        pairs = list(zip(ct_peaks_sorted, oct_peaks_sorted))
    else:
        pairs = []
    print(pairs,ang_big)
    return pairs,ang_big,an

class PeaksMatcherGUI(ttk.Frame):
    def __init__(self, master,Area_CT=None, Area_OCT=None,CT_image=None,OCT_image=None,or_ct=None,or_oct=None):
        super().__init__(master)
        #find local maxima
        self.angle_between_common = 0
        self.vector_ct = []
        self.vector_oct = []
        self.or_ct = or_ct  
        self.or_oct = or_oct
        self.plot_type = tk.StringVar(value='circle')
        #self.plot_type = tk.StringVar(value='circle')
        self.ct_frames = CT_image.detach().numpy()
        self.oct_frames = OCT_image.detach().numpy()
        self.ct_current_frame_line = None
        self.oct_current_frame_line = None
        self.ct_frame_index = 0  # Initialize frame index for CT
        self.oct_frame_index = 0  # Initialize frame index for OCT
        self.Area_CT,self.Area_OCT = Area_CT.numpy(),Area_OCT.numpy()
        peaks_CT, _ = find_peaks(Area_CT, height=None)  # You can adjust parameters here for sensitivity
        peaks_OCT, _ = find_peaks(Area_OCT, height=None)
        self.peaks_CT, self.peaks_OCT = np.array(peaks_CT), np.array(peaks_OCT)
        self.pack(fill=tk.BOTH, expand=tk.YES)
        self.matched_pairs = set()  # To store matched pairs
        self.x_shift_ct = 0
        self.y_shift_ct = 0
        self.x_shift_oct = 0
        self.y_shift_oct = 0
        self.saving_orientation = {'oct':{},'ct':{}}
        self.saving_orientation_angl_show = {'ct':{}}
        self.points = {'ct': [], 'oct': []}

        # Matched Pairs window - modified to use Treeview
        self.pairs_window = tk.Toplevel(self)
        self.pairs_window.title("Matched Pairs")

        
        # Define the columns
        self.pairs_tree = ttk.Treeview(self.pairs_window, columns=("CT", "OCT"))
        self.pairs_tree.heading("#0", text="ID", anchor=tk.W)
        self.pairs_tree.heading("CT", text="CT Peak")
        self.pairs_tree.heading("OCT", text="OCT Peak")
        
        # Format the columns
        self.pairs_tree.column("#0", anchor=tk.W, width=40)
        self.pairs_tree.column("CT", anchor=tk.CENTER, width=80)
        self.pairs_tree.column("OCT", anchor=tk.CENTER, width=80)
        
        self.pairs_tree.pack(padx=20, pady=20, fill=tk.BOTH, expand=tk.YES)
        

        # Bind double-click and selection event to the same method
        self.show_btn = ttk.Button(self.pairs_window, text='Show Selected Pair', command=self.on_item_action)
        self.show_btn.pack(pady=10)
        self.delete_btn = ttk.Button(self.pairs_window, text='Delete Selected Pair', command=self.delete_selected_pair)
        self.delete_btn.pack(pady=10)
        self.create_widgets()
        self.pairs_window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        

    def on_item_action(self):

        selected_items = self.pairs_tree.selection()

        if selected_items:
            for item in selected_items:
    
                # Extract information from the item to identify the corresponding pair in your data structure
                item_values = self.pairs_tree.item(item, 'values')
                print(f"Show pairs {item_values}")

                self.visualize_pair(item_values)

    def visualize_pair(self,item_values):
        
        # Add RadioButtons to choose plot type
        self.plot_window = tk.Toplevel(self)
        self.plot_window.title("Orientation")
        #self.save_btn = ttk.Button(self.plot_window, text='Save orientation', command=self.save_current_orientation)
        #self.save_btn.pack(pady=10)
        self.fig_or, self.axes = plt.subplots(2, 1, figsize=(8, 6))
        
        radio_frame = tk.Frame(self.plot_window)
        radio_frame.pack()
        options = [
            ("Circle", 'circle'), ("PCA", 'pca'), ("PCA Change Direction", 'pca_change_dir'),
            ("Ellipse", 'ellipse'), ("Convex Hull", 'convex_hull'), ("Triangle", 'triangle'),
            ("Triangle Max Edges", 'triangle_max_edges'), ("Manual Assignment", 'manual assignment')
        ]

        for text, value in options:
            radio = tk.Radiobutton(radio_frame, text=text, variable=self.plot_type, value=value,
                                   command=lambda: self.show_plots(item_values))
            radio.pack(side=tk.LEFT)


        #self.plot_window.mainloop()

        #remove_button = tk.Button(self.plot_window, text="Remove Label", command=self.plot_window.destroy())
        #remove_button.pack(pady=20)

        

        #self.show_plots()

 
        #self.show_plots(item_values)
        
        self.canvass = FigureCanvasTkAgg(self.fig_or, master=self.plot_window)
        self.canvass.draw()
        self.canvass.get_tk_widget().pack(fill=tk.BOTH, expand=tk.YES)
        self.save_btn = ttk.Button(self.plot_window, text='Save orientation', command=self.save_current_orientation)
        self.save_btn.pack(pady=10)
 
       
      
    
    def save_current_orientation(self):
        if self.plot_type.get() == 'manual assignment':
            self.saving_orientation_angl_show['ct'][self.ct_value]= [self.angle_between_common,self.direction]
            pass
            #saving happend in the on_click_draw function automatically
            #self.saving_orientation['ct'][self.ct_value]= [self.plot_type.get(),self.points['ct']]
            #self.saving_orientation['oct'][self.oct_value]= [self.plot_type.get(),self.points['oct']]
        else:
            self.saving_orientation['oct'][self.oct_value]= [self.plot_type.get()]
            self.saving_orientation['ct'][self.ct_value]= [self.plot_type.get()]
            self.saving_orientation_angl_show['ct'][self.ct_value]= [self.angle_between_common,self.direction]
            
        
        print("Save orientation button clicked",self.plot_type.get(),self.ct_value,self.oct_value)
        self.show_save_popup()

    def show_save_popup(self):
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        messagebox.showinfo("Save Points", "Points have been saved successfully!")
        root.destroy()
    def show_plots(self,item_values):
        
        self.axes[0].cla()
        self.axes[0].axis('off')
        self.axes[1].cla()
        self.axes[1].axis('off')
        # Create new subplots

        print("Showing plots...",self.plot_type.get())
        

        ct_value = int(item_values[0])
        self.ct_value = ct_value
        oct_value = int(item_values[1])
        self.oct_value = oct_value
        #plot_window = tk.Toplevel(self)
        #plot_window.title("CT and OCT Images")
        #fig, axes = plt.subplots(2, 1, figsize=(8, 6))

        # Customize CT plot
        
        def calculate_angle_between_vectors(v1, v2):
                dot_product = np.dot(v1, v2)
                magnitude_v1 = np.linalg.norm(v1)
                magnitude_v2 = np.linalg.norm(v2)
                cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
                angle = np.arccos(cos_angle)
                # Calculate the cross product and determine the direction
                cross_product = np.cross(v1, v2)
                if cross_product > 0:  # Assuming 2D vectors for simplicity
                    direction = "left"
                elif cross_product < 0:
                    direction = "right"
                else:
                    direction = "collinear"
                print(direction)
                
                return np.degrees(angle),direction

        if self.plot_type.get() == 'circle':


            # First set of calculations and plotting
            orientation1, centroid1, circle1, direction_vertex1 = calculate_orientation_circle(self.ct_frames[0, ct_value, ...])
            im1 = self.axes[0].imshow(self.ct_frames[0, ct_value, ...], cmap=plt.cm.gray, interpolation='nearest')
            im2 = self.axes[0].imshow(self.or_ct[0, ct_value, ...], cmap=plt.cm.viridis, alpha=0.7, interpolation='bilinear')
            self.axes[0].plot(centroid1[1], centroid1[0], 'ro')
            self.axes[0].arrow(centroid1[1], centroid1[0], direction_vertex1[1] - centroid1[1], direction_vertex1[0] - centroid1[0], 
                            head_width=5, head_length=5, fc='r', ec='r', label='Orientation')
            self.axes[0].add_artist(circle1)

            # Second set of calculations and plotting
            orientation2, centroid2, circle2, direction_vertex2 = calculate_orientation_circle(self.oct_frames[0, oct_value , ...])
            im3 = self.axes[1].imshow(self.oct_frames[0, oct_value , ...], cmap=plt.cm.gray, interpolation='nearest')
            im4 = self.axes[1].imshow(self.or_oct[0, oct_value , ...], cmap=plt.cm.viridis, alpha=0.7, interpolation='bilinear')
            self.axes[1].plot(centroid2[1], centroid2[0], 'ro')
            self.axes[1].arrow(centroid2[1], centroid2[0], direction_vertex2[1] - centroid2[1], direction_vertex2[0] - centroid2[0],
                            head_width=5, head_length=5, fc='r', ec='r', label='Orientation')

            # Calculate the angle between the two arrows
            vector1 = np.array([direction_vertex1[1] - centroid1[1], direction_vertex1[0] - centroid1[0]])
            vector2 = np.array([direction_vertex2[1] - centroid2[1], direction_vertex2[0] - centroid2[0]])
            self.angle_between_common,self.direction = calculate_angle_between_vectors(vector1, vector2)

            # Add the angle text to the first axes
            self.axes[0].text(0.05, 0.95, f'Angle: {self.angle_between_common:.2f}°', transform=self.axes[0].transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

            print(f'Angle between the two arrows: {self.angle_between_common:.2f} degrees')

            """
            orientation, centroid, circle, direction_vertex = calculate_orientation_circle(self.ct_frames[0, ct_value, ...])
            #self.axes[0].imshow(self.ct_frames[0, ct_value, ...], cmap='GnBu')
            im1 = self.axes[0].imshow(self.ct_frames[0, ct_value, ...], cmap=plt.cm.gray,interpolation='nearest')
            im2 = self.axes[0].imshow(self.or_ct[0, ct_value, ...], cmap=plt.cm.viridis, alpha=0.7, interpolation='bilinear')
            self.axes[0].plot(centroid[1], centroid[0], 'ro')
            self.axes[0].arrow(centroid[1], centroid[0], direction_vertex[1] - centroid[1], direction_vertex[0] - centroid[0], 
                          head_width=5, head_length=5, fc='r', ec='r', label='Orientation')
            self.axes[0].add_artist(circle)
            orientation, centroid, circle, direction_vertex = calculate_orientation_circle(self.oct_frames[0, oct_value , ...])
            #self.axes[1].imshow(self.oct_frames[0, oct_value , ...], cmap='Oranges')
            im3 = self.axes[1].imshow(self.oct_frames[0, oct_value , ...], cmap=plt.cm.gray,interpolation='nearest')
            im4 = self.axes[1].imshow(self.or_oct[0, oct_value , ...], cmap=plt.cm.viridis, alpha=0.7, interpolation='bilinear')

            self.axes[1].plot(centroid[1], centroid[0], 'ro')
            self.axes[1].arrow(centroid[1], centroid[0], direction_vertex[1] - centroid[1], direction_vertex[0] - centroid[0],
                            head_width=5, head_length=5, fc='r', ec='r', label='Orientation')
            #self.axes[1].add_artist(circle)
            """
        elif self.plot_type.get() == 'pca':
            """
            center_of_mass, x1_major, x0, y1_major, y0, x2_major, y2_major = pca(self.ct_frames[0, ct_value, ...])
            #self.axes[0].imshow(self.ct_frames[0, ct_value, ...], cmap='GnBu')
            im1 = self.axes[0].imshow(self.ct_frames[0, ct_value, ...], cmap=plt.cm.gray,interpolation='nearest')
            im2 = self.axes[0].imshow(self.or_ct[0, ct_value, ...], cmap=plt.cm.viridis, alpha=0.7, interpolation='bilinear')
            self.axes[0].plot(center_of_mass[1], center_of_mass[0], 'go')
            self.axes[0].arrow(center_of_mass[1], center_of_mass[0], x1_major - x0, y1_major - y0, head_width=5, head_length=5, fc='blue', ec='blue', label='Major Axis (Positive)')
            self.axes[0].arrow(center_of_mass[1], center_of_mass[0], x2_major - x0, y2_major - y0, head_width=5, head_length=5, fc='red', ec='red', label='Major Axis (Negative)')
            center_of_mass, x1_major, x0, y1_major, y0, x2_major, y2_major = pca(self.oct_frames[0, oct_value , ...])
            #self.axes[1].imshow(self.oct_frames[0, oct_value , ...], cmap='Oranges')
            im3 = self.axes[1].imshow(self.oct_frames[0, oct_value , ...], cmap=plt.cm.gray,interpolation='nearest')
            im4 = self.axes[1].imshow(self.or_oct[0, oct_value , ...], cmap=plt.cm.viridis, alpha=0.7, interpolation='bilinear')
            self.axes[1].plot(center_of_mass[1], center_of_mass[0], 'go')
            self.axes[1].arrow(center_of_mass[1], center_of_mass[0], x1_major - x0, y1_major - y0, head_width=5, head_length=5, fc='blue', ec='blue', label='Major Axis (Positive)')
            self.axes[1].arrow(center_of_mass[1], center_of_mass[0], x2_major - x0, y2_major - y0, head_width=5, head_length=5, fc='red', ec='red', label='Major Axis (Negative)')
            #plt.show()
            """

            center_of_mass1, x1_major1, x01, y1_major1, y01, x2_major1, y2_major1 = pca(self.ct_frames[0, ct_value, ...])
            im1 = self.axes[0].imshow(self.ct_frames[0, ct_value, ...], cmap=plt.cm.gray, interpolation='nearest')
            im2 = self.axes[0].imshow(self.or_ct[0, ct_value, ...], cmap=plt.cm.viridis, alpha=0.7, interpolation='bilinear')
            self.axes[0].plot(center_of_mass1[1], center_of_mass1[0], 'go')
            self.axes[0].arrow(center_of_mass1[1], center_of_mass1[0], x1_major1 - x01, y1_major1 - y01, head_width=5, head_length=5, fc='blue', ec='blue', label='Major Axis (Positive)')
            self.axes[0].arrow(center_of_mass1[1], center_of_mass1[0], x2_major1 - x01, y2_major1 - y01, head_width=5, head_length=5, fc='red', ec='red', label='Major Axis (Negative)')

            # Second set of calculations and plotting for oct_frames
            center_of_mass2, x1_major2, x02, y1_major2, y02, x2_major2, y2_major2 = pca(self.oct_frames[0, oct_value , ...])
            im3 = self.axes[1].imshow(self.oct_frames[0, oct_value , ...], cmap=plt.cm.gray, interpolation='nearest')
            im4 = self.axes[1].imshow(self.or_oct[0, oct_value , ...], cmap=plt.cm.viridis, alpha=0.7, interpolation='bilinear')
            self.axes[1].plot(center_of_mass2[1], center_of_mass2[0], 'go')
            self.axes[1].arrow(center_of_mass2[1], center_of_mass2[0], x1_major2 - x02, y1_major2 - y02, head_width=5, head_length=5, fc='blue', ec='blue', label='Major Axis (Positive)')
            self.axes[1].arrow(center_of_mass2[1], center_of_mass2[0], x2_major2 - x02, y2_major2 - y02, head_width=5, head_length=5, fc='red', ec='red', label='Major Axis (Negative)')

            # Calculate the angle between the positive major axes
            vector1_positive = np.array([x1_major1 - x01, y1_major1 - y01])
            vector2_positive = np.array([x1_major2 - x02, y1_major2 - y02])
            angle_between_positive_axes,self.direction = calculate_angle_between_vectors(vector1_positive, vector2_positive)

            # Calculate the angle between the negative major axes
            vector1_negative = np.array([x2_major1 - x01, y2_major1 - y01])
            vector2_negative = np.array([x2_major2 - x02, y2_major2 - y02])
            self.angle_between_common,self.direction = calculate_angle_between_vectors(vector1_negative, vector2_negative)

            # Add the angle text to the first axes
            self.axes[0].text(0.05, 0.95, f'Positive Angle: {self.angle_between_common:.2f}°\nNegative Angle: {self.angle_between_common:.2f}°',
                            transform=self.axes[0].transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

            print(f'Angle between the positive major axes: {angle_between_positive_axes:.2f} degrees')
            print(f'Angle between the negative major axes: {self.angle_between_common:.2f} degrees')
        elif self.plot_type.get() == 'pca_change_dir':
            """
            center_of_mass,change_point_start,change_point_end,dx,dy = pca_change_direction(self.ct_frames[0, ct_value, ...])
            #self.axes[0].imshow(self.ct_frames[0, ct_value, ...], cmap='GnBu')
            im1 = self.axes[0].imshow(self.ct_frames[0, ct_value, ...], cmap=plt.cm.gray,interpolation='nearest')
            im2 = self.axes[0].imshow(self.or_ct[0, ct_value, ...], cmap=plt.cm.viridis, alpha=0.7, interpolation='bilinear')
            self.axes[0].plot(center_of_mass[1], center_of_mass[0], 'go')
            self.axes[0].arrow(center_of_mass[1], center_of_mass[0], dx, dy, head_width=5, head_length=5, fc='blue', ec='blue', label='Major Axis (Positive)')
            self.axes[0].plot(change_point_start[1], change_point_start[0], 'ro')
            self.axes[0].plot(change_point_end[1], change_point_end[0], 'ro')
            center_of_mass,change_point_start,change_point_end,dx,dy = pca_change_direction(self.oct_frames[0, oct_value , ...])
            #self.axes[1].imshow(self.oct_frames[0, oct_value , ...], cmap='Oranges')
            im3 = self.axes[1].imshow(self.oct_frames[0, oct_value , ...], cmap=plt.cm.gray,interpolation='nearest')
            im4 = self.axes[1].imshow(self.or_oct[0, oct_value , ...], cmap=plt.cm.viridis, alpha=0.7, interpolation='bilinear')
            self.axes[1].plot(center_of_mass[1], center_of_mass[0], 'go')
            self.axes[1].arrow(center_of_mass[1], center_of_mass[0], dx, dy, head_width=5, head_length=5, fc='blue', ec='blue', label='Major Axis (Positive)')
            self.axes[1].plot(change_point_start[1], change_point_start[0], 'ro')
            self.axes[1].plot(change_point_end[1], change_point_end[0], 'ro')
            #plt.show()
            """
            # First set of calculations and plotting for ct_frames
            center_of_mass1, change_point_start1, change_point_end1, dx1, dy1 = pca_change_direction(self.ct_frames[0, ct_value, ...])
            im1 = self.axes[0].imshow(self.ct_frames[0, ct_value, ...], cmap=plt.cm.gray, interpolation='nearest')
            im2 = self.axes[0].imshow(self.or_ct[0, ct_value, ...], cmap=plt.cm.viridis, alpha=0.7, interpolation='bilinear')
            self.axes[0].plot(center_of_mass1[1], center_of_mass1[0], 'go')
            self.axes[0].arrow(center_of_mass1[1], center_of_mass1[0], dx1, dy1, head_width=5, head_length=5, fc='blue', ec='blue', label='Major Axis (Positive)')
            self.axes[0].plot(change_point_start1[1], change_point_start1[0], 'ro')
            self.axes[0].plot(change_point_end1[1], change_point_end1[0], 'ro')

            # Second set of calculations and plotting for oct_frames
            center_of_mass2, change_point_start2, change_point_end2, dx2, dy2 = pca_change_direction(self.oct_frames[0, oct_value , ...])
            im3 = self.axes[1].imshow(self.oct_frames[0, oct_value , ...], cmap=plt.cm.gray, interpolation='nearest')
            im4 = self.axes[1].imshow(self.or_oct[0, oct_value , ...], cmap=plt.cm.viridis, alpha=0.7, interpolation='bilinear')
            self.axes[1].plot(center_of_mass2[1], center_of_mass2[0], 'go')
            self.axes[1].arrow(center_of_mass2[1], center_of_mass2[0], dx2, dy2, head_width=5, head_length=5, fc='blue', ec='blue', label='Major Axis (Positive)')
            self.axes[1].plot(change_point_start2[1], change_point_start2[0], 'ro')
            self.axes[1].plot(change_point_end2[1], change_point_end2[0], 'ro')

            # Calculate the angle between the major axes
            vector1_major = np.array([dx1, dy1])
            vector2_major = np.array([dx2, dy2])
            self.angle_between_common,self.direction = calculate_angle_between_vectors(vector1_major, vector2_major)

            # Add the angle text to the first axes
            self.axes[0].text(0.05, 0.95, f'Angle: {self.angle_between_common:.2f}°', transform=self.axes[0].transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

            print(f'Angle between the major axes: {self.angle_between_common:.2f} degrees')
        elif self.plot_type.get() == 'ellipse':
            """
            center_of_mass,farthest_point,y0,x0,y1,x1,rr,cc = fit_ellipse(self.ct_frames[0, ct_value, ...])
            #self.axes[0].imshow(self.ct_frames[0, ct_value, ...], cmap='GnBu')
            im1 = self.axes[0].imshow(self.ct_frames[0, ct_value, ...], cmap=plt.cm.gray,interpolation='nearest')
            im2 = self.axes[0].imshow(self.or_ct[0, ct_value, ...], cmap=plt.cm.viridis, alpha=0.7, interpolation='bilinear')
            self.axes[0].plot(center_of_mass[1], center_of_mass[0], 'go')
            self.axes[0].arrow(y0, x0, y1 - y0, x1 - x0, head_width=5, head_length=5, fc='r', ec='r', label='Orientation')
            self.axes[0].plot(cc, rr, 'b-', label='Fitted Ellipse')
            self.axes[0].arrow(center_of_mass[1], center_of_mass[0], farthest_point[1] - center_of_mass[1], farthest_point[0] - center_of_mass[0],
            head_width=5, head_length=5, fc='g', ec='g', label='Farthest Point')
            center_of_mass,farthest_point,y0,x0,y1,x1,rr,cc = fit_ellipse(self.oct_frames[0, oct_value , ...])
            #self.axes[1].imshow(self.oct_frames[0, oct_value , ...], cmap='Oranges')
            im3 = self.axes[1].imshow(self.oct_frames[0, oct_value , ...], cmap=plt.cm.gray,interpolation='nearest')
            im4 = self.axes[1].imshow(self.or_oct[0, oct_value , ...], cmap=plt.cm.viridis, alpha=0.7, interpolation='bilinear')
            self.axes[1].plot(center_of_mass[1], center_of_mass[0], 'go')
            self.axes[1].arrow(y0, x0, y1 - y0, x1 - x0, head_width=5, head_length=5, fc='r', ec='r', label='Orientation')
            self.axes[1].plot(cc, rr, 'b-', label='Fitted Ellipse')
            self.axes[1].arrow(center_of_mass[1], center_of_mass[0], farthest_point[1] - center_of_mass[1], farthest_point[0] - center_of_mass[0],
            head_width=5, head_length=5, fc='g', ec='g', label='Farthest Point')
            #plt.show()
            """
            # First set of calculations and plotting for ct_frames
            center_of_mass1, farthest_point1, y01, x01, y11, x11, rr1, cc1 = fit_ellipse(self.ct_frames[0, ct_value, ...])
            im1 = self.axes[0].imshow(self.ct_frames[0, ct_value, ...], cmap=plt.cm.gray, interpolation='nearest')
            im2 = self.axes[0].imshow(self.or_ct[0, ct_value, ...], cmap=plt.cm.viridis, alpha=0.7, interpolation='bilinear')
            self.axes[0].plot(center_of_mass1[1], center_of_mass1[0], 'go')
            self.axes[0].arrow(y01, x01, y11 - y01, x11 - x01, head_width=5, head_length=5, fc='r', ec='r', label='Orientation')
            self.axes[0].plot(cc1, rr1, 'b-', label='Fitted Ellipse')
            self.axes[0].arrow(center_of_mass1[1], center_of_mass1[0], farthest_point1[1] - center_of_mass1[1], farthest_point1[0] - center_of_mass1[0],
                            head_width=5, head_length=5, fc='g', ec='g', label='Farthest Point')

            # Second set of calculations and plotting for oct_frames
            center_of_mass2, farthest_point2, y02, x02, y12, x12, rr2, cc2 = fit_ellipse(self.oct_frames[0, oct_value, ...])
            im3 = self.axes[1].imshow(self.oct_frames[0, oct_value, ...], cmap=plt.cm.gray, interpolation='nearest')
            im4 = self.axes[1].imshow(self.or_oct[0, oct_value, ...], cmap=plt.cm.viridis, alpha=0.7, interpolation='bilinear')
            self.axes[1].plot(center_of_mass2[1], center_of_mass2[0], 'go')
            self.axes[1].arrow(y02, x02, y12 - y02, x12 - x02, head_width=5, head_length=5, fc='r', ec='r', label='Orientation')
            self.axes[1].plot(cc2, rr2, 'b-', label='Fitted Ellipse')
            self.axes[1].arrow(center_of_mass2[1], center_of_mass2[0], farthest_point2[1] - center_of_mass2[1], farthest_point2[0] - center_of_mass2[0],
                            head_width=5, head_length=5, fc='g', ec='g', label='Farthest Point')

            # Calculate the angle between the major axes of the two ellipses
            vector1_major = np.array([farthest_point1[1] - center_of_mass1[1], farthest_point1[0] - center_of_mass1[0]])
            vector2_major = np.array([farthest_point2[1] - center_of_mass2[1], farthest_point2[0] - center_of_mass2[0]])
            self.angle_between_common,self.direction = calculate_angle_between_vectors(vector1_major, vector2_major)

            # Add the angle text to the first axes
            self.axes[0].text(0.05, 0.95, f'Angle: {self.angle_between_common:.2f}°', transform=self.axes[0].transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

            print(f'Angle between the major axes: {self.angle_between_common:.2f} degrees')
        elif self.plot_type.get() == 'convex_hull':
            """
            #self.axes[0].imshow(self.ct_frames[0, ct_value, ...], cmap='GnBu')
            im1 = self.axes[0].imshow(self.ct_frames[0, ct_value, ...], cmap=plt.cm.gray,interpolation='nearest')
            im2 = self.axes[0].imshow(self.or_ct[0, ct_value, ...], cmap=plt.cm.viridis, alpha=0.7, interpolation='bilinear')
            intersection,longest_edges, coords, hull = convex_hull(self.ct_frames[0, ct_value, ...])
            for simplex in hull.simplices:
                self.axes[0].plot(coords[simplex, 1], coords[simplex, 0], 'b-')
            for edge in longest_edges:
                self.axes[0].plot([edge[0][1], edge[1][1]], [edge[0][0], edge[1][0]], 'g-', linewidth=2)
            self.axes[0].plot(intersection[1], intersection[0], 'ro')
            intersection,longest_edges, coords, hull = convex_hull(self.oct_frames[0, oct_value , ...])
            #self.axes[1].imshow(self.oct_frames[0, oct_value , ...], cmap='Oranges')
            im3 = self.axes[1].imshow(self.oct_frames[0, oct_value , ...], cmap=plt.cm.gray,interpolation='nearest')
            im4 = self.axes[1].imshow(self.or_oct[0, oct_value , ...], cmap=plt.cm.viridis, alpha=0.7, interpolation='bilinear')
            for simplex in hull.simplices:
                self.axes[1].plot(coords[simplex, 1], coords[simplex, 0], 'b-')
            for edge in longest_edges:
                self.axes[1].plot([edge[0][1], edge[1][1]], [edge[0][0], edge[1][0]], 'g-', linewidth=2)
            self.axes[1].plot(intersection[1], intersection[0], 'ro')
            #plt.show()
            """
            # First set of calculations and plotting for ct_frames
            im1 = self.axes[0].imshow(self.ct_frames[0, ct_value, ...], cmap=plt.cm.gray, interpolation='nearest')
            im2 = self.axes[0].imshow(self.or_ct[0, ct_value, ...], cmap=plt.cm.viridis, alpha=0.7, interpolation='bilinear')
            intersection1, longest_edges1, coords1, hull1 = convex_hull(self.ct_frames[0, ct_value, ...])
            for simplex in hull1.simplices:
                self.axes[0].plot(coords1[simplex, 1], coords1[simplex, 0], 'b-')
            for edge in longest_edges1:
                self.axes[0].plot([edge[0][1], edge[1][1]], [edge[0][0], edge[1][0]], 'g-', linewidth=2)
            self.axes[0].plot(intersection1[1], intersection1[0], 'ro')

            # Second set of calculations and plotting for oct_frames
            im3 = self.axes[1].imshow(self.oct_frames[0, oct_value, ...], cmap=plt.cm.gray, interpolation='nearest')
            im4 = self.axes[1].imshow(self.or_oct[0, oct_value, ...], cmap=plt.cm.viridis, alpha=0.7, interpolation='bilinear')
            intersection2, longest_edges2, coords2, hull2 = convex_hull(self.oct_frames[0, oct_value, ...])
            for simplex in hull2.simplices:
                self.axes[1].plot(coords2[simplex, 1], coords2[simplex, 0], 'b-')
            for edge in longest_edges2:
                self.axes[1].plot([edge[0][1], edge[1][1]], [edge[0][0], edge[1][0]], 'g-', linewidth=2)
            self.axes[1].plot(intersection2[1], intersection2[0], 'ro')

            # Calculate the angle between the longest edges
            # Assuming longest_edges is a list of tuples containing the start and end points of the edges
            vector1_edge = np.array([longest_edges1[0][1][1] - longest_edges1[0][0][1], longest_edges1[0][1][0] - longest_edges1[0][0][0]])
            vector2_edge = np.array([longest_edges2[0][1][1] - longest_edges2[0][0][1], longest_edges2[0][1][0] - longest_edges2[0][0][0]])
            self.angle_between_common,self.direction = calculate_angle_between_vectors(vector1_edge, vector2_edge)

            # Add the angle text to the first axes
            self.axes[0].text(0.05, 0.95, f'Angle: {self.angle_between_common:.2f}°', transform=self.axes[0].transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

            print(f'Angle between the longest edges: {self.angle_between_common:.2f} degrees')
        elif self.plot_type.get() == 'triangle':
            """
            #self.axes[0].imshow(self.ct_frames[0, ct_value, ...], cmap='GnBu')
            im1 = self.axes[0].imshow(self.ct_frames[0, ct_value, ...], cmap=plt.cm.gray,interpolation='nearest')
            im2 = self.axes[0].imshow(self.or_ct[0, ct_value, ...], cmap=plt.cm.viridis, alpha=0.7, interpolation='bilinear')
            centroid, triangle_points, longest_edge_vertex = triangle(self.ct_frames[0, ct_value, ...])
            self.axes[0].plot(centroid[1], centroid[0], 'ro')
            self.axes[0].arrow(centroid[1], centroid[0], longest_edge_vertex[1] - centroid[1], longest_edge_vertex[0] - centroid[0], 
             head_width=5, head_length=5, fc='r', ec='r', label='Orientation')
            self.axes[0].plot(triangle_points[:, 1], triangle_points[:, 0], 'b-', label='Fitted Triangle')
            centroid, triangle_points, longest_edge_vertex = triangle(self.oct_frames[0, oct_value , ...])
            #self.axes[1].imshow(self.oct_frames[0, oct_value , ...], cmap='Oranges')
            im3 = self.axes[1].imshow(self.oct_frames[0, oct_value , ...], cmap=plt.cm.gray,interpolation='nearest')
            im4 = self.axes[1].imshow(self.or_oct[0, oct_value , ...], cmap=plt.cm.viridis, alpha=0.7, interpolation='bilinear')
            self.axes[1].plot(centroid[1], centroid[0], 'ro')
            self.axes[1].arrow(centroid[1], centroid[0], longest_edge_vertex[1] - centroid[1], longest_edge_vertex[0] - centroid[0],
                head_width=5, head_length=10, fc='r', ec='r', label='Orientation')
            self.axes[1].plot(triangle_points[:, 1], triangle_points[:, 0], 'b-', label='Fitted Triangle')
            #plt.show()
            """
            # First set of calculations and plotting for ct_frames
            im1 = self.axes[0].imshow(self.ct_frames[0, ct_value, ...], cmap=plt.cm.gray, interpolation='nearest')
            im2 = self.axes[0].imshow(self.or_ct[0, ct_value, ...], cmap=plt.cm.viridis, alpha=0.7, interpolation='bilinear')
            centroid1, triangle_points1, longest_edge_vertex1 = triangle(self.ct_frames[0, ct_value, ...])
            self.axes[0].plot(centroid1[1], centroid1[0], 'ro')
            self.axes[0].arrow(centroid1[1], centroid1[0], longest_edge_vertex1[1] - centroid1[1], longest_edge_vertex1[0] - centroid1[0], 
                            head_width=5, head_length=5, fc='r', ec='r', label='Orientation')
            self.axes[0].plot(triangle_points1[:, 1], triangle_points1[:, 0], 'b-', label='Fitted Triangle')

            # Second set of calculations and plotting for oct_frames
            im3 = self.axes[1].imshow(self.oct_frames[0, oct_value, ...], cmap=plt.cm.gray, interpolation='nearest')
            im4 = self.axes[1].imshow(self.or_oct[0, oct_value, ...], cmap=plt.cm.viridis, alpha=0.7, interpolation='bilinear')
            centroid2, triangle_points2, longest_edge_vertex2 = triangle(self.oct_frames[0, oct_value, ...])
            self.axes[1].plot(centroid2[1], centroid2[0], 'ro')
            self.axes[1].arrow(centroid2[1], centroid2[0], longest_edge_vertex2[1] - centroid2[1], longest_edge_vertex2[0] - centroid2[0],
                            head_width=5, head_length=10, fc='r', ec='r', label='Orientation')
            self.axes[1].plot(triangle_points2[:, 1], triangle_points2[:, 0], 'b-', label='Fitted Triangle')

            # Calculate the angle between the longest edge orientations
            vector1_longest_edge = np.array([longest_edge_vertex1[1] - centroid1[1], longest_edge_vertex1[0] - centroid1[0]])
            vector2_longest_edge = np.array([longest_edge_vertex2[1] - centroid2[1], longest_edge_vertex2[0] - centroid2[0]])
            self.angle_between_common,self.direction = calculate_angle_between_vectors(vector1_longest_edge, vector2_longest_edge)

            # Add the angle text to the first axes
            self.axes[0].text(0.05, 0.95, f'Angle: {self.angle_between_common:.2f}°', transform=self.axes[0].transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

            print(f'Angle between the longest edges: {self.angle_between_common:.2f} degrees')
        elif self.plot_type.get() == 'triangle_max_edges':
            #self.axes[0].imshow(self.ct_frames[0, ct_value, ...], cmap='GnBu')
            """
            im1 = self.axes[0].imshow(self.ct_frames[0, ct_value, ...], cmap=plt.cm.gray,interpolation='nearest')
            im2 = self.axes[0].imshow(self.or_ct[0, ct_value, ...], cmap=plt.cm.viridis, alpha=0.7, interpolation='bilinear')  
            centroid, triangle_points, common_vertex = triangle_max_edges(self.ct_frames[0, ct_value, ...])
            self.axes[0].plot(centroid[1], centroid[0], 'ro')
            self.axes[0].arrow(centroid[1], centroid[0], common_vertex[1] - centroid[1], common_vertex[0] - centroid[0], 
             head_width=5, head_length=5, fc='r', ec='r', label='Orientation')
            self.axes[0].plot(common_vertex[1], common_vertex[0], 'go', markersize=10, label='Common Vertex')
            self.axes[0].plot(triangle_points[:, 1], triangle_points[:, 0], 'b-', label='Fitted Triangle')
            centroid, triangle_points, common_vertex = triangle_max_edges(self.oct_frames[0, oct_value , ...])
            #self.axes[1].imshow(self.oct_frames[0, oct_value , ...], cmap='Oranges')
            im3 = self.axes[1].imshow(self.oct_frames[0, oct_value , ...], cmap=plt.cm.gray,interpolation='nearest')
            im4 = self.axes[1].imshow(self.or_oct[0, oct_value , ...], cmap=plt.cm.viridis, alpha=0.7, interpolation='bilinear')
            self.axes[1].plot(centroid[1], centroid[0], 'ro')
            self.axes[1].arrow(centroid[1], centroid[0], common_vertex[1] - centroid[1], common_vertex[0] - centroid[0],
                head_width=5, head_length=5, fc='r', ec='r', label='Orientation')
            self.axes[1].plot(common_vertex[1], common_vertex[0], 'go', markersize=10, label='Common Vertex')
            self.axes[1].plot(triangle_points[:, 1], triangle_points[:, 0], 'b-', label='Fitted Triangle')
            #plt.show()
            """
            # First set of calculations and plotting for ct_frames
            im1 = self.axes[0].imshow(self.ct_frames[0, ct_value, ...], cmap=plt.cm.gray, interpolation='nearest')
            im2 = self.axes[0].imshow(self.or_ct[0, ct_value, ...], cmap=plt.cm.viridis, alpha=0.7, interpolation='bilinear')  
            centroid1, triangle_points1, common_vertex1 = triangle_max_edges(self.ct_frames[0, ct_value, ...])
            self.axes[0].plot(centroid1[1], centroid1[0], 'ro')
            self.axes[0].arrow(centroid1[1], centroid1[0], common_vertex1[1] - centroid1[1], common_vertex1[0] - centroid1[0], 
                            head_width=5, head_length=5, fc='r', ec='r', label='Orientation')
            self.axes[0].plot(common_vertex1[1], common_vertex1[0], 'go', markersize=10, label='Common Vertex')
            self.axes[0].plot(triangle_points1[:, 1], triangle_points1[:, 0], 'b-', label='Fitted Triangle')

            # Second set of calculations and plotting for oct_frames
            im3 = self.axes[1].imshow(self.oct_frames[0, oct_value, ...], cmap=plt.cm.gray, interpolation='nearest')
            im4 = self.axes[1].imshow(self.or_oct[0, oct_value, ...], cmap=plt.cm.viridis, alpha=0.7, interpolation='bilinear')
            centroid2, triangle_points2, common_vertex2 = triangle_max_edges(self.oct_frames[0, oct_value, ...])
            self.axes[1].plot(centroid2[1], centroid2[0], 'ro')
            self.axes[1].arrow(centroid2[1], centroid2[0], common_vertex2[1] - centroid2[1], common_vertex2[0] - centroid2[0],
                            head_width=5, head_length=5, fc='r', ec='r', label='Orientation')
            self.axes[1].plot(common_vertex2[1], common_vertex2[0], 'go', markersize=10, label='Common Vertex')
            self.axes[1].plot(triangle_points2[:, 1], triangle_points2[:, 0], 'b-', label='Fitted Triangle')

            # Calculate the angle between the common vertex orientations
            vector1_common_vertex = np.array([common_vertex1[1] - centroid1[1], common_vertex1[0] - centroid1[0]])
            vector2_common_vertex = np.array([common_vertex2[1] - centroid2[1], common_vertex2[0] - centroid2[0]])
            self.angle_between_common,self.direction = calculate_angle_between_vectors(vector1_common_vertex, vector2_common_vertex)

            # Add the angle text to the first axes
            self.axes[0].text(0.05, 0.95, f'Angle: {self.angle_between_common:.2f}°', transform=self.axes[0].transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

            print(f'Angle between the common vertex orientations: {self.angle_between_common:.2f} degrees')

        elif self.plot_type.get() == 'manual assignment':
            im1 = self.axes[0].imshow(self.ct_frames[0, ct_value, ...], cmap=plt.cm.gray,interpolation='nearest')
            im2 = self.axes[0].imshow(self.or_ct[0, ct_value, ...], cmap=plt.cm.viridis, alpha=0.7, interpolation='bilinear')  
            im3 = self.axes[1].imshow(self.oct_frames[0, oct_value , ...], cmap=plt.cm.gray,interpolation='nearest')
            im4 = self.axes[1].imshow(self.or_oct[0, oct_value , ...], cmap=plt.cm.viridis, alpha=0.7, interpolation='bilinear')
            #self.axes[0].imshow(self.ct_frames[0, ct_value, ...], cmap='GnBu')
            #self.axes[1].imshow(self.oct_frames[0, oct_value , ...], cmap='Oranges')
            #im1 = self.axes[0].imshow(self.ct_frames[0, ct_value, ...], cmap=plt.cm.gray,interpolation='nearest')
            #im2 = self.axes[0].imshow(self.or_ct[0, ct_value, ...], cmap=plt.cm.viridis, alpha=0.7, interpolation='bilinear')
            #im3 = self.axes[1].imshow(self.oct_frames[0, oct_value , ...], cmap=plt.cm.gray,interpolation='nearest')
            #im4 = self.axes[1].imshow(self.or_oct[0, oct_value , ...], cmap=plt.cm.viridis, alpha=0.7, interpolation='bilinear')

            self.canvass.mpl_connect('button_press_event', self.on_click_draw)
            self.canvass.mpl_connect('key_press_event', self.on_key_press)
            
            self.draw_vectors('ct')
            self.draw_vectors('oct')

            #plt.show()
           
        self.axes[0].set_title(f'Adjusted bifurcation № {ct_value}', fontsize=12)
        #self.axes[0].set_xlabel('X-axis Label', fontsize=12)
        #self.axes[0].set_ylabel('Y-axis Label', fontsize=12)
        self.axes[0].grid(True)

        # Customize OCT plot
        
        self.axes[1].set_title(f'Ground truth bifurcation № {oct_value}', fontsize=12)
        #self.axes[1].set_xlabel('X-axis Label', fontsize=12)
        #self.axes[1].set_ylabel('Y-axis Label', fontsize=12)
        self.axes[1].grid(True)
        self.canvass.draw()

    def on_key_press(self, event):
        if event.key == 'r':  # Press 'r' to remove the last picked point
            if self.points['ct']:
                self.points['ct'].pop()
                print(f"Removed last CT point. Current CT points: {self.points['ct']}")
            if self.points['oct']:
                self.points['oct'].pop()
                print(f"Removed last OCT point. Current OCT points: {self.points['oct']}")
            self.update_display()

    def update_display(self):
        for image_type in ['ct', 'oct']:
            self.draw_vectors(image_type)
        self.canvass.draw()
    def on_click_draw(self, event):
        self.text_obj = None 
        def calculate_angle_between_vectors(v1, v2):
                dot_product = np.dot(v1, v2)
                magnitude_v1 = np.linalg.norm(v1)
                magnitude_v2 = np.linalg.norm(v2)
                cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
                angle = np.arccos(cos_angle)

                cross_product = np.cross(v1, v2)
                if cross_product > 0:  # Assuming 2D vectors for simplicity
                    direction = "left"
                elif cross_product < 0:
                    direction = "right"
                else:
                    direction = "collinear"
                print(direction)
                
                return np.degrees(angle),direction
        axe = event.inaxes
        if axe == self.axes[0]:
            self.points['ct'].append((event.xdata, event.ydata))
            print(f"CT points: {self.points['ct']}")
        elif axe == self.axes[1]:
            self.points['oct'].append((event.xdata, event.ydata))
        
        if len(self.points['ct']) == 2:
            self.saving_orientation['ct'][self.ct_value]= [self.plot_type.get(),self.points['ct']]
            start_point_ct = self.points['ct'][0]
            end_point_ct = self.points['ct'][1]

            self.vector_ct = np.array([end_point_ct[0] - start_point_ct[0], end_point_ct[1] - start_point_ct[1]])
            self.draw_vectors('ct')



        if len(self.points['oct']) == 2:
            self.saving_orientation['oct'][self.oct_value]= [self.plot_type.get(),self.points['oct']]

            start_point_oct = self.points['oct'][0]
            end_point_oct = self.points['oct'][1]
            self.vector_oct = np.array([end_point_oct[0] - start_point_oct[0], end_point_oct[1] - start_point_oct[1]])
            self.draw_vectors('oct')
        if  len(self.vector_ct)!=0 and len(self.vector_oct)!=0:
                self.angle_between_common,self.direction = calculate_angle_between_vectors(self.vector_ct, self.vector_oct)
                            # Clear the previous text if it exists
                if self.text_obj is not None:
                   self.text_obj.remove()
                self.text_obj = self.axes[0].text(0.05, 0.95, f'Angle: {self.angle_between_common:.2f}°', transform=self.axes[0].transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))
                print(f'Angle between the arrows: {self.angle_between_common:.2f} degrees')



        self.canvass.draw()
    
    def draw_vectors(self, image_type):
        """
        self.text_obj = None 
        def calculate_angle_between_vectors(v1, v2):
                dot_product = np.dot(v1, v2)
                magnitude_v1 = np.linalg.norm(v1)
                magnitude_v2 = np.linalg.norm(v2)
                cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
                angle = np.arccos(cos_angle)
                return np.degrees(angle)
        """
        ct_value = self.ct_value
        oct_value = self.oct_value
        axe = self.axes[0] if image_type == 'ct' else self.axes[1]
        axe.clear()
        #im1 = axe.imshow(self.ct_frames[0, ct_value, ...], cmap=plt.cm.gray,interpolation='nearest')
        #im2 = axe.imshow(self.or_ct[0, ct_value, ...], cmap=plt.cm.viridis, alpha=0.7, interpolation='bilinear')
        #im3 = axe.imshow(self.oct_frames[0, oct_value , ...], cmap=plt.cm.gray,interpolation='nearest')
        #im4 = axe.imshow(self.or_oct[0, oct_value , ...], cmap=plt.cm.viridis, alpha=0.7, interpolation='bilinear')
        image_data = self.ct_frames[0,ct_value,...] if image_type == 'ct' else self.oct_frames[0,oct_value,...]
        cmap = 'GnBu' if image_type == 'ct' else 'Oranges'
        #axe.imshow(image_data, cmap=cmap)
        im1 = self.axes[0].imshow(self.ct_frames[0, ct_value, ...], cmap=plt.cm.gray,interpolation='nearest')
        im2 = self.axes[0].imshow(self.or_ct[0, ct_value, ...], cmap=plt.cm.viridis, alpha=0.7, interpolation='bilinear')  
        im3 = self.axes[1].imshow(self.oct_frames[0, oct_value , ...], cmap=plt.cm.gray,interpolation='nearest')
        im4 = self.axes[1].imshow(self.or_oct[0, oct_value , ...], cmap=plt.cm.viridis, alpha=0.7, interpolation='bilinear')

        #plt.show()
   
        if len(self.points[image_type]) == 2:
            start_point = self.points[image_type][0]
            print(f"Start point: {start_point}")
            end_point = self.points[image_type][1]
            axe.arrow(start_point[0], start_point[1], end_point[0] - start_point[0], end_point[1] - start_point[1],
                     head_width=5, head_length=10, fc='red', ec='red')
            """
            if  len(self.vector_ct)!=0 and len(self.vector_oct)!=0:
                angle_between_arrows = calculate_angle_between_vectors(self.vector_ct, self.vector_oct)
                            # Clear the previous text if it exists
                if self.text_obj is not None:
                   self.text_obj.remove()
                self.text_obj = axe.text(0.05, 0.95, f'Angle: {angle_between_arrows:.2f}°', transform=axe.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))
                print(f'Angle between the arrows: {angle_between_arrows:.2f} degrees')
            """
            self.points[image_type] = []

     
        self.canvass.draw()

    def create_widgets(self):
        # Slider Frame for adjustments
        print("Creating widgets...")
        slider_frame = ttk.Frame(self)
        slider_frame.pack(fill=tk.Y)
        # Widget Frame for sliders
        widget_frame = ttk.Frame(self)
        widget_frame.pack(fill=tk.X)
        self.create_sliders(slider_frame,widget_frame)
        # Plotting Frame
        self.plot_frame = ttk.Frame(self)
        self.plot_frame.pack(fill=tk.BOTH, expand=tk.YES)
        self.figg = plt.figure(figsize=(12, 6))

        titles = [
        "Masks of Register frames",
        "Original Register frames",
        "Mask of Target frames",
        "Original Target frames"
        ]
        gs = GridSpec(2, 4, figure=self.figg, wspace=0.1, hspace=0.2)  # Adjust wspace and hspace as needed
  
        self.ax = [
            self.figg.add_subplot(gs[0, :]), 
            self.figg.add_subplot(gs[1, 0:1]), 
            self.figg.add_subplot(gs[1, 1:2]), 
            self.figg.add_subplot(gs[1, 2:3]), 
            self.figg.add_subplot(gs[1, 3:4])
        ]
        # Loop through each axis and apply the patch and title
        for i, ax in enumerate(self.ax[1:]):
            # Set the title
            ax.text(0.5, -0.1, titles[i], horizontalalignment='center', fontsize=12, fontname='Arial', transform=ax.transAxes)
            if i == 0 or i == 1:
                
                # Display the image (for demonstration, using the same image for all)
                ax.imshow(self.ct_frames[0][self.ct_frame_index], cmap='GnBu')
            if i == 2 or i == 3:
                ax.imshow(self.oct_frames[0][self.oct_frame_index], cmap='Oranges')
            # Add rounded boundaries around the plot
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Create a FancyBboxPatch with rounded corners
            bbox = patches.FancyBboxPatch((0.1, 0.1), 0.8, 0.8, boxstyle="round,pad=0.1", linewidth=1, edgecolor='black', facecolor='none', transform=ax.transAxes, clip_on=False)
            ax.add_patch(bbox)
            ax.axis('off')
      
 
        self.canvas = FigureCanvasTkAgg(self.figg, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=tk.YES)
        # Initial plot
        # Widget Frame for sliders (place below plots)  # Updated

        self.canvas.mpl_connect('pick_event', self.on_pick)
        self.update_plot()
        self.add_pairs_button = tk.Button(self.plot_frame, text="Add Pairs", command=self.add_pairs)
        self.add_pairs_button.pack(pady=20)
    def add_pairs(self):
        self.ct_frame_index
        self.ct_frame_index
        
            # Only add to matched_pairs if it's a new selection, following the alternation rule
        self.matched_pairs.add(("CT", self.ct_frame_index))
        self.matched_pairs.add(("OCT", self.oct_frame_index))
  
        print(f"Added (CT, {self.ct_frame_index}) to matched pairs.")
        print(f"Added (OCT, {self.oct_frame_index}) to matched pairs.")
        self.update_matched_pairs_window()
        print("Add Pairs button clicked")
    

    def create_sliders(self, frame,widget_frame):
        l = np.max(self.Area_CT) - np.max(self.Area_OCT)
        if l >= 0:
            v = np.max(self.Area_CT) /3 
        else:
            v = np.max(self.Area_OCT) /3
        d = len(self.Area_CT) /2

        # CT X Shift Slider
        CustomSlider(frame, from_=-d, to=d, label='X Shift CT', command=self.update_x_shift_ct).pack(side='left',padx=5, pady=5)
        # CT Y Shift Slider
        CustomSlider(frame, from_=-v, to=v, label='Y Shift CT', command=self.update_y_shift_ct).pack(side='left',padx=5, pady=5)
        # OCT X Shift Slider
        CustomSlider(frame, from_=-d, to=d, label='X Shift OCT', command=self.update_x_shift_oct).pack(side='left',padx=5, pady=5)
        # OCT Y Shift Slider
        CustomSlider(frame, from_=-v, to=v, label='Y Shift OCT', command=self.update_y_shift_oct).pack(side='left',padx=5, pady=5)
        # Slider for CT frames
        self.ct_frame_index_slider = tk.Scale(widget_frame, label="CT Frame Index",
                                              from_=0, to=self.ct_frames.shape[1] - 1,
                                              orient=tk.HORIZONTAL, command=self.update_ct_frame)
        self.ct_frame_index_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)

        # Slider for OCT frames
        self.oct_frame_index_slider = tk.Scale(widget_frame, label="OCT Frame Index",
                                               from_=0, to=self.oct_frames.shape[1] - 1,
                                               orient=tk.HORIZONTAL, command=self.update_oct_frame)
        self.oct_frame_index_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)


    def update_x_shift_ct(self, value):
        self.x_shift_ct = int(value)
        self.update_plot()

    def update_y_shift_ct(self, value):
        self.y_shift_ct = float(value)
        self.update_plot()

    def update_x_shift_oct(self, value):
        self.x_shift_oct = int(value)
        self.update_plot()

    def update_y_shift_oct(self, value):
        self.y_shift_oct = float(value)
        self.update_plot()

    def update_ct_frame(self, value):
        self.ct_frame_index = int(value)
        titles = [
        "Masks of Register frames",
        "Original Register frames",
        ]
        # Loop through each axis and apply the patch and title
        for i, ax in enumerate([self.ax[1],self.ax[2]]):
            # Set the title
            ax.clear()
            ax.text(0.5, -0.1, titles[i], horizontalalignment='center', fontsize=12, fontname='Arial', transform=ax.transAxes)
            if i == 0:  
                # Display the image (for demonstration, using the same image for all)
                ax.imshow(self.ct_frames[0, self.ct_frame_index, ...], cmap='GnBu')
                
            if i == 1:
                #ax.imshow(self.ct_frames[0, self.ct_frame_index, ...], cmap='Blues')
                ax.imshow(self.or_ct[0, self.ct_frame_index, ...])
            # Add rounded boundaries around the plot
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Create a FancyBboxPatch with rounded corners
            bbox = patches.FancyBboxPatch((0.1, 0.1), 0.8, 0.8, boxstyle="round,pad=0.1", linewidth=1, edgecolor='black', facecolor='none', transform=ax.transAxes, clip_on=False)
            ax.add_patch(bbox)
            ax.axis('off')  
        # Remove the previous vertical line if it exists
        if self.ct_current_frame_line is not None:
            self.ct_current_frame_line.remove()
        
        # Add the new vertical line and store its reference
        self.ct_current_frame_line = self.ax[0].axvline(self.ct_frame_index+self.x_shift_ct, color='blue', linestyle='--', label='Current Frame')
      
        self.canvas.draw()
        print('update ct frame')
    def on_closing(self):
        self.show_saving_orientation()
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            
            self.master.destroy()


    def show_saving_orientation(self):
        def overlap_vusial(idx,CT_image,center_ct,outermost_ct,OCT_image,center_oct,outermost_oct):
            CT_image=CT_image
            OCT_image=OCT_image
            original_height, original_width = OCT_image.shape[:2]

            OCT_image = resize(OCT_image[:, :], (500, 500), anti_aliasing=False)
            CT_image = resize(CT_image[:, :], (500, 500), anti_aliasing=False)
            scale_x = 500 / original_width
            scale_y = 500 / original_height
            
            # Adapt center_oct coordinates to the new resized dimensions
            adapted_center_oct_x = center_oct[0] * scale_x
            adapted_center_oct_y = center_oct[1] * scale_y
            center_oct = (adapted_center_oct_x, adapted_center_oct_y)
            adapted_outermost_oct_x = outermost_oct[0] * scale_x
            adapted_outermost_oct_y = outermost_oct[1] * scale_y
            outermost_oct = (adapted_outermost_oct_x, adapted_outermost_oct_y)
            outermost_ct=outermost_ct
            center_ct=center_ct
            shift_x, shift_y = np.array(center_ct) - np.array(center_oct)
            aligned_OCT_image = ndimage.shift(OCT_image, shift=(shift_x,shift_y))
            
            composite_image = 0.5 * CT_image + 0.5 * aligned_OCT_image
            self.axs[idx, 2].clear()
            self.axs[idx, 2].imshow(composite_image, cmap='gray')
            self.axs[idx, 2].set_title(f'Overlapping bifurcation')
            shifted_outermost_oct = np.array(outermost_oct) + [shift_x, shift_y]
                # Calculate angle between vectors in radians
            vector_ct = np.array([center_ct[0] - outermost_ct[1], center_ct[1] - outermost_ct[0]])
            vector_oct = np.array([center_ct[0] - shifted_outermost_oct[1], center_ct[1] - shifted_outermost_oct[0]])
            cos_theta = np.dot(vector_ct, vector_oct) / (np.linalg.norm(vector_ct) * np.linalg.norm(vector_oct))
            #angle_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Angle in radians
            angle_radians = np.arccos(cos_theta)  # Clipping for numerical stability
            #angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))  # Clipping for numerical stability
            angle = np.degrees(np.arccos(cos_theta))  # Angle in degrees
            # Determine the direction of rotation using the cross product
            cross_product = np.cross(vector_ct,vector_oct)
            print('cross product',cross_product)
            if cross_product > 0:
                direction = '+'  # Counterclockwise
            else:
                direction = '-'  # Clockwise

            # Apply the direction to the angle
            angle_radians *= -1 if direction == '-' else 1

            # Convert the angle to degrees
            angle = np.degrees(angle_radians)
            """
            """
            import math
            OCT_delta_x = shifted_outermost_oct[0] - center_ct[0]
            OCT_delta_y = center_ct[1] - shifted_outermost_oct[1]
            OCT_angle_in_radians = math.atan2(OCT_delta_y, OCT_delta_x)
            print('OCT_angle_in_radians',OCT_angle_in_radians)
            
            CT_delta_x = outermost_ct[0] - center_ct[0]
            CT_delta_y = center_ct[1] - outermost_ct[1]
            CT_angle_in_radians = math.atan2(CT_delta_y, CT_delta_x)
      
            print('CT_angle_in_radians',CT_angle_in_radians)
            ### cna be different here 
            rad = CT_angle_in_radians - OCT_angle_in_radians
            angl_deg= rad*180/math.pi
       
            self.axs[idx, 2].text(5, 40, f'Rad: {rad:.2f}°', color='red', fontsize=10)
            # Display angle in radians
            self.axs[idx, 2].text(5, 5, f'Degree: {angl_deg:.5f} ', color='red', fontsize=10)
            self.saving_orientation['ct'][i].append(rad)

            #shifted_outermost_oct = np.array(outermost_ct) + [shift_x, shift_y]
            print('outmostprint',outermost_ct)
            print('outmostprint',outermost_oct)
            print(center_ct)
            print(center_oct)
            
                # Draw vectors from the common center to the outermost points
            self.axs[idx, 2].arrow(center_ct[0], center_ct[1],
                           outermost_ct[1]-center_ct[1], outermost_ct[0]-center_ct[0],
                           head_width=5, head_length=10, fc='cyan', ec='cyan')
            self.axs[idx, 2].arrow(center_ct[0], center_ct[1],
                     shifted_outermost_oct[1] - center_ct[1], shifted_outermost_oct[0] - center_ct[0],
                     head_width=5, head_length=10, fc='magenta', ec='magenta')
            plt.axis('off')
            del CT_image
            del OCT_image
            del original_height
            del original_width
            del scale_x
            del scale_y
            del adapted_center_oct_x
            del adapted_center_oct_y
            del adapted_outermost_oct_x
            del adapted_outermost_oct_y
            del shift_x
            del shift_y
            del aligned_OCT_image
            del composite_image
            del shifted_outermost_oct
            del vector_ct
            del vector_oct
            del cos_theta
            del angle_radians
            del angle
        # Create a new window to display the saving orientation
        self.orientation_window = tk.Toplevel(self)
        self.orientation_window.title("Saved Data Visualization")
        self.orientation_window_plot_frame = ttk.Frame(self.orientation_window)
        self.orientation_window_plot_frame.pack(fill=tk.BOTH, expand=tk.YES)
        sorted_keys_ct = sorted(self.saving_orientation['ct'].keys())
        print("sorted keys ct",sorted_keys_ct)

        sorted_keys_oct = sorted(self.saving_orientation['oct'].keys())
        print("sorted keys oct",sorted_keys_oct)
       

        # Create a figure with subplots
        self.figz, self.axs = plt.subplots(len(sorted_keys_ct), 3, figsize=(15, 15))
 
        self.axs = self.axs.reshape(-1,3)
        # Generate plots for each sorted key
        for idx, (i, j) in enumerate(zip(sorted_keys_ct, sorted_keys_oct)):
            print(self.saving_orientation['ct'][i][0])
            if self.saving_orientation['ct'][i][0] == 'manual assignment':
                print(self.saving_orientation['ct'][i][1])
                print("ct",self.points['ct'])
                start_point_ct = self.saving_orientation['ct'][i][1][0]
                end_point_ct = self.saving_orientation['ct'][i][1][1]
                start_point_oct = self.saving_orientation['oct'][j][1][0]
                end_point_oct = self.saving_orientation['oct'][j][1][1]
                self.axs[idx, 0].clear()
                self.axs[idx, 1].clear()
                
                self.axs[idx, 0].imshow(self.ct_frames[0, i, ...], cmap='GnBu')  # Adjust key as necessary
                self.axs[idx, 0].arrow(start_point_ct[0], start_point_ct[1],end_point_ct[0] - start_point_ct[0], end_point_ct[1] - start_point_ct[1],
                     head_width=5, head_length=10, fc='red', ec='red')
                self.axs[idx, 0].set_title(f'Bifurcation pair # {idx} frame {i}')
                self.axs[idx, 1].imshow(self.oct_frames[0, j, ...], cmap='Oranges')  # Adjust key as necessary
                self.axs[idx, 1].arrow(start_point_oct[0], start_point_oct[1],end_point_oct[0] - start_point_oct[0], end_point_oct[1] - start_point_oct[1],
                                       head_width=5, head_length=10, fc='red', ec='red')
                self.axs[idx, 1].set_title(f'-- frame {j}')
                #self.axs[idx, 2].imshow(self.ct_frames[0, i, ...], cmap='GnBu')  # Adjust key as necessary
                print('show point',start_point_ct)
                print('show point',end_point_ct)
                print('show point',start_point_ct[::-1])
                print('show point',end_point_ct[::-1])
                overlap_vusial(idx,CT_image=self.ct_frames[0, i, ...],center_ct=start_point_ct,outermost_ct=end_point_ct[::-1],OCT_image=self.oct_frames[0, j, ...],
                               center_oct=start_point_oct,outermost_oct=end_point_oct[::-1])
                del start_point_ct
                del end_point_ct
                del start_point_oct
                del end_point_oct
            if self.saving_orientation['ct'][i][0] == 'circle':
                self.axs[idx, 0].clear()
                self.axs[idx, 1].clear()
                orientation, centroid_ct, circle, direction_vertex_ct = calculate_orientation_circle(self.ct_frames[0, i, ...])



                self.axs[idx, 0].imshow(self.ct_frames[0, i, ...], cmap='GnBu')
                self.axs[idx, 0].plot(centroid_ct[1], centroid_ct[0], 'ro')
                self.axs[idx, 0].arrow(centroid_ct[1], centroid_ct[0], direction_vertex_ct[1] - centroid_ct[1], direction_vertex_ct[0] - centroid_ct[0], 
                          head_width=5, head_length=10, fc='r', ec='r', label='Orientation')
                self.axs[idx, 0].add_artist(circle)
                self.axs[idx, 0].set_title(f'Bifurcation pair # {idx} frame {i}')
                orientation, centroid_oct, circle, direction_vertex_oct = calculate_orientation_circle(self.oct_frames[0, j , ...])
                self.axs[idx, 1].imshow(self.oct_frames[0, j , ...], cmap='Oranges')
                self.axs[idx, 1].plot(centroid_oct[1], centroid_oct[0], 'ro')
                self.axs[idx, 1].arrow(centroid_oct[1], centroid_oct[0], direction_vertex_oct[1] - centroid_oct[1], direction_vertex_oct[0] - centroid_oct[0],
                            head_width=5, head_length=10, fc='r', ec='r', label='Orientation')
                self.axs[idx, 1].add_artist(circle)
                self.axs[idx, 1].set_title(f'-- frame {j}')
                print('circle',direction_vertex_ct)
          
                print('circle',direction_vertex_ct[::-1])
                overlap_vusial(idx,CT_image=self.ct_frames[0, i, ...],center_ct=centroid_ct,outermost_ct=direction_vertex_ct,OCT_image=self.oct_frames[0, j, ...],center_oct=centroid_oct,outermost_oct=direction_vertex_oct)
                del orientation
                del direction_vertex_ct
                del circle
                del centroid_ct
            if self.saving_orientation['ct'][i][0] == 'pca':
                
                self.axs[idx, 0].clear()
                self.axs[idx, 1].clear()
                center_of_mass_ct, x1_major_ct, x0_ct, y1_major_ct, y0_ct, x2_major_ct, y2_major_ct = pca(self.ct_frames[0, i, ...])
                self.axs[idx, 0].imshow(self.ct_frames[0, i, ...], cmap='GnBu')
                self.axs[idx, 0].set_title(f'Bifurcation pair # {idx} frame {i}')
                self.axs[idx, 0].plot(center_of_mass_ct[1], center_of_mass_ct[0], 'go')
                self.axs[idx, 0].arrow(center_of_mass_ct[1], center_of_mass_ct[0], x1_major_ct - x0_ct, y1_major_ct - y0_ct, head_width=10, head_length=15, fc='blue', ec='blue', label='Major Axis (Positive)')
                self.axs[idx, 0].arrow(center_of_mass_ct[1], center_of_mass_ct[0], x2_major_ct - x0_ct, y2_major_ct - y0_ct, head_width=10, head_length=15, fc='red', ec='red', label='Major Axis (Negative)')
                center_of_mass_oct, x1_major_oct, x0_oct, y1_major_oct, y0_oct, x2_major_oct, y2_major_oct = pca(self.oct_frames[0, j , ...])
                self.axs[idx, 1].imshow(self.oct_frames[0, j , ...], cmap='Oranges')
                self.axs[idx, 1].plot(center_of_mass_oct[1], center_of_mass_oct[0], 'go')
                self.axs[idx, 1].arrow(center_of_mass_oct[1], center_of_mass_oct[0], x1_major_oct - x0_oct, y1_major_oct - y0_oct, head_width=10, head_length=15, fc='blue', ec='blue', label='Major Axis (Positive)')
                self.axs[idx, 1].arrow(center_of_mass_oct[1], center_of_mass_oct[0], x2_major_oct - x0_oct, y2_major_oct - y0_oct, head_width=10, head_length=15, fc='red', ec='red', label='Major Axis (Negative)')
                self.axs[idx, 1].set_title(f'-- frame {j}')
                overlap_vusial(idx,CT_image=self.ct_frames[0, i, ...],center_ct=center_of_mass_ct[::-1],outermost_ct=(y1_major_ct,x1_major_ct),OCT_image=self.oct_frames[0, j, ...],center_oct=center_of_mass_oct[::-1],outermost_oct=(y1_major_oct,x1_major_oct))
            if self.saving_orientation['ct'][i][0] == 'pca_change_dir':
                self.axs[idx, 0].clear()
                self.axs[idx, 1].clear()
                center_of_mass_ct,change_point_start,change_point_end,dx_ct,dy_ct = pca_change_direction(self.ct_frames[0, i, ...])
                self.axs[idx, 0].imshow(self.ct_frames[0, i, ...], cmap='GnBu')
                self.axs[idx, 0].set_title(f'Bifurcation pair # {idx} frame {i}')
                self.axs[idx, 0].plot(center_of_mass_ct[1], center_of_mass_ct[0], 'go')
                self.axs[idx, 0].arrow(center_of_mass_ct[1], center_of_mass_ct[0], dx_ct, dy_ct,
                head_width=10, head_length=15, fc='blue', ec='blue', label='Major Axis (Positive)')
                self.axs[idx, 0].plot(change_point_start[1], change_point_start[0], 'ro')
                self.axs[idx, 0].plot(change_point_end[1], change_point_end[0], 'ro')
                center_of_mass_oct,change_point_start,change_point_end,dx_oct,dy_oct = pca_change_direction(self.oct_frames[0, j , ...])
                self.axs[idx, 1].imshow(self.oct_frames[0, j , ...], cmap='Oranges')
                self.axs[idx, 1].plot(center_of_mass_oct[1], center_of_mass_oct[0], 'go')
                self.axs[idx, 1].arrow(center_of_mass_oct[1], center_of_mass_oct[0], dx_oct, dy_oct,
                head_width=10, head_length=15, fc='blue', ec='blue', label='Major Axis (Positive)')
                self.axs[idx, 1].plot(change_point_start[1], change_point_start[0], 'ro')
                self.axs[idx, 1].plot(change_point_end[1], change_point_end[0], 'ro')
                self.axs[idx, 1].set_title(f'-- frame {j}')
                overlap_vusial(idx,CT_image=self.ct_frames[0, i, ...],center_ct=center_of_mass_ct,outermost_ct=(dy_ct+center_of_mass_ct[0],dx_ct+center_of_mass_ct[1]),OCT_image=self.oct_frames[0, j, ...],center_oct=center_of_mass_oct,outermost_oct=(dy_oct+center_of_mass_oct[0],dx_oct+center_of_mass_oct[1]))

            if self.saving_orientation['ct'][i][0] == 'ellipse':
                self.axs[idx, 0].clear()
                
                center_of_mass_ct,farthest_point_ct,y0_ct,x0_ct,y1,x1,rr,cc = fit_ellipse(self.ct_frames[0, i, ...])
                self.axs[idx, 0].imshow(self.ct_frames[0, i, ...], cmap='GnBu')
                self.axs[idx, 0].set_title(f'Bifurcation pair # {idx} frame {i}')
                self.axs[idx, 0].plot(center_of_mass_ct[1], center_of_mass_ct[0], 'go')
                self.axs[idx, 0].arrow(y0_ct, x0_ct, y1 - y0_ct, x1 - x0_ct, head_width=5, head_length=10, fc='r', ec='r', label='Orientation')
                self.axs[idx, 0].plot(cc, rr, 'b-', label='Fitted Ellipse')
                self.axs[idx, 0].arrow(center_of_mass_ct[1], center_of_mass_ct[0], farthest_point_ct[1] - center_of_mass_ct[1], farthest_point_ct[0] - center_of_mass_ct[0],
                head_width=5, head_length=10, fc='g', ec='g', label='Farthest Point')
                center_of_mass_oct,farthest_point_oct,y0_ct,x0_ct,y1,x1,rr,cc = fit_ellipse(self.oct_frames[0, j , ...])
                self.axs[idx, 1].clear()
                self.axs[idx, 1].imshow(self.oct_frames[0, j , ...], cmap='Oranges')
                self.axs[idx, 1].plot(center_of_mass_oct[1], center_of_mass_oct[0], 'go')
                self.axs[idx, 1].arrow(y0_ct, x0_ct, y1 - y0_ct, x1 - x0_ct, head_width=5, head_length=10, fc='r', ec='r', label='Orientation')
                self.axs[idx, 1].plot(cc, rr, 'b-', label='Fitted Ellipse')
                self.axs[idx, 1].arrow(center_of_mass_oct[1], center_of_mass_oct[0], farthest_point_oct[1] - center_of_mass_oct[1], farthest_point_oct[0] - center_of_mass_oct[0],
                head_width=5, head_length=10, fc='g', ec='g', label='Farthest Point')
                self.axs[idx, 1].set_title(f'-- frame {j}')
                overlap_vusial(idx,CT_image=self.ct_frames[0, i, ...],center_ct=center_of_mass_ct[::-1],outermost_ct=farthest_point_ct,OCT_image=self.oct_frames[0, j, ...],center_oct=center_of_mass_oct,outermost_oct=farthest_point_oct)
            if self.saving_orientation['ct'][i][0] == 'convex_hull':
                self.axs[idx, 0].clear()
                self.axs[idx, 1].clear()
                self.axs[idx, 0].imshow(self.ct_frames[0, i, ...], cmap='GnBu')
                self.axs[idx, 0].set_title(f'Bifurcation pair # {idx} frame {i}')
                intersection_ct,longest_edges, coords, hull = convex_hull(self.ct_frames[0, i, ...])
                centroid_ct = np.mean(coords, axis=0)
                for simplex in hull.simplices:
                    self.axs[idx, 0].plot(coords[simplex, 1], coords[simplex, 0], 'b-')
                for edge in longest_edges:
                    self.axs[idx, 0].plot([edge[0][1], edge[1][1]], [edge[0][0], edge[1][0]], 'g-', linewidth=2)
                self.axs[idx, 0].plot(intersection_ct[1], intersection_ct[0], 'ro')
                self.axs[idx, 0].plot(centroid_ct[1], centroid_ct[0], 'go')
                
                intersection_oct,longest_edges, coords, hull = convex_hull(self.oct_frames[0, j , ...])
                self.axs[idx, 1].imshow(self.oct_frames[0, j , ...], cmap='Oranges')
                for simplex in hull.simplices:
                    self.axs[idx, 1].plot(coords[simplex, 1], coords[simplex, 0], 'b-')
                for edge in longest_edges:
                    self.axs[idx, 1].plot([edge[0][1], edge[1][1]], [edge[0][0], edge[1][0]], 'g-', linewidth=2)
                self.axs[idx, 1].plot(intersection_oct[1], intersection_oct[0], 'ro')
                centroid_oct = np.mean(coords, axis=0)
                self.axs[idx, 1].plot(centroid_oct[1], centroid_oct[0], 'go')
                self.axs[idx, 1].set_title(f'-- frame {j}')
                overlap_vusial(idx,CT_image=self.ct_frames[0, i, ...],center_ct=centroid_ct,outermost_ct=intersection_ct,OCT_image=self.oct_frames[0, j, ...],center_oct=centroid_oct,outermost_oct=intersection_oct)
            if self.saving_orientation['ct'][i][0] == 'triangle':
                self.axs[idx, 0].clear()
                self.axs[idx, 1].clear()
                self.axs[idx, 0].imshow(self.ct_frames[0, i, ...], cmap='GnBu')
                self.axs[idx, 0].set_title(f'Bifurcation pair # {idx} frame {i}')
                centroid_ct, triangle_points, longest_edge_vertex_ct = triangle(self.ct_frames[0, i, ...])
                self.axs[idx, 0].plot(centroid_ct[1], centroid_ct[0], 'ro')
                self.axs[idx, 0].arrow(centroid_ct[1], centroid_ct[0], longest_edge_vertex_ct[1] - centroid_ct[1], longest_edge_vertex_ct[0] - centroid_ct[0], 
                head_width=5, head_length=10, fc='r', ec='r', label='Orientation')
                self.axs[idx, 0].plot(triangle_points[:, 1], triangle_points[:, 0], 'b-', label='Fitted Triangle')
                centroid_oct, triangle_points, longest_edge_vertex_oct = triangle(self.oct_frames[0, j , ...])
                self.axs[idx, 1].imshow(self.oct_frames[0, j , ...], cmap='Oranges')
                self.axs[idx, 1].plot(centroid_oct[1], centroid_oct[0], 'ro')
                self.axs[idx, 1].arrow(centroid_oct[1], centroid_oct[0], longest_edge_vertex_oct[1] - centroid_oct[1], longest_edge_vertex_oct[0] - centroid_oct[0],
                head_width=5, head_length=10, fc='r', ec='r', label='Orientation')
                self.axs[idx, 1].plot(triangle_points[:, 1], triangle_points[:, 0], 'b-', label='Fitted Triangle')
                self.axs[idx, 1].set_title(f'-- frame {j}')
                overlap_vusial(idx,CT_image=self.ct_frames[0, i, ...],center_ct=centroid_ct,outermost_ct=longest_edge_vertex_ct,OCT_image=self.oct_frames[0, j, ...],center_oct=centroid_oct,outermost_oct=longest_edge_vertex_oct)
            if self.saving_orientation['ct'][i][0] == 'triangle_max_edges':
                self.axs[idx, 0].clear()
                self.axs[idx, 1].clear()
                self.axs[idx, 0].imshow(self.ct_frames[0, i, ...], cmap='GnBu')
                self.axs[idx, 0].set_title(f'Bifurcation pair # {idx} frame {i}')  
                centroid_ct, triangle_points, common_vertex_ct = triangle_max_edges(self.ct_frames[0, i, ...])
                self.axs[idx, 0].plot(centroid_ct[1], centroid_ct[0], 'ro')
                self.axs[idx, 0].arrow(centroid_ct[1], centroid_ct[0], common_vertex_ct[1] - centroid_ct[1], common_vertex_ct[0] - centroid_ct[0], 
                head_width=5, head_length=10, fc='r', ec='r', label='Orientation')
                self.axs[idx, 0].plot(common_vertex_ct[1], common_vertex_ct[0], 'go', markersize=10, label='Common Vertex')
                self.axs[idx, 0].plot(triangle_points[:, 1], triangle_points[:, 0], 'b-', label='Fitted Triangle')
                centroid_oct, triangle_points, common_vertex_oct = triangle_max_edges(self.oct_frames[0, j , ...])
                self.axs[idx, 1].imshow(self.oct_frames[0, j , ...], cmap='Oranges')
                self.axs[idx, 1].plot(centroid_oct[1], centroid_oct[0], 'ro')
                self.axs[idx, 1].arrow(centroid_oct[1], centroid_oct[0], common_vertex_oct[1] - centroid_oct[1], common_vertex_oct[0] - centroid_oct[0],
                head_width=5, head_length=10, fc='r', ec='r', label='Orientation')
                self.axs[idx, 1].plot(common_vertex_oct[1], common_vertex_oct[0], 'go', markersize=10, label='Common Vertex')
                self.axs[idx, 1].plot(triangle_points[:, 1], triangle_points[:, 0], 'b-', label='Fitted Triangle')
                self.axs[idx, 1].set_title(f'-- frame {j}')
                overlap_vusial(idx,CT_image=self.ct_frames[0, i, ...],center_ct=centroid_ct,outermost_ct=common_vertex_ct,OCT_image=self.oct_frames[0, j, ...],center_oct=centroid_oct,outermost_oct=common_vertex_oct)

                
        
        canvasz = FigureCanvasTkAgg(self.figz, master=self.orientation_window_plot_frame )
        canvasz.draw()
        canvasz.get_tk_widget().pack(fill=tk.BOTH, expand=tk.YES)


        cl_btn = ttk.Button(self.orientation_window, text="Close", command=self.orientation_window.destroy)
        cl_btn.pack(pady=10)


    def update_oct_frame(self, value):
        self.oct_frame_index = int(value)
        titles = [
        "Masks of Target frames",
        "Original  Target frames",
        ]
        # Loop through each axis and apply the patch and title
        for i, ax in enumerate([self.ax[3],self.ax[4]]):
            # Set the title
            ax.clear()
            ax.text(0.5, -0.1, titles[i], horizontalalignment='center', fontsize=12, fontname='Arial', transform=ax.transAxes)
            if i == 0:  
                # Display the image (for demonstration, using the same image for all)
                ax.imshow(self.oct_frames[0, self.oct_frame_index, ...], cmap='Oranges')
            if i == 1:
                #ax.imshow(self.oct_frames[0, self.oct_frame_index, ...], cmap='Reds')
                ax.imshow(self.or_oct[0, self.oct_frame_index, ...])
            # Add rounded boundaries around the plot
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Create a FancyBboxPatch with rounded corners
            bbox = patches.FancyBboxPatch((0.1, 0.1), 0.8, 0.8, boxstyle="round,pad=0.1", linewidth=1, edgecolor='black', facecolor='none', transform=ax.transAxes, clip_on=False)
            ax.add_patch(bbox)
            ax.axis('off')
        # Remove the previous vertical line if it exists
        if self.oct_current_frame_line is not None:
            self.oct_current_frame_line.remove()
        
        # Add the new vertical line and store its reference
        self.oct_current_frame_line = self.ax[0].axvline(self.oct_frame_index+self.x_shift_oct, color='red', linestyle='--', label='Current Frame')
 
        self.canvas.draw()
        print('update oct frame')

   
    def update_plot(self):
     
        Area_CT, Area_OCT = self.Area_CT, self.Area_OCT

        
        self.ax[0].clear()
        
        adjusted_x_ct = np.arange(len(self.Area_CT)) + self.x_shift_ct
        adjusted_x_oct = np.arange(len(self.Area_OCT)) + self.x_shift_oct
  
        adjusted_y_ct = self.Area_CT + self.y_shift_ct
        adjusted_y_oct = self.Area_OCT + self.y_shift_oct
        peaks_CT, _ = find_peaks(self.Area_CT, height=None)  # You can adjust parameters here for sensitivity
        peaks_OCT, _ = find_peaks(self.Area_OCT, height=None)
        self.peaks_CT, self.peaks_OCT = np.array(peaks_CT), np.array(peaks_OCT)
        self.ax[0].plot(self.peaks_CT + self.x_shift_ct, self.Area_CT[self.peaks_CT] + self.y_shift_ct, "x", color='magenta', markersize=5,label='Potential bifurcation',)  # Highlighted in green
        self.ax[0].plot(self.peaks_OCT + self.x_shift_oct, self.Area_OCT[self.peaks_OCT] + self.y_shift_oct, "x", color='magenta', markersize=5)
        self.ax[0].plot(adjusted_x_ct, adjusted_y_ct, label='Areas of Register frames', picker=5)
        self.ax[0].plot(adjusted_x_oct, adjusted_y_oct, label='Area of Target frames', picker=5)
        
        # Plot each peak, highlight if it's in highlighted_points
        organized_pairs = organize_matched_pairs(self.matched_pairs)
        print(organized_pairs)
        # Add organized pairs to the Treeview
     
        for color,sets in zip(distinct_colors_extended[:len(organized_pairs)],organized_pairs):

                peak_ct = sets[0]

                self.ax[0].plot(peak_ct + self.x_shift_ct, Area_CT[peak_ct] + self.y_shift_ct, "o", color=color, markersize=10)  # Highlighted in red
                peak_oct = sets[1]
                self.ax[0].plot(peak_oct + self.x_shift_oct, Area_OCT[peak_oct] + self.y_shift_oct, "o", color=color, markersize=10)
        
        self.ax[0].legend(fontsize=10)
        
  
        self.ax[0].tick_params(axis='both', which='major', labelsize=10)  # Updated tick labels font size
        self.ax[0].xaxis.label.set_size(8)  # Updated x-axis label font size
        self.ax[0].yaxis.label.set_size(8)  # Updated y-axis label font size
        
        self.canvas.draw()


    def on_pick(self, event):
        Area_CT, Area_OCT = self.Area_CT, self.Area_OCT

        # This method is called when a plot line is clicked near a data point.
        picked_x = event.mouseevent.xdata  # Get the x-coordinate of the pick event in data coordinates.

        # Determine which dataset the point belongs to
        if event.artist.get_label() == 'Areas of Register frames':

            # Adjust the picked x-value based on current shift and find the closest peak
            adjusted_x = picked_x - self.x_shift_ct
            # Find the peak closest to this adjusted x-value
            point_index = np.argmin(np.abs(np.arange(len(Area_CT)) - adjusted_x))

            self.ct_frame_index_slider.set(point_index)

        else:
  
            adjusted_x = picked_x - self.x_shift_oct
            point_index = np.argmin(np.abs(np.arange(len(Area_OCT)) - adjusted_x))
        
            self.oct_frame_index_slider.set(point_index)
     
        self.update_plot()


    def update_matched_pairs_window(self):
        self.pairs_tree.delete(*self.pairs_tree.get_children())  # Clear existing entries
        
        # Organize matched pairs into CT-OCT pairs
        organized_pairs = organize_matched_pairs(self.matched_pairs)
        print(self.matched_pairs)
        # Add organized pairs to the Treeview
        for ct_peak, oct_peak in organized_pairs:
          
            self.pairs_tree.insert("", tk.END, values=(ct_peak, oct_peak))
        
        self.update_plot()


    def delete_selected_pair(self):
        selected_items = self.pairs_tree.selection()
        if selected_items:
            for item in selected_items:
    
                # Extract information from the item to identify the corresponding pair in your data structure
                item_values = self.pairs_tree.item(item, 'values')
                print(f"Deleting pair {item_values}")
                ct_value = item_values[0]
                oct_value = item_values[1]

                # Find the corresponding pair in matched_pairs and highlighted_points to delete
                # This assumes matched_pairs and highlighted_points store information in a way that can be matched to the Treeview values
                # You may need to adjust this logic based on how you're storing these pairs
                for pair in list(self.matched_pairs):  # Iterate over a copy to allow modification of the original list
                    if pair[0] == "CT" and str(pair[1]) == ct_value:
                        self.matched_pairs.remove(pair)
                        if pair in self.matched_pairs:
                            self.matched_pairs.remove(pair)
                    elif pair[0] == "OCT" and str(pair[1]) == oct_value:
                        self.matched_pairs.remove(pair)
                        if pair in self.matched_pairs:
                            self.matched_pairs.remove(pair)

                self.pairs_tree.delete(item)  # Delete the item from the Treeview

        self.update_matched_pairs_window()  # Update the display
        self.update_plot()  # Refresh the plot to reflect any changes



