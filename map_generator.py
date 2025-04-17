import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageOps, EpsImagePlugin
import io
import os

# Ensure Ghostscript is available for EPS conversion (for Linux users)
EpsImagePlugin.gs_windows_binary = r'gs'  # For Linux or if ghostscript is in your PATH

class ShapeDrawer:
    def __init__(self, root, width=1000, height=1000):
        self.root = root
        self.width = width
        self.height = height

        # Set up the main window
        self.root.title("2D Bitmap Shape Builder")

        # Initialize shape selection and point list
        self.current_shape = None  # Options: "triangle" or "quadrilateral"
        self.points = []  # List to accumulate clicked points

        # Create the top frame with control buttons
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        # Button to select Triangle
        btn_triangle = tk.Button(control_frame, text="Triangle", width=12,
                                 command=lambda: self.select_shape("triangle"))
        btn_triangle.pack(side=tk.LEFT, padx=5, pady=5)

        # Button to select Quadrilateral
        btn_quad = tk.Button(control_frame, text="Quadrilateral", width=12,
                                 command=lambda: self.select_shape("quadrilateral"))
        btn_quad.pack(side=tk.LEFT, padx=5, pady=5)

        # Save Button
        btn_save = tk.Button(control_frame, text="Save Image", width=12,
                               command=self.save_canvas)
        btn_save.pack(side=tk.LEFT, padx=5, pady=5)

        # Reset Button to clear the canvas
        btn_reset = tk.Button(control_frame, text="Reset", width=12,
                              command=self.reset_canvas)
        btn_reset.pack(side=tk.LEFT, padx=5, pady=5)

        # Create a status label
        self.status_label = tk.Label(control_frame, text="Select a shape to draw.")
        self.status_label.pack(side=tk.LEFT, padx=10)

        # Create the canvas with white background
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, bg="white")
        self.canvas.pack(side=tk.TOP)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

    def select_shape(self, shape):
        """Set the current shape for drawing."""
        self.current_shape = shape
        self.points = []  # Reset points whenever a new shape is selected
        self.status_label.config(text=f"Selected shape: {shape.capitalize()}. Click on canvas to set points.")

    def on_canvas_click(self, event):
        """Handle canvas mouse clicks to record points and draw the shape if ready."""
        if not self.current_shape:
            messagebox.showinfo("Info", "Please select a shape first.")
            return

        # Save the clicked point (canvas coordinates)
        self.points.append((event.x, event.y))
        self.status_label.config(text=f"{len(self.points)} point(s) selected for {self.current_shape}.")

        # Determine required number of points
        required_points = 3 if self.current_shape == "triangle" else 4

        # If we have enough points, draw the polygon
        if len(self.points) == required_points:
            self.canvas.create_polygon(self.points, fill="black")
            self.status_label.config(text=f"{self.current_shape.capitalize()} drawn. Select a shape to start over.")
            # Reset current shape and points so you can choose another shape
            self.current_shape = None
            self.points = []

    def save_canvas(self):
        """Save the contents of the canvas to a PNG image using PostScript conversion."""
        # Get a file name to save the image using filedialog
        filename = filedialog.asksaveasfilename(defaultextension=".png",
                                                filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if not filename:
            return

        try:
            # Save canvas content as a PostScript file in memory
            ps_data = self.canvas.postscript(colormode='color')
            # Use BytesIO to avoid file I/O and load the postscript data into PIL directly
            ps_bytes = io.BytesIO(ps_data.encode('utf-8'))
            # Open the image using PIL (requires Ghostscript installed)
            image = Image.open(ps_bytes)
            # Convert to RGBA to prevent issues and save as PNG
            image = image.convert("RGBA")
            image.save(filename, "png")
            messagebox.showinfo("Image Saved", f"Image has been saved as {os.path.basename(filename)}")
        except Exception as e:
            messagebox.showerror("Save Error", f"An error occurred while saving the image:\n{e}")

    def reset_canvas(self):
        """Clear the canvas and reset shape selections."""
        self.canvas.delete("all")
        self.current_shape = None
        self.points = []
        self.status_label.config(text="Canvas reset. Select a shape to draw.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ShapeDrawer(root)
    root.mainloop()
