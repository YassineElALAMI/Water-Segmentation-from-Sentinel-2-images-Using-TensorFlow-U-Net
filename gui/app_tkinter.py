# gui/app_tkinter.py
import sys
from pathlib import Path
from tkinter import (
    Tk, Label, Button, Frame, filedialog, Scale,
    HORIZONTAL, messagebox, Canvas
)
from tkinter.ttk import Progressbar, Notebook
from PIL import Image, ImageTk
import threading
import numpy as np

# allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_image, preprocess_for_model
from src.inference import load_model, predict_mask, calculate_water_coverage
from src.visualization import (
    create_mask_visualization,
    create_overlay,
    create_probability_heatmap,
    blend_heatmap_with_image,
    save_mask,
    save_overlay,
    save_heatmap
)


class WaterDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🌊 GeoAI Water Body Detection")
        self.root.geometry("1400x900")
        self.root.configure(bg="#f0f0f0")

        # state
        self.input_image_path = None
        self.original_image = None          # PIL Image (full size)
        self.preprocessed_image = None      # numpy array (1,H,W,3)
        self.binary_mask = None             # (256,256) 0/1
        self.prob_map = None                # (256,256) float
        self.overlay_arr = None             # overlay uint8 arr
        self.heatmap_arr = None             # heatmap uint8 arr
        self.heatmap_blend = None           # blended uint8 arr
        self.model = None
        self.threshold = 0.5
        self.alpha = 0.4

        self.setup_ui()
        self.load_model_async()

    def setup_ui(self):
        """Setup UI with tabs including heatmap."""
        title_frame = Frame(self.root, bg='#1E3A8A', height=80)
        title_frame.pack(fill='x')
        title_frame.pack_propagate(False)

        Label(title_frame, text="🌊 GeoAI Water Body Detection", font=("Arial", 26, "bold"),
              bg='#1E3A8A', fg='white').pack(pady=6)
        Label(title_frame, text="U-Net Semantic Segmentation • RGB only", font=("Arial", 12),
              bg='#1E3A8A', fg='#CCCCCC').pack()

        main_frame = Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # left controls
        left = Frame(main_frame, bg='white', relief='solid', bd=1, width=320)
        left.pack(side='left', fill='y', padx=(0,10))
        left.pack_propagate(False)

        Label(left, text="⚙️ Controls", font=("Arial", 16, "bold"), bg='white').pack(pady=12)

        Button(left, text="📁 Upload Image", font=("Arial",12,"bold"), bg="#3B82F6", fg="white",
               command=self.upload_image, height=2).pack(padx=20, pady=8, fill="x")

        self.threshold_label = Label(left, text="Detection Threshold: 0.50", bg="white")
        self.threshold_label.pack(pady=(12,4))
        self.threshold_slider = Scale(left, from_=0.0, to=1.0, resolution=0.05, orient=HORIZONTAL,
                                      command=self.on_threshold_change)
        self.threshold_slider.set(0.5)
        self.threshold_slider.pack(padx=20, fill="x")

        self.alpha_label = Label(left, text="Overlay Transparency: 0.40", bg="white")
        self.alpha_label.pack(pady=(12,4))
        self.alpha_slider = Scale(left, from_=0.0, to=1.0, resolution=0.05, orient=HORIZONTAL,
                                  command=self.on_alpha_change)
        self.alpha_slider.set(0.4)
        self.alpha_slider.pack(padx=20, fill="x")

        self.process_btn = Button(left, text="🔍 Detect Water", font=("Arial",12,"bold"),
                                  bg="#10B981", fg="white", height=2, state="disabled",
                                  command=self.process_image)
        self.process_btn.pack(pady=16, padx=20, fill="x")

        # stats
        stats = Frame(left, bg='#ECFDF5', relief='solid', bd=1)
        stats.pack(padx=20, pady=10, fill="x")
        Label(stats, text="📊 Statistics", bg='#ECFDF5', font=("Arial",12,"bold")).pack(pady=6)
        self.water_label = Label(stats, text="Water: ---%", bg="#ECFDF5")
        self.water_label.pack()
        self.land_label = Label(stats, text="Land: ---%", bg="#ECFDF5")
        self.land_label.pack()

        # save buttons
        self.save_mask_btn = Button(left, text="💾 Save Mask", bg="#6366F1", fg="white", state="disabled",
                                    command=self.save_mask_file)
        self.save_mask_btn.pack(padx=20, pady=6, fill="x")

        self.save_overlay_btn = Button(left, text="💾 Save Overlay", bg="#6366F1", fg="white", state="disabled",
                                       command=self.save_overlay_file)
        self.save_overlay_btn.pack(padx=20, pady=6, fill="x")

        self.save_heatmap_btn = Button(left, text="💾 Save Heatmap", bg="#F97316", fg="white", state="disabled",
                                       command=self.save_heatmap_file)
        self.save_heatmap_btn.pack(padx=20, pady=6, fill="x")

        self.status_label = Label(left, text="Status: Ready", bg="white")
        self.status_label.pack(side="bottom", pady=12)

        # right panel - tabs
        right = Frame(main_frame)
        right.pack(side='right', fill='both', expand=True)

        self.notebook = Notebook(right)
        self.notebook.pack(fill='both', expand=True)

        # original tab
        tab_orig = Frame(self.notebook, bg="white")
        self.notebook.add(tab_orig, text="📷 Original")
        self.orig_canvas = Canvas(tab_orig, bg="white")
        self.orig_canvas.pack(fill="both", expand=True)

        # mask tab
        tab_mask = Frame(self.notebook, bg="white")
        self.notebook.add(tab_mask, text="🎭 Mask")
        self.mask_canvas = Canvas(tab_mask, bg="white")
        self.mask_canvas.pack(fill="both", expand=True)

        # overlay tab
        tab_overlay = Frame(self.notebook, bg="white")
        self.notebook.add(tab_overlay, text="🌈 Overlay")
        self.overlay_canvas = Canvas(tab_overlay, bg="white")
        self.overlay_canvas.pack(fill="both", expand=True)

        # heatmap tab
        tab_heat = Frame(self.notebook, bg="white")
        self.notebook.add(tab_heat, text="🔥 Heatmap")
        self.heat_canvas = Canvas(tab_heat, bg="white")
        self.heat_canvas.pack(fill="both", expand=True)

        # progress bar
        self.progress = Progressbar(right, mode="indeterminate", length=280)

    def on_threshold_change(self, v):
        self.threshold = float(v)
        self.threshold_label.config(text=f"Detection Threshold: {self.threshold:.2f}")

    def on_alpha_change(self, v):
        self.alpha = float(v)
        self.alpha_label.config(text=f"Overlay Transparency: {self.alpha:.2f}")

    def load_model_async(self):
        def target():
            try:
                self.status_label.config(text="Status: Loading model...")
                model_path = Path(__file__).parent.parent / "models" / "unet_water_best.keras"
                if not model_path.exists():
                    messagebox.showerror("Model missing", f"Model not found at:\n{model_path}")
                    return
                self.model = load_model(model_path)
                self.status_label.config(text="Status: Model loaded ✓")
            except Exception as e:
                messagebox.showerror("Error loading model", str(e))
                self.status_label.config(text="Status: Model load error")
        threading.Thread(target=target, daemon=True).start()

    def upload_image(self):
        fp = filedialog.askopenfilename(title="Select image (RGB)", filetypes=[("Images","*.jpg *.jpeg *.png")])
        if not fp:
            return
        try:
            self.input_image_path = fp
            self.original_image = load_image(fp)  # PIL Image
            self.display_image(self.original_image, self.orig_canvas)
            self.status_label.config(text="Status: Image loaded ✓")
            if self.model is not None:
                self.process_btn.config(state="normal")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def process_image(self):
        if self.original_image is None:
            messagebox.showwarning("No image", "Please upload an image first.")
            return

        def work():
            try:
                self.progress.pack(pady=8)
                self.progress.start(12)
                self.status_label.config(text="Status: Processing...")

                # preprocess
                self.preprocessed_image = preprocess_for_model(self.original_image, target_size=(256,256))

                # predict: returns binary mask (256x256) and prob_map (256x256)
                binary, prob = predict_mask(self.preprocessed_image, self.threshold, self.model)
                self.binary_mask = binary
                self.prob_map = prob

                # stats
                stats = calculate_water_coverage(binary)
                self.water_label.config(text=f"Water: {stats['water_percentage']:.2f}%")
                self.land_label.config(text=f"Land: {stats['land_percentage']:.2f}%")

                # visuals
                mask_vis = create_mask_visualization(binary)
                overlay_arr = create_overlay(self.original_image, binary, alpha=self.alpha)
                heatmap_rgb = create_probability_heatmap(prob, colormap="jet")
                heat_blend = blend_heatmap_with_image(self.original_image, heatmap_rgb, alpha=self.alpha)

                self.overlay_arr = overlay_arr
                self.heatmap_arr = heatmap_rgb
                self.heatmap_blend = heat_blend

                # display
                self.display_image(Image.fromarray(mask_vis), self.mask_canvas)
                self.display_image(Image.fromarray(overlay_arr), self.overlay_canvas)
                self.display_image(Image.fromarray(heat_blend), self.heat_canvas)

                # enable save buttons
                self.save_mask_btn.config(state="normal")
                self.save_overlay_btn.config(state="normal")
                self.save_heatmap_btn.config(state="normal")

                self.status_label.config(text="Status: Done ✓")
                self.notebook.select(self.mask_canvas.master)  # go to mask tab
            except Exception as e:
                messagebox.showerror("Processing failed", str(e))
                self.status_label.config(text="Status: Error")
            finally:
                self.progress.stop()
                self.progress.pack_forget()

        threading.Thread(target=work, daemon=True).start()

    def display_image(self, pil_image: Image.Image, canvas: Canvas):
        """Safely display a PIL Image on the given canvas."""
        canvas.update_idletasks()
        cw, ch = canvas.winfo_width(), canvas.winfo_height()
        if cw < 50 or ch < 50:
            cw, ch = 800, 600
        iw, ih = pil_image.size
        ratio = min(cw/iw, ch/ih) * 0.95
        new_w, new_h = max(1, int(iw*ratio)), max(1, int(ih*ratio))
        resized = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(resized)
        canvas.delete("all")
        canvas.create_image(cw//2, ch//2, image=tk_img, anchor="center")
        canvas.image = tk_img

    def save_mask_file(self):
        if self.binary_mask is None:
            return
        fp = filedialog.asksaveasfilename(defaultextension=".png")
        if not fp:
            return
        mask_vis = create_mask_visualization(self.binary_mask)
        save_mask(mask_vis, fp)
        messagebox.showinfo("Saved", f"Mask saved to:\n{fp}")

    def save_overlay_file(self):
        if self.overlay_arr is None:
            return
        fp = filedialog.asksaveasfilename(defaultextension=".png")
        if not fp:
            return
        save_overlay(self.overlay_arr, fp)
        messagebox.showinfo("Saved", f"Overlay saved to:\n{fp}")

    def save_heatmap_file(self):
        if self.heatmap_blend is None:
            return
        fp = filedialog.asksaveasfilename(defaultextension=".png")
        if not fp:
            return
        # save blended heatmap (visual)
        save_heatmap(self.heatmap_blend, fp)
        messagebox.showinfo("Saved", f"Heatmap saved to:\n{fp}")


def main():
    root = Tk()
    WaterDetectionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
