import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from PIL import Image, ImageTk
import xgboost as xgb
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import os


class ICHPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced ICH Risk Predictor")
        self.root.geometry("800x900")
        self.root.configure(bg="#F0F0F0")  # Light gray background (Windows 10/11 style)

        # Style configuration
        self.style = ttk.Style()
        self.style.configure("TFrame", background="#F0F0F0")
        self.style.configure("TLabel", background="#F0F0F0", font=("Segoe UI", 12))
        self.style.configure("TButton", font=("Segoe UI", 12))
        self.style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"), background="#F0F0F0", foreground="#1A1A1A")
        self.style.configure("SubTitle.TLabel", font=("Segoe UI", 10), background="#F0F0F0", foreground="#1A1A1A")
        self.style.configure("Result.TLabel", font=("Segoe UI", 12), background="#FFFFFF", foreground="#1A1A1A")
        self.style.configure("Advice.TLabel", font=("Segoe UI", 12, "italic"), background="#FFFFFF",
                             foreground="#1A1A1A")
        self.style.configure("Custom.TButton", background="#0078D4", foreground="black", padding=10)
        self.style.map("Custom.TButton", background=[("active", "#005EA6")], foreground=[("active", "white")])

        # Load models
        self.resnet_model = self.load_resnet()
        self.xgb_model = self.load_xgb_model()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512, 512)),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # Create a canvas and scrollbar
        self.canvas = tk.Canvas(self.root, bg="#F0F0F0", highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas, padding=10)

        # Configure canvas and scrollbar
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Place canvas and scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Add the scrollable frame to the canvas
        self.canvas_frame = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # Main frame with border (inside scrollable frame)
        self.main_frame = ttk.Frame(self.scrollable_frame, padding=10, relief="groove", borderwidth=1)
        self.main_frame.pack(fill="x", padx=10, pady=10)

        # Title
        ttk.Label(self.main_frame, text="Advanced Intracranial Hemorrhage Risk Predictor", style="Title.TLabel").grid(
            row=0, column=0, columnspan=2, pady=(0, 15))

        # Patient info frame
        self.info_frame = ttk.Frame(self.main_frame, padding=8, relief="groove", borderwidth=1)
        self.info_frame.grid(row=1, column=0, columnspan=2, pady=10, sticky="ew")

        ttk.Label(self.info_frame, text="Age:").grid(row=0, column=0, padx=5, sticky="e")
        self.age_entry = ttk.Entry(self.info_frame, font=("Segoe UI", 12), width=15)
        self.age_entry.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(self.info_frame, text="BMI:").grid(row=1, column=0, padx=5, sticky="e")
        self.bmi_entry = ttk.Entry(self.info_frame, font=("Segoe UI", 12), width=15)
        self.bmi_entry.grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(self.info_frame, text="Blood Pressure (Systolic):").grid(row=2, column=0, padx=5, sticky="e")
        self.bp_entry = ttk.Entry(self.info_frame, font=("Segoe UI", 12), width=15)
        self.bp_entry.grid(row=2, column=1, padx=5, pady=2)

        # CT scan upload
        self.upload_button = ttk.Button(self.main_frame, text="Upload CT Scan (.npy)", command=self.upload_ct,
                                        style="Custom.TButton")
        self.upload_button.grid(row=2, column=0, columnspan=2, pady=10)
        self.ct_label = ttk.Label(self.main_frame, text="No file selected", style="SubTitle.TLabel")
        self.ct_label.grid(row=3, column=0, columnspan=2)

        # CT and predicted CT display
        self.ct_frame = ttk.Frame(self.main_frame, padding=8, relief="groove", borderwidth=1)
        self.ct_frame.grid(row=4, column=0, columnspan=2, pady=10)

        # Original CT
        ttk.Label(self.ct_frame, text="Original CT", style="SubTitle.TLabel").grid(row=0, column=0, padx=20)
        self.ct_canvas = tk.Canvas(self.ct_frame, width=256, height=256, bg="white", highlightthickness=1,
                                   highlightbackground="#D3D3D3")
        self.ct_canvas.grid(row=1, column=0, padx=20)

        # Predicted CT
        ttk.Label(self.ct_frame, text="5-Year Predicted CT", style="SubTitle.TLabel").grid(row=0, column=1, padx=20)
        self.predicted_ct_canvas = tk.Canvas(self.ct_frame, width=256, height=256, bg="white", highlightthickness=1,
                                             highlightbackground="#D3D3D3")
        self.predicted_ct_canvas.grid(row=1, column=1, padx=20)

        # Predict button
        self.predict_button = ttk.Button(self.main_frame, text="Predict ICH Risk", command=self.predict_risk,
                                         style="Custom.TButton")
        self.predict_button.grid(row=5, column=0, columnspan=2, pady=10)

        # Result and advice frame
        self.result_frame = ttk.Frame(self.main_frame, padding=8, relief="groove", borderwidth=1)
        self.result_frame.grid(row=6, column=0, columnspan=2, pady=10, sticky="ew")

        self.result_label = ttk.Label(self.result_frame, text="Prediction results will appear here",
                                      style="Result.TLabel", wraplength=700, justify="center")
        self.result_label.pack(fill="x", pady=5)

        self.advice_label = ttk.Label(self.result_frame, text="Medical advice will appear here", style="Advice.TLabel",
                                      wraplength=700, justify="center")
        self.advice_label.pack(fill="x", pady=5)

        self.ct_path = None
        self.ct_image = None
        self.predicted_ct_image = None

        # Bind mouse wheel to scrolling
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def load_resnet(self):
        model_path = "resnet50_best_100000.pth"
        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"ResNet50 model not found at {model_path}")
            return None
        model = models.resnet50(weights=None)
        try:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            if 'conv1.weight' in state_dict:
                conv1_weight = state_dict['conv1.weight']  # Shape: [64, 3, 7, 7]
                conv1_weight_1ch = conv1_weight.sum(dim=1, keepdim=True)  # Shape: [64, 1, 7, 7]
                state_dict['conv1.weight'] = conv1_weight_1ch
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model.fc = nn.Identity()  # 2048 features
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load ResNet50: {str(e)}")
            return None
        model.eval()
        return model

    def load_xgb_model(self):
        model_path = "models/xgboost_v3.model"
        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"XGBoost model not found at {model_path}")
            return None
        try:
            model = xgb.Booster()
            model.load_model(model_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load XGBoost model: {str(e)}")
            return None
        return model

    def upload_ct(self):
        self.ct_path = filedialog.askopenfilename(filetypes=[("NumPy files", "*.npy")])
        if self.ct_path:
            self.ct_label.config(text=os.path.basename(self.ct_path))
            try:
                ct_data = np.load(self.ct_path)
                if len(ct_data.shape) == 3:
                    ct_data = ct_data[0]  # Assume first channel
                ct_data_display = (ct_data - ct_data.min()) / (ct_data.max() - ct_data.min() + 1e-8)
                ct_img = Image.fromarray((ct_data_display * 255).astype(np.uint8))
                ct_img = ct_img.resize((256, 256))
                self.ct_image = ImageTk.PhotoImage(ct_img)
                self.ct_canvas.create_image(0, 0, anchor="nw", image=self.ct_image)
                # Clear predicted CT and results when uploading a new CT
                self.predicted_ct_canvas.delete("all")
                self.result_label.config(text="Prediction results will appear here")
                self.advice_label.config(text="Medical advice will appear here")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load CT scan: {str(e)}")
                self.ct_path = None
                self.ct_label.config(text="No file selected")

    def predict_risk(self):
        if not self.resnet_model or not self.xgb_model:
            messagebox.showerror("Error", "Models not loaded. Check file paths.")
            return

        try:
            age = float(self.age_entry.get()) if self.age_entry.get() else 50
            bmi = float(self.bmi_entry.get()) if self.bmi_entry.get() else 25
            bp = float(self.bp_entry.get()) if self.bp_entry.get() else 120

            if not self.ct_path:
                messagebox.showerror("Error", "Please upload a CT scan (.npy)")
                return

            ct_data = np.load(self.ct_path)
            if len(ct_data.shape) == 3:
                ct_data = ct_data[0]
            ct_tensor = self.transform(ct_data).unsqueeze(0)

            with torch.no_grad():
                features = self.resnet_model(ct_tensor).numpy().flatten()  # 2048 features

            dmatrix = xgb.DMatrix(features.reshape(1, -1))
            prob = self.xgb_model.predict(dmatrix)[0]

            risk_factor = 1 + (age / 100) + (bmi / 50) + (bp / 200)
            five_year_risk = min(prob * risk_factor, 1.0)

            risk_level = "Low" if five_year_risk < 0.3 else "Medium" if five_year_risk < 0.6 else "High"
            result_text = (
                f"Current ICH Probability: {prob:.2%}\n"
                f"5-Year ICH Risk: {five_year_risk:.2%} ({risk_level})\n"
                f"Risk Factors: Age ({age}), BMI ({bmi}), BP ({bp})"
            )
            self.result_label.config(text=result_text)

            # Simulate 5-year predicted CT by degrading the original CT based on risk
            ct_data_display = (ct_data - ct_data.min()) / (ct_data.max() - ct_data.min() + 1e-8)
            degradation_factor = five_year_risk * 0.5  # Scale degradation (0 to 0.5)
            predicted_ct = ct_data_display - (ct_data_display * degradation_factor) + np.random.normal(0, 0.1,
                                                                                                       ct_data_display.shape) * degradation_factor
            predicted_ct = np.clip(predicted_ct, 0, 1)
            predicted_img = Image.fromarray((predicted_ct * 255).astype(np.uint8))
            predicted_img = predicted_img.resize((256, 256))
            self.predicted_ct_image = ImageTk.PhotoImage(predicted_img)
            self.predicted_ct_canvas.create_image(0, 0, anchor="nw", image=self.predicted_ct_image)

            # Medical advice based on risk factors
            advice = self.generate_medical_advice(age, bmi, bp, five_year_risk)
            self.advice_label.config(text=advice)

        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")

    def generate_medical_advice(self, age, bmi, bp, five_year_risk):
        advice = "General recommendations for ICH prevention:\n"
        if five_year_risk >= 0.3:
            advice += "- Consult a neurologist for regular monitoring.\n"
        if bp > 120:
            advice += "- Manage blood pressure with medication (e.g., ACE inhibitors) and reduce salt intake.\n"
        if bmi > 25:
            advice += "- Aim for a healthy weight through diet and exercise (target BMI < 25).\n"
        if age > 50:
            advice += "- Consider aspirin therapy after consulting a doctor.\n"
        advice += "- Maintain a balanced diet rich in fruits and vegetables.\n- Avoid smoking and limit alcohol.\n"
        return advice


if __name__ == "__main__":
    root = tk.Tk()
    app = ICHPredictorApp(root)
    root.mainloop()