import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk  # For handling images
import pandas as pd
import joblib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Load the saved model
model_path = "model_rf_smote_compatible.sav"
try:
    model = joblib.load(model_path)
except Exception as e:
    messagebox.showerror("Error", f"Failed to load the model: {e}")


# Function to predict churn and display results
def predict_churn():
    try:
        # Collect raw user inputs
        raw_data = {
            "SeniorCitizen": int(senior_citizen_var.get()),
            "Partner": "Yes" if partner_var.get() else "No",
            "Dependents": "Yes" if dependents_var.get() else "No",
            "tenure": float(tenure_var.get()),
            "MonthlyCharges": float(monthly_charges_var.get()),
            "Contract": contract_var.get(),
        }

        # Convert input to DataFrame
        input_data = pd.DataFrame([raw_data])

        # Preprocess input data (one-hot encoding)
        categorical_features = ["Partner", "Dependents", "Contract"]
        input_data = pd.get_dummies(input_data, columns=categorical_features)

        # Align input features with the training features
        training_features = model.feature_names_in_
        input_data = input_data.reindex(columns=training_features, fill_value=0)

        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        # Calculate ELTV
        monthly_revenue = float(monthly_charges_var.get())
        tenure = float(tenure_var.get())
        eltv = monthly_revenue * (1 - probability) * tenure

        # Display result
        result = (
            f"Prediction: {'Churn' if prediction == 1 else 'No Churn'}\n"
            f"Probability: {probability:.2f}\n"
            f"Expected Lifetime Value (ELTV): ${eltv:.2f}"
        )
        result_label.config(text=result)

        # Update the graph and the image
        update_graphs(probability, tenure)
        update_image(probability)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during prediction: {e}")


# Function to update graphs
def update_graphs(probability, tenure):
    try:
        # Create a figure with subplots
        fig = Figure(figsize=(8, 4), dpi=100)
        ax1 = fig.add_subplot(121)  # Cumulative Hazard
        ax2 = fig.add_subplot(122)  # Survival Probability

        # Line graph for cumulative hazard over time
        time = np.linspace(0, 70, 100)
        hazard = np.cumsum(np.exp(-0.1 * time))  # Mock hazard function
        ax1.plot(time, hazard, color="#3F5E78", label="Hazard")
        ax1.axvline(x=tenure, color="#EEBA30", linestyle="--", label="Current Position")
        ax1.set_title("Cumulative Hazard Over Time")
        ax1.set_xlabel("Tenure (months)")
        ax1.set_ylabel("Hazard")
        ax1.legend()

        # Line graph for survival probability over time
        survival = np.exp(-0.05 * time)  # Mock survival function
        ax2.plot(time, survival, color="#3F5E78", label="Survival Function")
        ax2.axvline(x=tenure, color="#EEBA30", linestyle="--", label="Current Position")
        ax2.set_title("Survival Probability Over Time")
        ax2.set_xlabel("Tenure (months)")
        ax2.set_ylabel("Survival Probability")
        ax2.legend()

        # Clear the existing graphs (if any)
        for widget in graph_frame.winfo_children():
            widget.destroy()

        # Embed the Matplotlib figure in the Tkinter GUI
        canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to update graphs: {e}")


# Function to update the image based on probability
def update_image(probability):
    try:
        if probability < 0.5:
            displayed_image.config(image=low_risk_image)
            displayed_image.image = low_risk_image
        else:
            displayed_image.config(image=high_risk_image)
            displayed_image.image = high_risk_image
    except Exception as e:
        messagebox.showerror("Error", f"Failed to update the image: {e}")


# Tkinter GUI setup
root = tk.Tk()
root.title("Customer Churn Prediction")
root.geometry("1000x800")

# Header
header = tk.Label(
    root,
    text="Customer Churn Predictor",
    font=("Arial", 20, "bold"),
    fg="white",
    pady=10,
)
header.pack(fill="x")

# Variables for inputs
senior_citizen_var = tk.BooleanVar()
partner_var = tk.BooleanVar()
dependents_var = tk.BooleanVar()
tenure_var = tk.StringVar()
monthly_charges_var = tk.StringVar()
contract_var = tk.StringVar()

# Main frame for layout
main_frame = tk.Frame(root)
main_frame.pack(pady=20)

# Row 1: Senior Citizen, Tenure
tk.Checkbutton(main_frame, text="Senior Citizen", variable=senior_citizen_var).grid(
    row=0, column=0, sticky="w", padx=10, pady=5
)
tk.Label(main_frame, text="Tenure (months):").grid(
    row=0, column=1, sticky="e", padx=10, pady=5
)
tk.Entry(main_frame, textvariable=tenure_var).grid(
    row=0, column=2, sticky="w", padx=10, pady=5
)

# Row 2: Partner, Monthly Charges
tk.Checkbutton(main_frame, text="Partner", variable=partner_var).grid(
    row=1, column=0, sticky="w", padx=10, pady=5
)
tk.Label(main_frame, text="Monthly Charges:").grid(
    row=1, column=1, sticky="e", padx=10, pady=5
)
tk.Entry(main_frame, textvariable=monthly_charges_var).grid(
    row=1, column=2, sticky="w", padx=10, pady=5
)

# Row 3: Dependents, Contract Type
tk.Checkbutton(main_frame, text="Dependents", variable=dependents_var).grid(
    row=2, column=0, sticky="w", padx=10, pady=5
)
tk.Label(main_frame, text="Contract Type:").grid(
    row=2, column=1, sticky="e", padx=10, pady=5
)
ttk.Combobox(
    main_frame,
    textvariable=contract_var,
    values=["Month-to-Month", "One year", "Two year"],
).grid(row=2, column=2, sticky="w", padx=10, pady=5)

# Predict button
tk.Button(main_frame, text="Predict", command=predict_churn).grid(
    row=3, column=0, columnspan=3, pady=20
)

# Output label
result_label = tk.Label(root, text="", font=("Arial", 16), fg="white")
result_label.pack(pady=20)

# Image display section
image_frame = tk.Frame(root)
image_frame.pack(pady=10)

displayed_image = tk.Label(image_frame)
displayed_image.pack()

# Load and resize images
low_risk_image = ImageTk.PhotoImage(
    Image.open(
        "/Users/karylam/Desktop/Customer Churn/interface_pictures/low_risk.png"
    ).resize((150, 150))
)
high_risk_image = ImageTk.PhotoImage(
    Image.open(
        "/Users/karylam/Desktop/Customer Churn/interface_pictures/high_risk.png"
    ).resize((150, 150))
)

# Graph section
graph_frame = tk.Frame(root)
graph_frame.pack(pady=20, fill="both", expand=True)

# Set default image
displayed_image.config(image="")

# Run the Tkinter application
root.mainloop()
