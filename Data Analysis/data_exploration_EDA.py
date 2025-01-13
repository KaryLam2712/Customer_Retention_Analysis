import tkinter as tk
from tkinter import ttk, messagebox
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


# Function to predict churn and generate graphs
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
        # (fill missing columns with 0 and ensure the order matches the training data)
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

        # Update the graphs
        update_graphs(probability, tenure)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during prediction: {e}")


# Function to update graphs
def update_graphs(probability, tenure):
    try:
        # Create a figure with subplots
        fig = Figure(figsize=(12, 6), dpi=100)
        ax1 = fig.add_subplot(131)  # Gauge Chart
        ax2 = fig.add_subplot(132)  # Cumulative Hazard
        ax3 = fig.add_subplot(133)  # Survival Probability

        # Gauge chart for churn probability
        ax1.pie(
            [probability, 1 - probability],
            labels=["Churn", "No Churn"],
            colors=["#3F5E78", "#EEBA30"],
            startangle=90,
            autopct="%1.1f%%",
        )
        ax1.set_title("Churn Probability")

        # Line graph for cumulative hazard over time
        time = np.linspace(0, 70, 100)
        hazard = np.cumsum(np.exp(-0.1 * time))  # Mock hazard function
        ax2.plot(time, hazard, color="#3F5E78", label="Hazard")
        ax2.axvline(x=tenure, color="#EEBA30", linestyle="--", label="Current Position")
        ax2.set_title("Cumulative Hazard Over Time")
        ax2.set_xlabel("Tenure (months)")
        ax2.set_ylabel("Hazard")
        ax2.legend()

        # Line graph for survival probability over time
        survival = np.exp(-0.05 * time)  # Mock survival function
        ax3.plot(time, survival, color="#3F5E78", label="Survival Function")
        ax3.axvline(x=tenure, color="#EEBA30", linestyle="--", label="Current Position")
        ax3.set_title("Survival Probability Over Time")
        ax3.set_xlabel("Tenure (months)")
        ax3.set_ylabel("Survival Probability")
        ax3.legend()

        # Clear the existing graphs (if any)
        for widget in graph_frame.winfo_children():
            widget.destroy()

        # Embed the Matplotlib figure in the Tkinter GUI
        canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to update graphs: {e}")


# Tkinter GUI setup
root = tk.Tk()
root.title("Customer Churn Prediction")
root.geometry("1200x800")

# Variables for inputs
senior_citizen_var = tk.BooleanVar()
partner_var = tk.BooleanVar()
dependents_var = tk.BooleanVar()
tenure_var = tk.StringVar()
monthly_charges_var = tk.StringVar()
contract_var = tk.StringVar()

# Widgets
tk.Label(root, text="Customer Churn Prediction", font=("Arial", 16, "bold")).pack(
    pady=10
)

# Input options
tk.Checkbutton(root, text="Senior Citizen", variable=senior_citizen_var).pack(
    anchor="w", padx=20
)
tk.Checkbutton(root, text="Partner", variable=partner_var).pack(anchor="w", padx=20)
tk.Checkbutton(root, text="Dependents", variable=dependents_var).pack(
    anchor="w", padx=20
)

tk.Label(root, text="Tenure (months):").pack(anchor="w", padx=20)
tk.Entry(root, textvariable=tenure_var).pack(anchor="w", padx=20)

tk.Label(root, text="Monthly Charges:").pack(anchor="w", padx=20)
tk.Entry(root, textvariable=monthly_charges_var).pack(anchor="w", padx=20)

tk.Label(root, text="Contract Type:").pack(anchor="w", padx=20)
ttk.Combobox(
    root, textvariable=contract_var, values=["Month-to-Month", "One year", "Two year"]
).pack(anchor="w", padx=20)

# Predict button
tk.Button(root, text="Predict", command=predict_churn).pack(pady=20)

# Output label with white text color
result_label = tk.Label(root, text="", font=("Arial", 12), fg="white", bg="black")
result_label.pack(pady=20)

# Graph section
graph_frame = tk.Frame(root)
graph_frame.pack(pady=20, fill="both", expand=True)

# Run the Tkinter application
root.mainloop()
