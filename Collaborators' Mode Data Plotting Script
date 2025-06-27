import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame, skipping the first row and handling mixed types
file_path = '/content/20250521-520RPM-ringdown-1hour-scope.csv' # Replace with your actual file path
data = pd.read_csv(file_path, skiprows=1)

# Convert columns to numeric, coercing errors
data.iloc[:, 0] = pd.to_numeric(data.iloc[:, 0], errors='coerce')
data.iloc[:, 1] = pd.to_numeric(data.iloc[:, 1], errors='coerce')


# Display the first few rows of the DataFrame (optional)
print(data.head())

# Plotting
x = data.iloc[:, 0]  # First column
y = data.iloc[:, 1]  # Second column

plt.figure(figsize=(10, 6))
#plt.plot(x, y, marker='.', linestyle='-', color='b', markersize = 2, linewidth = 1)  # You can customize the plot style
plt.plot(x, y, color = 'blue')
plt.title('Plot of mode position vs time')
plt.xlabel('Time')
plt.ylabel('Mode Position')
plt.grid(True)

# Show the plot
plt.show()
