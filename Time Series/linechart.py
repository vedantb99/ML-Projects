import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as anime

# Function to load and process a dataset
def load_and_process_csv(filepath):
    # Load the dataset
    df = pd.read_csv(filepath)
    
    # Clean the data
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')  # Ensure the date format is correct
    df['Close'] = df['Close/Last'].str.replace('$', '').astype(float)
    
    # Sort by Date in ascending order
    df = df.sort_values(by='Date', ascending=True)
    
    return df

# Load all datasets
amd_df = load_and_process_csv('./data/amd.csv')
nvidia_df = load_and_process_csv('./data/nvidia.csv')
intel_df = load_and_process_csv('./data/intel.csv')

# Prepare the figure and axis
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_title('Stock Closing Prices Over Time (Animated)', fontsize=16)
ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('Close Price (USD)', fontsize=14)
plt.xticks(rotation=45)
ax.grid(True)

# Plot placeholders for animation
amd_line, = ax.plot([], [], color='red', linewidth=2, label='AMD Close Price')
nvidia_line, = ax.plot([], [], color='green', linewidth=2, label='NVIDIA Close Price')
intel_line, = ax.plot([], [], color='blue', linewidth=2, label='Intel Close Price')

# Ticker placeholders
amd_ticker = ax.text(0, 0, '', color='red', fontsize=10)
nvidia_ticker = ax.text(0, 0, '', color='green', fontsize=10)
intel_ticker = ax.text(0, 0, '', color='blue', fontsize=10)

# Set initial x and y axis limits
min_date = min(amd_df['Date'].min(), nvidia_df['Date'].min(), intel_df['Date'].min())
max_date = max(amd_df['Date'].max(), nvidia_df['Date'].max(), intel_df['Date'].max())
min_close = min(amd_df['Close'].min(), nvidia_df['Close'].min(), intel_df['Close'].min())
max_close = max(amd_df['Close'].max(), nvidia_df['Close'].max(), intel_df['Close'].max())

ax.set_xlim(min_date, max_date)
ax.set_ylim(min_close * 0.95, max_close * 1.05)

# Animation update function
def update(frame):
    # Select data up to the current frame
    amd_dates, amd_closes = amd_df['Date'][:frame], amd_df['Close'][:frame]
    nvidia_dates, nvidia_closes = nvidia_df['Date'][:frame], nvidia_df['Close'][:frame]
    intel_dates, intel_closes = intel_df['Date'][:frame], intel_df['Close'][:frame]

    # Update the line data
    amd_line.set_data(amd_dates, amd_closes)
    nvidia_line.set_data(nvidia_dates, nvidia_closes)
    intel_line.set_data(intel_dates, intel_closes)

    # Update ticker marks for the last frame
    if frame == len(amd_df):
        amd_ticker.set_position((amd_dates.iloc[-1], amd_closes.iloc[-1]))
        amd_ticker.set_text(f"{amd_closes.iloc[-1]:.2f}")
        nvidia_ticker.set_position((nvidia_dates.iloc[-1], nvidia_closes.iloc[-1]))
        nvidia_ticker.set_text(f"{nvidia_closes.iloc[-1]:.2f}")
        intel_ticker.set_position((intel_dates.iloc[-1], intel_closes.iloc[-1]))
        intel_ticker.set_text(f"{intel_closes.iloc[-1]:.2f}")

    return amd_line, nvidia_line, intel_line, amd_ticker, nvidia_ticker, intel_ticker

# Create animation
frames = max(len(amd_df), len(nvidia_df), len(intel_df))
ani = anime.FuncAnimation(fig, update, frames=frames, interval=0.001, blit=True)

# Save the animation as a GIF
ani.save('stock_prices_with_tickers.gif', writer='pillow', fps=120)

# Show the animation
plt.tight_layout()
plt.show()
