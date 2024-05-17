import pandas as pd
import numpy as np


def generate_large_synthetic_data_with_smaller_chunks(df, start_date, num_chunks, chunk_size):
    synthetic_data = pd.DataFrame()

    for i in range(num_chunks):
        # Create a date range for the current chunk
        chunk_start_date = pd.to_datetime(
            start_date) + pd.DateOffset(days=i * chunk_size)
        date_range = pd.date_range(
            start=chunk_start_date, periods=chunk_size, freq='D')

        chunk_data = pd.DataFrame(index=range(chunk_size), columns=df.columns)
        for j in range(chunk_size):
            # Pick a random row from the original data to replicate
            random_row = df.sample(1).iloc[0]
            random_row['Leave_date'] = date_range[j].strftime('%m/%d/%Y')
            random_row['Return_date'] = date_range[j].strftime('%m/%d/%Y')

            # Add the row to the chunk data
            chunk_data.iloc[j] = random_row

        # Correct the month and week based on the new dates
        chunk_data['Leave_date'] = pd.to_datetime(
            chunk_data['Leave_date'], format='%m/%d/%Y')
        chunk_data['Return_date'] = pd.to_datetime(
            chunk_data['Return_date'], format='%m/%d/%Y')
        chunk_data['month'] = chunk_data['Leave_date'].dt.strftime('%B')
        chunk_data['week'] = chunk_data['Leave_date'].dt.isocalendar().week
        chunk_data['day'] = chunk_data['Leave_date'].dt.strftime('%A')

        # Append the chunk to the synthetic data
        synthetic_data = pd.concat(
            [synthetic_data, chunk_data], ignore_index=True)

    # Ensure sufficient anomalies are included
    # Assume 1% of the rows should be anomalies
    anomaly_count = len(synthetic_data) // 100
    anomalies = pd.DataFrame({
        'Sno': range(len(synthetic_data) + 1, len(synthetic_data) + anomaly_count + 1),
        'Name': ['Aania']*anomaly_count,
        'month': ['March', 'April', 'May', 'June', 'July'] * (anomaly_count // 5),
        'week': [1, 2, 3, 4, 1] * (anomaly_count // 5),
        'day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'] * (anomaly_count // 5),
        'Leave_date': pd.to_datetime(['2024-03-01', '2024-04-08', '2024-05-15', '2024-06-22', '2024-07-29'] * (anomaly_count // 5)),
        'Leave_time': [np.nan, '08:00:00', '05:00:00', '08:00:00', np.nan] * (anomaly_count // 5),
        'Return_date': pd.to_datetime(['2024-03-01', '2024-04-08', '2024-05-15', '2024-06-22', '2024-07-29'] * (anomaly_count // 5)),
        'Return_time': [np.nan, '16:00:00', '16:00:00', '21:00:00', np.nan] * (anomaly_count // 5)
    })

    # Append anomalies to the synthetic data
    synthetic_data = pd.concat([synthetic_data, anomalies], ignore_index=True)

    return synthetic_data


# Generate 100,000 rows of large synthetic data in chunks starting from January 1, 2024
num_chunks = 100
chunk_size = 1000
large_synthetic_data = generate_large_synthetic_data_with_smaller_chunks(
    original_data, '2024-01-01', num_chunks, chunk_size)

# Save to CSV
large_synthetic_data.to_csv(
    'D:\Synthetic-data-generation-using-langchain\Data/aania_schedule.csv', index=False)

# Display the first and last 10 rows
large_synthetic_data.head(10), large_synthetic_data.tail(10)
