
import matplotlib.pyplot as plt



def plot_lof_2d_circle(df, cols):
    # Add a column to indicate outliers
    df['Outlier'] = df['lof'] == -1

    # Filter outliers
    outliers = df[df['Outlier']]

    # Normalize LOF factors for marker sizes
    max_score = outliers['lof_factor'].max()
    min_score = outliers['lof_factor'].min()
    outliers['Normalized_Score'] = (outliers['lof_factor'] - min_score) / (max_score - min_score) * 100  # Scale for visibility

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(df[cols[0]], df[cols[1]], color='k', s=3, label='Data points')
    plt.scatter(outliers[cols[0]], outliers[cols[1]], edgecolor='r', facecolor='none', s=outliers['Normalized_Score'], label='Outlier scores')

    # Add title and labels
    plt.title('Local Outlier Factor (LOF)')
    plt.xlabel(cols[0])
    plt.ylabel(cols[1])
    plt.legend(loc='upper right')
    plt.show()

def plot_lof_2d_grad(df, cols, cat):
    # Normalize LOF scores for marker sizes
    max_score = df['lof_factor'].max()
    min_score = df['lof_factor'].min()
    df['Normalized_Score'] = (df['lof_factor'] - min_score) / (max_score - min_score) * 100  # Scale for visibility

    # Add a column to indicate outliers
    df['Outlier'] = df['lof'] == -1

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(df[cols[0]], df[cols[1]], c=df['Outlier'], s=df['Normalized_Score'], cmap='coolwarm', edgecolor='k', alpha=0.7)

    # Add colorbar and labels
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Outlier Factor')
    ax.set_title(f'LOF Outlier Detection For {cat}')
    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])

    plt.show()

def plot_lof_3d_circle(df, cols, cat):
    # Add a column to indicate outliers
    df['Outlier'] = df['lof'] == -1

    # Filter outliers
    outliers = df[df['Outlier']]

    # Normalize LOF factors for marker sizes
    max_score = outliers['lof_factor'].max()
    min_score = outliers['lof_factor'].min()
    outliers['Normalized_Score'] = (outliers['lof_factor'] - min_score) / (max_score - min_score) * 100  # Scale for visibility

    # Plotting
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df[cols[0]], df[cols[1]], df[cols[2]], color='k', s=3, label='Data points')
    ax.scatter(outliers[cols[0]], outliers[cols[1]], outliers[cols[2]], edgecolor='r', facecolor='none', s=outliers['Normalized_Score'], label='Outlier scores')

    # Add title and labels
    ax.set_title(f'Local Outlier Factor (LOF) For {cat}')
    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])
    ax.set_zlabel(cols[2])
    ax.legend(loc='upper right')
    plt.show()


def plot_lof_3d_grad(df, cols, cat):

    # Normalize LOF scores for marker sizes
    max_score = df['lof_factor'].max()
    min_score = df['lof_factor'].min()
    df['Normalized_Score'] = (df['lof_factor'] - min_score) / (max_score - min_score) * 100  # Scale for visibility

    # Add a column to indicate outliers
    df['Outlier'] = df['lof'] == -1

    # Plotting
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(df[cols[0]], df[cols[1]], df[cols[2]], c=df['Outlier'], s=df['Normalized_Score'], cmap='coolwarm', edgecolor='k', alpha=0.7)

    # Add colorbar and labels
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Outlier Factor')
    ax.set_title(f'LOF Outlier Detection For {cat}')
    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])
    ax.set_zlabel(cols[2])

    plt.show()