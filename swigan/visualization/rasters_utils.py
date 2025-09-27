"""Module containing visualization methods for teh rasters."""

import numpy as np
import pandas as pd
import plotly.express as px


def plot_rasters_from_dataframe(
    input_df: pd.DataFrame,
    variable: str,
    date_col: str = "date_str",
    x_dim_col: str = "x",
    y_dim_col: str = "y",
) -> None:
    """Visualize the rasters over time steps."""
    frames = []
    unique_dates = sorted(input_df[date_col].unique())

    for date in unique_dates:
        df_date = input_df[input_df[date_col] == date]

        # Transform into a 2D [y, x]
        grid = df_date.pivot(index=y_dim_col, columns=x_dim_col, values=variable).sort_index(
            ascending=False
        )

        frames.append(grid.values)

    # Stack over time steps [T, H, W]
    frames_array = np.stack(frames, axis=0)

    fig = px.imshow(
        frames_array,
        animation_frame=0,
        color_continuous_scale="Viridis",
        labels={"animation_frame": "Mois"},
        origin="upper",
        aspect="equal",
        title=f"Animation de la variable {variable} (mois par mois)",
    )

    # Rename the frames with real dates.
    for i, f in enumerate(fig.frames):
        f.name = unique_dates[i]
    fig.layout.sliders[0].steps = [
        {
            "label": unique_dates[i],
            "method": "animate",
            "args": [
                [unique_dates[i]],
                {
                    "mode": "immediate",
                    "frame": {"duration": 500, "redraw": True},
                    "transition": {"duration": 0},
                },
            ],
        }
        for i in range(len(unique_dates))
    ]
    fig.show()
