import plotly.graph_objects as go
import plotly.colors as pc

img_path="../../img/2/"

layout=go.Layout(
    template="plotly_dark",
    colorway=pc.qualitative.Plotly + pc.qualitative.Dark24,
    autosize=True,
    margin=go.layout.Margin(
        l=0,
        r=0,
        b=0,
        t=0,
        pad=0
    ),
    paper_bgcolor='rgb(40,40,40)',
    plot_bgcolor='rgb(40,40,40)',
)

config = {
    'displayModeBar': False,
    'scrollZoom': True
}
