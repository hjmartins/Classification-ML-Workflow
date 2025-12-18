# models/charts_models.py
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, roc_curve


def confusion_matrix_plot(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=["Pred 0", "Pred 1"],
        y=["True 0", "True 1"],
        colorscale="Blues",
    )
    fig.update_layout(title="Confusion Matrix")
    return fig


def roc_curve_plot(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig = px.line(
        x=fpr,
        y=tpr,
        labels={"x": "False Positive Rate", "y": "True Positive Rate"},
        title="ROC Curve",
    )
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(dash="dash"),
    )
    return fig
