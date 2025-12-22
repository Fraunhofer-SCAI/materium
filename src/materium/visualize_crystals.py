import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from pymatgen.core.structure import Structure
import crystal_toolkit.components as ctc
import os
import re


def find_data_structure(base_path):
    """
    Scans the base_path for model and generation subdirectories and the CIF files within them.
    Returns a dictionary structured as: {model: {generation: [file_indices]}}.
    """
    data = {}
    if not os.path.isdir(base_path):
        print(f"Warning: Directory not found at {base_path}")
        return data

    for model_name in sorted(os.listdir(base_path)):
        model_path = os.path.join(base_path, model_name)
        if os.path.isdir(model_path):
            data[model_name] = {}
            for gen_name in sorted(os.listdir(model_path)):
                gen_path = os.path.join(model_path, gen_name)
                if os.path.isdir(gen_path):
                    cif_indices = []
                    # Use regex to find files like 'material_sample_gen_0.cif', 'material_sample_gen_12.cif' etc.
                    for filename in sorted(os.listdir(gen_path)):
                        match = re.match(r"material_sample_gen_(\d+)\.cif", filename)
                        if match:
                            cif_indices.append(int(match.group(1)))
                    if cif_indices:
                        data[model_name][gen_name] = sorted(cif_indices)
    return data


app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Define the base path to the data relative to this script's location
# The folder structure is:
# /your_project/
# |-- app.py (this file)
# |-- qe_input/
#     |-- model_A/
#         |-- generation_1/
#             |-- material_sample_gen_0.cif
#             |-- material_sample_gen_1.cif
#         |-- generation_2/
#     |-- model_B/
#         |-- ...
base_path = os.path.join(os.path.dirname(__file__), "qe_input")

data_structure = find_data_structure(base_path)
model_names = list(data_structure.keys())

structure_component = ctc.StructureMoleculeComponent(id="structure_viewer")


my_layout = html.Div(
    [
        html.H1("Crystal Structure Viewer"),
        html.P("Dynamically select a model and generation to view crystal structures."),
        html.Hr(),
        html.Div(
            [
                html.Label("Select Model:"),
                dcc.Dropdown(
                    id="model-dropdown",
                    options=[{"label": name, "value": name} for name in model_names],
                    value=model_names[0] if model_names else None,
                    clearable=False,
                ),
            ],
            style={"marginBottom": "20px"},
        ),
        html.Div(
            [
                html.Label("Select Generation:"),
                dcc.Dropdown(id="generation-dropdown", clearable=False),
            ],
            style={"marginBottom": "20px"},
        ),
        html.Div(
            [
                html.Label("Select Sample Index:"),
                dcc.Dropdown(id="sample-dropdown", clearable=False),
            ],
            style={"marginBottom": "20px"},
        ),
        html.Div(
            [html.H3("Structure"), structure_component.layout()],
            style={"border": "1px solid #ddd", "padding": "20px", "marginTop": "20px"},
        ),
    ],
    style={"fontFamily": "sans-serif", "padding": "20px"},
)

ctc.register_crystal_toolkit(app=app, layout=my_layout)


@app.callback(
    Output("generation-dropdown", "options"),
    Output("generation-dropdown", "value"),
    Input("model-dropdown", "value"),
)
def update_generation_dropdown(selected_model):
    if not selected_model:
        return [], None
    generations = list(data_structure.get(selected_model, {}).keys())
    options = [{"label": gen, "value": gen} for gen in generations]
    value = generations[0] if generations else None
    return options, value


@app.callback(
    Output("sample-dropdown", "options"),
    Output("sample-dropdown", "value"),
    Input("generation-dropdown", "value"),
    State("model-dropdown", "value"),
)
def update_sample_dropdown(selected_generation, selected_model):
    if not selected_generation or not selected_model:
        return [], None
    indices = data_structure.get(selected_model, {}).get(selected_generation, [])
    options = [{"label": f"Sample {i}", "value": i} for i in indices]
    value = indices[0] if indices else None
    return options, value


# Callback 3: Update the Structure Viewer based on all dropdowns
@app.callback(
    Output(structure_component.id(), "data"),
    Input("sample-dropdown", "value"),
    State("model-dropdown", "value"),
    State("generation-dropdown", "value"),
    prevent_initial_call=True,  # Don't run this on page load
)
def update_structure(selected_sample, selected_model, selected_generation):
    # This check ensures that all dropdowns have a valid selection before proceeding
    if selected_sample is None or not selected_model or not selected_generation:
        return None  # Return None to clear the viewer or keep it unchanged

    # Construct the full path to the selected CIF file
    file_name = f"material_sample_gen_{selected_sample}.cif"
    cif_path = os.path.join(base_path, selected_model, selected_generation, file_name)

    # Try to load the structure from the file and return it to the component
    try:
        if os.path.exists(cif_path):
            structure = Structure.from_file(cif_path)
            return structure
        else:
            print(f"Error: File not found at {cif_path}")
            return None
    except Exception as e:
        print(f"Error loading CIF file {cif_path}: {e}")
        return None


# --- 5. Run the App ---
if __name__ == "__main__":
    print("Starting server on http://localhost:8050")
    app.run(debug=True, port=8050)
