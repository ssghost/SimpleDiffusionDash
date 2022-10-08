module layout

import Dash
import ./components as Cpn

export create_layout = Dash.html_div() do
    Dash.title("Simple Diffusion Page"),
    Cpn.image_in(),
    Cpn.train_button(),
    Cpn.image_out(),
    Cpn.test_button()
end