module components

import ../JP-interface/interface as Jpi
import Dash

export image_in = Dash.html_div(
    Dash.dcc_upload(
        id = "upload_image",
        children=html_div([
            "Drag and Drop or ",
            html_a("Select Files")
        ]),
        multiple = false ),
    Dash.htmlImg(
        id = "image_in_show", 
        style = Dict(
            "width" => "64px",
            "height" => "64px",
        ),
    ) do 
    Jpi.init()
end

export image_out = Dash.html_div(
    Dash.dcc_input(
        id = "out_path"
        placeholder = "output.png"
    ),
    Dash.htmlImg(
        id = "image_out_show",
        style = Dict(
            "width" => "64px",
            "height" => "64px",
        ),
) 
end

export train_button = Dash.html_button(
    id = "train_btn"
    value = "train"
) 
end

export test_button = Dash.html_button(
    id = "test_btn"
    value = "test"
) 
end