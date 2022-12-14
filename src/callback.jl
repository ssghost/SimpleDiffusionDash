module Callback

import ../JP-interface/interface as Jpi
import Dash

export callback!(
    app,
    [Dash.Input('upload_image', 'content')
     Dash.Input('train_button', 'n_clicks'),
     Dash.Input('test_button', 'n_clicks'),
     Dash.Input('out_path', 'value')],
    Dash.Output('image_in_show', 'children')
    Dash.Output('image_out_show', 'children')
) do image, btn1, btn2, opath, ishow, oshow
    if !(image isa Nothing)
        global ipath = Dash.get_asset_url(image)
        ishow(src=ipath)
    end

    if !(opath isa Nothing)
        oshow(src=opath)
    end

    if Dash.ctx.triggered_id == 'train_button'
        Jpi.train()
    end
    if Dash.ctx.triggered_id == 'test_button'
        Jpi.test(ipath, opath)
    end
end