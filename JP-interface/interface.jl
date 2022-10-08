module interface

import PyCall

Dmodel = PyCall.pyimport("../model/diffusion.Diffusion")

export async function init()
    global dmodel = Dmodel()
    await dmodel.create_model()
    await dmodel.load_dataset()
end

export function train()
    dmodel.train()
end

export function test(ipath, opath)
    dmodel.test(ipath, opath)
end