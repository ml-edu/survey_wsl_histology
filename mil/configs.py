from mil.deepmil import DeepMIL, DeepMILMulti
from mil.poolings import Average, Max, LogSumExp, Wildcat, GAP, GradCamPooling, GradCamPlusPooling

poolings = {
    'deepmil': DeepMIL,
    'deepmil_multi': DeepMILMulti,
    'average': Average,
    'max': Max,
    'lse': LogSumExp,
    'wildcat': Wildcat,
    # 'gradcam': GAP,
    'gradcam': GradCamPooling,
    # 'gradcampp': GAP
    'gradcampp': GradCamPlusPooling
}
