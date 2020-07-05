from mil.deepmil import DeepMIL, DeepMILMulti
from mil.poolings import Average, Max, LogSumExp, Wildcat, GradCAMPP, GAP

poolings = {
    'deepmil': DeepMIL,
    'deepmil_multi': DeepMILMulti,
    'average': Average,
    'max': Max,
    'lse': LogSumExp,
    'wildcat': Wildcat,
    'gradcampp': GAP
    # 'gradcampp': GradCAMPP
}
